#!/usr/bin/env python
"""A24 (configs/experiments/video_context_relation_v1.yaml): A23's Other / Same updates (fixed A23 manifests, 12 originals
+ 12 copies, CE + KL on originals, CE on copies) plus lambda * Rel, a teacher-referenced relation between consecutive
original windows of the same group: the student's probability change should follow the frozen own A08's change.

  --check            relation cases, groups, lambda = 0 replay of A23 Other_s0, teacher parity, forced OtherRel = SameRel,
                     optimiser / encoder, resume, Rel size and gradient share, timing and budget projection
  --queue K [K ...]  OtherRel_s0, SameRel_s0, OtherRel_s2, SameRel_s2 (seed 1 only after a met rule)
"""
from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.prepare_raw_video_context as prv  # noqa: E402
import scripts.train_context_intervention as tci  # noqa: E402
import scripts.train_ensemble_teacher as tet  # noqa: E402
import scripts.train_video_switch_head as vsh  # noqa: E402
from scripts._common import load_cfg  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import echo_routing.vit.target_mvit as tm  # noqa: E402
from echo_routing.evaluate.complementarity import sha256_file  # noqa: E402

A23_CONFIG = "configs/experiments/video_context_intervention_v1.yaml"
ARMS = {"OtherRel": "Other", "SameRel": "Same"}
BASE, GROUP = tci.BASE, 4
now, write_json, ledger_hours, ledger_append, determinism, max_diff = tci.now, tci.write_json, tci.ledger_hours, tci.ledger_append, tci.determinism, tci.max_diff


# ----------------------------------------------------------------------------------------------- relation
def relation(z: torch.Tensor, zt: torch.Tensor) -> torch.Tensor:
    """Mean over the 9 within-group consecutive pairs of 0.5 * L1( (p_i - p_{i-1}) - (q_i - q_{i-1}) ); q detached."""
    p = F.softmax(z[:BASE].float(), -1).view(BASE // GROUP, GROUP, -1); q = F.softmax(zt.float().detach(), -1).view(BASE // GROUP, GROUP, -1)
    return 0.5 * ((p[:, 1:] - p[:, :-1]) - (q[:, 1:] - q[:, :-1])).abs().sum(-1).mean()


def relation_np(p: np.ndarray, q: np.ndarray) -> float:
    """Independent loop restatement on probability arrays (12, K)."""
    tot, n = 0.0, 0
    for g in range(BASE // GROUP):
        for i in range(1, GROUP):
            a, b = g * GROUP + i, g * GROUP + i - 1
            tot += 0.5 * float(np.abs((p[a] - p[b]) - (q[a] - q[b])).sum()); n += 1
    return tot / n


def relation_cases() -> dict:
    rng = np.random.default_rng(20261022); zq = rng.normal(size=(BASE, 5)); q = np.exp(zq) / np.exp(zq).sum(1, keepdims=True); out = {}
    z = torch.tensor(zq, dtype=torch.float32); out["p equals q gives 0"] = float(relation(z, z)) < 1e-7
    zs = rng.normal(size=(BASE, 5)); p = np.exp(zs) / np.exp(zs).sum(1, keepdims=True)
    out["torch equals the loop restatement"] = abs(float(relation(torch.tensor(zs, dtype=torch.float32), torch.tensor(zq, dtype=torch.float32))) - relation_np(p, q)) < 1e-6
    qs = np.tile(np.array([0.9, 0.025, 0.025, 0.025, 0.025]), (BASE, 1)); qs[2:4] = [0.025, 0.9, 0.025, 0.025, 0.025]   # a teacher switch inside group 0
    zt = torch.tensor(np.log(qs), dtype=torch.float32); out["a teacher switch followed by the student gives 0"] = float(relation(zt, zt)) < 1e-7
    off = qs.copy(); off[:, 0] -= 0.01; off[:, 4] += 0.01                                                                  # a constant offset from q
    out["a constant offset from q gives 0"] = float(relation(torch.tensor(np.log(off), dtype=torch.float32), zt)) < 1e-6
    jump = qs.copy(); jump[6] = [0.025, 0.025, 0.9, 0.025, 0.025]                                                          # a new student jump in group 1
    out["a new student jump is penalised"] = float(relation(torch.tensor(np.log(jump), dtype=torch.float32), zt)) > 0.1
    zero = qs.copy(); zero[2:4] = zero[0]                                                                                   # the student removes the teacher's switch
    out["removing the teacher's switch is penalised"] = abs(float(relation(torch.tensor(np.log(zero), dtype=torch.float32), zt)) - 0.5 * 2 * 0.875 / 9) < 1e-5   # one of 9 pairs
    cross = np.tile(np.array([0.9, 0.025, 0.025, 0.025, 0.025]), (BASE, 1)); cross[4:] = [0.025, 0.025, 0.025, 0.025, 0.9]   # a change between groups only
    out["a change between groups is never a pair"] = float(relation(torch.tensor(np.log(cross), dtype=torch.float32), torch.tensor(np.log(np.tile(cross[:1], (BASE, 1))), dtype=torch.float32))) < 1e-6
    out["ok"] = bool(all(out.values()))
    return out


def train_step(model, heads, opt, b, zt: torch.Tensor, lrs, o: dict, device, lam: float, keep: bool = False) -> dict:
    """A23's update (one forward over 24 windows) plus lam * Rel on the originals."""
    u8, slots, target = b[1], b[2], b[3]; t = target.to(device); model.train(); heads.train()
    z = model.classifier(model.embed(tm.to_rgb(u8.to(device, non_blocking=True)), tm.marker_planes(slots.to(device)))); zt = zt.to(device)
    lp = F.log_softmax(z[:BASE], -1); lpt = F.log_softmax(zt, -1); ce_o = F.cross_entropy(z[:BASE], t[:BASE]); kl = (lpt.exp() * (lpt - lp)).sum(-1).mean()
    ce_c = F.cross_entropy(z[BASE:], t[BASE:]); rel = relation(z, zt); loss = ce_o + kl + ce_c + lam * rel
    if not torch.isfinite(loss):
        raise FloatingPointError("non-finite loss")
    opt.zero_grad(set_to_none=True); loss.backward(); params = [p for g in opt.param_groups for p in g["params"] if p.grad is not None]
    gn = float(torch.nn.utils.clip_grad_norm_(params, float(o["grad_clip"])))
    if not np.isfinite(gn):
        raise FloatingPointError("non-finite gradient norm")
    enc = float(max((p.grad.abs().max() for p in model.encoder_parameters() if p.grad is not None), default=torch.tensor(0.0))) if keep else None
    opt.step(); pred = z.detach().argmax(1).cpu().numpy(); tn = target.numpy()
    rec = {"loss": float(loss.detach()), "ce_original": float(ce_o.detach()), "kl": float(kl.detach()), "ce_copy": float(ce_c.detach()), "rel": float(rel.detach()), "grad_norm": gn,
           "lr_encoder": lrs[0], "lr_new": lrs[1], "correct_original": int((pred[:BASE] == tn[:BASE]).sum()), "correct_copy": int((pred[BASE:] == tn[BASE:]).sum())}
    if keep:
        rec["encoder_grad_max"] = enc
    return rec


# ----------------------------------------------------------------------------------------------- context
class Ctx:
    def __init__(self, cfg: dict, config: str) -> None:
        self.cfg, self.config, self.vr = cfg, config, cfg["video_context_relation"]; p = self.vr["pinned"]
        for k in ("A23_config", "A23_manifest_index", "A23_screen"):
            if sha256_file(ROOT / p[k]["path"]) != p[k]["sha256"]:
                raise SystemExit(f"{k}: differs from its pin")
        self.a23 = tci.Ctx(load_cfg(SimpleNamespace(config=A23_CONFIG, set=[])), A23_CONFIG); self.total, self.warmup, self.o = self.a23.total, self.a23.warmup, self.a23.o
        self.lam = float(self.vr["relation"]["coefficient"].split("=")[1].split(",")[0])
        o = self.vr["outputs"]; self.out, self.ckpt_root, self.results, self.ledger = ROOT / o["root"], ROOT / o["checkpoints"], ROOT / o["results_dir"], ROOT / self.vr["budget"]["ledger"]

    def manifest(self, arm: str, seed: int) -> tuple[dict, str]:
        base = ARMS.get(arm, arm); m, sha = self.a23.manifest(base, seed)
        if sha != self.vr["pinned"]["A23_manifests"][f"{base}_s{seed}"]:
            raise SystemExit(f"{base}_s{seed}: the A23 manifest differs from its pin")
        return m, sha


def code_digest() -> str:
    h = hashlib.sha256()
    for p in (Path(__file__), Path(tci.__file__), Path(tet.__file__), Path(vsh.__file__), Path(tm.__file__)):
        h.update(Path(p).read_bytes())
    return h.hexdigest()


def identity(ctx: Ctx, arm: str, seed: int, msha: str, ssha: str, csha: str, lam: float) -> str:
    payload = {"experiment": ctx.cfg.get("experiment"), "sections": {k: ctx.vr[k] for k in ("arms", "relation", "loss", "training")}, "arm": arm, "seed": int(seed), "lambda": lam,
               "manifest_sha256": msha, "start_sha256": ssha, "teacher_cache_sha256": csha, "environment": prv.environment(), "total_updates": int(ctx.total), "code": code_digest()}
    return hashlib.sha256(json.dumps(json.loads(json.dumps(payload, default=str)), sort_keys=True).encode()).hexdigest()


def train(ctx: Ctx, arm: str, seed: int, device, workers: int) -> dict:
    m, msha = ctx.manifest(arm, seed); cache = ctx.a23.teacher(seed); model, heads, ssha = tci.student(ctx.a23, seed, device); ident = identity(ctx, arm, seed, msha, ssha, cache["sha256"], ctx.lam)
    key = f"{arm}_s{seed}"; rd, ck = ctx.out / "train" / key, ctx.ckpt_root / key; rd.mkdir(parents=True, exist_ok=True); ck.mkdir(parents=True, exist_ok=True)
    if (rd / "run_manifest.json").is_file() and json.loads((rd / "run_manifest.json").read_text()).get("identity") == ident and (ck / "final.pt").is_file():
        return {"key": key, "status": "finished"}
    opt = vsh.make_optimizer(model, heads, ctx.o)
    if not vsh.bound(opt, model, heads):
        raise SystemExit("the optimiser is not bound")
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); done, log = 0, []; ul = rd / "updates.jsonl"
    if (ck / "resume.pt").is_file():
        done, log = vsh.load_state(ck / "resume.pt", model, heads, opt, ident); lines = ul.read_text().splitlines()[:done] if ul.is_file() else []; ul.write_text("".join(x + "\n" for x in lines))
        with (rd / "resumes.jsonl").open("a") as fh:
            fh.write(json.dumps({"at_update": done, "time": now()}) + "\n")
    else:
        for p in (ul, rd / "train_log.jsonl", rd / "resumes.jsonl"):
            p.unlink(missing_ok=True)
    print(f"[{key}] identity {ident[:12]}; lambda {ctx.lam}; updates {done} -> {ctx.total}", flush=True); t0 = time.time(); acc = []; torch.cuda.reset_peak_memory_stats(device)
    for b in tci.loader(m, ctx.a23.frame_paths(m), list(range(done, ctx.total)), workers):
        tci.check_batch(m, b); u = int(b[0]); lrs = vsh.apply_schedule(model, opt, u, ctx.warmup, ctx.total, ctx.o)
        r = train_step(model, heads, opt, b, torch.from_numpy(cache["logits"][seed, u].copy()), lrs, ctx.o, device, ctx.lam); acc.append(r); done = u + 1
        with ul.open("a") as fh:
            fh.write(json.dumps({"u": u, **{k: (round(v, 7) if isinstance(v, float) else v) for k, v in r.items()}}) + "\n")
        if done % 25 == 0:
            write_json(rd / "progress.json", {"stage": f"{arm} student training (backward, GPU)", "identity": ident, "updates": done, "total": ctx.total, "seconds": time.time() - t0, "loss": r["loss"]})
        if done % ctx.warmup == 0:
            mean = lambda k: float(np.mean([a[k] for a in acc]))  # noqa: E731
            rec = {"updates": done, **{k: mean(k) for k in ("loss", "ce_original", "kl", "ce_copy", "rel", "grad_norm")}, "accuracy_original": sum(a["correct_original"] for a in acc) / (BASE * len(acc)),
                   "accuracy_copy": sum(a["correct_copy"] for a in acc) / (BASE * len(acc)), "lr_encoder_last": acc[-1]["lr_encoder"], "seconds_this_process": time.time() - t0,
                   "peak_gpu_memory_mib": torch.cuda.max_memory_allocated(device) / 2**20}
            log.append(rec); vsh.save_state(ck / "resume.pt", model, heads, opt, done, ident, log)
            with (rd / "train_log.jsonl").open("a") as fh:
                fh.write(json.dumps(rec) + "\n")
            print(f"[{key}] {done}/{ctx.total} loss {rec['loss']:.4f} CE orig {rec['ce_original']:.4f} KL {rec['kl']:.4f} CE copy {rec['ce_copy']:.4f} Rel {rec['rel']:.4f} "
                  f"acc orig {rec['accuracy_original']:.3f} copy {rec['accuracy_copy']:.3f} {rec['seconds_this_process']:.0f}s", flush=True); acc = []
    if done != ctx.total:
        raise SystemExit(f"stopped at update {done}")
    final = ck / "final.pt"; torch.save({"model": model.state_dict(), "heads": heads.state_dict()}, final)
    man = {"identity": ident, "arm": arm, "key": key, "seed": seed, "lambda": ctx.lam, "config_sha256": sha256_file(ROOT / ctx.config), "manifest_sha256": msha, "cache_sha256": cache["sha256"],
           "start_sha256": ssha, "environment": prv.environment(), "code": code_digest(), "updates": done, "epochs": log,
           "resumes": (rd / "resumes.jsonl").read_text().splitlines() if (rd / "resumes.jsonl").is_file() else [], "final_checkpoint": str(final), "final_sha256": sha256_file(final),
           "parameters": int(sum(p.numel() for p in model.parameters())), "encoder_forwards_per_window": 1, "seconds_last_process": time.time() - t0}
    write_json(rd / "run_manifest.json", man); (ck / "resume.pt").unlink(missing_ok=True); print(f"[{key}] done: final {man['final_sha256'][:12]}", flush=True)
    return {"key": key, "status": "finished", "final_sha256": man["final_sha256"]}


def queue(ctx: Ctx, keys: list[str], device, workers: int) -> None:
    chk = json.loads((ctx.out / "checks.json").read_text())
    if not chk["ok"]:
        raise SystemExit("the checks did not pass")
    spu = float(chk["timing"]["seconds_per_update"]); ev = float(chk["budget"]["evaluation_per_model_hours"])
    for i, key in enumerate(keys):
        spent = ledger_hours(ctx.ledger, "gpu"); proj = spent + len(keys[i:]) * (ctx.total * spu / 3600 + ev)
        print(f"[queue] before {key}: spent {spent:.3f}, projection with evaluation {proj:.3f} of {ctx.vr['budget']['gpu_hours_max']}", flush=True)
        if proj > float(ctx.vr["budget"]["gpu_hours_max"]):
            raise SystemExit("the projection exceeds the cap; nothing new starts")
        arm, s = key.rsplit("_s", 1); t0, started, ok, note = time.time(), now(), False, ""
        try:
            r = train(ctx, arm, int(s), device, workers); ok = True; note = json.dumps(r)
        except BaseException as e:  # noqa: BLE001 - recorded, then raised again
            note = f"{type(e).__name__}: {str(e)[:200]}"; raise
        finally:
            ledger_append(ctx.ledger, f"train {key}", "gpu", started, time.time() - t0, ok, note)


# ----------------------------------------------------------------------------------------------- checks
def batch_of(ctx: Ctx, arm: str, seed: int, u: int):
    m, _ = ctx.manifest(arm, seed); b = tci.Inputs(m, ctx.a23.frame_paths(m), [u])[0]; tci.check_batch(m, b); return b


def run(ctx: Ctx, arm: str, seed: int, us: list[int], device, lam: float, state_file: Path | None = None, inputs_of: str | None = None) -> dict:
    cache = ctx.a23.teacher(seed); model, heads, _ = tci.student(ctx.a23, seed, device); opt = vsh.make_optimizer(model, heads, ctx.o)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); losses, secs, enc, rels = [], [], [], []
    if state_file is not None and state_file.is_file():
        vsh.load_state(state_file, model, heads, opt, "check")
    for u in us:
        b = batch_of(ctx, inputs_of or arm, seed, u); lrs = vsh.apply_schedule(model, opt, u, ctx.warmup, ctx.total, ctx.o); torch.cuda.synchronize(); a = time.time()
        r = train_step(model, heads, opt, b, torch.from_numpy(cache["logits"][seed, u].copy()), lrs, ctx.o, device, lam, keep=True); torch.cuda.synchronize(); secs.append(time.time() - a)
        losses.append(r["loss"]); enc.append(r["encoder_grad_max"]); rels.append(r["rel"])
    if state_file is not None and not state_file.is_file():
        vsh.save_state(state_file, model, heads, opt, len(us), "check", [])
    return {"model": model, "losses": losses, "seconds": secs, "encoder_grads": enc, "rels": rels}


def gradient_share(ctx: Ctx, seed: int, us: list[int], device) -> list[dict]:
    """At the A08 start (schedule of update u): gradient norms of the A23 loss part and of lam * Rel, separately."""
    cache = ctx.a23.teacher(seed); out = []
    for u in us:
        model, heads, _ = tci.student(ctx.a23, seed, device); opt = vsh.make_optimizer(model, heads, ctx.o); vsh.apply_schedule(model, opt, u, ctx.warmup, ctx.total, ctx.o); model.train()
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); b = batch_of(ctx, "OtherRel", seed, u); zt = torch.from_numpy(cache["logits"][seed, u].copy()).to(device); t = b[3].to(device)
        z = model.classifier(model.embed(tm.to_rgb(b[1].to(device)), tm.marker_planes(b[2].to(device)))); lp = F.log_softmax(z[:BASE], -1); lpt = F.log_softmax(zt, -1)
        base = F.cross_entropy(z[:BASE], t[:BASE]) + (lpt.exp() * (lpt - lp)).sum(-1).mean() + F.cross_entropy(z[BASE:], t[BASE:]); rel = ctx.lam * relation(z, zt)
        params = [p for p in model.parameters() if p.requires_grad]
        gb = torch.autograd.grad(base, params, retain_graph=True, allow_unused=True); gr = torch.autograd.grad(rel, params, allow_unused=True)
        nb = float(torch.sqrt(sum((g.float() ** 2).sum() for g in gb if g is not None))); nr = float(torch.sqrt(sum((g.float() ** 2).sum() for g in gr if g is not None)))
        dot = float(sum((a.float() * c.float()).sum() for a, c in zip(gb, gr) if a is not None and c is not None))
        out.append({"u": u, "base_loss": float(base), "lambda_rel": float(rel), "base_grad_norm": nb, "rel_grad_norm": nr, "cosine": dot / max(1e-12, nb * nr)}); del model, heads, opt; torch.cuda.empty_cache()
    return out


def check(ctx: Ctx, device, workers: int) -> None:
    t0, started, ok, note, res = time.time(), now(), False, "", {"config_sha256": sha256_file(ROOT / ctx.config), "started": now(), "lambda": ctx.lam}
    try:
        res["relation_cases"] = relation_cases()
        for k, sha in ctx.vr["pinned"]["A23_weights"].items():
            man = json.loads((ctx.a23.out / "train" / k / "run_manifest.json").read_text()); fz = ROOT / "models_frozen/video_context_intervention_v1" / k / "final.pt"
            if man["final_sha256"] != sha or sha256_file(fz) != sha:
                raise SystemExit(f"A23 {k}: frozen weights differ from the pin")
        groups = {}
        for s in (0, 2):
            m, _ = ctx.manifest("OtherRel", s); m10, _ = ctx.a23.a10.manifest(s); sq, ci = m10["seq"][:ctx.total, :BASE], m10["centre_index"][:ctx.total, :BASE]; good = True
            for g in range(BASE // GROUP):
                a, c = sq[:, GROUP * g:GROUP * (g + 1)], ci[:, GROUP * g:GROUP * (g + 1)]; d = np.diff(c, axis=1); good &= bool((a == a[:, :1]).all() and ((d == 0) | (d == 1)).all())
            same = bool(np.array_equal(m["vid"][:, :BASE].shape, m10["vid"][:ctx.total].shape) and np.array_equal(m["target"][:, :BASE], m10["target"][:ctx.total]))
            groups[f"s{s}"] = {"groups_one_sequence_consecutive": good, "originals_equal_A10": same, "repeated_pairs": int(sum((np.diff(ci[:, GROUP * g:GROUP * (g + 1)], axis=1) == 0).sum() for g in range(3)))}
        res["groups"] = {**groups, "ok": all(v["groups_one_sequence_consecutive"] and v["originals_equal_A10"] for v in groups.values())}
        us = list(range(0, ctx.warmup + 2)); r0 = run(ctx, "OtherRel", 0, us, device, 0.0)
        log = {x["u"]: x for x in (json.loads(l) for l in (ctx.a23.out / "train" / "Other_s0" / "updates.jsonl").read_text().splitlines())}; d = [abs(a - float(log[u]["loss"])) for u, a in zip(us, r0["losses"])]
        res["lambda0_replay_of_A23_Other_s0"] = {"updates": [us[0], us[-1]], "loss_max_abs": max(d), "after_unfreezing_abs": d[-2:], "rel_at_start": r0["rels"][:3], "ok": max(d) <= 1e-4}; del r0; torch.cuda.empty_cache()
        state, _ = ctx.a23.start(0); cache = ctx.a23.teacher(0); a08 = tet.make_model(state, device).eval(); d0 = tet.state_digest(a08); par = []
        with torch.no_grad():
            for u in range(4):
                b = batch_of(ctx, "OtherRel", 0, u); par.append(float((a08(tm.to_rgb(b[1][:BASE].to(device)), tm.marker_planes(b[2][:BASE].to(device))).float().cpu() - torch.from_numpy(cache["logits"][0, u])).abs().max()))
        res["teacher"] = {"cached_against_online_max_abs": max(par), "windows": 48, "unchanged": tet.state_digest(a08) == d0, "ok": max(par) <= 1e-4 and tet.state_digest(a08) == d0}; del a08
        f1 = run(ctx, "OtherRel", 0, [0, 1], device, ctx.lam, inputs_of="SameRel"); r1 = run(ctx, "SameRel", 0, [0, 1], device, ctx.lam)
        res["forced_otherrel_equals_samerel"] = {"loss_max_abs": float(max(abs(a - b2) for a, b2 in zip(f1["losses"], r1["losses"]))), "params_max_abs": max_diff(f1["model"], r1["model"])}
        res["forced_otherrel_equals_samerel"]["ok"] = res["forced_otherrel_equals_samerel"]["loss_max_abs"] <= 1e-5 and res["forced_otherrel_equals_samerel"]["params_max_abs"] <= 1e-5; del f1, r1; torch.cuda.empty_cache()
        us = list(range(ctx.warmup - 2, ctx.warmup + 4)); sf = ctx.out / "check_resume.pt"; sf.unlink(missing_ok=True); straight = run(ctx, "OtherRel", 0, us, device, ctx.lam)
        repeat = run(ctx, "OtherRel", 0, us, device, ctx.lam); run(ctx, "OtherRel", 0, us[:3], device, ctx.lam, sf); second = run(ctx, "OtherRel", 0, us[3:], device, ctx.lam, sf)
        rp = max_diff(straight["model"], repeat["model"]); dif = max_diff(straight["model"], second["model"]); ld = float(max(abs(a - b2) for a, b2 in zip(straight["losses"][3:], second["losses"])))
        res["resume"] = {"updates": us, "loss_max_abs": ld, "params_max_abs": dif, "repeat_params_max_abs": rp, "ok": bool(ld <= 1e-4 and dif <= max(1e-5, 2 * rp))}
        res["encoder"] = {"grad_max_before_warmup": straight["encoder_grads"][:2], "grad_max_after": straight["encoder_grads"][2:], "ok": bool(all(g == 0.0 for g in straight["encoder_grads"][:2]) and all(g > 0 for g in straight["encoder_grads"][2:]))}
        model, heads, _ = tci.student(ctx.a23, 0, device); opt = vsh.make_optimizer(model, heads, ctx.o); res["optimiser_bound"] = {"ok": bool(vsh.bound(opt, model, heads))}; del model, heads, opt
        sf.unlink(missing_ok=True); spu = float(np.mean(straight["seconds"][2:])); res["rel_values_at_unfreezing"] = straight["rels"]; del straight, repeat, second; torch.cuda.empty_cache()
        res["gradient_share"] = gradient_share(ctx, 0, [0, ctx.warmup, ctx.warmup + 100], device)
        res["timing"] = {"seconds_per_update": spu, "peak_gpu_memory_mib": torch.cuda.max_memory_allocated(device) / 2**20}
        a08 = tet.make_model(state, device).eval(); x = torch.randint(0, 256, (32, 16, 224, 224, 3), dtype=torch.uint8); mk = tm.marker_planes(torch.zeros(32, dtype=torch.long).to(device))
        with torch.no_grad():
            a08(tm.to_rgb(x.to(device)), mk); torch.cuda.synchronize(); a = time.time()
            for _ in range(3):
                a08(tm.to_rgb(x.to(device)), mk)
            torch.cuda.synchronize(); spw = (time.time() - a) / 96
        ev = (7666 + 7021 + 18175) * spw / 3600; spent = ledger_hours(ctx.ledger, "gpu"); per_run = ctx.total * spu / 3600
        res["budget"] = {"spent": spent, "inference_seconds_per_window": spw, "evaluation_per_model_hours": ev, "per_run_hours": per_run, "projection_two_seeds": spent + 4 * (per_run + ev),
                         "seed1_reserve": 2 * (per_run + ev) + 0.05, "cap": ctx.vr["budget"]["gpu_hours_max"]}
        res["budget"]["fits_with_seed1"] = bool(res["budget"]["projection_two_seeds"] + res["budget"]["seed1_reserve"] <= res["budget"]["cap"])
        res["ok"] = bool(all(res[k]["ok"] for k in ("relation_cases", "groups", "lambda0_replay_of_A23_Other_s0", "teacher", "forced_otherrel_equals_samerel", "resume", "encoder", "optimiser_bound"))
                         and res["budget"]["projection_two_seeds"] <= res["budget"]["cap"]); res["finished"] = now()
        flat = json.loads(json.dumps(res, default=str))
        for p in (ctx.out / "checks.json", ctx.results / "checks.json"):
            p.parent.mkdir(parents=True, exist_ok=True); write_json(p, flat)
        print(json.dumps(flat, indent=1)[:6000], flush=True)
        if not res["ok"]:
            raise SystemExit("a check failed")
        ok = True; note = f"{spu:.3f} s/update, projection {res['budget']['projection_two_seeds']:.3f}"
    except BaseException as e:  # noqa: BLE001 - recorded, then raised again
        note = f"{type(e).__name__}: {str(e)[:200]}"; raise
    finally:
        ledger_append(ctx.ledger, "check (GPU job wall time)", "gpu", started, time.time() - t0, ok, note)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True); ap.add_argument("--workers", type=int, default=7)
    g = ap.add_mutually_exclusive_group(required=True); g.add_argument("--check", action="store_true"); g.add_argument("--queue", nargs="+"); g.add_argument("--cases", action="store_true")
    a = ap.parse_args(); a.set = []
    if a.cases:
        print(json.dumps(relation_cases(), indent=1)); return
    ctx = Ctx(load_cfg(a), a.config); device = torch.device("cuda"); determinism()
    if a.check:
        return check(ctx, device, a.workers)
    queue(ctx, a.queue, device, a.workers)


if __name__ == "__main__":
    main()
