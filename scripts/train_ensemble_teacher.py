#!/usr/bin/env python
"""A18 (configs/experiments/video_ensemble_teacher_v1.yaml): the three-teacher cache and the student E, which is A10 R's
computation with the KL target replaced by the averaged probabilities of the three frozen A08 models.

Every run strictly loads its seed's frozen A08 all_s{k} into the unchanged TargetMViT, reads A10's pinned VSH rows
0-1,349 (12 windows) through A10's loader, keeps A10's unused head group in AdamW, and follows A10's schedule and model RNG.
The teachers' logits come from a cache computed on the same loader batches, so no teacher forward runs during training.

  --cache            GPU: all three teachers' logits for every update and window of the E_s0 / E_s2 manifests
  --diagnose         CPU: teacher disagreement, q against the own teacher and CE by group on training windows
  --check            GPU: cache, q, the matched-control step against A10's live train_step, identical teachers, resume, timing
  --queue K [K ...]  runs in order (E_s0, E_s2)
"""
from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.prepare_raw_video_context as prv  # noqa: E402
import scripts.train_raw_video_context as rvc  # noqa: E402
import scripts.train_video_switch_head as vsh  # noqa: E402
from scripts._common import load_cfg  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import echo_routing.vit.target_mvit as tm  # noqa: E402
from echo_routing.evaluate.complementarity import sha256_file  # noqa: E402

A10_CONFIG = "configs/experiments/video_switch_head_v1.yaml"
SEEDS, TEACHERS, WINDOWS, C = (0, 2), (0, 1, 2), vsh.WINDOWS, 5
now, write_json, ledger_hours, ledger_append, determinism, make_model, max_diff = vsh.now, vsh.write_json, vsh.ledger_hours, vsh.ledger_append, vsh.determinism, vsh.make_model, vsh.max_diff


class Ctx:
    def __init__(self, cfg: dict, config: str) -> None:
        self.cfg, self.config, self.ve = cfg, config, cfg["video_ensemble_teacher"]; p = self.ve["pinned"]
        for k in ("A10_config", "A10_manifest_index"):
            if sha256_file(ROOT / p[k]["path"]) != p[k]["sha256"]:
                raise SystemExit(f"{k}: differs from its pin")
        self.a10 = vsh.Ctx(load_cfg(SimpleNamespace(config=A10_CONFIG, set=[])), A10_CONFIG); self.total, self.warmup, self.o = self.a10.total, self.a10.warmup, self.a10.o
        for j in TEACHERS:
            if self.a10.src["a08_weights"][j]["sha256"] != p["A08_teachers"][j]:
                raise SystemExit(f"teacher all_s{j}: A10's pin differs from A18's")
        o = self.ve["outputs"]; self.out, self.ckpt_root, self.results = ROOT / o["root"], ROOT / o["checkpoints"], ROOT / o["results_dir"]
        self.ledger = ROOT / self.ve["budget"]["ledger"]

    def manifest(self, seed: int) -> tuple[dict, str]:
        return self.a10.manifest(seed)


def teacher(ctx: Ctx, j: int, device):
    state, sha = ctx.a10.start_state(j); m = make_model(state, device).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m, sha


def state_digest(model) -> str:
    h = hashlib.sha256()
    for k, v in model.state_dict().items():                  # parameters and buffers
        h.update(k.encode()); h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def log_q(zs: torch.Tensor) -> torch.Tensor:
    """zs (3, N, 5) teacher logits -> log of the equal-weight probability average, by logsumexp over the members."""
    return torch.logsumexp(F.log_softmax(zs, -1), 0) - math.log(zs.shape[0])


def cache_path(ctx: Ctx, seed: int) -> Path:
    return ctx.out / "cache" / f"teachers_s{seed}.npz"


def check_batch(m: dict, b) -> None:
    u = int(b[0]); rvc.check_digests(m, vsh.vso.GROUP_NAME, u, b[4])
    if not np.array_equal(b[3].numpy(), m["target"][u]) or not np.array_equal(b[2].numpy(), m["marker"][u]):
        raise SystemExit(f"update {u}: targets or markers differ from the manifest")


# ----------------------------------------------------------------------------------------------- teacher cache (GPU)
def build_cache(ctx: Ctx, workers: int) -> None:
    t0, started, ok, note, gsec = time.time(), now(), False, "", 0.0
    try:
        determinism(); device = torch.device("cuda"); T = {j: teacher(ctx, j, device) for j in TEACHERS}; d0 = {j: state_digest(T[j][0]) for j in TEACHERS}; U = ctx.total; index = {}
        (ctx.out / "cache").mkdir(parents=True, exist_ok=True)
        for s in SEEDS:
            m, msha = ctx.manifest(s); lg = np.zeros((len(TEACHERS), U, WINDOWS, C), np.float32); dig = np.zeros((U, WINDOWS, 32), np.uint8); seen = np.zeros(U, bool)
            for b in vsh.loader(m, ctx.a10.frame_paths(), list(range(U)), workers):
                u = int(b[0]); check_batch(m, b); rgb, mk = tm.to_rgb(b[1].to(device, non_blocking=True)), tm.marker_planes(b[2].to(device))
                torch.cuda.synchronize(); a = time.time()
                with torch.no_grad():
                    for j in TEACHERS:
                        lg[j, u] = T[j][0](rgb, mk).float().cpu().numpy()
                torch.cuda.synchronize(); gsec += time.time() - a; dig[u] = b[4].numpy(); seen[u] = True
                if (u + 1) % 135 == 0:
                    print(f"cache s{s}: {u + 1}/{U} updates ({time.time() - t0:.0f} s)", flush=True)
                    write_json(ctx.out / "cache" / "progress.json", {"stage": "teacher cache (three frozen A08 teachers, eval, no_grad; no training)", "seed": s, "updates": u + 1, "of": U,
                                                                     "windows_cached": int((u + 1) * WINDOWS), "seconds": round(time.time() - t0, 1)})
            if not seen.all():
                raise SystemExit(f"seed {s}: {int((~seen).sum())} updates were not cached")
            zs = torch.from_numpy(lg).reshape(len(TEACHERS), U * WINDOWS, C); lq = log_q(zs); q_direct = torch.softmax(zs, -1).mean(0)
            dq = float((lq.exp() - q_direct).abs().max()); sums = float((lq.exp().sum(-1) - 1).abs().max())
            if dq > 1e-6 or sums > 1e-5 or float(lq.exp().min()) < 0:
                raise SystemExit(f"seed {s}: the stable average differs from direct averaging ({dq}, {sums})")
            p = cache_path(ctx, s); p.parent.mkdir(parents=True, exist_ok=True)
            np.savez(p, logits=lg, logq=lq.reshape(U, WINDOWS, C).numpy(), digests=dig, manifest_sha256=msha, teacher_sha256=np.asarray([T[j][1] for j in TEACHERS]), rows=U)
            index[str(s)] = {"file": p.name, "sha256": sha256_file(p), "manifest_sha256": msha, "rows_used": U, "rows_stored_in_manifest": int(len(m["target"])), "windows": int(U * WINDOWS),
                             "q_minus_direct_max_abs": dq, "q_sum_minus_1_max_abs": sums}
            print(f"cache s{s}: done {json.dumps(index[str(s)])}", flush=True)
        if any(state_digest(T[j][0]) != d0[j] for j in TEACHERS):
            raise SystemExit("a teacher's state changed")
        rec = {"teachers": {f"all_s{j}": T[j][1] for j in TEACHERS}, "seeds": index, "teacher_states_unchanged": True, "config_sha256": sha256_file(ROOT / ctx.config), "gpu_seconds": gsec, "time": now()}
        write_json(ctx.out / "cache" / "index.json", rec); ok = True; note = f"{gsec:.0f} GPU s"
    except BaseException as e:  # noqa: BLE001 - recorded, then raised again
        note = f"{type(e).__name__}: {str(e)[:200]}"; raise
    finally:
        ledger_append(ctx.ledger, f"teacher cache (three frozen A08 teachers, GPU job wall time; {workers} loader workers inside)", "gpu", started, time.time() - t0, ok, note)


def load_cache(ctx: Ctx, seed: int) -> dict:
    idx = json.loads((ctx.out / "cache" / "index.json").read_text())["seeds"][str(seed)]; p = cache_path(ctx, seed)
    if sha256_file(p) != idx["sha256"] or idx["manifest_sha256"] != ctx.manifest(seed)[1]:
        raise SystemExit(f"seed {seed}: the teacher cache differs from its index or manifest")
    with np.load(p) as z:
        return {"logits": z["logits"], "digests": z["digests"], "sha256": idx["sha256"]}


# ----------------------------------------------------------------------------------------------- one update
def train_step(model, heads, opt, b, zt: torch.Tensor, mode: str, lrs, o: dict, device, keep: bool = False) -> dict:
    """A10 R's update with a cached KL target: mode 'own' uses the student's own A08 teacher logits (A10 R's target);
    mode 'E' uses the three teachers' averaged probabilities. zt is (12, 5) for 'own' and (3, 12, 5) for 'E'."""
    u8, slots, target = b[1], b[2], b[3]; t = target.to(device); model.train(); heads.train()
    rgb, mk = tm.to_rgb(u8.to(device, non_blocking=True)), tm.marker_planes(slots.to(device))
    e = model.embed(rgb, mk); z = model.classifier(e); zt = zt.to(device)
    ce = F.cross_entropy(z, t); lp = F.log_softmax(z, -1); lpt = F.log_softmax(zt, -1) if mode == "own" else log_q(zt)
    kl = (lpt.exp() * (lpt - lp)).sum(-1).mean(); loss = ce + kl
    if not torch.isfinite(loss):
        raise FloatingPointError("non-finite loss")
    opt.zero_grad(set_to_none=True); loss.backward(); params = [p for g in opt.param_groups for p in g["params"] if p.grad is not None]
    gn = float(torch.nn.utils.clip_grad_norm_(params, float(o["grad_clip"])))
    if not np.isfinite(gn):
        raise FloatingPointError("non-finite gradient norm")
    opt.step(); pred = z.detach().argmax(1).cpu().numpy()
    rec = {"loss": float(loss.detach()), "ce": float(ce.detach()), "kl": float(kl.detach()), "grad_norm": gn, "lr_encoder": lrs[0], "lr_new": lrs[1], "lr_head": lrs[2],
           "correct": int((pred == target.numpy()).sum())}
    if keep:
        rec["logits"] = z.detach().cpu()
    return rec


def target_of(cache: dict, seed: int, u: int, mode: str) -> torch.Tensor:
    return torch.from_numpy(cache["logits"][seed if mode == "own" else slice(None), u].copy())


def code_digest() -> str:
    h = hashlib.sha256()
    for p in (Path(__file__), Path(vsh.__file__), Path(rvc.__file__), Path(tm.__file__)):
        h.update(Path(p).read_bytes())
    return h.hexdigest()


def identity(ctx: Ctx, arm: str, seed: int, msha: str, ssha: str, csha: str) -> str:
    payload = {"experiment": ctx.cfg.get("experiment"), "sections": {k: ctx.ve[k] for k in ("predecessors", "student_E", "teacher", "loss", "data", "training")}, "arm": arm, "seed": int(seed),
               "manifest_sha256": msha, "cache_sha256": csha, "start_sha256": ssha, "environment": prv.environment(), "total_updates": int(ctx.total), "code": code_digest()}
    return hashlib.sha256(json.dumps(json.loads(json.dumps(payload, default=str)), sort_keys=True).encode()).hexdigest()


def student(ctx: Ctx, seed: int, device):
    state, ssha = ctx.a10.start_state(seed); return make_model(state, device), vsh.build_heads(seed).to(device), ssha


def train(ctx: Ctx, arm: str, seed: int, device, workers: int) -> dict:
    m, msha = ctx.manifest(seed); cache = load_cache(ctx, seed); model, heads, ssha = student(ctx, seed, device); ident = identity(ctx, arm, seed, msha, ssha, cache["sha256"])
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
    print(f"[{key}] identity {ident[:12]}; updates {done} -> {ctx.total}", flush=True); t0 = time.time(); acc = []; torch.cuda.reset_peak_memory_stats(device)
    for b in vsh.loader(m, ctx.a10.frame_paths(), list(range(done, ctx.total)), workers):
        check_batch(m, b); u = int(b[0])
        if not np.array_equal(b[4].numpy(), cache["digests"][u]):
            raise SystemExit(f"update {u}: the batch differs from the cached teacher request")
        lrs = vsh.apply_schedule(model, opt, u, ctx.warmup, ctx.total, ctx.o); r = train_step(model, heads, opt, b, target_of(cache, seed, u, "E"), "E", lrs, ctx.o, device)
        acc.append(r); done = u + 1
        with ul.open("a") as fh:
            fh.write(json.dumps({"u": u, **{k: (round(v, 7) if isinstance(v, float) else v) for k, v in r.items()}}) + "\n")
        if done % 25 == 0:
            write_json(rd / "progress.json", {"stage": "E student training (backward, GPU)", "identity": ident, "updates": done, "total": ctx.total, "seconds": time.time() - t0, "loss": r["loss"]})
        if done % ctx.warmup == 0:
            mean = lambda k: float(np.mean([a[k] for a in acc]))  # noqa: E731
            rec = {"updates": done, **{k: mean(k) for k in ("loss", "ce", "kl", "grad_norm")}, "accuracy": sum(a["correct"] for a in acc) / (WINDOWS * len(acc)),
                   "lr_encoder_last": acc[-1]["lr_encoder"], "seconds_this_process": time.time() - t0, "peak_gpu_memory_mib": torch.cuda.max_memory_allocated(device) / 2**20}
            log.append(rec); vsh.save_state(ck / "resume.pt", model, heads, opt, done, ident, log)
            with (rd / "train_log.jsonl").open("a") as fh:
                fh.write(json.dumps(rec) + "\n")
            print(f"[{key}] {done}/{ctx.total} loss {rec['loss']:.4f} CE {rec['ce']:.4f} KL {rec['kl']:.4f} acc {rec['accuracy']:.3f} {rec['seconds_this_process']:.0f}s", flush=True); acc = []
    if done != ctx.total:
        raise SystemExit(f"stopped at update {done}")
    final = ck / "final.pt"; torch.save({"model": model.state_dict(), "heads": heads.state_dict()}, final)
    man = {"identity": ident, "arm": arm, "key": key, "seed": seed, "config_sha256": sha256_file(ROOT / ctx.config), "manifest_sha256": msha, "cache_sha256": cache["sha256"], "start_sha256": ssha,
           "environment": prv.environment(), "code": code_digest(), "updates": done, "epochs": log, "resumes": (rd / "resumes.jsonl").read_text().splitlines() if (rd / "resumes.jsonl").is_file() else [],
           "final_checkpoint": str(final), "final_sha256": sha256_file(final), "seconds_last_process": time.time() - t0}
    write_json(rd / "run_manifest.json", man); (ck / "resume.pt").unlink(missing_ok=True); print(f"[{key}] done: final {man['final_sha256'][:12]}", flush=True)
    return {"key": key, "status": "finished", "final_sha256": man["final_sha256"]}


# ----------------------------------------------------------------------------------------------- diagnostic and checks
def diagnose(ctx: Ctx) -> None:
    t0, started, ok, note = time.time(), now(), False, ""
    try:
        out = {"note": "training windows only; descriptive, not evidence that the teacher is stronger; nothing is chosen from it", "seeds": {}}
        for s in SEEDS:
            m, _ = ctx.manifest(s); cache = load_cache(ctx, s); U = ctx.total; z = torch.from_numpy(cache["logits"][:, :U]); t = torch.from_numpy(m["target"][:U].astype(np.int64))
            p = torch.softmax(z, -1); am = p.argmax(-1); q = log_q(z.reshape(3, -1, C)).exp().reshape(U, WINDOWS, C); own = p[s]
            grp = np.repeat(np.arange(3), WINDOWS // 3)[None, :].repeat(U, 0)                    # windows 0-3 native, 4-7 change, 8-11 same-label join
            ent = lambda x: -(x * torch.log(x.clamp_min(1e-12))).sum(-1)  # noqa: E731
            r = {"windows": int(U * WINDOWS)}
            for g, name in ((0, "native"), (1, "change"), (2, "join"), (None, "all")):
                sel = torch.from_numpy(grp == g) if g is not None else torch.ones(U, WINDOWS, dtype=torch.bool)
                ce = lambda x: float(-torch.log(x.gather(-1, t.unsqueeze(-1)).squeeze(-1).clamp_min(1e-12))[sel].mean())  # noqa: E731
                r[name] = {"windows": int(sel.sum()), "members_disagree_argmax": float((~((am[0] == am[1]) & (am[1] == am[2])))[sel].float().mean()),
                           "q_argmax_differs_from_own": float((q.argmax(-1) != own.argmax(-1))[sel].float().mean()), "mean_abs_q_minus_own": float((q - own).abs().sum(-1)[sel].mean()),
                           "entropy_own": float(ent(own)[sel].mean()), "entropy_q": float(ent(q)[sel].mean()), "ce_own": ce(own), "ce_q": ce(q),
                           "ce_members": [ce(p[j]) for j in TEACHERS], "accuracy_own": float((own.argmax(-1) == t)[sel].float().mean()), "accuracy_q": float((q.argmax(-1) == t)[sel].float().mean())}
            out["seeds"][str(s)] = r; print(f"seed {s}: {json.dumps(r)}", flush=True)
        for p_ in (ctx.out / "training_diagnostic.json", ctx.results / "training_diagnostic.json"):
            p_.parent.mkdir(parents=True, exist_ok=True); write_json(p_, out)
        ok = True
    except BaseException as e:  # noqa: BLE001 - recorded, then raised again
        note = f"{type(e).__name__}: {str(e)[:200]}"; raise
    finally:
        ledger_append(ctx.ledger, "training-window teacher diagnostic (CPU)", "cpu", started, time.time() - t0, ok, note)


def run_path(ctx: Ctx, seed: int, us: list[int], device, live: bool, mode: str = "own", cache: dict | None = None, state_file: Path | None = None, keep: bool = True, stack_own: bool = False) -> dict:
    """Updates us from the seed's start through A10's own live-teacher train_step (live) or this round's cached step."""
    m, _ = ctx.manifest(seed); model, heads, _ = student(ctx, seed, device); own = teacher(ctx, seed, device)[0] if live else None
    opt = vsh.make_optimizer(model, heads, ctx.o); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    if state_file is not None:
        vsh.load_state(state_file, model, heads, opt, "check")
    recs, grads = [], []
    for u in us:
        b = vsh.Inputs(m, ctx.a10.frame_paths(), [u])[0]; check_batch(m, b); lrs = vsh.apply_schedule(model, opt, u, ctx.warmup, ctx.total, ctx.o)
        if live:
            r = vsh.train_step(model, own, heads, opt, b, m["edge"][u], "R", 0.0, {"pos": 1.0, "neg": 1.0}, lrs, ctx.o, device, keep=keep)
        else:
            zt = target_of(cache, seed, u, mode)
            if stack_own:
                zt = torch.from_numpy(np.stack([cache["logits"][seed, u]] * 3))
            r = train_step(model, heads, opt, b, zt, "E" if (mode == "E" or stack_own) else "own", lrs, ctx.o, device, keep=keep)
        recs.append(r); grads.append({n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None})
    return {"model": model, "heads": heads, "opt": opt, "records": recs, "grads": grads, "rng": torch.cuda.get_rng_state_all(), "teacher": own}


def grad_diff(a: list, b: list) -> float:
    return float(max(max(float((x[k] - y[k]).abs().max()) for k in x) if set(x) == set(y) else float("inf") for x, y in zip(a, b)))


def check(ctx: Ctx, device, workers: int) -> None:
    t0, started, ok, note = time.time(), now(), False, ""; out = {"config_sha256": sha256_file(ROOT / ctx.config), "started": started}
    try:
        determinism(); s = 0; m, _ = ctx.manifest(s); cache = load_cache(ctx, s); fp = ctx.a10.frame_paths()
        T = {j: teacher(ctx, j, device) for j in TEACHERS}; d0 = {j: state_digest(T[j][0]) for j in TEACHERS}; live_diff = 0.0; ident_ok = True
        for u in (0, 1, ctx.warmup, 700, ctx.total - 1):
            b = vsh.Inputs(m, fp, [u])[0]; check_batch(m, b); ident_ok &= bool(np.array_equal(b[4].numpy(), cache["digests"][u]))
            with torch.no_grad():
                rgb, mk = tm.to_rgb(b[1].to(device)), tm.marker_planes(b[2].to(device))
                for j in TEACHERS:
                    live_diff = max(live_diff, float((T[j][0](rgb, mk).float().cpu() - torch.from_numpy(cache["logits"][j, u])).abs().max()))
        out["cache_against_live"] = {"updates": 5, "max_abs_logit_difference": live_diff, "request_identity": ident_ok}
        out["teachers"] = {"unchanged": all(state_digest(T[j][0]) == d0[j] for j in TEACHERS), "with_grad": int(sum(p.requires_grad for j in TEACHERS for p in T[j][0].parameters()))}
        stu, _, _ = student(ctx, s, device); stu.eval(); b = vsh.Inputs(m, fp, [0])[0]
        with torch.no_grad():
            mk = tm.marker_planes(b[2].to(device)); out["own_teacher_equals_start"] = bool(torch.equal(stu(tm.to_rgb(b[1].to(device)), mk), T[s][0](tm.to_rgb(b[1].to(device)), mk)))
        del stu, T; torch.cuda.empty_cache()
        z = torch.from_numpy(cache["logits"][:, :50]).reshape(3, -1, C); lq = log_q(z); qd = torch.softmax(z, -1).mean(0)
        out["q"] = {"min": float(lq.exp().min()), "sum_minus_1_max_abs": float((lq.exp().sum(-1) - 1).abs().max()), "stable_minus_direct_max_abs": float((lq.exp() - qd).abs().max())}
        res = {}
        for name, us in (("initial", [0, 1]), ("after_unfreezing", [ctx.warmup, ctx.warmup + 1])):
            A = run_path(ctx, s, us, device, live=True); A2 = run_path(ctx, s, us, device, live=True); B = run_path(ctx, s, us, device, live=False, cache=cache)
            rep = max_diff(A["model"], A2["model"])
            res[name] = {"loss_max_abs": float(max(abs(x["loss"] - y["loss"]) for x, y in zip(A["records"], B["records"]))), "ce_kl_max_abs": float(max(max(abs(x["ce"] - y["ce"]), abs(x["kl"] - y["kl"])) for x, y in zip(A["records"], B["records"]))),
                         "logits_max_abs": float(max(float((x["logits"] - y["logits"]).abs().max()) for x, y in zip(A["records"], B["records"]))), "grads_max_abs": grad_diff(A["grads"], B["grads"]),
                         "params_max_abs": max_diff(A["model"], B["model"]), "repeat_params_max_abs": rep, "cuda_rng_equal": all(torch.equal(x, y) for x, y in zip(A["rng"], B["rng"])),
                         "encoder_grad_present": bool(any(n.startswith("encoder") and n in B["grads"][-1] for n, _ in B["model"].named_parameters()))}
            res[name]["ok"] = bool(res[name]["loss_max_abs"] <= 1e-4 and res[name]["params_max_abs"] <= max(1e-5, 2 * rep) and res[name]["cuda_rng_equal"])
            del A, A2, B; torch.cuda.empty_cache()
        out["matched_control"] = res
        X = run_path(ctx, s, [ctx.warmup], device, live=False, cache=cache); Y = run_path(ctx, s, [ctx.warmup], device, live=False, cache=cache, stack_own=True)
        out["identical_teachers_reduce_to_one"] = {"kl_abs": abs(X["records"][0]["kl"] - Y["records"][0]["kl"]), "grads_max_abs": grad_diff(X["grads"], Y["grads"]), "params_max_abs": max_diff(X["model"], Y["model"])}
        out["identical_teachers_reduce_to_one"]["ok"] = bool(out["identical_teachers_reduce_to_one"]["kl_abs"] <= 1e-5 and out["identical_teachers_reduce_to_one"]["params_max_abs"] <= 1e-5)
        mdl, hd, _ = student(ctx, s, device); out["optimiser_bound"] = bool(vsh.bound(vsh.make_optimizer(mdl, hd, ctx.o), mdl, hd)); del X, Y, mdl, hd; torch.cuda.empty_cache()
        us = list(range(ctx.warmup, ctx.warmup + 6)); full = run_path(ctx, s, us, device, live=False, mode="E", cache=cache, keep=False); a3 = run_path(ctx, s, us[:3], device, live=False, mode="E", cache=cache, keep=False)
        tmp = ctx.out / "check_resume.pt"; vsh.save_state(tmp, a3["model"], a3["heads"], a3["opt"], 3, "check", []); del a3
        b3 = run_path(ctx, s, us[3:], device, live=False, mode="E", cache=cache, state_file=tmp, keep=False); tmp.unlink()
        rerun = run_path(ctx, s, us, device, live=False, mode="E", cache=cache, keep=False); rr_ = max_diff(full["model"], rerun["model"]); del rerun
        out["resume_E"] = {"loss_max_abs": float(max(abs(a["loss"] - c["loss"]) for a, c in zip(full["records"][3:], b3["records"]))), "params_max_abs": max_diff(full["model"], b3["model"]), "rerun_params_max_abs": rr_}
        out["resume_E"]["ok"] = bool(out["resume_E"]["loss_max_abs"] <= 1e-4 and out["resume_E"]["params_max_abs"] <= max(1e-5, 2 * rr_)); del full, b3; torch.cuda.empty_cache()
        model, heads, _ = student(ctx, s, device); opt = vsh.make_optimizer(model, heads, ctx.o); torch.manual_seed(s); torch.cuda.manual_seed_all(s); ts = []
        for i, bb in enumerate(vsh.loader(m, fp, list(range(ctx.warmup, ctx.warmup + 23)), workers)):
            torch.cuda.synchronize(); a = time.time(); u = int(bb[0]); lrs = vsh.apply_schedule(model, opt, u, ctx.warmup, ctx.total, ctx.o)
            train_step(model, heads, opt, bb, target_of(cache, s, u, "E"), "E", lrs, ctx.o, device); torch.cuda.synchronize()
            if i >= 3:
                ts.append(time.time() - a)
        spu = float(np.mean(ts)) * 1.05; del model, heads, opt
        spent = ledger_hours(ctx.ledger, "gpu") + (time.time() - t0) / 3600; proj = spent + 2 * ctx.total * spu / 3600 + 0.5; reserve = 2 * ctx.total * spu / 3600 + 0.4
        out["timing"] = {"seconds_per_update": spu}; out["budget"] = {"spent": spent, "projection_two_students_plus_evaluation": proj, "seed1_reserve": reserve, "cap": float(ctx.ve["budget"]["gpu_hours_max"])}
        mc = out["matched_control"]
        out["ok"] = bool(live_diff == 0.0 and ident_ok and out["teachers"]["unchanged"] and out["teachers"]["with_grad"] == 0 and out["own_teacher_equals_start"] and out["q"]["min"] >= 0
                         and out["q"]["sum_minus_1_max_abs"] <= 1e-5 and out["q"]["stable_minus_direct_max_abs"] <= 1e-6 and mc["initial"]["ok"] and mc["after_unfreezing"]["ok"]
                         and mc["after_unfreezing"]["encoder_grad_present"] and out["identical_teachers_reduce_to_one"]["ok"] and out["optimiser_bound"] and out["resume_E"]["ok"] and proj <= out["budget"]["cap"])
        out["finished"] = now(); ok = out["ok"]; note = f"{spu:.3f} s/update, projection {proj:.3f}"
        for p in (ctx.out / "checks.json", ctx.results / "checks.json"):
            p.parent.mkdir(parents=True, exist_ok=True); write_json(p, out)
        print(json.dumps(out, indent=1, default=str), flush=True)
        if not ok:
            raise SystemExit("a check failed")
    except BaseException as e:  # noqa: BLE001 - recorded, then raised again
        note = f"{type(e).__name__}: {str(e)[:200]}"; raise
    finally:
        ledger_append(ctx.ledger, "check", "gpu", started, time.time() - t0, ok, note)


def queue(ctx: Ctx, keys: list[str], device, workers: int) -> None:
    chk = json.loads((ctx.out / "checks.json").read_text())
    if not chk["ok"]:
        raise SystemExit("the checks did not pass")
    spu = float(chk["timing"]["seconds_per_update"])
    for i, key in enumerate(keys):
        arm, s = key.rsplit("_s", 1); spent = ledger_hours(ctx.ledger, "gpu"); proj = spent + len(keys[i:]) * ctx.total * spu / 3600 + 0.5
        print(f"[queue] before {key}: spent {spent:.3f}, projection {proj:.3f} of {ctx.ve['budget']['gpu_hours_max']}", flush=True)
        if proj > float(ctx.ve["budget"]["gpu_hours_max"]):
            raise SystemExit("the projection exceeds the cap; nothing new starts")
        t0, started, ok, note = time.time(), now(), False, ""
        try:
            r = train(ctx, arm, int(s), device, workers); ok = True; note = json.dumps(r)
        except BaseException as e:  # noqa: BLE001 - recorded, then raised again
            note = f"{type(e).__name__}: {str(e)[:200]}"; raise
        finally:
            ledger_append(ctx.ledger, f"train {key}", "gpu", started, time.time() - t0, ok, note)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True); ap.add_argument("--workers", type=int, default=8)
    g = ap.add_mutually_exclusive_group(required=True); g.add_argument("--cache", action="store_true"); g.add_argument("--diagnose", action="store_true"); g.add_argument("--check", action="store_true")
    g.add_argument("--queue", nargs="+")
    a = ap.parse_args(); a.set = []; ctx = Ctx(load_cfg(a), a.config)
    if a.cache:
        return build_cache(ctx, a.workers)
    if a.diagnose:
        return diagnose(ctx)
    device = torch.device("cuda"); determinism()
    if a.check:
        return check(ctx, device, a.workers)
    queue(ctx, a.queue, device, a.workers)


if __name__ == "__main__":
    main()
