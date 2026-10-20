#!/usr/bin/env python
"""Check, train and queue A12's paired arms (configs/experiments/video_processed_input_v1.yaml).

A10 R's computation, unchanged: the seed's frozen A08 all_s{k} strictly loaded, A10's unused head group kept in AdamW for an
identical optimiser structure, A10's VSH rows 0-1,349 (12 windows), CE + KL(teacher || student), A10's schedule and model RNG.
Raw: the student reads V01's original pixels. Pro: the student reads Pro pixels (fixed fallback). The teacher always reads
the original pixels of the same windows. Both pixel streams are digest-checked against the prepared manifests.

  --check            GPU: source parity, pairing, digests, gradients, teacher, optimiser, resume, timing, projection
  --queue K [K ...]  runs in order (Raw_s0, Pro_s0, ...)
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
import scripts.prepare_video_sequence_objective as vpr  # noqa: E402
import scripts.processed_input as pi  # noqa: E402
import scripts.train_raw_video_context as rvc  # noqa: E402
import scripts.train_video_switch_head as vsh  # noqa: E402
from scripts._common import load_cfg  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

import echo_routing.vit.target_mvit as tm  # noqa: E402
from echo_routing.evaluate.complementarity import sha256_file  # noqa: E402

ARMS = ("Raw", "Pro")
WINDOWS = 12
now, write_json, ledger_hours, ledger_append, determinism, make_model, max_diff = vsh.now, vsh.write_json, vsh.ledger_hours, vsh.ledger_append, vsh.determinism, vsh.make_model, vsh.max_diff


class Ctx:
    def __init__(self, cfg: dict, config: str) -> None:
        self.cfg, self.config, self.vp = cfg, config, cfg["video_processed_input"]; p10 = "configs/experiments/video_switch_head_v1.yaml"
        self.a10 = vsh.Ctx(load_cfg(SimpleNamespace(config=p10, set=[])), p10); self.total, self.warmup, self.o = self.a10.total, self.a10.warmup, self.a10.o
        o = self.vp["outputs"]; self.out, self.ckpt_root, self.results = ROOT / o["root"], ROOT / o["checkpoints"], ROOT / o["results_dir"]
        self.ledger = ROOT / self.vp["budget"]["ledger"]; self._m, self._cache = {}, None

    def manifest(self, seed: int) -> tuple[dict, str]:
        if seed not in self._m:
            e = json.loads((self.out / "manifests" / "index.json").read_text())["seeds"][str(seed)]; path = self.out / "manifests" / e["file"]
            if (got := sha256_file(path)) != e["sha256"] or e["a10_manifest_sha256"] != self.a10.manifest(seed)[1]:
                raise SystemExit(f"{path.name}: differs from its index or from A10's manifest")
            with np.load(path) as z:
                m = {k: z[k] for k in z.files}
            self._m[seed] = (m, got)
        return self._m[seed]

    def cache(self) -> pi.ProCache:
        if self._cache is None:
            self._cache = pi.ProCache(self.out / "cache", self.a10.frame_paths())
        return self._cache


class Inputs(Dataset):
    def __init__(self, ctx: Ctx, m: dict, updates: list[int], pro: bool) -> None:
        self.ctx, self.m, self.updates, self.pro = ctx, m, list(updates), pro

    def __len__(self) -> int:
        return len(self.updates)

    def __getitem__(self, i: int):
        u = self.updates[i]; pairs = [(u, w) for w in range(WINDOWS)]
        raw = rvc.window_inputs(self.m, self.m["order_ordered"], self.ctx.a10.frame_paths(), pairs)
        pro = pi.window_inputs(self.m, self.m["order_ordered"], self.ctx.cache(), pairs) if self.pro else None
        return (u, raw[0], raw[1], raw[2], raw[3], None if pro is None else pro[0], None if pro is None else pro[3])


def loader(ctx: Ctx, m: dict, updates: list[int], pro: bool, workers: int) -> DataLoader:
    return DataLoader(Inputs(ctx, m, updates, pro), batch_size=None, num_workers=workers, prefetch_factor=2 if workers else None, pin_memory=False)


def check_batch(m: dict, b, pro: bool) -> None:
    u = int(b[0]); rvc.check_digests(m, vsh.vso.GROUP_NAME, u, b[4])
    if pro and not np.array_equal(b[6].numpy(), m["digest_pro"][u]):
        raise SystemExit(f"update {u}: the Pro pixels differ from the prepared Pro digests")
    if not np.array_equal(b[3].numpy(), m["target"][u]) or not np.array_equal(b[2].numpy(), m["marker"][u]):
        raise SystemExit(f"update {u}: targets or markers differ from the manifest")


def models(ctx: Ctx, seed: int, device):
    state, ssha = ctx.a10.start_state(seed); student = make_model(state, device); teacher = make_model(state, device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return student, teacher, vsh.build_heads(seed).to(device), ssha


def train_step(model, teacher, opt, b, pro: bool, lrs, o: dict, device, keep: bool = False) -> dict:
    """A10 R's update: one student forward on the arm's pixels, the teacher on the original pixels, CE + KL."""
    raw_u8, slots, target = b[1], b[2], b[3]; t = target.to(device); mk = tm.marker_planes(slots.to(device)); model.train()
    stu = tm.to_rgb((b[5] if pro else raw_u8).to(device, non_blocking=True)); z = model(stu, mk)
    with torch.no_grad():
        zt = teacher(tm.to_rgb(raw_u8.to(device, non_blocking=True)), mk)
    ce = F.cross_entropy(z, t); lp, lpt = F.log_softmax(z, -1), F.log_softmax(zt, -1); kl = (lpt.exp() * (lpt - lp)).sum(-1).mean(); loss = ce + kl
    if not torch.isfinite(loss):
        raise FloatingPointError("non-finite loss")
    opt.zero_grad(set_to_none=True); loss.backward(); params = [p for g in opt.param_groups for p in g["params"] if p.grad is not None]
    gn = float(torch.nn.utils.clip_grad_norm_(params, float(o["grad_clip"])))
    if not np.isfinite(gn):
        raise FloatingPointError("non-finite gradient norm")
    opt.step(); pred = z.detach().argmax(1).cpu().numpy()
    rec = {"loss": float(loss.detach()), "ce": float(ce.detach()), "kl": float(kl.detach()), "grad_norm": gn, "lr_encoder": lrs[0], "lr_new": lrs[1], "correct": int((pred == target.numpy()).sum())}
    if keep:
        rec["logits"] = z.detach().cpu(); rec["teacher_logits"] = zt.detach().cpu()
    return rec


def identity(ctx: Ctx, arm: str, seed: int, msha: str, ssha: str) -> str:
    h = hashlib.sha256()
    for p in (Path(__file__), Path(pi.__file__), Path(vsh.__file__), Path(rvc.__file__), Path(tm.__file__)):
        h.update(Path(p).read_bytes())
    cache_sha = sha256_file(ctx.out / "cache" / "index.json")
    payload = {"experiment": ctx.cfg.get("experiment"), "sections": {k: ctx.vp[k] for k in ("predecessors", "pro", "mapping", "arms", "training")}, "arm": arm, "seed": int(seed),
               "manifest_sha256": msha, "cache_index_sha256": cache_sha, "start_sha256": ssha, "environment": prv.environment(), "total_updates": int(ctx.total), "code": h.hexdigest()}
    return hashlib.sha256(json.dumps(json.loads(json.dumps(payload, default=str)), sort_keys=True).encode()).hexdigest()


def train(ctx: Ctx, arm: str, seed: int, device, workers: int) -> dict:
    pro = arm == "Pro"; m, msha = ctx.manifest(seed); model, teacher, heads, ssha = models(ctx, seed, device); ident = identity(ctx, arm, seed, msha, ssha)
    key = f"{arm}_s{seed}"; rd, ck = ctx.out / "train" / key, ctx.ckpt_root / key; rd.mkdir(parents=True, exist_ok=True); ck.mkdir(parents=True, exist_ok=True)
    if (rd / "run_manifest.json").is_file() and json.loads((rd / "run_manifest.json").read_text()).get("identity") == ident and (ck / "final.pt").is_file():
        return {"key": key, "status": "finished"}
    opt = vsh.make_optimizer(model, heads, ctx.o)
    if not vsh.bound(opt, model, heads):
        raise SystemExit("the optimiser is not bound")
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); done, log = 0, []; ul = rd / "updates.jsonl"
    if (ck / "resume.pt").is_file():
        done, log = vsh.load_state(ck / "resume.pt", model, heads, opt, ident); lines = ul.read_text().splitlines()[:done] if ul.is_file() else []; ul.write_text("".join(x + "\n" for x in lines))
    else:
        for p in (ul, rd / "train_log.jsonl"):
            p.unlink(missing_ok=True)
    print(f"[{key}] identity {ident[:12]}; updates {done} -> {ctx.total}", flush=True); t0 = time.time(); acc = []; torch.cuda.reset_peak_memory_stats(device)
    for b in loader(ctx, m, list(range(done, ctx.total)), pro, workers):
        check_batch(m, b, pro); u = int(b[0]); lrs = vsh.apply_schedule(model, opt, u, ctx.warmup, ctx.total, ctx.o)
        r = train_step(model, teacher, opt, b, pro, lrs, ctx.o, device); acc.append(r); done = u + 1
        with ul.open("a") as fh:
            fh.write(json.dumps({"u": u, **{k: (round(v, 7) if isinstance(v, float) else v) for k, v in r.items()}}) + "\n")
        if done % 25 == 0:
            write_json(rd / "progress.json", {"identity": ident, "updates": done, "total": ctx.total, "seconds": time.time() - t0, "loss": r["loss"]})
        if done % ctx.warmup == 0:
            mean = lambda k: float(np.mean([a[k] for a in acc]))  # noqa: E731
            rec = {"updates": done, **{k: mean(k) for k in ("loss", "ce", "kl", "grad_norm")}, "accuracy": sum(a["correct"] for a in acc) / (WINDOWS * len(acc)),
                   "seconds_this_process": time.time() - t0, "peak_gpu_memory_mib": torch.cuda.max_memory_allocated(device) / 2**20}
            log.append(rec); vsh.save_state(ck / "resume.pt", model, heads, opt, done, ident, log)
            with (rd / "train_log.jsonl").open("a") as fh:
                fh.write(json.dumps(rec) + "\n")
            print(f"[{key}] {done}/{ctx.total} loss {rec['loss']:.4f} CE {rec['ce']:.4f} KL {rec['kl']:.4f} acc {rec['accuracy']:.3f} {rec['seconds_this_process']:.0f}s", flush=True); acc = []
    final = ck / "final.pt"; torch.save(model.state_dict(), final)
    man = {"identity": ident, "arm": arm, "key": key, "seed": seed, "config_sha256": sha256_file(ROOT / ctx.config), "manifest_sha256": msha, "start_sha256": ssha, "updates": done,
           "epochs": log, "final_checkpoint": str(final), "final_sha256": sha256_file(final), "fallback_windows_in_manifest": int(m["fallback_window"].sum())}
    write_json(rd / "run_manifest.json", man); (ck / "resume.pt").unlink(missing_ok=True); print(f"[{key}] done: final {man['final_sha256'][:12]}", flush=True)
    return {"key": key, "status": "finished", "final_sha256": man["final_sha256"]}


def run_updates(ctx: Ctx, arm: str, us: list[int], device, state_file: Path | None = None, keep: bool = False) -> dict:
    model, teacher, heads, _ = models(ctx, 0, device); opt = vsh.make_optimizer(model, heads, ctx.o); torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    if state_file is not None:
        vsh.load_state(state_file, model, heads, opt, "check")
    m, _ = ctx.manifest(0); recs = []
    for u in us:
        b = Inputs(ctx, m, [u], arm == "Pro")[0]; check_batch(m, b, arm == "Pro"); lrs = vsh.apply_schedule(model, opt, u, ctx.warmup, ctx.total, ctx.o)
        recs.append(train_step(model, teacher, opt, b, arm == "Pro", lrs, ctx.o, device, keep=keep))
    return {"model": model, "heads": heads, "opt": opt, "teacher": teacher, "records": recs}


def check(ctx: Ctx, device, workers: int) -> None:
    t0, started, ok, note = time.time(), now(), False, ""; out = {"config_sha256": sha256_file(ROOT / ctx.config), "started": started}
    try:
        determinism(); m, _ = ctx.manifest(0)
        for s in (0, 1, 2):
            ms, _ = ctx.manifest(s); m10 = ctx.a10.manifest(s)[0]; U = ctx.total
            for k in ("vid", "frame", "gamma", "target", "marker", "digest_ordered"):
                if not np.array_equal(ms[k], m10[k][:U]):
                    raise SystemExit(f"seed {s}: {k} differs from A10's rows")
        out["inputs_equal_a10_rows"] = True
        rec = {}
        for arm in ARMS:
            x = run_updates(ctx, arm, [0], device, keep=True); rec[arm] = (x["records"][0], torch.cuda.get_rng_state_all()); del x; torch.cuda.empty_cache()
        out["pairing"] = {"teacher_logits_identical": bool(torch.equal(rec["Raw"][0]["teacher_logits"], rec["Pro"][0]["teacher_logits"])),
                          "cuda_rng_identical": all(torch.equal(p, q) for p, q in zip(rec["Raw"][1], rec["Pro"][1])),
                          "student_logits_differ": not bool(torch.equal(rec["Raw"][0]["logits"], rec["Pro"][0]["logits"]))}
        b = Inputs(ctx, m, [0], True)[0]; model, teacher, heads, _ = models(ctx, 0, device); model.eval()
        with torch.no_grad():
            mk = tm.marker_planes(b[2].to(device)); out["teacher_equals_start"] = bool(torch.equal(model(tm.to_rgb(b[1].to(device)), mk), teacher(tm.to_rgb(b[1].to(device)), mk)))
        out["optimiser_bound"] = bool(vsh.bound(vsh.make_optimizer(model, heads, ctx.o), model, heads)); del model, teacher, heads
        x = run_updates(ctx, "Pro", [ctx.warmup], device); mx = lambda p: None if p.grad is None else float(p.grad.abs().max())  # noqa: E731
        out["gradients"] = {"projection": mx(x["model"].encoder.head[-1].weight), "conv_proj": mx(x["model"].encoder.conv_proj.weight), "classifier": mx(x["model"].classifier.weight),
                            "teacher_with_grad": int(sum(p.grad is not None for p in x["teacher"].parameters()))}; del x; torch.cuda.empty_cache()
        us = list(range(ctx.warmup, ctx.warmup + 6)); full = run_updates(ctx, "Pro", us, device); a3 = run_updates(ctx, "Pro", us[:3], device); tmp = ctx.out / "check_resume.pt"
        vsh.save_state(tmp, a3["model"], a3["heads"], a3["opt"], 3, "check", []); del a3; b3 = run_updates(ctx, "Pro", us[3:], device, state_file=tmp); tmp.unlink()
        rerun = run_updates(ctx, "Pro", us, device); rr_ = max_diff(full["model"], rerun["model"]); del rerun
        out["resume"] = {"loss_max_abs": float(max(abs(a["loss"] - c["loss"]) for a, c in zip(full["records"][3:], b3["records"]))), "params_max_abs": max_diff(full["model"], b3["model"]),
                         "rerun_params_max_abs": rr_}
        out["resume"]["ok"] = bool(out["resume"]["loss_max_abs"] <= 1e-4 and out["resume"]["params_max_abs"] <= max(1e-5, 2 * rr_)); del full, b3; torch.cuda.empty_cache()
        model, teacher, heads, _ = models(ctx, 0, device); opt = vsh.make_optimizer(model, heads, ctx.o); torch.manual_seed(0); torch.cuda.manual_seed_all(0); ts = []
        for i, bb in enumerate(loader(ctx, m, list(range(ctx.warmup, ctx.warmup + 23)), True, workers)):
            torch.cuda.synchronize(); a = time.time(); lrs = vsh.apply_schedule(model, opt, int(bb[0]), ctx.warmup, ctx.total, ctx.o); train_step(model, teacher, opt, bb, True, lrs, ctx.o, device)
            torch.cuda.synchronize()
            if i >= 3:
                ts.append(time.time() - a)
        spu = float(np.mean(ts)) * 1.05; del model, teacher, heads, opt
        spent = ledger_hours(ctx.ledger, "gpu") + (time.time() - t0) / 3600; proj = spent + 4 * ctx.total * spu / 3600 + 0.6
        out["timing"] = {"seconds_per_update": spu}; out["budget"] = {"spent": spent, "projection_four_runs_plus_evaluation": proj, "cap": float(ctx.vp["budget"]["gpu_hours_max"])}
        g = out["gradients"]
        out["ok"] = bool(out["pairing"]["teacher_logits_identical"] and out["pairing"]["cuda_rng_identical"] and out["teacher_equals_start"] and out["optimiser_bound"]
                         and g["projection"] and g["conv_proj"] and g["classifier"] and g["teacher_with_grad"] == 0 and out["resume"]["ok"] and proj <= out["budget"]["cap"])
        out["finished"] = now(); ok = out["ok"]; note = f"{spu:.3f} s/update, projection {proj:.3f}"
        for p in (ctx.out / "checks.json", ctx.results / "checks.json"):
            p.parent.mkdir(parents=True, exist_ok=True); write_json(p, out)
        print(json.dumps(out, indent=1, default=str), flush=True)
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
        arm, s = key.rsplit("_s", 1); spent = ledger_hours(ctx.ledger, "gpu"); proj = spent + len(keys[i:]) * ctx.total * spu / 3600 + 0.6
        print(f"[queue] before {key}: spent {spent:.3f}, projection {proj:.3f} of {ctx.vp['budget']['gpu_hours_max']}", flush=True)
        if proj > float(ctx.vp["budget"]["gpu_hours_max"]):
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
    g = ap.add_mutually_exclusive_group(required=True); g.add_argument("--check", action="store_true"); g.add_argument("--queue", nargs="+")
    a = ap.parse_args(); a.set = []; ctx = Ctx(load_cfg(a), a.config); determinism(); device = torch.device("cuda")
    if a.check:
        return check(ctx, device, a.workers)
    queue(ctx, a.queue, device, a.workers)


if __name__ == "__main__":
    main()
