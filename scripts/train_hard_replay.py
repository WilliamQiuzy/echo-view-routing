#!/usr/bin/env python
"""A20 training (configs/experiments/video_hard_replay_v1.yaml): A10 R's recipe on 16-window updates (A10's 12 pinned
windows + the arm's 4 replay windows), CE + KL to the cached own A08 teacher on all 16 windows, through A18's verified
own-teacher step.

  --check            GPU: cached against online teacher logits, teacher state, base windows, U/H constraints, forced H = U,
                     optimiser, encoder gradients, resume, timing and budget projection
  --queue K [K ...]  runs in order (U_s0, H_s0, U_s2, H_s2; U_s1, H_s1 only after a met promotion rule)
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
import scripts.b3_coverage as bc  # noqa: E402
import scripts.prepare_hard_replay as phr  # noqa: E402
import scripts.prepare_raw_video_context as prv  # noqa: E402
import scripts.report_raw_video_context as rr  # noqa: E402
import scripts.report_video_within_view as rw  # noqa: E402
import scripts.train_ensemble_teacher as tet  # noqa: E402
import scripts.train_raw_video_context as rvc  # noqa: E402
import scripts.train_video_switch_head as vsh  # noqa: E402
from scripts._common import load_cfg  # noqa: E402
from scripts.train_joint_visual_temporal import FramePaths  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

import echo_routing.vit.target_mvit as tm  # noqa: E402
from echo_routing.evaluate.complementarity import sha256_file  # noqa: E402

A10_CONFIG = "configs/experiments/video_switch_head_v1.yaml"
WINDOWS, BASE = 16, 12
now, write_json, ledger_hours, ledger_append, determinism, max_diff = vsh.now, vsh.write_json, vsh.ledger_hours, vsh.ledger_append, vsh.determinism, vsh.max_diff


class Ctx:
    def __init__(self, cfg: dict, config: str) -> None:
        self.cfg, self.config, self.vh = cfg, config, cfg["video_hard_replay"]
        self.a10 = vsh.Ctx(load_cfg(SimpleNamespace(config=A10_CONFIG, set=[])), A10_CONFIG); self.total, self.warmup, self.o = self.a10.total, self.a10.warmup, self.a10.o
        o = self.vh["outputs"]; self.out, self.ckpt_root, self.results, self.ledger = ROOT / o["root"], ROOT / o["checkpoints"], ROOT / o["results_dir"], ROOT / self.vh["budget"]["ledger"]
        self._m, self._fp = {}, None

    def manifest(self, arm: str, seed: int) -> tuple[dict, str]:
        key = f"{arm}_s{seed}"
        if key not in self._m:
            idx = json.loads((self.out / "manifests" / "index.json").read_text())[key]; p = self.out / "manifests" / idx["file"]
            if (got := sha256_file(p)) != idx["sha256"]:
                raise SystemExit(f"{key}: the manifest differs from its index")
            with np.load(p) as z:
                self._m[key] = ({k: z[k] for k in z.files}, got)
        return self._m[key]

    def frame_paths(self, m: dict) -> FramePaths:
        if self._fp is None:
            vids = {str(v) for v in m["videos"].tolist()}
            if vids - set(prv.allowlist(self.a10.rv, ROOT)):
                raise SystemExit("a manifest recording is not a training clip")
            self._fp = FramePaths(prv.frame_dirs(self.a10.paths, self.a10.rv, vids))
        return self._fp


class Inputs(Dataset):
    def __init__(self, m: dict, fp, updates: list[int]) -> None:
        self.m, self.fp, self.updates = m, fp, list(updates)

    def __len__(self) -> int:
        return len(self.updates)

    def __getitem__(self, i: int):
        u = self.updates[i]; return (u, *rvc.window_inputs(self.m, self.m["order_ordered"], self.fp, [(u, w) for w in range(WINDOWS)]))


def loader(m: dict, fp, updates: list[int], workers: int) -> DataLoader:
    return DataLoader(Inputs(m, fp, updates), batch_size=None, num_workers=workers, prefetch_factor=2 if workers else None, pin_memory=True)


def check_batch(m: dict, b) -> None:
    u = int(b[0])
    if not np.array_equal(b[4].numpy(), m["digest_ordered"][u]):
        raise SystemExit(f"update {u}: the decoded pixels differ from the prepared digests")
    if not np.array_equal(b[3].numpy(), m["target"][u]) or not np.array_equal(b[2].numpy(), m["marker"][u]):
        raise SystemExit(f"update {u}: targets or markers differ from the manifest")


def step(model, heads, opt, b, zt: torch.Tensor, lrs, o: dict, device) -> dict:
    """A18's own-teacher update on all 16 windows; base-12 and replay-4 CE / KL / accuracy logged from the same logits."""
    r = tet.train_step(model, heads, opt, b, zt, "own", lrs, o, device, keep=True); z = r.pop("logits"); t = b[3]
    lp, lpt = F.log_softmax(z, -1), F.log_softmax(zt, -1); ce = -lp[torch.arange(len(t)), t]; kl = (lpt.exp() * (lpt - lp)).sum(-1); ok = (z.argmax(1) == t).float()
    for name, sl in (("base", slice(0, BASE)), ("replay", slice(BASE, WINDOWS))):
        r[f"ce_{name}"], r[f"kl_{name}"], r[f"acc_{name}"] = float(ce[sl].mean()), float(kl[sl].mean()), float(ok[sl].mean())
    return r


def code_digest() -> str:
    h = hashlib.sha256()
    for p in (Path(__file__), Path(tet.__file__), Path(vsh.__file__), Path(rvc.__file__), Path(tm.__file__), Path(phr.__file__)):
        h.update(Path(p).read_bytes())
    return h.hexdigest()


def identity(ctx: Ctx, arm: str, seed: int, msha: str, ssha: str) -> str:
    payload = {"experiment": ctx.cfg.get("experiment"), "sections": {k: ctx.vh[k] for k in ("predecessors", "arms", "replay_pool", "sampling", "training")}, "arm": arm, "seed": int(seed),
               "manifest_sha256": msha, "start_sha256": ssha, "environment": prv.environment(), "total_updates": int(ctx.total), "code": code_digest()}
    return hashlib.sha256(json.dumps(json.loads(json.dumps(payload, default=str)), sort_keys=True).encode()).hexdigest()


def student(ctx: Ctx, seed: int, device):
    state, ssha = ctx.a10.start_state(seed); return vsh.make_model(state, device), vsh.build_heads(seed).to(device), ssha


def train(ctx: Ctx, arm: str, seed: int, device, workers: int) -> dict:
    m, msha = ctx.manifest(arm, seed); model, heads, ssha = student(ctx, seed, device); ident = identity(ctx, arm, seed, msha, ssha); key = f"{arm}_s{seed}"
    rd, ck = ctx.out / "train" / key, ctx.ckpt_root / key; rd.mkdir(parents=True, exist_ok=True); ck.mkdir(parents=True, exist_ok=True)
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
    for b in loader(m, ctx.frame_paths(m), list(range(done, ctx.total)), workers):
        check_batch(m, b); u = int(b[0]); lrs = vsh.apply_schedule(model, opt, u, ctx.warmup, ctx.total, ctx.o)
        r = step(model, heads, opt, b, torch.from_numpy(m["teacher"][u].copy()), lrs, ctx.o, device); acc.append(r); done = u + 1
        with ul.open("a") as fh:
            fh.write(json.dumps({"u": u, **{k: (round(v, 7) if isinstance(v, float) else v) for k, v in r.items()}}) + "\n")
        if done % 25 == 0:
            write_json(rd / "progress.json", {"stage": f"{arm} student training (backward, GPU)", "identity": ident, "updates": done, "total": ctx.total, "seconds": time.time() - t0, "loss": r["loss"]})
        if done % ctx.warmup == 0:
            mean = lambda k: float(np.mean([a[k] for a in acc]))  # noqa: E731
            rec = {"updates": done, **{k: mean(k) for k in ("loss", "ce", "kl", "grad_norm", "ce_base", "ce_replay", "kl_base", "kl_replay", "acc_base", "acc_replay")},
                   "accuracy": sum(a["correct"] for a in acc) / (WINDOWS * len(acc)), "lr_encoder_last": acc[-1]["lr_encoder"], "seconds_this_process": time.time() - t0,
                   "peak_gpu_memory_mib": torch.cuda.max_memory_allocated(device) / 2**20}
            log.append(rec); vsh.save_state(ck / "resume.pt", model, heads, opt, done, ident, log)
            with (rd / "train_log.jsonl").open("a") as fh:
                fh.write(json.dumps(rec) + "\n")
            print(f"[{key}] {done}/{ctx.total} loss {rec['loss']:.4f} CE base {rec['ce_base']:.4f} replay {rec['ce_replay']:.4f} acc base {rec['acc_base']:.3f} replay {rec['acc_replay']:.3f} "
                  f"{rec['seconds_this_process']:.0f}s", flush=True); acc = []
    if done != ctx.total:
        raise SystemExit(f"stopped at update {done}")
    final = ck / "final.pt"; torch.save({"model": model.state_dict(), "heads": heads.state_dict()}, final)
    man = {"identity": ident, "arm": arm, "key": key, "seed": seed, "config_sha256": sha256_file(ROOT / ctx.config), "manifest_sha256": msha, "start_sha256": ssha, "environment": prv.environment(),
           "code": code_digest(), "updates": done, "epochs": log, "resumes": (rd / "resumes.jsonl").read_text().splitlines() if (rd / "resumes.jsonl").is_file() else [],
           "final_checkpoint": str(final), "final_sha256": sha256_file(final), "seconds_last_process": time.time() - t0}
    write_json(rd / "run_manifest.json", man); (ck / "resume.pt").unlink(missing_ok=True); print(f"[{key}] done: final {man['final_sha256'][:12]}", flush=True)
    return {"key": key, "status": "finished", "final_sha256": man["final_sha256"]}


# ----------------------------------------------------------------------------------------------- checks
def run_updates(ctx: Ctx, m: dict, seed: int, us: list[int], device, state_file: Path | None = None) -> dict:
    model, heads, _ = student(ctx, seed, device); opt = vsh.make_optimizer(model, heads, ctx.o); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    if state_file is not None:
        vsh.load_state(state_file, model, heads, opt, "check")
    recs, grads, fp = [], [], ctx.frame_paths(m)
    for u in us:
        b = Inputs(m, fp, [u])[0]; check_batch(m, b); lrs = vsh.apply_schedule(model, opt, u, ctx.warmup, ctx.total, ctx.o)
        recs.append(step(model, heads, opt, b, torch.from_numpy(m["teacher"][u].copy()), lrs, ctx.o, device)); grads.append({n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None})
    return {"model": model, "heads": heads, "opt": opt, "records": recs, "grads": grads}


def check(ctx: Ctx, device, workers: int) -> None:
    t0, started, ok, note = time.time(), now(), False, ""; out = {"config_sha256": sha256_file(ROOT / ctx.config), "started": started}
    try:
        determinism(); s = 0; mU, _ = ctx.manifest("U", s); mH, _ = ctx.manifest("H", s); fp = ctx.frame_paths(mU); a10, _ = ctx.a10.manifest(s)
        same_base = all(np.array_equal(mx[k][:, :BASE], a10[k][:ctx.total]) for mx in (mU, mH) for k in ("frame", "gamma", "order_ordered", "marker", "target", "digest_ordered"))
        vmap = all(np.array_equal(mx["videos"][mx["vid"][:, :BASE]], a10["videos"][a10["vid"][:ctx.total]]) for mx in (mU, mH)); out["base_12_equal_A10"] = bool(same_base and vmap)
        own, sha = tet.teacher(_A18Ctx(ctx), s, device); d0 = tet.state_digest(own); par = {}
        for arm, mx in (("U", mU), ("H", mH)):
            mx_ = []
            for u in (0, 1, 700, ctx.total - 1):
                b = Inputs(mx, fp, [u])[0]; check_batch(mx, b)
                with torch.no_grad():
                    z = own(tm.to_rgb(b[1].to(device)), tm.marker_planes(b[2].to(device))).float().cpu().numpy()
                mx_.append((z, mx["teacher"][u]))
            par[arm] = tet.vsh.vso.parity(np.concatenate([a for a, _ in mx_]), np.concatenate([b for _, b in mx_]), 1e-4)
        out["cached_teacher_against_online"] = par; out["teacher"] = {"unchanged": tet.state_digest(own) == d0, "with_grad": int(sum(p.requires_grad for p in own.parameters()))}; del own
        P = bc.pool(_B3Ctx()); seqs = P["seqs"]; refs = rw.half_refs(seqs); at = {r: i for i, r in enumerate(refs)}; Wf = rr.half_windows(seqs, at, 5)
        W = {"seq": Wf["seq"], "centre": Wf["centre"], "s": Wf["s"], "marker": Wf["marker"], "labels16": Wf["labels16"]}; C = phr.candidates(P, W)
        target = np.asarray([int(seqs[int(W["seq"][r])]["labels"][int(W["centre"][r])]) for r in range(len(W["seq"]))]); ce = phr.difficulty(C, bc.pool_logits(_B3Ctx(), P, f"A08_s{s}"), target)
        S = phr.sample(C, ce, s); key = {tuple(r): g for g, r in enumerate(C["rows"].tolist())}
        gU = np.asarray([key[tuple(r)] for r in mU["pool_rows"].tolist()]); gH = np.asarray([key[tuple(r)] for r in mH["pool_rows"].tolist()])
        out["sampling_recomputed_equal"] = bool(np.array_equal(gU, S["U"]) and np.array_equal(gH, S["H"]))
        same_stream = bool(np.all(C["stream"][gU] == C["stream"][gH])); same_bucket = bool(np.all(C["short"][gU] == C["short"][gH]) and np.all(C["short"][gU] == (np.arange(ctx.total) % 2 == 0)))
        same_tuple = all(C["tuple"][a] == C["tuple"][b] for a, b in zip(gU, gH)); mixed = bool(np.all(C["mixed"][gU]) and np.all(C["mixed"][gH]))
        in_top = []
        for gu, gh in zip(gU, gH):
            strat = sorted([g for g in np.flatnonzero((C["stream"] == C["stream"][gu]) & (C["short"] == C["short"][gu]) & C["mixed"]) if C["tuple"][g] == C["tuple"][gu]], key=lambda g: (-float(ce[g]), int(C["k"][g])))
            in_top.append(gh in strat[:max(1, math.ceil(len(strat) / 4))])
        fit_only = bool(all(P["group"][int(C["stream"][g])] == "fit" for g in np.concatenate([gU, gH])))
        out["U_H_constraints"] = {"same_stream": same_stream, "same_bucket_and_parity": same_bucket, "same_tuple": same_tuple, "all_mixed": mixed, "H_in_top_quarter": bool(all(in_top)), "large_fit_only": fit_only}
        forced = {k: v.copy() for k, v in mH.items()}
        for k in ("vid", "frame", "gamma", "order_ordered", "marker", "target", "digest_ordered", "teacher"):
            forced[k][:, BASE:] = mU[k][:, BASE:]
        forced["pool_rows"], forced["videos"] = mU["pool_rows"].copy(), mU["videos"]
        A = run_updates(ctx, mU, s, [0, ctx.warmup], device); B = run_updates(ctx, forced, s, [0, ctx.warmup], device)
        out["forced_H_equals_U"] = {"loss_max_abs": float(max(abs(x["loss"] - y["loss"]) for x, y in zip(A["records"], B["records"]))), "params_max_abs": max_diff(A["model"], B["model"]),
                                    "grads_max_abs": tet.grad_diff(A["grads"], B["grads"])}
        out["encoder_gradient_after_unfreezing"] = bool(any(n.startswith("encoder") for n in A["grads"][1])); out["encoder_frozen_before"] = not any(n.startswith("encoder") for n in A["grads"][0])
        out["optimiser_bound"] = bool(vsh.bound(A["opt"], A["model"], A["heads"])); del A, B; torch.cuda.empty_cache()
        us = list(range(ctx.warmup, ctx.warmup + 6)); full = run_updates(ctx, mU, s, us, device); a3 = run_updates(ctx, mU, s, us[:3], device); tmp = ctx.out / "check_resume.pt"
        vsh.save_state(tmp, a3["model"], a3["heads"], a3["opt"], 3, "check", []); del a3; b3 = run_updates(ctx, mU, s, us[3:], device, state_file=tmp); tmp.unlink()
        rerun = run_updates(ctx, mU, s, us, device); rep = max_diff(full["model"], rerun["model"]); del rerun
        out["resume"] = {"loss_max_abs": float(max(abs(a["loss"] - c["loss"]) for a, c in zip(full["records"][3:], b3["records"]))), "params_max_abs": max_diff(full["model"], b3["model"]), "repeat_params_max_abs": rep}
        out["resume"]["ok"] = bool(out["resume"]["loss_max_abs"] <= 1e-4 and out["resume"]["params_max_abs"] <= max(1e-5, 2 * rep)); del full, b3; torch.cuda.empty_cache()
        model, heads, _ = student(ctx, s, device); opt = vsh.make_optimizer(model, heads, ctx.o); torch.manual_seed(s); torch.cuda.manual_seed_all(s); ts = []
        for i, bb in enumerate(loader(mU, fp, list(range(ctx.warmup, ctx.warmup + 23)), workers)):
            torch.cuda.synchronize(); a = time.time(); u = int(bb[0]); lrs = vsh.apply_schedule(model, opt, u, ctx.warmup, ctx.total, ctx.o)
            step(model, heads, opt, bb, torch.from_numpy(mU["teacher"][u].copy()), lrs, ctx.o, device); torch.cuda.synchronize()
            if i >= 3:
                ts.append(time.time() - a)
        spu = float(np.mean(ts)) * 1.05; del model, heads, opt
        spent = ledger_hours(ctx.ledger, "gpu") + (time.time() - t0) / 3600; proj = spent + 4 * ctx.total * spu / 3600 + 0.7; seed1 = 2 * ctx.total * spu / 3600 + 0.35
        out["timing"] = {"seconds_per_update": spu}; out["budget"] = {"spent": spent, "projection_four_runs_plus_evaluation": proj, "seed1_reserve": seed1, "cap": float(ctx.vh["budget"]["gpu_hours_max"])}
        f = out["forced_H_equals_U"]; u_h = out["U_H_constraints"]
        out["ok"] = bool(out["base_12_equal_A10"] and par["U"]["ok"] and par["H"]["ok"] and out["teacher"]["unchanged"] and out["teacher"]["with_grad"] == 0 and out["sampling_recomputed_equal"]
                         and all(u_h.values()) and f["loss_max_abs"] <= 1e-4 and f["params_max_abs"] <= max(1e-5, 2 * rep) and out["encoder_gradient_after_unfreezing"] and out["encoder_frozen_before"]
                         and out["optimiser_bound"] and out["resume"]["ok"] and proj <= out["budget"]["cap"])
        out["finished"] = now(); ok = out["ok"]; note = f"{spu:.3f} s/update, projection {proj:.3f}"
        for p in (ctx.out / "checks.json", ctx.results / "checks.json"):
            p.parent.mkdir(parents=True, exist_ok=True); write_json(p, json.loads(json.dumps(out, default=str)))
        print(json.dumps(out, indent=1, default=str), flush=True)
        if not ok:
            raise SystemExit("a check failed")
    except BaseException as e:  # noqa: BLE001 - recorded, then raised again
        note = f"{type(e).__name__}: {str(e)[:200]}"; raise
    finally:
        ledger_append(ctx.ledger, "check (GPU job wall time)", "gpu", started, time.time() - t0, ok, note)


class _A18Ctx:
    """The minimal context tet.teacher needs (A10's pinned A08 start states)."""

    def __init__(self, ctx: Ctx) -> None:
        self.a10 = ctx.a10


def _B3Ctx() -> bc.Ctx:
    return bc.Ctx(load_cfg(SimpleNamespace(config=phr.B3C_CONFIG, set=[])), phr.B3C_CONFIG)


def queue(ctx: Ctx, keys: list[str], device, workers: int) -> None:
    chk = json.loads((ctx.out / "checks.json").read_text())
    if not chk["ok"]:
        raise SystemExit("the checks did not pass")
    spu = float(chk["timing"]["seconds_per_update"])
    for i, key in enumerate(keys):
        arm, s = key.rsplit("_s", 1); spent = ledger_hours(ctx.ledger, "gpu"); proj = spent + len(keys[i:]) * ctx.total * spu / 3600 + 0.7
        print(f"[queue] before {key}: spent {spent:.3f}, projection {proj:.3f} of {ctx.vh['budget']['gpu_hours_max']}", flush=True)
        if proj > float(ctx.vh["budget"]["gpu_hours_max"]):
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
    a = ap.parse_args(); a.set = []; ctx = Ctx(load_cfg(a), a.config); device = torch.device("cuda"); determinism()
    if a.check:
        return check(ctx, device, a.workers)
    queue(ctx, a.queue, device, a.workers)


if __name__ == "__main__":
    main()
