#!/usr/bin/env python
"""A43 (configs/experiments/video_a5c_witness_mil_v1.yaml): a binary A5C witness specialist (A08 copy, marker-time local
tokens, gated attention, scalar score) trained end to end on whole native A4C / A5C bags with dense (D) or multiple-instance
(M) positive supervision, deployed only as a one-way change of original Base-A4C runs.

  --stage check    bags / plans / pixels / markers, Init (copy, head, score -2, online fit parity, patched forward), local
                   tokens / time alignment / hook removal, attention and loss references, two-pass vs retained-graph
                   gradients, skip-pool parity, binding, first-step zero / genuine gradients, repeat, forced dense M, 3 + 3
                   resume, one-way edit / eligible counts / Init = Base, timing / memory / disk and the projection (GPU)
  --stage train    --seed s: D_f0, M_f0, D_f1, M_f1, resumable (GPU)
  --stage freeze   --seed s --commit c --manifest-sha h: record that weights_s{s}.json (sha h) was committed in c
  --stage readout  --seed s: after the weights are committed, one opposite-fold scoring of the eligible rows (+ rule) (GPU);
                   --dry-run reads Init (every score -2) through the same tables and publishes nothing (CPU)
The frozen A08 + B3 path supplies q, Base and its runs; nothing here changes it.
"""
from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")     # A27's CPU-only trainer import would otherwise hide the GPU

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections import Counter  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.b3_coverage as bc  # noqa: E402
import scripts.diagnose_a5c_evidence as d32  # noqa: E402
import scripts.report_video_switch_head as rsh  # noqa: E402
import scripts.train_ensemble_teacher as tet  # noqa: E402
import scripts.train_interval_visual_adaptation as t28  # noqa: E402
import scripts.train_masked_feature as t40  # noqa: E402
import scripts.train_predicted_run_context as t38  # noqa: E402
import scripts.train_session_correction as t31  # noqa: E402
import scripts.train_wider_temporal_evidence as t37  # noqa: E402
from scripts._common import load_cfg  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

import echo_routing.temporal.interval_recognition as ir  # noqa: E402
import echo_routing.temporal.joint_visual as jv  # noqa: E402
import echo_routing.temporal.masked_feature as mf  # noqa: E402
import echo_routing.temporal.session_pairs as sp  # noqa: E402
import echo_routing.temporal.video_sequence_features as vsf  # noqa: E402
import echo_routing.temporal.witness_mil as wm  # noqa: E402
import echo_routing.vit.target_mvit as tm  # noqa: E402
import echo_routing.vit.target_patch_readout as tp  # noqa: E402
from echo_routing.evaluate.complementarity import sha256_file  # noqa: E402
from echo_routing.temporal.dev_train import lr_factor  # noqa: E402
from echo_routing.evaluate.failure_units import session_of  # noqa: E402

A31_CONFIG, A32_CONFIG = t40.A31_CONFIG, t40.A32_CONFIG
FOLDS, ARMS, U, WARM, SAVE_EVERY, MON_EVERY, LOG_EVERY, N_MON = (0, 1), ("D", "M"), 600, 60, 150, 150, 50, 8
LR = {"encoder_trunk": 1e-5, "marker_stem": 1e-4, "witness": 1e-3}
CLIP, HEAD_PARAMS, FROZEN = 1.0, 197_761, ("encoder.head.", "classifier.")
SCH = {"updates": U, "warmup": WARM}
LIMITS = {"loss": 1e-4, "params": 1e-5}
TOL = {"grad_rel": 1e-5, "formula": 1e-8, "attention_fp64": 1e-10, "replay_abs": 1e-5, "parity": 1e-4, "peak_gpu_gb": 40.0}
EXPECTED_BAGS = {0: {"A4C": 200, "A5C": 67, "mean": (11.59, 10.09), "max": (102, 100)}, 1: {"A4C": 200, "A5C": 75, "mean": (12.00, 12.37), "max": (72, 58)}}
EXPECTED_ELIGIBLE = {0: {0: {"runs": 749, "rows": 5030, "grid": 24585}, 1: {"runs": 771, "rows": 5437, "grid": 25774}}}   # source seed -> held-out fold
REQUIRED_GATES = {"isolation", "bags", "plan", "numerics", "pixels", "markers", "init_state", "init_head", "init_score_constant", "init_online_parity", "patched_forward", "base_from_cache",
                  "tokens", "time_alignment", "hook_removal", "attention_reference", "formula", "two_pass_fp32", "replay_scores", "skip_pool_parity", "binding", "first_step_zero",
                  "genuine_gradients", "real_update", "repeat", "forced_dense", "resume", "M_not_D", "one_way_synthetic", "eligible_counts", "run_alignment", "init_equals_base",
                  "readout_label_free", "a08_files", "memory", "projection", "disk"}
A4C, A5C = wm.A4C, wm.A5C
now, write_json, ledger_append, a5c_extras, ceil_pct, tables = t31.now, t31.write_json, t31.ledger_append, t40.a5c_extras, t40.ceil_pct, t40.tables


def model_seed(s: int, f: int) -> int:
    return 2026102743 + 10 * int(s) + int(f)


class Ctx:
    def __init__(self, cfg: dict, config: str) -> None:
        self.config, self.v = config, cfg["a5c_witness_mil"]; o = self.v["outputs"]
        self.out, self.results, self.ledger = ROOT / o["root"], ROOT / o["results_dir"], ROOT / self.v["budget"]["ledger"]
        self.a31 = t31.Ctx(load_cfg(SimpleNamespace(config=A31_CONFIG, set=[])), A31_CONFIG); self.d = d32.Ctx(load_cfg(SimpleNamespace(config=A32_CONFIG, set=[])), A32_CONFIG)


def staged(ctx: Ctx, job: str, kind: str, fn) -> None:
    t40.staged(ctx, job, kind, fn)


def publish(ctx: Ctx, name: str, rec: dict) -> None:
    t40.publish(ctx, name, rec)


# ----------------------------------------------------------------------------------------------- bags, plan, frames
class Fold:
    def __init__(self, ctx: Ctx, f: int) -> None:
        self.f, self.seqs = f, t31.load_pool(ctx.a31, f); self.W = t31.pool_W(self.seqs); self.fp = t28.frame_paths(self.seqs); self.B = wm.native_bags(self.seqs, self.W)
        n4, n5 = len(self.B["a4c"]), len(self.B["a5c"]); self.pi = (n4 / (n4 + n5), n5 / (n4 + n5)); self.mon = (self.B["a4c"][:N_MON], self.B["a5c"][:N_MON]); self._plans = {}

    def plan(self, s: int) -> np.ndarray:
        if s not in self._plans:
            self._plans[s] = wm.plan(self.B, s, self.f, U)
        return self._plans[s]

    def weights(self, i4: int, i5: int, device) -> torch.Tensor:
        return torch.from_numpy(np.concatenate([self.B["weights"][int(i4)], self.B["weights"][int(i5)]])).float().to(device)


def bag_frames(F_: Fold, idxs, workers: int = 0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Unique decoded frames of the bags' windows (A31's chunk_frames path), each window's 16 indices, marker slots, first-bag size."""
    rows = np.concatenate([F_.B["rows"][int(i)] for i in idxs]); fr, idx = t37.chunk_frames(F_.seqs, {"seq": F_.W["seq"][rows]}, F_.W["s"][rows], F_.fp, workers)
    return torch.from_numpy(fr), torch.from_numpy(idx), torch.from_numpy(F_.W["marker"][rows].astype(np.int64)), len(F_.B["rows"][int(idxs[0])])


class Bags(Dataset):
    """Item = (update, A4C sequence, A5C sequence, unique uint8 frames, (n, 16) indices, marker slots, A4C window count)."""

    def __init__(self, F_: Fold, s: int, us) -> None:
        self.F, self.pl, self.us = F_, F_.plan(s), list(us)

    def __len__(self) -> int:
        return len(self.us)

    def __getitem__(self, k: int):
        u = self.us[k]; i4, i5 = (int(x) for x in self.pl[u]); return (u, i4, i5, *bag_frames(self.F, [i4, i5]))


def loader(F_: Fold, s: int, us, workers: int):
    return DataLoader(Bags(F_, s, us), batch_size=None, num_workers=workers, prefetch_factor=2 if workers else None, pin_memory=True)


def plan_sha(pl: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(pl, np.int64).tobytes()).hexdigest()


# ----------------------------------------------------------------------------------------------- model, two-pass update
def build(ctx: Ctx, s: int, f: int, device, patched: bool = True):
    """The A08 copy (projection and five-class classifier frozen, everything else trainable, eval mode; deterministic skip
    pools on training copies) and the witness head constructed from the model RNG right after the copy."""
    m = ctx.a31.a08(s, device)
    for n, p in m.named_parameters():
        p.requires_grad_(not n.startswith(FROZEN))
    if patched:
        sp.deterministic_skip_pools(m.encoder)
    torch.manual_seed(model_seed(s, f)); head = wm.GatedWitness().to(device)
    return m.eval(), head.eval()


def groups_of(m, head) -> dict:
    return {"encoder_trunk": [p for n, p in m.encoder.named_parameters() if not n.startswith("head.")], "marker_stem": [m.marker_stem.weight],
            "attention": [head.V.weight, head.V.bias, head.U.weight, head.U.bias, head.w.weight], "scorer": [head.scorer.weight, head.scorer.bias]}


def make_opt(m, head):
    G = groups_of(m, head)
    return t31.opt_of([("encoder_trunk", G["encoder_trunk"], LR["encoder_trunk"]), ("marker_stem", G["marker_stem"], LR["marker_stem"]), ("witness", G["attention"] + G["scorer"], LR["witness"])])


def trainable(m, head) -> list:
    return [p for ps in groups_of(m, head).values() for p in ps]


def sq_norm(gs) -> float:
    return float(sum(0.0 if g is None else float(g.double().pow(2).sum()) for g in gs)) ** 0.5


def window_score(m, head, frd: torch.Tensor, idx: torch.Tensor, mk: torch.Tensor, w: int) -> torch.Tensor:
    """(1,) witness score of window w: one full original marker forward, microbatch 1."""
    return wm.score(m, head, tm.to_rgb(frd[idx[w:w + 1]]), tm.marker_planes(mk[w:w + 1]), mk[w:w + 1], mf.embed_with_tokens_grad)


def two_pass(m, head, frd: torch.Tensor, idx: torch.Tensor, mk: torch.Tensor, v: torch.Tensor, n4: int, pi, positive: str) -> dict:
    """Pass 1: every window's score without a graph; dL/ds from the bag objective on a leaf vector. Pass 2: every window again
    with autograd and s_j.backward(dL/ds_j) (zero-gradient windows included); parameter .grad accumulates."""
    n = idx.shape[0]
    with torch.no_grad():
        s1 = torch.cat([window_score(m, head, frd, idx, mk, w) for w in range(n)])
    sl = s1.detach().clone().requires_grad_(True); L, l4, l5 = wm.bag_loss(sl[:n4], v[:n4], sl[n4:], v[n4:], pi[0], pi[1], positive); L.backward(); g = sl.grad.detach()
    s2 = torch.empty_like(s1)
    for w in range(n):
        sw = window_score(m, head, frd, idx, mk, w); sw.backward(g[w:w + 1]); s2[w] = sw.detach()[0]
    return {"s1": s1, "s2": s2, "g": g, "L": float(L.detach()), "l4": float(l4.detach()), "l5": float(l5.detach())}


def update_record(u: int, n4: int, r: dict, v: torch.Tensor) -> dict:
    s, g, vv = (x.double().cpu().numpy() for x in (r["s1"], r["g"], v)); s4, s5 = s[:n4], s[n4:]; sp_ = lambda x: np.logaddexp(0.0, x)  # noqa: E731
    return {"u": u, "n4": int(n4), "n5": int(len(s5)), "loss": r["L"], "l4": r["l4"], "l5": r["l5"], "l5_dense": float((vv[n4:] * sp_(-s5)).sum()), "l5_max": float(sp_(-s5.max())),
            "pos4": float((s4 > 0).mean()), "pos5": float((s5 > 0).mean()), "pos4_w": float(vv[:n4][s4 > 0].sum()), "pos5_w": float(vv[n4:][s5 > 0].sum()), "max4": float(s4.max()), "max5": float(s5.max()),
            "margin": float(s5.max() - s4.max()), "gsum4": float(np.abs(g[:n4]).sum()), "gsum5": float(np.abs(g[n4:]).sum()), "replay_max_abs": float((r["s2"] - r["s1"]).abs().max())}


def update(m, head, opt, b, positive: str, F_: Fold, device, on_grads=None) -> dict:
    """One real update: two-pass whole-bag gradient of one A4C and one A5C bag, one clip, one AdamW step."""
    u, i4, i5, fr, idx, mk, n4 = b; u, n4 = int(u), int(n4); t31.sched(opt, u, SCH); opt.zero_grad(set_to_none=True)
    frd, idx, mk = fr.to(device, non_blocking=True), idx.to(device), mk.to(device); v = F_.weights(i4, i5, device)
    r = two_pass(m, head, frd, idx, mk, v, n4, F_.pi, positive); rec = update_record(u, n4, r, v)
    rec.update({f"g_{k}": sq_norm([p.grad for p in ps]) for k, ps in groups_of(m, head).items()})
    if on_grads is not None:
        on_grads(u)
    rec["grad_norm"] = float(torch.nn.utils.clip_grad_norm_(trainable(m, head), CLIP)); opt.step(); return rec


def run_updates(m, head, opt, F_: Fold, s: int, us, positive: str, device, workers: int, log: list, hook=None, on_grads=None) -> float:
    gsec = 0.0
    for b in loader(F_, s, us, workers):
        torch.cuda.synchronize(); a = time.time(); rec = update(m, head, opt, b, positive, F_, device, on_grads); torch.cuda.synchronize(); rec["gpu_s"] = time.time() - a; gsec += rec["gpu_s"]; log.append(rec)
        if hook is not None:
            hook(rec["u"], rec)
    return gsec


def state_of(m, head, opt) -> dict:
    return {"model": {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}, "head": {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}, "opt": opt.state_dict(),
            "cpu_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all()}


def load_state(m, head, opt, st: dict) -> None:
    m.load_state_dict(st["model"]); head.load_state_dict(st["head"]); opt.load_state_dict(st["opt"]); torch.set_rng_state(st["cpu_rng"]); torch.cuda.set_rng_state_all(st["cuda_rng"])


def flat_state(m, head) -> dict:
    return {**{f"m.{k}": v.detach().cpu() for k, v in m.state_dict().items()}, **{f"h.{k}": v.detach().cpu() for k, v in head.state_dict().items()}}


def params_max_diff(a: dict, b: dict) -> float:
    return float(max((a[k].float() - b[k].float()).abs().max() for k in a))


@torch.no_grad()
def bag_scores(m, head, F_: Fold, idxs, device, workers: int) -> list[np.ndarray]:
    """Pass-1 scores of whole bags (microbatch 1), one array per bag."""
    out = []
    for i in idxs:
        fr, idx, mk, n = bag_frames(F_, [i], workers); frd, idx, mk = fr.to(device), idx.to(device), mk.to(device)
        out.append(torch.cat([window_score(m, head, frd, idx, mk, w) for w in range(n)]).double().cpu().numpy())
    return out


def fit_monitor(m, head, F_: Fold, device, workers: int) -> dict:
    """Fit only: the fixed monitoring bags (first 8 A4C, first 8 A5C in pool order, all windows); both loss forms, positive
    fractions, bag maxima and bag-level r > 0 rates. Never used to select or retune."""
    s4, s5 = bag_scores(m, head, F_, F_.mon[0], device, workers), bag_scores(m, head, F_, F_.mon[1], device, workers); v4, v5 = [F_.B["weights"][i] for i in F_.mon[0]], [F_.B["weights"][i] for i in F_.mon[1]]
    sp_ = lambda x: np.logaddexp(0.0, x)  # noqa: E731
    l4 = float(np.mean([(v * sp_(s)).sum() for s, v in zip(s4, v4)])); l5d = float(np.mean([(v * sp_(-s)).sum() for s, v in zip(s5, v5)])); l5m = float(np.mean([sp_(-s.max()) for s in s5]))
    return {"l4": l4, "l5_dense": l5d, "l5_max": l5m, "total_D_form": F_.pi[0] * l4 + F_.pi[1] * l5d, "total_M_form": F_.pi[0] * l4 + F_.pi[1] * l5m,
            "pos_frac_A4C": float(np.mean([(s > 0).mean() for s in s4])), "pos_frac_A5C": float(np.mean([(s > 0).mean() for s in s5])),
            "mean_max_A4C": float(np.mean([s.max() for s in s4])), "mean_max_A5C": float(np.mean([s.max() for s in s5])), "r_pos_rate_A4C": float(np.mean([s.max() > 0 for s in s4])),
            "r_pos_rate_A5C": float(np.mean([s.max() > 0 for s in s5])), "windows": int(sum(len(s) for s in s4 + s5)), "bags": "first 8 A4C and first 8 A5C native bags of the fit pool (pool order)"}


def drift(m, head, s: int, f: int, device) -> dict:
    """Per-group drift from the pinned A08 and the initial head; tensors that did not move at all."""
    a08 = torch.load(ROOT / f"models_frozen/video_within_view_v1/all_s{s}/final.pt", map_location="cpu", weights_only=True); cur = {k: v.detach().cpu() for k, v in m.state_dict().items()}
    torch.manual_seed(model_seed(s, f)); h0 = wm.GatedWitness().state_dict(); hc = {k: v.detach().cpu() for k, v in head.state_dict().items()}; out = {}
    for name, keys, ref, now_ in (("encoder_trunk", [k for k in cur if k.startswith("encoder.") and not k.startswith("encoder.head.") and cur[k].is_floating_point()], a08, cur),
                                  ("marker_stem", ["marker_stem.weight"], a08, cur), ("attention", ["V.weight", "V.bias", "U.weight", "U.bias", "w.weight"], h0, hc), ("scorer", ["scorer.weight", "scorer.bias"], h0, hc)):
        d = sum(float((now_[k].double() - ref[k].double()).pow(2).sum()) for k in keys) ** 0.5; n0 = sum(float(ref[k].double().pow(2).sum()) for k in keys) ** 0.5
        out[name] = {"l2": d, "relative": d / max(n0, 1e-12), "max_abs": float(max((now_[k] - ref[k]).abs().max() for k in keys))}
    out["frozen_unchanged"] = all(torch.equal(cur[k], a08[k]) for k in cur if k.startswith(FROZEN)); out["scorer_now"] = {"bias": float(hc["scorer.bias"][0]), "weight_l2": float(hc["scorer.weight"].norm())}
    out["unchanged_tensors"] = sorted(k for k in cur if cur[k].is_floating_point() and torch.equal(cur[k], a08[k])); return out


# ----------------------------------------------------------------------------------------------- train
def identity(ctx: Ctx, arm: str, s: int, f: int, psha: str) -> str:
    h = hashlib.sha256()
    for p in (Path(wm.__file__), Path(mf.__file__), Path(sp.__file__), Path(tm.__file__), Path(__file__)):
        h.update(p.read_bytes())
    spl = json.loads((ctx.a31.out / "split.json").read_text())
    h.update(json.dumps({"arm": arm, "positive": wm.POSITIVE[arm], "seed": s, "fold": f, "a08": t31.A08[s], "pools": spl["pool_sha256"], "plan_sha256": psha, "config": sha256_file(ROOT / ctx.config)}, sort_keys=True).encode())
    return h.hexdigest()


def train_one(ctx: Ctx, arm: str, s: int, F_: Fold, ident: str, psha: str, device, workers: int) -> dict:
    key = f"{arm}_s{s}_f{F_.f}"; rd = ctx.out / "train" / key; rd.mkdir(parents=True, exist_ok=True); rp = rd / "resume.pt"; pos = wm.POSITIVE[arm]
    m, head = build(ctx, s, F_.f, device); opt = make_opt(m, head); init_digest = {"model": tet.state_digest(m), "head": tet.state_digest(head)}; start, log, resumes = 0, [], []
    mons = [{"after_update": 0, **fit_monitor(m, head, F_, device, workers)}]
    if rp.exists():
        st = torch.load(rp, map_location="cpu", weights_only=False)
        if st["identity"] != ident:
            raise SystemExit(f"{key}: the resume file belongs to another identity")
        load_state(m, head, opt, st); start, log, mons, init_digest = st["update"], st["log"], st["monitors"], st["init_digest"]; resumes.append({"from_update": start, "at": now()})
    prog = ctx.out / f"progress_s{s}.json"; t0 = time.time(); torch.cuda.reset_peak_memory_stats()

    def hook(u: int, rec: dict) -> None:
        if (u + 1) % LOG_EVERY == 0:
            last = log[-LOG_EVERY:]; mean = lambda k: float(np.mean([x[k] for x in last]))  # noqa: E731
            print(f"[{key}] update {u + 1}/{U}: loss {mean('loss'):.4f} l4 {mean('l4'):.4f} l5 {mean('l5'):.4f} max4 {mean('max4'):+.2f} max5 {mean('max5'):+.2f} pos4 {mean('pos4'):.2f} pos5 {mean('pos5'):.2f} "
                  f"g_trunk {mean('g_encoder_trunk'):.3g} grad {mean('grad_norm'):.3f} win {sum(x['n4'] + x['n5'] for x in last)}  {time.time() - t0:.0f} s", flush=True)
            write_json(prog, {"seed": s, "model": key, "update": u + 1, "total": U, "updated_at": now(), "pid": os.getpid(), "device": torch.cuda.get_device_name(0)})
        if (u + 1) % MON_EVERY == 0:
            mons.append({"after_update": u + 1, **fit_monitor(m, head, F_, device, workers)})
        if (u + 1) % SAVE_EVERY == 0 and u + 1 < U:
            tmp = rp.with_suffix(".tmp"); torch.save({**state_of(m, head, opt), "update": u + 1, "log": log, "monitors": mons, "identity": ident, "init_digest": init_digest}, tmp); tmp.replace(rp)

    gsec = run_updates(m, head, opt, F_, s, range(start, U), pos, device, workers, log, hook)
    steps = [int(st_["step"]) for st_ in opt.state_dict()["state"].values()]
    if len(log) != U or [x["u"] for x in log] != list(range(U)) or not steps or set(steps) != {U} or len(steps) != len(trainable(m, head)):
        raise SystemExit(f"{key}: {len(log)} logged updates, optimiser steps {sorted(set(steps))} over {len(steps)} tensors")
    torch.save({k: v.detach().cpu().clone() for k, v in m.state_dict().items()}, rd / "final.pt"); torch.save({k: v.detach().cpu().clone() for k, v in head.state_dict().items()}, rd / "head.pt")
    (rd / "updates.jsonl").write_text("".join(json.dumps(x) + "\n" for x in log)); keys = ("loss", "l4", "l5", "l5_dense", "l5_max", "pos4", "pos5", "max4", "max5", "margin", "g_encoder_trunk", "g_marker_stem", "g_attention", "g_scorer", "grad_norm", "replay_max_abs")
    man = {"arm": arm, "positive": pos, "seed": s, "fit_fold": F_.f, "pi": list(F_.pi), "updates": len(log), "optimizer_steps": U, "identity": ident, "plan_sha256": psha, "a08_sha256": t31.A08[s], "init_digest": init_digest,
           "final_sha256": sha256_file(rd / "final.pt"), "head_sha256": sha256_file(rd / "head.pt"), "updates_sha256": sha256_file(rd / "updates.jsonl"), "monitors": mons, "drift": drift(m, head, s, F_.f, device),
           "windows_trained": int(sum(x["n4"] + x["n5"] for x in log)), "replay_max_abs_max": float(max(x["replay_max_abs"] for x in log)), "resumes": resumes,
           "gpu_seconds_this_process": gsec, "seconds_this_process": time.time() - t0, "peak_gpu_gb": torch.cuda.max_memory_allocated() / 1e9,
           "loss": {k: {"first_60": float(np.mean([x[k] for x in log[:WARM]])), "last_60": float(np.mean([x[k] for x in log[-WARM:]]))} for k in keys}}
    write_json(rd / "run_manifest.json", man); return man


def train(ctx: Ctx, s: int, device, workers: int) -> None:
    for f in FOLDS:
        F_ = Fold(ctx, f); psha = plan_sha(F_.plan(s))
        for arm in ARMS:
            key = f"{arm}_s{s}_f{f}"; rd = ctx.out / "train" / key; ident = identity(ctx, arm, s, f, psha); mp = rd / "run_manifest.json"
            if mp.exists() and json.loads(mp.read_text())["identity"] == ident and sha256_file(rd / "final.pt") == json.loads(mp.read_text())["final_sha256"]:
                print(f"[{key}] finished earlier", flush=True); continue
            t0, started, ok, note = time.time(), now(), False, ""
            try:
                man = train_one(ctx, arm, s, F_, ident, psha, device, workers); torch.cuda.empty_cache(); ok = True; note = f"final {man['final_sha256'][:12]} head {man['head_sha256'][:12]}"
                if sha256_file(rd / "final.pt") == man["final_sha256"] and sha256_file(rd / "head.pt") == man["head_sha256"] and (rd / "resume.pt").exists():
                    (rd / "resume.pt").unlink()                                  # regenerable; the final weights are verified
                print(json.dumps({k: man[k] for k in ("arm", "seed", "fit_fold", "updates", "final_sha256", "head_sha256", "loss")}), flush=True)
            except BaseException as e:  # noqa: BLE001 - recorded, then raised again
                note = f"{type(e).__name__}: {str(e)[:200]}"; raise
            finally:
                ledger_append(ctx.ledger, f"train {key} (GPU job wall time)", "gpu", started, time.time() - t0, ok, note)
    man = {f"{arm}_s{s}_f{f}": json.loads((ctx.out / "train" / f"{arm}_s{s}_f{f}" / "run_manifest.json").read_text()) for f in FOLDS for arm in ARMS}
    publish(ctx, f"weights_s{s}.json", {"seed": s, "models": man, "config_sha256": sha256_file(ROOT / ctx.config), "loaded_code_sha256": t38.loaded_code(), "written": now()})


def freeze(ctx: Ctx, s: int, commit: str, manifest_sha: str) -> str:
    p = ctx.results / f"weights_s{s}.json"
    if sha256_file(p) != manifest_sha or sha256_file(ctx.out / f"weights_s{s}.json") != manifest_sha:
        raise SystemExit("the committed weight manifest differs from the server's")
    models = json.loads(p.read_text())["models"]; com = ctx.out / "weights_committed.json"; cur = json.loads(com.read_text()) if com.exists() else {}
    for k, man in models.items():
        if sha256_file(ctx.out / "train" / k / "final.pt") != man["final_sha256"] or sha256_file(ctx.out / "train" / k / "head.pt") != man["head_sha256"]:
            raise SystemExit(f"{k}: weights differ from the manifest")
        cur[k] = {"final": man["final_sha256"], "head": man["head_sha256"]}
    cur[f"commit_s{s}"] = commit; write_json(com, cur); return f"{len(models)} models pinned at {commit}"


# ----------------------------------------------------------------------------------------------- check
def nearest_loop(n: int, centres) -> np.ndarray:
    """Independent explicit loop: each sample's nearest centre, ties to the earlier."""
    out = np.empty(n, np.int64)
    for t in range(n):
        best, bd = 0, abs(t - int(centres[0]))
        for k in range(1, len(centres)):
            d = abs(t - int(centres[k]))
            if d < bd:
                best, bd = k, d
        out[t] = best
    return out


def bag_checks(ctx: Ctx, s: int) -> dict:
    """(1) Bag counts / membership / weights / feasibility values, 600 plans, exposure, cross-fold isolation."""
    out = {}; spl = json.loads((ctx.a31.out / "split.json").read_text())
    for f in FOLDS:
        F_ = Fold(ctx, f); B, seqs, W = F_.B, F_.seqs, F_.W; ex = EXPECTED_BAGS[f]; nw = {c: np.asarray([len(B["rows"][i]) for i in B[k]]) for c, k in (("A4C", "a4c"), ("A5C", "a5c"))}
        member = single = grid = weights_ok = True; wsum = 0.0; minw = 1.0
        for i in B["a4c"] + B["a5c"]:
            q = seqs[i]; r = B["rows"][i]; n = len(q["labels"]); w = vsf.windows(n)
            member &= bool(np.array_equal(r, np.flatnonzero(W["seq"] == i)) and np.all(np.diff(r) > 0))
            single &= len({str(x[0]) for x in q["refs"]}) == 1
            grid &= bool(np.array_equal(w["centres"], W["centre"][r]) and np.array_equal(w["samples"], W["s"][r]) and np.array_equal(w["markers"], W["marker"][r]))
            cnt = np.bincount(nearest_loop(n, W["centre"][r]), minlength=len(r)); weights_ok &= bool(np.array_equal(np.rint(B["weights"][i] * n).astype(np.int64), cnt) and abs(B["weights"][i].sum() - 1) < 1e-12)
            wsum = max(wsum, abs(B["weights"][i].sum() - 1)); minw = min(minw, float(B["weights"][i].min()))
        native_codes = Counter(int(np.asarray(q["labels"])[0]) for q in seqs if q["kind"] == "native")
        rec_bag = {str(seqs[i]["refs"][0][0]) for i in B["a4c"] + B["a5c"]}; other = t31.load_pool(ctx.a31, 1 - f); rec_other = {str(r[0]) for q in other for r in q["refs"]}
        pl = F_.plan(s); pl2 = wm.plan(B, s, f, U); a4n = np.asarray([len(B["rows"][i]) for i in pl[:, 0]]); a5n = np.asarray([len(B["rows"][i]) for i in pl[:, 1]]); d4, d5 = Counter(pl[:, 0].tolist()), Counter(pl[:, 1].tolist())
        explicit = True
        for u in range(U):
            g = np.random.default_rng([20261027, 43, s, f, u]); a = int(g.integers(len(B["a4c"]))); b = int(g.integers(len(B["a5c"]))); explicit &= (int(pl[u, 0]), int(pl[u, 1])) == (B["a4c"][a], B["a5c"][b])
        out[str(f)] = {"counts": {"A4C": len(B["a4c"]), "A5C": len(B["a5c"])}, "native_code_counts": {str(k): int(v) for k, v in sorted(native_codes.items())},
                       "counts_equal_expected": len(B["a4c"]) == ex["A4C"] and len(B["a5c"]) == ex["A5C"],
                       "mean_windows": [float(nw["A4C"].mean()), float(nw["A5C"].mean())], "max_windows": [int(nw["A4C"].max()), int(nw["A5C"].max())], "min_windows": [int(nw["A4C"].min()), int(nw["A5C"].min())],
                       "feasibility_equal": bool(abs(nw["A4C"].mean() - ex["mean"][0]) < 0.006 and abs(nw["A5C"].mean() - ex["mean"][1]) < 0.006 and (int(nw["A4C"].max()), int(nw["A5C"].max())) == ex["max"]),
                       "whole_bags_pool_order": member, "single_recording_natives": single, "grid_equals_A07_windows": grid, "weights_equal_independent_loop": weights_ok, "weight_sum_max_dev": wsum, "min_window_weight": minw,
                       "pi": list(F_.pi), "bag_recordings_in_other_pool": len(rec_bag & rec_other), "bag_sessions_in_other_pool": len({session_of(x) for x in rec_bag} & {session_of(x) for x in rec_other}),
                       "plan": {"updates": int(len(pl)), "repeat_equal": bool(np.array_equal(pl, pl2)), "explicit_rng_equal": bool(explicit), "sha256": plan_sha(pl),
                                "a4c_in_list": bool(set(pl[:, 0].tolist()) <= set(B["a4c"])), "a5c_in_list": bool(set(pl[:, 1].tolist()) <= set(B["a5c"])),
                                "windows_total": int(a4n.sum() + a5n.sum()), "windows_A4C": int(a4n.sum()), "windows_A5C": int(a5n.sum()), "windows_per_update_max": int((a4n + a5n).max()),
                                "windows_per_update_mean": float((a4n + a5n).mean()), "distinct_A4C_recordings": len(d4), "distinct_A5C_recordings": len(d5),
                                "draws_per_A4C_min_median_max": [int(min(d4.values())), float(np.median(list(d4.values()))), int(max(d4.values()))],
                                "draws_per_A5C_min_median_max": [int(min(d5.values())), float(np.median(list(d5.values()))), int(max(d5.values()))],
                                "selection_inputs": "kinds, clip codes and the window grid of the fit pool (no model output, Base mistake, S1 / harm set, session matching or held-out result); one plan for both arms"}}
    out["isolation"] = spl["isolation"]; return out


def frame_checks(F0: Fold, s: int) -> dict:
    """(1) Update 0's bags: dataset frames = an independent per-ref decode (A42 Batches path); markers = A07 marker slots."""
    item = Bags(F0, s, [0])[0]; u, i4, i5, fr, idx, mk, n4 = item; rows = np.concatenate([F0.B["rows"][i4], F0.B["rows"][i5]]); memo, same = {}, True
    for w, r in enumerate(rows):
        q = F0.seqs[int(F0.W["seq"][r])]
        for k, p in enumerate(F0.W["s"][r]):
            ref = tuple(q["refs"][int(p)])
            if ref not in memo:
                v, fi, g = ref; pth = F0.fp(v); memo[ref] = jv.load_frame_u8(pth[min(int(fi), len(pth) - 1)], round(float(g), 6))
            same &= bool(np.array_equal(memo[ref], fr[idx[w, k]].numpy()))
    exp_mk = np.asarray([tm.marker_slot(F0.W["s"][r], int(F0.W["centre"][r])) for r in rows])
    return {"pixels_equal_independent_decode": same, "windows": int(len(rows)), "unique_frames": int(len(fr)), "markers_equal_W": bool(np.array_equal(mk.numpy(), F0.W["marker"][rows])),
            "markers_equal_marker_slot": bool(np.array_equal(mk.numpy(), exp_mk)), "first_bag_windows": int(n4), "repeat_equal": bool(torch.equal(Bags(F0, s, [0])[0][3], fr))}


def check_head(device, seed: int = 4343) -> wm.GatedWitness:
    """A check-only head with a nonzero scorer weight (N(0, 0.05^2) from its own generator), so the trunk receives gradient."""
    h = wm.GatedWitness(); g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        h.scorer.weight.copy_(torch.randn(1, wm.DIM, generator=g) * 0.05)
    return h.to(device).eval()


def init_checks(ctx: Ctx, s: int, F0: Fold, device) -> dict:
    """(2) Init: A08 copy bitwise, identical head across arms, 197,761 head parameters, every first-pass score -2, online fit
    logits = A31 cache (no held-out pixel), patched = original forward, tokens / time alignment / hook removal."""
    a08 = torch.load(ROOT / f"models_frozen/video_within_view_v1/all_s{s}/final.pt", map_location="cpu", weights_only=True); res = {}
    m, head = build(ctx, s, 0, device); m2, head2 = build(ctx, s, 0, device)
    res["init_state_equals_A08"] = all(torch.equal(v.cpu(), a08[k]) for k, v in m.state_dict().items()) and sorted(m.state_dict()) == sorted(a08)
    res["identical_arm_start"] = tet.state_digest(m) == tet.state_digest(m2) and tet.state_digest(head) == tet.state_digest(head2)
    res["patched_blocks"] = [i for i, bl in enumerate(m.encoder.blocks) if isinstance(getattr(bl.pool_skip, "pool", None), sp.GatherMaxPool3d)]
    res["head"] = {"parameters": int(sum(p.numel() for p in head.parameters())), "scorer_weight_all_zero": bool((head.scorer.weight == 0).all()), "scorer_bias": float(head.scorer.bias[0]),
                   "attention_nonzero": bool(all(float(p.abs().sum()) > 0 for p in (head.V.weight, head.U.weight, head.w.weight))), "w_has_bias": head.w.bias is not None,
                   "fold1_head_differs": tet.state_digest(build(ctx, s, 1, device)[1]) != tet.state_digest(head)}
    item = Bags(F0, s, [0])[0]; u, i4, i5, fr, idx, mk, n4 = item; frd, idxd, mkd = fr.to(device), idx.to(device), mk.to(device); rows = np.concatenate([F0.B["rows"][i4], F0.B["rows"][i5]]); n = len(rows)
    with torch.no_grad():
        s0 = torch.cat([window_score(m, head, frd, idxd, mkd, w) for w in range(n)])
    res["init_scores_all_minus_2"] = {"windows": int(n), "all_exactly_minus_2": bool((s0 == -2.0).all()), "max_abs_dev": float((s0 + 2).abs().max())}
    O = t31.outputs(ctx.a31, s, 0); k = min(n, 16)
    with torch.no_grad():
        z = torch.cat([m(tm.to_rgb(frd[idxd[w:w + 1]]), tm.marker_planes(mkd[w:w + 1])) for w in range(k)])
        e, uu = mf.embed_with_tokens_grad(m, tm.to_rgb(frd[idxd[:1]]), tm.marker_planes(mkd[:1]))
    res["init_online_fit_logits_vs_A31_cache_max_abs"] = float(np.abs(z.cpu().numpy() - O["logits"][rows[:k]]).max())
    res["hooked_embed_logits_equal_plain"] = bool(torch.equal(m.classifier(e), z[:1]))
    mo, _ = build(ctx, s, 0, device, patched=False); hc = check_head(device)
    with torch.no_grad():
        sp_ = torch.cat([window_score(m, hc, frd, idxd, mkd, w) for w in range(k)]); so = torch.cat([window_score(mo, hc, frd, idxd, mkd, w) for w in range(k)])
        zo = torch.cat([mo(tm.to_rgb(frd[idxd[w:w + 1]]), tm.marker_planes(mkd[w:w + 1])) for w in range(k)])
    res["patched_equals_original_forward_bitwise"] = bool(torch.equal(sp_, so) and torch.equal(zo, z))
    res["tokens"] = token_checks(m, head, frd, idxd, mkd)
    del m, m2, mo; torch.cuda.empty_cache(); res["Base_from_cached_q"] = t40.init_base(ctx, s); return res


def token_checks(m, head, frd: torch.Tensor, idx: torch.Tensor, mk: torch.Tensor) -> dict:
    """(3) encoder.norm tokens: shape, network grid, explicit selection loop, nearest-time loop, hook removal."""
    enc = m.encoder; w = int(np.argmax(mk.cpu().numpy() % 2 == 1)) if bool((mk % 2 == 1).any()) else 0; rgb, pl = tm.to_rgb(frd[idx[w:w + 1]]), tm.marker_planes(mk[w:w + 1])
    with torch.no_grad():
        e0 = m.embed(rgb, pl)
    e, u = mf.embed_with_tokens_grad(m, rgb, pl); h = wm.local_tokens(u, mk[w:w + 1]); t = int(mk[w]) // 2
    loop = max(float((h[0, y * 7 + x] - u[0, 1 + t * 49 + y * 7 + x]).abs().max()) for y in range(7) for x in range(7))
    near = []
    for k in range(tm.N_SLOTS):
        best, bd = 0, abs(k - 0)
        for tt in range(1, wm.GRID[0]):
            if abs(k - 2 * tt) < bd:
                best, bd = tt, abs(k - 2 * tt)
        near.append(best)
    err = False
    try:
        mf.embed_with_tokens_grad(m, rgb, pl[:, :, :8])
    except ValueError:
        err = True
    return {"U_shape": list(u.shape), "final_grid_from_network": list(tp.token_grid(enc)[0]), "U_keeps_autograd": bool(u.requires_grad and u.grad_fn is not None), "e_equals_plain_embed": bool(torch.equal(e.detach(), e0)),
            "local_tokens_shape": list(h.shape), "selection_vs_loop_max_abs": loop, "window_marker_slot": int(mk[w]), "token_time": t,
            "nearest_time_loop": near, "token_time_rule": [wm.token_time(k) for k in range(tm.N_SLOTS)], "time_rule_equal": near == [wm.token_time(k) for k in range(tm.N_SLOTS)],
            "batch_selection_equal_marker_div_2": bool(torch.equal(torch.div(mk, 2, rounding_mode="floor"), torch.tensor([wm.token_time(int(x)) for x in mk], device=mk.device))),
            "hook_removed_after_error": err and not enc.norm._forward_hooks and not enc.conv_proj._forward_hooks, "no_hook_left": not enc.norm._forward_hooks and not enc.conv_proj._forward_hooks}


def attention_checks(device) -> dict:
    """(4) Float64 reference of the gated attention / score, token permutation invariance, finite gradients, Init score -2."""
    g = torch.Generator().manual_seed(4344); h64 = check_head("cpu").double()
    with torch.no_grad():
        for p in h64.parameters():
            p.add_(torch.randn(p.shape, generator=g, dtype=torch.float64) * 0.01)
    x = torch.randn(1, wm.N_SPATIAL, wm.DIM, generator=g, dtype=torch.float64) * 2.0; sd = {k: v.numpy() for k, v in h64.state_dict().items()}
    s_t, a_t = h64(x); s_r, a_r = wm.reference_head(x[0].numpy(), sd); perm = torch.randperm(wm.N_SPATIAL, generator=g); s_p, a_p = h64(x[:, perm])
    xg = x.clone().requires_grad_(True); h64.zero_grad(); h64(xg)[0].sum().backward(); grads = [xg.grad] + [p.grad for p in h64.parameters()]
    h32 = check_head(device); x32 = x.float().to(device)
    with torch.no_grad():
        d32_ = float((h32(x32)[0] - h32(x32[:, perm.to(device)])[0]).abs().max())
    hi = wm.GatedWitness(); xi = torch.randn(3, wm.N_SPATIAL, wm.DIM, generator=g, requires_grad=True); si = hi(xi)[0]; si.sum().backward()
    return {"score_abs_vs_reference": abs(float(s_t[0]) - s_r), "attention_max_abs_vs_reference": float(np.abs(a_t[0].detach().numpy() - a_r).max()), "attention_sums_to_1": abs(float(a_t.sum()) - 1.0),
            "permutation_score_abs_fp64": abs(float(s_p[0]) - float(s_t[0])), "permutation_attention_follows": float((a_p[0] - a_t[0][perm]).abs().max()), "permutation_score_abs_fp32": d32_,
            "gradients_finite_nonzero": all(gg is not None and bool(torch.isfinite(gg).all()) and float(gg.abs().sum()) > 0 for gg in grads),
            "init_scores_minus_2_any_input": bool((si == -2.0).all()), "init_token_gradient_zero": bool((xi.grad == 0).all()), "init_scorer_gradient_nonzero": float(hi.scorer.weight.grad.abs().sum()) > 0}


def formula_checks() -> dict:
    """(5) Independent float64 BCE / amax losses and score gradients: variable sizes, unequal weights, exact ties, +/- 1e3 margins."""
    g = np.random.default_rng([20261027, 43, 99]); out = {}; pi = (200 / 267, 67 / 267)
    cases = {"1x1": (1, 1, 2.0, 0), "3x5": (3, 5, 2.0, 0), "17x40": (17, 40, 3.0, 0), "102x100": (102, 100, 3.0, 0), "ties2": (7, 9, 2.0, 2), "ties3": (4, 6, 2.0, 3), "large_margins": (6, 8, 1e3, 0)}
    for name, (n4, n5, sc, ties) in cases.items():
        s4, s5 = g.normal(size=n4) * sc, g.normal(size=n5) * sc; v4, v5 = g.dirichlet(np.ones(n4)), g.dirichlet(np.ones(n5))
        if ties:
            s5[g.choice(n5, ties, replace=False)] = s5.max() + 0.5
        for pos in ("dense", "max"):
            a = torch.tensor(s4, requires_grad=True); b = torch.tensor(s5, requires_grad=True); L, l4, l5 = wm.bag_loss(a, torch.tensor(v4), b, torch.tensor(v5), pi[0], pi[1], pos); L.backward()
            r = wm.reference_loss(s4, v4, s5, v5, pi[0], pi[1], pos)
            out[f"{name}_{pos}"] = {"l4_abs": abs(float(l4) - r["l4"]), "l5_abs": abs(float(l5) - r["l5"]), "total_abs": abs(float(L) - r["total"]),
                                    "grad_max_abs": float(max(np.abs(a.grad.numpy() - r["g4"]).max(), np.abs(b.grad.numpy() - r["g5"]).max())), "finite": bool(torch.isfinite(L) and torch.isfinite(a.grad).all() and torch.isfinite(b.grad).all()),
                                    "nonmax_A5C_gradient_zero": bool((b.grad.numpy()[s5 < s5.max()] == 0).all()) if pos == "max" else None, "tied_windows": int((s5 == s5.max()).sum())}
    s4, v4 = g.normal(size=5), g.dirichlet(np.ones(5)); s5 = np.asarray([0.37]); res = {}
    for pos in ("dense", "max"):
        a = torch.tensor(s4, requires_grad=True); b = torch.tensor(s5, requires_grad=True); L = wm.bag_loss(a, torch.tensor(v4), b, torch.tensor([1.0], dtype=torch.float64), pi[0], pi[1], pos)[0]; L.backward(); res[pos] = (float(L), a.grad.clone(), b.grad.clone())
    out["one_instance_M_equals_D"] = {"loss_abs": abs(res["dense"][0] - res["max"][0]), "grad_equal": bool(torch.equal(res["dense"][1], res["max"][1]) and torch.allclose(res["dense"][2], res["max"][2], atol=1e-15, rtol=0))}
    return out


def grads(m, head) -> dict:
    return {**{f"m.{n}": (torch.zeros_like(p) if p.grad is None else p.grad.detach().clone()) for n, p in m.named_parameters() if p.requires_grad},
            **{f"h.{n}": (torch.zeros_like(p) if p.grad is None else p.grad.detach().clone()) for n, p in head.named_parameters()}}


def rel(ga: dict, gb: dict) -> dict:
    mx = max(float(v.abs().max()) for v in gb.values()); d = max(float((ga[k] - gb[k]).abs().max()) for k in gb); return {"max_abs": d, "grad_max": mx, "relative": d / mx}


def short_bags(F_: Fold) -> tuple[int, int]:
    """The first A4C and first A5C bag in pool order with 3-6 windows and unequal weights (bounded retained graphs)."""
    pick = lambda lst: next(i for i in lst if 3 <= len(F_.B["rows"][i]) <= 6 and len(set(np.round(F_.B["weights"][i], 12).tolist())) > 1)  # noqa: E731
    return pick(F_.B["a4c"]), pick(F_.B["a5c"])


def zero_all(m, head) -> None:
    m.zero_grad(set_to_none=True); head.zero_grad(set_to_none=True)


def gradient_checks(ctx: Ctx, s: int, F_: Fold, device) -> dict:
    """(6) Two-pass production FP32 gradient vs one retained-graph backward (check-only nonzero scorer weight), D and M; pass-1
    vs replayed scores; patched vs original MaxPool3d backward; batched vs microbatch FP32 (diagnostic only)."""
    i4, i5 = short_bags(F_); fr, idx, mk, n4 = bag_frames(F_, [i4, i5]); frd, idx, mk = fr.to(device), idx.to(device), mk.to(device); v = F_.weights(i4, i5, device); n = idx.shape[0]; out = {"bags_windows": [int(n4), int(n - n4)]}
    m, _ = build(ctx, s, F_.f, device); hc = check_head(device)
    for pos in ("dense", "max"):
        zero_all(m, hc); r = two_pass(m, hc, frd, idx, mk, v, n4, F_.pi, pos); ga = grads(m, hc); zero_all(m, hc)
        sr = torch.cat([window_score(m, hc, frd, idx, mk, w) for w in range(n)]); L = wm.bag_loss(sr[:n4], v[:n4], sr[n4:], v[n4:], F_.pi[0], F_.pi[1], pos)[0]; L.backward(); gb = grads(m, hc); zero_all(m, hc)
        nz = {k: sq_norm([ga[f"{'m' if k in ('encoder_trunk', 'marker_stem') else 'h'}.{nm}"] for nm in names]) for k, names in group_names(m, hc).items()}
        out[f"two_pass_vs_retained_{pos}"] = {**rel(ga, gb), "pass1_vs_retained_scores_max_abs": float((r["s1"] - sr.detach()).abs().max()), "pass1_vs_replay_max_abs": float((r["s1"] - r["s2"]).abs().max()),
                                              "loss_abs": abs(r["L"] - float(L)), "group_grad_norms": nz, "all_groups_nonzero": all(x > 0 for x in nz.values()), "status": "gated (production FP32 two-pass gradient)"}
        if pos == "dense":
            gb_dense = gb
    with torch.no_grad():
        sb = hc(wm.local_tokens(mf.embed_with_tokens_grad(m, tm.to_rgb(frd[idx]), tm.marker_planes(mk))[1], mk))[0]
    out["batched_vs_microbatch_scores_fp32"] = {"max_abs": float((sb - r["s1"]).abs().max()), "status": "diagnostic round-off, not a gate"}
    del m; torch.cuda.empty_cache()
    mo, _ = build(ctx, s, F_.f, device, patched=False); torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        sr = torch.cat([window_score(mo, hc, frd, idx, mk, w) for w in range(n)]); wm.bag_loss(sr[:n4], v[:n4], sr[n4:], v[n4:], F_.pi[0], F_.pi[1], "dense")[0].backward(); out["patched_vs_original_maxpool_backward"] = rel(gb_dense, grads(mo, hc))
    finally:
        torch.use_deterministic_algorithms(True, warn_only=False)
    del mo; torch.cuda.empty_cache(); out["warn_only_restored_false"] = not torch.is_deterministic_algorithms_warn_only_enabled(); return out


def group_names(m, head) -> dict:
    ids = {id(p): n for n, p in m.named_parameters()}; hid = {id(p): n for n, p in head.named_parameters()}; G = groups_of(m, head)
    return {k: [ids.get(id(p)) or hid[id(p)] for p in ps] for k, ps in G.items()}


def mech_run(ctx: Ctx, s: int, F_: Fold, arm: str, us, device, workers: int, state: dict | None = None, track: bool = False, force_positive: str | None = None):
    """Check-only training path of an arm (the positive term from wm.POSITIVE unless forced)."""
    positive = force_positive or wm.POSITIVE[arm]; m, head = build(ctx, s, F_.f, device); opt = make_opt(m, head)
    if state is not None:
        load_state(m, head, opt, state)
    log, seen = [], {}
    names = {id(p): f"m.{n}" for n, p in m.named_parameters()} | {id(p): f"h.{n}" for n, p in head.named_parameters()}

    def on_grads(u: int) -> None:
        seen[u] = {names[id(p)] for p in trainable(m, head) if p.grad is not None and float(p.grad.abs().sum()) > 0}

    g = run_updates(m, head, opt, F_, s, us, positive, device, workers, log, on_grads=on_grads if track else None); return m, head, opt, log, g, seen


def training_checks(ctx: Ctx, s: int, F_: Fold, device, workers: int) -> dict:
    """(7 / 8) Binding, update 0 zero / later genuine gradients, movement, repeat, forced dense M, M vs D, 3 + 3 resume, timing."""
    us = range(0, 6); res = {}; m, head = build(ctx, s, F_.f, device); opt = make_opt(m, head); init = {k: v.clone() for k, v in flat_state(m, head).items()}
    ids = sorted(id(p) for g in opt.param_groups for p in g["params"]); names = {id(p): n for n, p in m.named_parameters()} | {id(p): f"head.{n}" for n, p in head.named_parameters()}
    res["binding"] = {"binds_exactly_trainable": ids == sorted(id(p) for p in trainable(m, head)) and len(ids) == len(set(ids)),
                      "trainable_equals_requires_grad": {id(p) for p in trainable(m, head)} == {id(p) for p in list(m.parameters()) + list(head.parameters()) if p.requires_grad},
                      "frozen": sorted(n for n, p in m.named_parameters() if not p.requires_grad), "frozen_only_projection_and_classifier": all(n.startswith(FROZEN) for n, p in m.named_parameters() if not p.requires_grad)
                      and all(not p.requires_grad for n, p in m.named_parameters() if n.startswith(FROZEN)), "eval_mode": not m.training and not head.training and not any(x.training for x in list(m.modules()) + list(head.modules())),
                      "groups": {g["name"]: {"tensors": len(g["params"]), "parameters": int(sum(p.numel() for p in g["params"])), "lr": g["base"], "weight_decay": g["weight_decay"]} for g in opt.param_groups},
                      "witness_group": [names[id(p)] for p in opt.param_groups[2]["params"]], "marker_group": [names[id(p)] for p in opt.param_groups[1]["params"]],
                      "head_parameters": int(sum(p.numel() for p in head.parameters())), "frozen_parameters": int(sum(p.numel() for p in m.parameters() if not p.requires_grad)), "betas_eps": [list(opt.defaults["betas"]), opt.defaults["eps"]],
                      "lr_update0": {g["name"]: g["base"] * lr_factor(0, WARM, U) for g in opt.param_groups}, "lr_update59_peak": {g["name"]: g["base"] * lr_factor(WARM - 1, WARM, U) for g in opt.param_groups}}
    del m, head, opt; torch.cuda.empty_cache()
    L = lambda a, b: max(max(abs(x["loss"] - y["loss"]), abs(x["l4"] - y["l4"]), abs(x["l5"] - y["l5"])) for x, y in zip(a[3], b[3]))  # noqa: E731
    P = lambda a, b: params_max_diff(flat_state(a[0], a[1]), flat_state(b[0], b[1]))  # noqa: E731
    da = mech_run(ctx, s, F_, "D", us, device, workers, track=True); fin = flat_state(da[0], da[1]); lg = da[3]
    res["update0"] = {k: lg[0][k] for k in ("g_encoder_trunk", "g_marker_stem", "g_attention", "g_scorer", "n4", "n5", "loss")}
    res["updates_1_5"] = [{k: x[k] for k in ("u", "g_encoder_trunk", "g_marker_stem", "g_attention", "g_scorer", "grad_norm", "n4", "n5", "loss", "max4", "max5", "replay_max_abs")} for x in lg[1:]]
    tr = {f"m.{n}" for n, p in da[0].named_parameters() if p.requires_grad} | {f"h.{n}" for n, _ in da[1].named_parameters()}; nz15 = set().union(*[da[5][u] for u in range(1, 6)])
    grp = {"encoder_trunk": lambda k: k.startswith("m.encoder.") and not k.startswith("m.encoder.head."), "marker_stem": lambda k: k == "m.marker_stem.weight", "attention": lambda k: k.startswith(("h.V.", "h.U.", "h.w.")), "scorer": lambda k: k.startswith("h.scorer.")}
    res["real_update_D_6"] = {"moved": {g_: {"tensors_moved": int(sum(1 for k in fin if f_(k) and not torch.equal(fin[k], init[k]))), "tensors": int(sum(1 for k in fin if f_(k))),
                                             "max_abs": float(max((fin[k] - init[k]).abs().max() for k in fin if f_(k)))} for g_, f_ in grp.items()},
                              "unchanged_tensors": sorted(k for k in fin if torch.equal(fin[k], init[k])), "optimizer_steps": sorted({int(v["step"]) for v in da[2].state_dict()["state"].values()}),
                              "optimizer_state_tensors": len(da[2].state_dict()["state"]), "zero_gradient_trainable_tensors_updates_1_5": sorted(tr - nz15), "update0_nonzero_gradient_tensors": sorted(da[5][0])}
    db = mech_run(ctx, s, F_, "D", us, device, workers); res["repeat_6_D"] = {"loss_max_abs": L(da, db), "params_max_abs": P(da, db)}; del db; torch.cuda.empty_cache()
    ma = mech_run(ctx, s, F_, "M", us, device, workers); res["M_differs_from_D"] = {"params_max_abs": P(ma, da), "l5_first_updates": [[x["l5"] for x in ma[3][:3]], [x["l5"] for x in da[3][:3]]]}
    fd = mech_run(ctx, s, F_, "M", us, device, workers, force_positive="dense")        # the M arm's path, positive term forced to dense
    res["forced_dense_M_vs_D"] = {"loss_max_abs": L(fd, da), "params_max_abs": P(fd, da), "formal_map_unchanged": wm.POSITIVE == {"D": "dense", "M": "max"}}; del fd, da; torch.cuda.empty_cache()
    m3 = mech_run(ctx, s, F_, "M", range(0, 3), device, workers); p = ctx.out / "check_resume.pt"; torch.save(state_of(m3[0], m3[1], m3[2]), p); log3 = m3[3]; del m3
    st = torch.load(p, map_location="cpu", weights_only=False); mr = mech_run(ctx, s, F_, "M", range(3, 6), device, workers, st); p.unlink()
    res["resume_3_plus_3_M"] = {"loss_max_abs": max(max(abs(x["loss"] - y["loss"]), abs(x["l5"] - y["l5"])) for x, y in zip(ma[3], log3 + mr[3])), "params_max_abs": P(ma, mr)}
    res["timing"] = {"updates_0_5": {"windows": int(sum(x["n4"] + x["n5"] for x in ma[3])), "gpu_seconds": ma[4], "seconds_per_window_gpu": ma[4] / sum(x["n4"] + x["n5"] for x in ma[3])}}; del ma, mr; torch.cuda.empty_cache()
    pl = F_.plan(s); nwin = np.asarray([len(F_.B["rows"][a]) + len(F_.B["rows"][b]) for a, b in pl]); ul, ush = int(nwin.argmax()), int(nwin.argmin())
    torch.cuda.reset_peak_memory_stats(); lo = mech_run(ctx, s, F_, "M", [ul], device, 0); sh = mech_run(ctx, s, F_, "M", [ush], device, 0)
    res["timing"].update({"longest_update": {"u": ul, "windows": int(nwin[ul]), "gpu_seconds": lo[4], "seconds_per_window": lo[4] / nwin[ul]}, "shortest_update": {"u": ush, "windows": int(nwin[ush]), "gpu_seconds": sh[4]},
                          "peak_gpu_gb_training": torch.cuda.max_memory_allocated() / 1e9}); del lo, sh; torch.cuda.empty_cache()
    t0 = time.time(); w12 = mech_run(ctx, s, F_, "D", range(6, 18), device, workers); wall = time.time() - t0; nw12 = int(sum(x["n4"] + x["n5"] for x in w12[3]))
    res["timing"]["updates_6_17_wall"] = {"windows": nw12, "gpu_seconds": w12[4], "wall_seconds_incl_build_loader": wall, "seconds_per_window_wall": wall / nw12, "seconds_per_window_gpu": w12[4] / nw12}; del w12
    torch.cuda.empty_cache(); return res


def one_way_checks(ctx: Ctx, s: int) -> dict:
    """(9) Synthetic edit / merge invariants; eligible counts from cached P runs; run alignment; Init = Base; label-free decision."""
    out = {}; base = np.asarray([0, 1, 2, 3, 4, 2, 2, 2]); r = np.asarray([5.0, 5.0, 1.0, 5.0, 5.0, 0.0, -1.0, 1e-9])
    out["synthetic_rule"] = {"result": wm.one_way(base, r).tolist(), "expected": [0, 1, 3, 3, 4, 2, 2, 3], "input_unchanged": base.tolist() == [0, 1, 2, 3, 4, 2, 2, 2]}
    seq = np.asarray([0, 0, 2, 2, 3, 3, 2, 2, 2, 1]); pk = {"seq": np.zeros(5, int), "a": np.asarray([0, 2, 4, 6, 9]), "b": np.asarray([2, 4, 6, 9, 10]), "base": np.asarray([0, 2, 3, 2, 1])}
    ed = wm.edit_sequences({"q": seq}, pk, [{"id": "q"}], np.asarray([0, 3, 3, 2, 1]))["q"]
    out["synthetic_merge"] = {"edited": ed.tolist(), "expected": [0, 0, 3, 3, 3, 3, 2, 2, 2, 1], "runs_after": [list(x) for x in ir.runs(ed)], "boundaries_before_after": [int((np.diff(seq) != 0).sum()), int((np.diff(ed) != 0).sum())]}
    X = t40.full_q(ctx, s); a8 = rsh.a08_ctx(); data = t31.vpr.training_data(ctx.a31.a15.a10.a07p); g = np.random.default_rng([20261027, 43, 7])
    for h in FOLDS:
        D = d32.fold_data(ctx.d, s, h); D["M"] = d32.set_masks(D); pk = D["pk"]; off = X["offset"][h]; nW = len(t31.pool_W(D["seqs"])["seq"]); el = pk["base"] == A4C; need = eligible_rows(pk, el)
        al = True
        for qi, q in enumerate(D["seqs"]):
            ks = np.flatnonzero(pk["seq"] == qi); al &= [tuple(int(v) for v in x) for x in ir.runs(D["base_preds"][q["id"]])] == [(int(pk["a"][k]), int(pk["b"][k]), int(pk["base"][k])) for k in ks] and bool(np.all(np.diff(ks) == 1))
        sc = np.full(nW, -2.0); r = wm.run_max(pk, sc, off, el); p0 = wm.one_way(pk["base"], r); P = {"seqs": D["seqs"], "n_of": dict(data.n), "label_of": dict(data.label)}; ev = bc.evalset(P, list(range(len(D["seqs"]))), f"heldout_f{h}")
        t0_ = tables_pred(p0, D, a8.cfg, P, ev); tb = tables(X["q"], D, off, a8.cfg, P, ev); js = lambda x: json.loads(json.dumps(x, default=float))  # noqa: E731
        same = all(js(t0_[k]) == js(tb[k]) for k in ("runs", "changes", "correct_A5C_kept", "S1_fixed", "S2_kept", "sequence_row", "counts")) and bool(np.array_equal(t0_["pred"], tb["pred"]))
        rnd = g.normal(size=nW).astype(np.float32); rr_ = wm.run_max(pk, rnd, off, el); pr = wm.one_way(pk["base"], rr_); pk2 = {**pk, "labels": np.zeros_like(pk["labels"])}; pr2 = wm.one_way(pk2["base"], wm.run_max(pk2, rnd, off, pk2["base"] == A4C))
        edr = wm.edit_sequences(D["base_preds"], pk, D["seqs"], pr)
        out[f"heldout_f{h}"] = {"eligible_runs": int(el.sum()), "eligible_unique_rows": int(len(need)), "full_grid": int(nW), "rows_inside_fold": bool(need.min() >= off and need.max() < off + nW),
                                "expected": EXPECTED_ELIGIBLE.get(s, {}).get(h), "run_alignment": al, "init_r_all_minus_2": bool(np.all(r[el] == -2.0)), "init_pred_equals_base": bool(np.array_equal(p0, pk["base"])),
                                "init_sequences_equal_base": all(np.array_equal(t0_["preds"][q["id"]], D["base_preds"][q["id"]]) for q in D["seqs"]), "init_tables_equal_Base_tables": same,
                                "random_scores_non_A4C_unchanged": bool(np.array_equal(pr[~el], pk["base"][~el])), "random_scores_changes_only_2_to_3": bool(set(zip(pk["base"][pr != pk["base"]].tolist(), pr[pr != pk["base"]].tolist())) <= {(A4C, A5C)}),
                                "random_scores_changed": int((pr != pk["base"]).sum()), "label_free_decision": bool(np.array_equal(pr, pr2)),
                                "random_sequences_change_only_changed_runs": all(np.array_equal(edr[q["id"]][int(pk["a"][k]):int(pk["b"][k])], np.full(int(pk["b"][k] - pk["a"][k]), pr[k])) for k in range(len(pr)) for q in [D["seqs"][int(pk["seq"][k])]])}
    return out


def eligible_rows(pk: dict, el: np.ndarray) -> np.ndarray:
    """Global request rows of the eligible runs (the union of their unique nearest-centre rows), sorted."""
    o = pk["offsets"]; return np.unique(np.concatenate([pk["rows"][o[i]:o[i + 1]] for i in np.flatnonzero(el)])) if el.any() else np.zeros(0, np.int64)


def readout_timing(ctx: Ctx, s: int, device, workers: int) -> dict:
    """Fit rows (first 6 pool sequences): decode once, two specialists (microbatch 1, no graph)."""
    F_ = Fold(ctx, 0); rows = np.flatnonzero(F_.W["seq"] < 6)[:64]; t0 = time.time(); fr, idx = t37.chunk_frames(F_.seqs, {"seq": F_.W["seq"][rows]}, F_.W["s"][rows], F_.fp, workers); td = time.time() - t0
    ms = [build(ctx, s, 0, device, patched=False) for _ in range(2)]; frd, idxd, mkd = torch.from_numpy(fr).to(device), torch.from_numpy(idx).to(device), torch.from_numpy(F_.W["marker"][rows].astype(np.int64)).to(device)
    torch.cuda.synchronize(); t1 = time.time()
    with torch.no_grad():
        for m, h in ms:
            for w in range(len(rows)):
                window_score(m, h, frd, idxd, mkd, w)
    torch.cuda.synchronize(); gs = time.time() - t1; del ms; torch.cuda.empty_cache()
    return {"seconds_per_row_two_models": (gs + td) / len(rows), "gpu_ms_per_row_one_model": 1e3 * gs / 2 / len(rows), "decode_ms_per_row": 1e3 * td / len(rows), "rows": int(len(rows))}


def check(ctx: Ctx, device, workers: int) -> str:
    s = 0; t_start = time.time(); res = {"config_sha256": sha256_file(ROOT / ctx.config), "tolerances": TOL, "limits": LIMITS, "contract": wm.CONTRACT, "expected_bags": EXPECTED_BAGS, "expected_eligible": EXPECTED_ELIGIBLE,
                                         "numerics": {"tf32_matmul": torch.backends.cuda.matmul.allow_tf32, "tf32_cudnn": torch.backends.cudnn.allow_tf32, "cudnn_deterministic": torch.backends.cudnn.deterministic,
                                                      "cudnn_benchmark": torch.backends.cudnn.benchmark, "deterministic": torch.are_deterministic_algorithms_enabled(), "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled()}}
    spl = json.loads((ctx.a31.out / "split.json").read_text()); res["pool_sha256"] = spl["pool_sha256"]; res["bags"] = bag_checks(ctx, s)
    if not all(res["bags"][str(f)]["counts_equal_expected"] and res["bags"][str(f)]["feasibility_equal"] for f in FOLDS):
        publish(ctx, "checks.json", {**res, "ok": False, "stop": "native bags differ from the asserted counts / feasibility", "finished": now()}); raise SystemExit("native bags differ from the asserted counts: engineering stop before pixels")
    res["one_way"] = one_way_checks(ctx, s)
    F0 = Fold(ctx, 0); res["frames"] = frame_checks(F0, s); res["init"] = init_checks(ctx, s, F0, device); res["attention"] = attention_checks(device); res["formula"] = formula_checks()
    res["gradients"] = gradient_checks(ctx, s, F0, device); res["training"] = training_checks(ctx, s, F0, device, workers)
    res["a08_files_unchanged"] = all(sha256_file(ROOT / f"models_frozen/video_within_view_v1/all_s{k}/final.pt") == t31.A08[k] for k in (0, 2)); tm_ = readout_timing(ctx, s, device, workers); T = res["training"]["timing"]
    spw = max(T["longest_update"]["seconds_per_window"], T["updates_0_5"]["seconds_per_window_gpu"], T["updates_6_17_wall"]["seconds_per_window_wall"]); ovh = max(0.0, T["shortest_update"]["gpu_seconds"] - T["shortest_update"]["windows"] * spw)
    win = {f: res["bags"][str(f)]["plan"]["windows_total"] for f in FOLDS}; mon_w = 2 * N_MON * 12; spent = t31.vsh.ledger_hours(ctx.ledger, "gpu") + (time.time() - t_start) / 3600
    train_h = sum(2 * (win[f] * spw + U * ovh + 5 * mon_w * tm_["gpu_ms_per_row_one_model"] / 1e3 * 2) for f in FOLDS) * 1.15 / 3600
    nrows = sum(v["rows"] for v in EXPECTED_ELIGIBLE[s].values()); read_h = nrows * tm_["seconds_per_row_two_models"] * 1.15 / 3600; proj = spent + train_h + read_h + 0.4
    res["projection"] = {"spent_gpu_h_incl_this_check": spent, "train_4x600_h": train_h, "readout_eligible_rows_h": read_h, "reserve_h": 0.4, "source0_total_h": proj, "source2_if_triggered_h_approx": train_h + read_h,
                         "cap_h": ctx.v["budget"]["gpu_hours_max"], "seconds_per_window_used": spw, "per_update_overhead_s": ovh, "plan_windows": win, "eligible_rows": nrows, **tm_}
    npar = res["training"]["binding"]["groups"]; ntr = sum(g["parameters"] for g in npar.values()); nall = ntr + res["training"]["binding"]["frozen_parameters"]
    res["disk"] = {"free_gb": shutil.disk_usage(ctx.out).free / 1e9, "per_model_final_gb": 4 * nall / 1e9, "resume_gb": (4 * nall + 8 * ntr) / 1e9, "min_free_gb": ctx.v["budget"]["min_free_disk_gb"]}
    gs = gates(res); res["enforced_gates"] = {k: bool(v) for k, v in sorted(gs.items())}; res["required_gates_missing"] = sorted(REQUIRED_GATES - set(gs))
    res["loaded_code_sha256"] = t38.loaded_code(); res["failed"] = sorted(k for k, v in gs.items() if not v) + res["required_gates_missing"]; res["ok"] = not res["failed"]; res["finished"] = now(); publish(ctx, "checks.json", res)
    print(json.dumps({k: v for k, v in res.items() if k not in ("loaded_code_sha256", "bags")}, indent=1, default=str)[:14000], flush=True)
    if not res["ok"]:
        raise SystemExit(f"checks failed: {res['failed']}")
    return f"{spw:.3f} s per window; source-0 projection {proj:.2f} GPU h"


def gates(r: dict) -> dict:
    """Every enforced check gate, one dictionary item each; a missing required gate fails the check."""
    B, O, Fr, I, A, Fm, G, T, Pr, N = r["bags"], r["one_way"], r["frames"], r["init"], r["attention"], r["formula"], r["gradients"], r["training"], r["projection"], r["numerics"]; tk = I["tokens"]
    lim = lambda d: d["loss_max_abs"] <= LIMITS["loss"] and d["params_max_abs"] <= LIMITS["params"]  # noqa: E731
    bk = lambda f: B[str(f)]  # noqa: E731
    ex = EXPECTED_ELIGIBLE[0]; up = T["updates_1_5"]
    return {"isolation": not any(B["isolation"].values()),
            "bags": all(bk(f)["counts_equal_expected"] and bk(f)["feasibility_equal"] and bk(f)["whole_bags_pool_order"] and bk(f)["single_recording_natives"] and bk(f)["grid_equals_A07_windows"]
                        and bk(f)["weights_equal_independent_loop"] and bk(f)["weight_sum_max_dev"] < 1e-12 and bk(f)["bag_recordings_in_other_pool"] == 0 and bk(f)["bag_sessions_in_other_pool"] == 0 for f in FOLDS),
            "plan": all(bk(f)["plan"]["updates"] == U and bk(f)["plan"]["repeat_equal"] and bk(f)["plan"]["explicit_rng_equal"] and bk(f)["plan"]["a4c_in_list"] and bk(f)["plan"]["a5c_in_list"] for f in FOLDS),
            "numerics": not N["tf32_matmul"] and not N["tf32_cudnn"] and N["cudnn_deterministic"] and not N["cudnn_benchmark"] and N["deterministic"] and not N["deterministic_warn_only"],
            "pixels": Fr["pixels_equal_independent_decode"] and Fr["repeat_equal"], "markers": Fr["markers_equal_W"] and Fr["markers_equal_marker_slot"],
            "init_state": I["init_state_equals_A08"] and I["identical_arm_start"] and I["patched_blocks"] == [1, 3, 14],
            "init_head": I["head"]["parameters"] == HEAD_PARAMS and I["head"]["scorer_weight_all_zero"] and I["head"]["scorer_bias"] == wm.INIT_BIAS and I["head"]["attention_nonzero"] and not I["head"]["w_has_bias"] and I["head"]["fold1_head_differs"],
            "init_score_constant": I["init_scores_all_minus_2"]["all_exactly_minus_2"] and A["init_scores_minus_2_any_input"] and A["init_token_gradient_zero"] and A["init_scorer_gradient_nonzero"],
            "init_online_parity": I["init_online_fit_logits_vs_A31_cache_max_abs"] <= TOL["parity"] and I["hooked_embed_logits_equal_plain"], "patched_forward": I["patched_equals_original_forward_bitwise"],
            "base_from_cache": all(all(v[k] for k in ("runs_equal_base", "sequences_equal_base", "rows_equal_A32_base")) for v in I["Base_from_cached_q"].values()),
            "tokens": tk["U_shape"] == [1, wm.N_TOKENS, wm.DIM] and tk["final_grid_from_network"] == list(wm.GRID) and tk["U_keeps_autograd"] and tk["e_equals_plain_embed"] and tk["local_tokens_shape"] == [1, wm.N_SPATIAL, wm.DIM] and tk["selection_vs_loop_max_abs"] == 0,
            "time_alignment": tk["time_rule_equal"] and tk["batch_selection_equal_marker_div_2"], "hook_removal": tk["hook_removed_after_error"] and tk["no_hook_left"],
            "attention_reference": A["score_abs_vs_reference"] <= TOL["attention_fp64"] and A["attention_max_abs_vs_reference"] <= TOL["attention_fp64"] and A["attention_sums_to_1"] <= 1e-12 and A["permutation_score_abs_fp64"] <= 1e-12
            and A["permutation_attention_follows"] <= 1e-12 and A["permutation_score_abs_fp32"] <= 1e-4 and A["gradients_finite_nonzero"],
            "formula": all(v["l4_abs"] <= TOL["formula"] and v["l5_abs"] <= TOL["formula"] and v["total_abs"] <= TOL["formula"] and v["grad_max_abs"] <= TOL["formula"] and v["finite"] and v["nonmax_A5C_gradient_zero"] is not False
                           for k, v in Fm.items() if k != "one_instance_M_equals_D") and Fm["ties2_max"]["tied_windows"] == 2 and Fm["ties3_max"]["tied_windows"] == 3
            and Fm["one_instance_M_equals_D"]["loss_abs"] <= 1e-15 and Fm["one_instance_M_equals_D"]["grad_equal"],
            "two_pass_fp32": all(G[f"two_pass_vs_retained_{p}"]["relative"] <= TOL["grad_rel"] and G[f"two_pass_vs_retained_{p}"]["all_groups_nonzero"] and G[f"two_pass_vs_retained_{p}"]["loss_abs"] <= 1e-5 for p in ("dense", "max")),
            "replay_scores": all(G[f"two_pass_vs_retained_{p}"]["pass1_vs_replay_max_abs"] <= TOL["replay_abs"] and G[f"two_pass_vs_retained_{p}"]["pass1_vs_retained_scores_max_abs"] <= TOL["replay_abs"] for p in ("dense", "max"))
            and all(x["replay_max_abs"] <= TOL["replay_abs"] for x in up),
            "skip_pool_parity": G["patched_vs_original_maxpool_backward"]["relative"] <= TOL["grad_rel"] and G["warn_only_restored_false"],
            "binding": T["binding"]["binds_exactly_trainable"] and T["binding"]["trainable_equals_requires_grad"] and T["binding"]["frozen_only_projection_and_classifier"] and T["binding"]["eval_mode"]
            and T["binding"]["head_parameters"] == HEAD_PARAMS and T["binding"]["groups"]["witness"]["parameters"] == HEAD_PARAMS and T["binding"]["marker_group"] == ["marker_stem.weight"]
            and set(T["binding"]["frozen"]) == {"encoder.head.1.weight", "encoder.head.1.bias", "classifier.weight", "classifier.bias"},
            "first_step_zero": T["update0"]["g_encoder_trunk"] == 0 and T["update0"]["g_marker_stem"] == 0 and T["update0"]["g_attention"] == 0 and T["update0"]["g_scorer"] > 0,
            "genuine_gradients": all(all(np.isfinite(x[k]) and x[k] > 0 for k in ("g_encoder_trunk", "g_marker_stem", "g_attention", "g_scorer")) for x in up),
            "real_update": T["real_update_D_6"]["optimizer_steps"] == [6] and all(v["tensors_moved"] > 0 for v in T["real_update_D_6"]["moved"].values())
            and all(k.startswith(("m.encoder.head.", "m.classifier.")) or k in T["real_update_D_6"]["zero_gradient_trainable_tensors_updates_1_5"] for k in T["real_update_D_6"]["unchanged_tensors"]),
            "repeat": lim(T["repeat_6_D"]), "forced_dense": lim(T["forced_dense_M_vs_D"]) and T["forced_dense_M_vs_D"]["formal_map_unchanged"], "resume": lim(T["resume_3_plus_3_M"]),
            "M_not_D": T["M_differs_from_D"]["params_max_abs"] > 0,
            "one_way_synthetic": O["synthetic_rule"]["result"] == O["synthetic_rule"]["expected"] and O["synthetic_rule"]["input_unchanged"] and O["synthetic_merge"]["edited"] == O["synthetic_merge"]["expected"]
            and O["synthetic_merge"]["boundaries_before_after"] == [4, 3],
            "eligible_counts": all(O[f"heldout_f{h}"]["eligible_runs"] == ex[h]["runs"] and O[f"heldout_f{h}"]["eligible_unique_rows"] == ex[h]["rows"] and O[f"heldout_f{h}"]["full_grid"] == ex[h]["grid"]
                                   and O[f"heldout_f{h}"]["rows_inside_fold"] for h in FOLDS),
            "run_alignment": all(O[f"heldout_f{h}"]["run_alignment"] for h in FOLDS),
            "init_equals_base": all(O[f"heldout_f{h}"][k] for h in FOLDS for k in ("init_r_all_minus_2", "init_pred_equals_base", "init_sequences_equal_base", "init_tables_equal_Base_tables")),
            "readout_label_free": all(O[f"heldout_f{h}"][k] for h in FOLDS for k in ("label_free_decision", "random_scores_non_A4C_unchanged", "random_scores_changes_only_2_to_3", "random_sequences_change_only_changed_runs")),
            "a08_files": r["a08_files_unchanged"], "memory": T["timing"]["peak_gpu_gb_training"] <= TOL["peak_gpu_gb"], "projection": Pr["source0_total_h"] <= Pr["cap_h"], "disk": r["disk"]["free_gb"] >= r["disk"]["min_free_gb"]}


# ----------------------------------------------------------------------------------------------- readout
def tables_pred(pred: np.ndarray, D: dict, a8cfg, P: dict, ev) -> dict:
    """A36's tables from run labels directly: the run table at true prevalence and the complete-sequence evaluation of Base's
    per-sample labels with the changed runs relabelled (equal neighbours merge)."""
    pk, seqs = D["pk"], D["seqs"]; y, b = pk["labels"], pk["base"]; M = D["M"]; ok5 = (y == A5C) & (b == A5C); preds = wm.edit_sequences(D["base_preds"], pk, seqs, pred); sr = t31.seq_rows(a8cfg, P, preds, ev)
    return {"runs": t31.run_table(np.eye(5)[pred], pk, seqs, D["sessions"]), "changes": d32.change_table(y, b, pred), "correct_A5C_kept": {"runs": int(ok5.sum()), "kept": int((ok5 & (pred == A5C)).sum())},
            "S1_fixed": {"runs": int(M["S1"].sum()), "fixed": int((M["S1"] & (pred == A5C)).sum())}, "S2_kept": {"runs": int(M["S2"].sum()), "kept": int((M["S2"] & (pred == A5C)).sum())},
            "sequence_row": sr["row"], "counts": rsh.counts_of(seqs, sr), "pred": pred, "preds": preds}


def bounds(D: dict, preds: dict) -> dict:
    nb = sum(int((np.diff(D["base_preds"][q["id"]]) != 0).sum()) for q in D["seqs"]); nf = sum(int((np.diff(preds[q["id"]]) != 0).sum()) for q in D["seqs"])
    return {"base_boundaries": nb, "final_boundaries": nf, "deleted_by_merge": nb - nf}


def load_model(ctx: Ctx, arm: str, s: int, f: int, device):
    """The committed final weights in the original A08 module (original nn.MaxPool3d; no backward at readout) and head."""
    key = f"{arm}_s{s}_f{f}"; rd = ctx.out / "train" / key; man = json.loads((rd / "run_manifest.json").read_text()); com = json.loads((ctx.out / "weights_committed.json").read_text()).get(key, {})
    if sha256_file(rd / "final.pt") != man["final_sha256"] or sha256_file(rd / "head.pt") != man["head_sha256"] or com != {"final": man["final_sha256"], "head": man["head_sha256"]}:
        raise SystemExit(f"{key}: weights not committed or changed")
    m = ctx.a31.a08(s, device); m.load_state_dict(torch.load(rd / "final.pt", map_location="cpu", weights_only=True), strict=True); head = wm.GatedWitness().to(device)
    head.load_state_dict(torch.load(rd / "head.pt", map_location="cpu", weights_only=True), strict=True); return m.eval(), head.eval()


def specialist_scores(ctx: Ctx, h: int, models: dict, need: np.ndarray, device, workers: int, acc: dict) -> dict:
    """Online scores of the frozen specialists at the eligible rows of held-out fold h only (fold-local rows; decoded once)."""
    seqs = t31.load_pool(ctx.a31, h); W = t31.pool_W(seqs); fp = t28.frame_paths(seqs); S = {a: np.full(len(W["seq"]), np.nan, np.float32) for a in models}; done = 0
    for c0 in range(0, len(seqs), 100):
        rows = need[(W["seq"][need] >= c0) & (W["seq"][need] < c0 + 100)]
        if not len(rows):
            continue
        a = time.time(); fr, idx = t37.chunk_frames(seqs, {"seq": W["seq"][rows]}, W["s"][rows], fp, workers); acc["decode_seconds"] += time.time() - a
        frd, idxd, mkd = torch.from_numpy(fr).to(device), torch.from_numpy(idx).to(device), torch.from_numpy(W["marker"][rows].astype(np.int64)).to(device)
        for arm, (m, head) in models.items():
            torch.cuda.synchronize(); a = time.time()
            with torch.no_grad():
                S[arm][rows] = torch.cat([window_score(m, head, frd, idxd, mkd, w) for w in range(len(rows))]).float().cpu().numpy()
            torch.cuda.synchronize(); acc["gpu_seconds"] += time.time() - a
        done += len(rows); del fr, frd; print(f"held-out fold {h}: {done}/{len(need)} eligible rows; GPU {acc['gpu_seconds']:.0f} s", flush=True)
    if not all(np.isfinite(S[a][need]).all() for a in S) or done != len(need):
        raise SystemExit("an eligible-row score is missing or not finite")
    return S


def score_summary(r: np.ndarray, y: np.ndarray, el: np.ndarray) -> dict:
    """Aggregate r of the eligible runs by reference class (A5C = S1 targets, A4C = correct, other)."""
    out = {}
    for name, m_ in (("true_A5C", el & (y == A5C)), ("true_A4C", el & (y == A4C)), ("other", el & (y != A5C) & (y != A4C))):
        x = r[m_]; out[name] = {"runs": int(m_.sum()), "r_positive": int((x > 0).sum()), **({"median": float(np.median(x)), "q90": float(np.quantile(x, 0.9)), "max": float(x.max())} if len(x) else {})}
    return out


def readout(ctx: Ctx, s: int, device, workers: int, dry: bool = False) -> str:
    name = f"readout_s{s}.json"
    if (ctx.out / name).exists() and not dry:
        raise SystemExit("this seed's readout exists; it runs once")
    X = t40.full_q(ctx, s); a8 = rsh.a08_ctx(); data = t31.vpr.training_data(ctx.a31.a15.a10.a07p); rec = {"seed": s, "cells": {}}; acc = {m_: [] for m_ in ("Base", *ARMS)}; ys, bs = [], []
    cov = {a: [] for a in ARMS}; s1recs = {a: [] for a in ARMS}; old = json.loads((ROOT / "docs/results/2026-10-25_a5c_source_external_evidence/readout.json").read_text())["sequences"]; tim = {"gpu_seconds": 0.0, "decode_seconds": 0.0}
    a42 = json.loads((ROOT / "docs/results/2026-10-27_session_matched_discrimination/readout_s0.json").read_text())["cells"] if s == 0 else None; cache = {}; nrows = 0
    for h in FOLDS:
        f = 1 - h; D = d32.fold_data(ctx.d, s, h); D["M"] = d32.set_masks(D); off = X["offset"][h]; pk = D["pk"]; P = {"seqs": D["seqs"], "n_of": dict(data.n), "label_of": dict(data.label)}; ev = bc.evalset(P, list(range(len(D["seqs"]))), f"heldout_f{h}")
        el = pk["base"] == A4C; need = eligible_rows(pk, el) - off; nW = len(t31.pool_W(D["seqs"])["seq"]); nrows += len(need)
        if s in EXPECTED_ELIGIBLE and (int(el.sum()), len(need), nW) != tuple(EXPECTED_ELIGIBLE[s][h][k] for k in ("runs", "rows", "grid")):
            raise SystemExit(f"held-out fold {h}: eligible runs / rows differ from the asserted counts")
        if dry:
            S = {a: np.full(nW, -2.0, np.float32) for a in ARMS}
        else:
            S = specialist_scores(ctx, h, {a: load_model(ctx, a, s, f, device) for a in ARMS}, need, device, workers, tim); torch.cuda.empty_cache()
        cell = {}; t = tables(X["q"], D, off, a8.cfg, P, ev); p = t.pop("pred"); acc["Base"].append(p); cell["Base"] = t; cell["Base"]["boundaries"] = t38.boundaries(X["q"], D, off); cell["Base"].update(a5c_extras(D, p)[0])
        for arm in ARMS:
            r = wm.run_max(pk, S[arm], off, el); p = wm.one_way(pk["base"], r); t = tables_pred(p, D, a8.cfg, P, ev); preds = t.pop("preds"); p = t.pop("pred"); acc[arm].append(p); cell[arm] = t
            ch = p != pk["base"]; cell[arm]["structural"] = {"eligible_runs": int(el.sum()), "eligible_rows": int(len(need)), "changed_runs": int(ch.sum()), "non_A4C_runs_unchanged": bool(np.array_equal(p[~el], pk["base"][~el])),
                                                         "changes_only_A4C_to_A5C": bool(set(zip(pk["base"][ch].tolist(), p[ch].tolist())) <= {(A4C, A5C)}),
                                                         "changed_by_true_class": {str(c): int((ch & (pk["labels"] == c)).sum()) for c in range(5)}}
            cell[arm]["eligible_scores"] = score_summary(r, pk["labels"], el); cell[arm]["boundaries"] = bounds(D, preds)
            cell[arm]["coverage"], sets = t37.coverage(D, pk["labels"], pk["base"], p); cov[arm].append(sets); ex, recs = a5c_extras(D, p); cell[arm].update(ex); s1recs[arm].append(recs)
            cache[f"scores_{arm}_heldout_f{h}"] = S[arm][need]; cache[f"r_{arm}_heldout_f{h}"] = r[el]; cache[f"run_pred_{arm}_heldout_f{h}"] = p
        cache[f"rows_heldout_f{h}"] = need; cache[f"eligible_runs_heldout_f{h}"] = np.flatnonzero(el); cache[f"run_pred_Base_heldout_f{h}"] = acc["Base"][-1]; cache[f"run_label_heldout_f{h}"] = pk["labels"]
        s1 = np.flatnonzero(D["M"]["S1"]); cell["S1_denominator"] = {"runs": int(len(s1)), **d32.concentration([(D, int(i)) for i in s1], "true_src", None)}
        ob = old[f"s{s}_heldout_f{h}"]["A08 Base"]; cell["Base_reproduced"] = {"A32_row": cell["Base"]["sequence_row"] == ob["sequence_row"] and cell["Base"]["counts"] == ob["counts"]}
        if a42 is not None:
            js = lambda x: json.loads(json.dumps(x, default=float))  # noqa: E731
            cell["Base_reproduced"]["A42_Base_cell"] = all(js(cell["Base"][k]) == a42[f"heldout_f{h}"]["Base"][k] for k in a42[f"heldout_f{h}"]["Base"] if k in cell["Base"])
        row = lambda m_: cell[m_]["sequence_row"]  # noqa: E731
        cell["comparisons"] = {f"{a} - {bb}": {"net_fixed": cell[a]["runs"]["all"]["net_fixed"] - (cell[bb]["runs"]["all"]["net_fixed"] if bb != "Base" else 0), "S1_fixed": cell[a]["S1_fixed"]["fixed"] - cell[bb]["S1_fixed"]["fixed"],
                                               "correct_A5C_kept": cell[a]["correct_A5C_kept"]["kept"] - cell[bb]["correct_A5C_kept"]["kept"], "correct_A4C_kept": cell[a]["correct_A4C_kept"]["kept"] - cell[bb]["correct_A4C_kept"]["kept"],
                                               **{k: row(a)[k] - row(bb)[k] for k in ("segment_f1_50_multi", "native_macro_f1", "boundary_recall_multi", "boundary_precision_multi")},
                                               **{k: cell[a]["counts"][k] - cell[bb]["counts"][k] for k in ("fp", "fp_short", "fn", "fn_short")}} for a, bb in (("M", "D"), ("M", "Base"), ("D", "Base"))}
        if not dry:
            mans = {arm: json.loads((ctx.out / "train" / f"{arm}_s{s}_f{f}" / "run_manifest.json").read_text()) for arm in ARMS}
            cell["fit_only"] = {arm: {k: mans[arm][k] for k in ("positive", "pi", "monitors", "drift", "loss", "windows_trained", "replay_max_abs_max")} for arm in ARMS}
        rec["cells"][f"heldout_f{h}"] = cell; ys.append(pk["labels"]); bs.append(pk["base"])
    y, b = np.concatenate(ys), np.concatenate(bs); wr = b != y; rec["oof"] = {}
    for m_ in ARMS:
        p = np.concatenate(acc[m_]); fx, br = int((wr & (p == y)).sum()), int((~wr & (p != y)).sum()); un = {k: set().union(*[c[k] for c in cov[m_]]) for k in ("fixed", "broken")}
        rec["oof"][m_] = {"runs": int(len(y)), "fixed": fx, "broken": br, "net_fixed": fx - br, **{f"{k}_recordings": len(v) for k, v in un.items()}, **{f"{k}_sessions": len({session_of(x) for x in v}) for k, v in un.items()},
                          "S1_fixed_true_class_recordings": len(set().union(*s1recs[m_])), "broken_correct_A5C": int(((b == y) & (y == A5C) & (p != y)).sum()), "broken_correct_A4C": int(((b == y) & (y == A4C) & (p != y)).sum()),
                          "fixed_true_A5C_read_A4C": int(((y == A5C) & (b == A4C) & (p == A5C)).sum()), "changed_runs": int((p != b).sum()), "non_A4C_runs_unchanged": bool(np.array_equal(p[b != A4C], b[b != A4C]))}
    rec["rule"], rec["values"], rec["summary"] = rule_of(rec, s1recs)
    if dry:
        print(json.dumps({"dry_run_init": True, "Base_reproduced": {k: v["Base_reproduced"] for k, v in rec["cells"].items()}, "summary": rec["summary"], "oof": rec["oof"], "rule": rec["rule"]}, indent=1)); return "dry run (Init); nothing published"
    cp = ctx.out / f"readout_cache_s{s}.npz"; np.savez(cp, **cache); pr = json.loads((ctx.a31.out / "cache" / f"prefix_s{s}.json").read_text())
    rec["private_cache"] = {"path": f"{ctx.v['outputs']['root']}/readout_cache_s{s}.npz (server only)", "sha256": sha256_file(cp), "arrays": sorted(cache), "rows": "fold-local pool_W rows of the eligible runs (sorted unique)"}
    rec["deployment_cost"] = {"forwards": "Base's one 224 forward per window plus one specialist forward (an A08-size encoder + 197,761-parameter head) per unique row of a Base-A4C run only", "parameters": "A08 34.6M (Base) + 34.6M + 0.198M (specialist)",
                              "eligible_rows": nrows, "gpu_ms_per_row_specialist": 1e3 * tim["gpu_seconds"] / (2 * nrows), "cpu_decode_ms_per_row_shared": 1e3 * tim["decode_seconds"] / nrows,
                              "gpu_ms_per_window_A31_base_pass": 1e3 * pr["gpu_seconds"] / sum(pr["rows"].values()), "readout_gpu_seconds": tim["gpu_seconds"], "readout_decode_seconds": tim["decode_seconds"], "note": "two specialists (D, M) per eligible row share one decode; microbatch 1"}
    rec["written"] = now(); publish(ctx, name, rec)
    print(json.dumps({"summary": rec["summary"], "oof": rec["oof"], "rule": rec["rule"], "values": rec["values"], "Base_reproduced": {k: v["Base_reproduced"] for k, v in rec["cells"].items()}}, indent=1)); return f"absolute gate D {rec['rule']['D']['passed']} M {rec['rule']['M']['passed']}"


def rule_of(rec: dict, s1recs: dict) -> tuple[dict, dict, dict]:
    C = lambda h: rec["cells"][f"heldout_f{h}"]  # noqa: E731
    tot = lambda m_, k, sub: sum(C(h)[m_][k][sub] for h in FOLDS)  # noqa: E731
    d_ = lambda a, bb, h, k: C(h)[a]["sequence_row"][k] - C(h)[bb]["sequence_row"][k]  # noqa: E731
    net = lambda a, h: C(h)[a]["runs"]["all"]["net_fixed"]  # noqa: E731
    n1, n4 = tot("Base", "S1_fixed", "runs"), tot("Base", "correct_A4C_kept", "runs"); out = {}
    for arm in ARMS:
        s1 = tot(arm, "S1_fixed", "fixed"); rf = [len(r) for r in s1recs[arm]]; rall = len(set().union(*s1recs[arm])); k4 = tot(arm, "correct_A4C_kept", "kept"); dF = [d_(arm, "Base", h, "segment_f1_50_multi") for h in FOLDS]
        g = {"S1_fixed": f"{s1}/{n1}", "S1_needed": ceil_pct(10, n1), "S1_ok": s1 >= ceil_pct(10, n1), "S1_recordings_total": rall, "S1_recordings_by_fold": rf, "S1_runs_by_fold": [C(h)[arm]["S1_fixed"]["fixed"] for h in FOLDS],
             "S1_sources_ok": rall >= 3 and all(C(h)[arm]["S1_fixed"]["fixed"] >= 1 and len(s1recs[arm][i]) >= 1 for i, h in enumerate(FOLDS)),
             "non_A4C_invariant_ok": all(C(h)[arm]["structural"]["non_A4C_runs_unchanged"] for h in FOLDS), "correct_A4C_kept": f"{k4}/{n4}", "A4C_needed": ceil_pct(99, n4), "A4C_ok": k4 >= ceil_pct(99, n4),
             "net_by_fold": [net(arm, h) for h in FOLDS], "net_each_fold_ge_0": all(net(arm, h) >= 0 for h in FOLDS), "net_oof": rec["oof"][arm]["net_fixed"], "net_oof_ge_5": rec["oof"][arm]["net_fixed"] >= 5,
             "dF_by_fold": dF, "dF_each_fold_ge_0": all(x >= 0 for x in dF), "dF_mean": float(np.mean(dF)), "dF_mean_ge_0.001": float(np.mean(dF)) >= 0.001,
             "native_each_fold_ge_-0.003": all(d_(arm, "Base", h, "native_macro_f1") >= -0.003 for h in FOLDS), "boundary_recall_each_fold_ge_-0.005": all(d_(arm, "Base", h, "boundary_recall_multi") >= -0.005 for h in FOLDS),
             "missed_short": tot(arm, "counts", "fn_short"), "missed_short_Base": tot("Base", "counts", "fn_short"), "missed_short_le_Base_plus_3": tot(arm, "counts", "fn_short") <= tot("Base", "counts", "fn_short") + 3}
        g["passed"] = all(v for v in g.values() if isinstance(v, bool)); out[arm] = g
    k4M, k4D = tot("M", "correct_A4C_kept", "kept"), tot("D", "correct_A4C_kept", "kept")
    mil = {"M_minus_D_S1_by_fold": [C(h)["M"]["S1_fixed"]["fixed"] - C(h)["D"]["S1_fixed"]["fixed"] for h in FOLDS]}
    mil.update({"S1_each_fold_ge_1": all(x >= 1 for x in mil["M_minus_D_S1_by_fold"]), "S1_oof_ge_5": sum(mil["M_minus_D_S1_by_fold"]) >= 5, "net_M_ge_D_each_fold": all(net("M", h) >= net("D", h) for h in FOLDS),
                "F1_M_ge_D_each_fold": all(d_("M", "D", h, "segment_f1_50_multi") >= 0 for h in FOLDS), "A4C_kept_M_ge_D_minus_3": k4M >= k4D - 3})
    mil["met"] = all(v for v in mil.values() if isinstance(v, bool)); out["mil_signal"] = mil
    out["source2_by_rule"] = out["D"]["passed"] or out["M"]["passed"]; out["note"] = "source 2 additionally requires the remaining shared budget (checked by the owner before launch)"
    values = {"M_minus_D_net": [net("M", h) - net("D", h) for h in FOLDS], "D_minus_Base_net": [net("D", h) for h in FOLDS], "M_minus_Base_net": [net("M", h) for h in FOLDS],
              "M_minus_D_dF": [d_("M", "D", h, "segment_f1_50_multi") for h in FOLDS], "D_minus_Base_dF": [d_("D", "Base", h, "segment_f1_50_multi") for h in FOLDS], "M_minus_Base_dF": [d_("M", "Base", h, "segment_f1_50_multi") for h in FOLDS],
              "native": {a: [d_(a, "Base", h, "native_macro_f1") for h in FOLDS] for a in ARMS}, "boundary_recall": {a: [d_(a, "Base", h, "boundary_recall_multi") for h in FOLDS] for a in ARMS},
              "missed_short": {m_: tot(m_, "counts", "fn_short") for m_ in ("Base", *ARMS)}}
    summary = {f"heldout_f{h}": {m_: {"net": net(m_, h) if m_ != "Base" else 0, "F1": C(h)[m_]["sequence_row"]["segment_f1_50_multi"], "S1": C(h)[m_]["S1_fixed"]["fixed"], "A5C_kept": C(h)[m_]["correct_A5C_kept"]["kept"],
                                       "A4C_kept": C(h)[m_]["correct_A4C_kept"]["kept"]} for m_ in ("Base", *ARMS)} for h in FOLDS}
    return out, values, summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True); ap.add_argument("--stage", choices=("check", "train", "freeze", "readout"), required=True); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6); ap.add_argument("--commit", default=""); ap.add_argument("--manifest-sha", default=""); ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(); a.set = []; ctx = Ctx(load_cfg(a), a.config); ctx.out.mkdir(parents=True, exist_ok=True)
    if a.stage == "freeze":
        return staged(ctx, f"freeze s{a.seed} (CPU)", "cpu", lambda: freeze(ctx, a.seed, a.commit, a.manifest_sha))
    if a.stage == "readout" and a.dry_run:
        return staged(ctx, f"readout dry run s{a.seed} (Init, CPU)", "cpu", lambda: readout(ctx, a.seed, None, a.workers, True))
    t31.vsh.determinism(); torch.use_deterministic_algorithms(True, warn_only=False); device = torch.device("cuda")
    if a.stage == "train":
        return train(ctx, a.seed, device, a.workers)
    fn = {"check": lambda: check(ctx, device, a.workers), "readout": lambda: readout(ctx, a.seed, device, a.workers)}[a.stage]
    staged(ctx, f"{a.stage}{'' if a.stage == 'check' else ' s' + str(a.seed)} (GPU job wall time)", "gpu", fn)


if __name__ == "__main__":
    main()
