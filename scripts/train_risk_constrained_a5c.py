#!/usr/bin/env python
"""A44 (configs/experiments/video_risk_constrained_a5c_v1.yaml): an end-to-end repair policy for original Base-A4C runs (A08
copy, every parameter but the five-class classifier trainable, Linear(512, 1) head 0 / -2 on the unchanged embed output),
trained on the fit fold's actual Base-A4C P runs with their original multiplicity weights. N (hinge false-promotion
constraint on correct A4C with an adaptive dual variable) against C (class-balanced binary cross-entropy), same plan and
compute, deployed only as a one-way change of original Base-A4C runs (weighted-mean score > 0).

  --stage plan     lists / plans / exposure against the asserted counts and plan sha256 (CPU, metadata only)
  --stage check    inputs, lists, membership, plans, pixels, markers, Init, formula / dual references, two-pass vs
                   retained-graph gradients, skip-pool parity, binding, first-step zero / genuine gradients, repeat,
                   forced C-objective N path, 3 + 3 resume, one-way edit / eligible counts / Init = Base, timing /
                   memory / disk and the projection (GPU)
  --stage train    --seed s: C_f0, N_f0, C_f1, N_f1, resumable (GPU); full-fit scoring at update 0 and 600
  --stage freeze   --seed s --commit c --manifest-sha h: record that weights_s{s}.json (sha h) was committed in c
  --stage readout  --seed s: after the weights are committed, one opposite-fold scoring of the eligible rows (+ rule) (GPU);
                   --dry-run reads Init (every score -2) through the same tables and publishes nothing (CPU)
The frozen A08 + B3 path supplies q, Base and its runs; nothing here changes it.
"""
from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

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
import scripts.train_a5c_witness_mil as t43  # noqa: E402
import scripts.train_ensemble_teacher as tet  # noqa: E402
import scripts.train_interval_recognition as t27  # noqa: E402
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
import echo_routing.temporal.risk_constrained as rc  # noqa: E402
import echo_routing.temporal.session_pairs as sp  # noqa: E402
import echo_routing.vit.target_mvit as tm  # noqa: E402
from echo_routing.evaluate.complementarity import sha256_file  # noqa: E402
from echo_routing.evaluate.failure_units import session_of  # noqa: E402
from echo_routing.temporal.dev_train import lr_factor  # noqa: E402

A31_CONFIG, A32_CONFIG = t40.A31_CONFIG, t40.A32_CONFIG
FOLDS, ARMS, U, WARM, SAVE_EVERY, LOG_EVERY = (0, 1), rc.ARMS, 600, 60, 150, 50
LR = {"encoder_trunk": 1e-5, "marker_stem": 1e-4, "projection": 1e-4, "scalar_head": 1e-3}
CLIP, FROZEN = 1.0, ("classifier.",)
SCH = {"updates": U, "warmup": WARM}
LIMITS = {"loss": 1e-4, "params": 1e-5}
TOL = {"grad_rel": 1e-5, "formula": 1e-8, "replay_abs": 1e-5, "parity": 1e-4, "peak_gpu_gb": 40.0, "roundoff_grad_rel": 1e-6, "membership": 1e-12}
PINS = {"pools": {"0": "96d1fb683e1a261d5be9a3cc7ec393637577bbf83b56140cec41f544c4f3b486", "1": "08255192dae3a90644ea062880bf6b2b298181ed12e9890d56ca04630f5adef4"},
        "split": "e03100941d77e1fe64a825e4f2a4f7ef038a5725bcf943a55db15a983ef29d6d",
        "outputs": {0: {"0": "5884435770e8b65959b22ff77f74b0c2053cadf7fd2a5a28c842104df2f1a512", "1": "0a99be35b0cd78d650f5014c8ddc783173e6fb849ee556d6fa2a52c73f33d360"},
                    2: {"0": "5e8042a75e7c38d2f8345c3c3bbf357039efeed530f8758a1a7c7838fe5507fb", "1": "2b1f1e13dce2a769f4091745e531bb998bc122b58cd5f3c09d2c364d904f4a33"}},
        "runs": {0: {"0": "7b28ac6c224f594ef7bf49d6307186807d91bd8577dbc192e8cece07bea7452c", "1": "98ce5c572aa29576a098685c1d28220684e755cbea537b6739020c0387ddc026"},
                 2: {"0": "91d04d4f255b7e892343b102c736f1b69c8126a69f2e0df1f5c23f1c60d8860f", "1": "ef76e587a0db2c7b18dcbdaee5631497963ad8eb9988a929e4526d5949259ef9"}}}
EXPECTED_LISTS = {0: {0: {"positives": 34, "negatives": 714, "other": 1, "eligible": 749}, 1: {"positives": 69, "negatives": 699, "other": 3, "eligible": 771}}}
EXPECTED_PLAN = {0: {0: {"sha256": "3b3be337e7d22a1f1a96196cfc4460b4425ae367f407f6a16a01e5b2cc5c3efc", "occurrences_total": 14778, "occurrences_max_update": 129},
                     1: {"sha256": "2a1dd83d9e8b5f2025ccb6e4bd90587bae15ee9b42bb27fe7f6c71ab6ed463a8", "occurrences_total": 15528, "occurrences_max_update": 88}}}
EXPECTED_ELIGIBLE = t43.EXPECTED_ELIGIBLE                                                   # source seed -> pool -> runs / rows / grid
KEY_NORM_BIAS = t43.KEY_NORM_BIAS
REQUIRED_GATES = {"isolation", "inputs", "lists", "membership", "plan", "numerics", "pixels", "markers", "init_state", "init_head", "init_score_constant", "init_online_parity", "patched_forward",
                  "base_from_cache", "formula", "dual", "two_pass_fp32", "replay_scores", "zero_loss_negatives", "skip_pool_parity", "binding", "first_step_zero", "genuine_gradients", "real_update",
                  "repeat", "forced_C_objective", "resume", "N_not_C", "one_way_synthetic", "eligible_counts", "run_alignment", "init_equals_base", "readout_label_free", "deploy_mean", "a08_files",
                  "memory", "projection", "disk"}
A4C, A5C = rc.A4C, rc.A5C
now, write_json, ledger_append, a5c_extras, ceil_pct, tables = t31.now, t31.write_json, t31.ledger_append, t40.a5c_extras, t40.ceil_pct, t40.tables
nearest_loop, tables_pred, bounds, eligible_rows, score_summary, rel, sq_norm, params_max_diff = t43.nearest_loop, t43.tables_pred, t43.bounds, t43.eligible_rows, t43.score_summary, t43.rel, t43.sq_norm, t43.params_max_diff


def model_seed(s: int, f: int) -> int:
    return 2026102844 + 10 * int(s) + int(f)


class Ctx:
    def __init__(self, cfg: dict, config: str) -> None:
        self.config, self.v = config, cfg["risk_constrained_a5c"]; o = self.v["outputs"]
        self.out, self.results, self.ledger = ROOT / o["root"], ROOT / o["results_dir"], ROOT / self.v["budget"]["ledger"]
        self.a31 = t31.Ctx(load_cfg(SimpleNamespace(config=A31_CONFIG, set=[])), A31_CONFIG); self.d = d32.Ctx(load_cfg(SimpleNamespace(config=A32_CONFIG, set=[])), A32_CONFIG)


def staged(ctx: Ctx, job: str, kind: str, fn) -> None:
    t40.staged(ctx, job, kind, fn)


def publish(ctx: Ctx, name: str, rec: dict) -> None:
    t40.publish(ctx, name, rec)


# ----------------------------------------------------------------------------------------------- fit fold, plan, frames
class Fold:
    """One fit fold under source seed s: pool, window grid, Base P runs (fold-local rows), training lists and the plan."""

    def __init__(self, ctx: Ctx, s: int, f: int) -> None:
        self.s, self.f, self.seqs = s, f, t31.load_pool(ctx.a31, f); self.W = t31.pool_W(self.seqs); self.fp = t28.frame_paths(self.seqs)
        R = t31.runs(ctx.a31, s, f); self.pk, self.base_labels = R["P"], R["base_labels"]
        self.off = int(json.loads((ctx.a31.out / "cache" / f"prefix_s{s}.json").read_text())["offsets"][str(f)])
        self.L = rc.training_lists(self.pk); self.pl = rc.plan(self.L["positives"], self.L["negatives"], s, f, U); self.psha = rc.plan_sha(self.pl)
        self.el = self.pk["base"] == A4C; self.need = eligible_rows(self.pk, self.el) - self.off

    def occ(self, u: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return rc.occurrences(self.pk, self.pl[int(u)], self.off)

    def frames(self, rows: np.ndarray, workers: int = 0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        fr, idx = t37.chunk_frames(self.seqs, {"seq": self.W["seq"][rows]}, self.W["s"][rows], self.fp, workers)
        return torch.from_numpy(fr), torch.from_numpy(idx), torch.from_numpy(self.W["marker"][rows].astype(np.int64))


class Updates(Dataset):
    """Item = (update, rows, weights, run slots, unique uint8 frames, (n, 16) indices, marker slots)."""

    def __init__(self, F_: Fold, us) -> None:
        self.F, self.us = F_, list(us)

    def __len__(self) -> int:
        return len(self.us)

    def __getitem__(self, k: int):
        u = self.us[k]; rows, v, seg = self.F.occ(u); fr, idx, mk = self.F.frames(rows)
        return (u, torch.from_numpy(rows), torch.from_numpy(v).float(), torch.from_numpy(seg), fr, idx, mk)


def loader(F_: Fold, us, workers: int):
    return DataLoader(Updates(F_, us), batch_size=None, num_workers=workers, prefetch_factor=2 if workers else None, pin_memory=True)


def lists_report(F_: Fold) -> dict:
    """Counts, kinds, purity, lengths and source exposure of the training lists (no ids)."""
    pk, seqs = F_.pk, F_.seqs; out = {}
    for name, idx in F_.L.items():
        if not len(idx):
            out[name] = {"runs": 0}; continue
        kinds = Counter(seqs[int(pk["seq"][i])]["kind"] for i in idx); pur = pk["purity"][idx]; L = (pk["b"] - pk["a"])[idx]
        srcs = [Counter(r[0] for r in seqs[int(pk["seq"][i])]["refs"][int(pk["a"][i]):int(pk["b"][i])]) for i in idx]
        per_rec = Counter(v for c in srcs for v in c); recs = set(per_rec)
        out[name] = {"runs": int(len(idx)), "kinds": dict(kinds), "purity_mean": float(pur.mean()), "pure_share": float((pur >= 1.0).mean()), "length_mean": float(L.mean()), "length_max": int(L.max()),
                     "short_le16": int((L <= 16).sum()), "distinct_recordings": len(recs), "distinct_sessions": len({session_of(v) for v in recs}), "multi_source_runs": int(sum(len(c) > 1 for c in srcs)),
                     "runs_per_recording_max": int(max(per_rec.values())), "runs_per_recording_mean": float(np.mean(list(per_rec.values()))), "unique_rows": int(sum(pk["offsets"][i + 1] - pk["offsets"][i] for i in idx))}
    return out


# ----------------------------------------------------------------------------------------------- model, two-pass update
def build(ctx: Ctx, s: int, f: int, device, patched: bool = True):
    """The A08 copy (classifier frozen, everything else trainable, eval mode; deterministic skip pools on training copies)
    and the scalar head constructed from the model RNG right after the copy (then set to 0 / -2)."""
    m = ctx.a31.a08(s, device)
    for n, p in m.named_parameters():
        p.requires_grad_(not n.startswith(FROZEN))
    if patched:
        sp.deterministic_skip_pools(m.encoder)
    torch.manual_seed(model_seed(s, f)); head = rc.ScalarHead().to(device)
    return m.eval(), head.eval()


def groups_of(m, head) -> dict:
    return {"encoder_trunk": [p for n, p in m.encoder.named_parameters() if not n.startswith("head.")], "marker_stem": [m.marker_stem.weight],
            "projection": [m.encoder.head[1].weight, m.encoder.head[1].bias], "scalar_head": [head.fc.weight, head.fc.bias]}


def make_opt(m, head):
    G = groups_of(m, head); return t31.opt_of([(k, G[k], LR[k]) for k in ("encoder_trunk", "marker_stem", "projection", "scalar_head")])


def trainable(m, head) -> list:
    return [p for ps in groups_of(m, head).values() for p in ps]


def window_score(m, head, frd: torch.Tensor, idx: torch.Tensor, mk: torch.Tensor, w: int) -> torch.Tensor:
    """(1,) score of occurrence w: one full original marker forward through the unchanged embed, microbatch 1."""
    return rc.score(m, head, tm.to_rgb(frd[idx[w:w + 1]]), tm.marker_planes(mk[w:w + 1]))


def two_pass(m, head, frd, idx, mk, v: torch.Tensor, seg: torch.Tensor, arm: str, lam: float) -> dict:
    """Pass 1: every occurrence's score without a graph; dL/ds from the objective on the weighted run means of a leaf vector.
    Pass 2: every occurrence again with autograd and s.backward(dL/ds) (zero-gradient windows included); .grad accumulates."""
    n = idx.shape[0]
    with torch.no_grad():
        s1 = torch.cat([window_score(m, head, frd, idx, mk, w) for w in range(n)])
    sl = s1.detach().clone().requires_grad_(True); r = rc.run_means(sl, v, seg, 1 + rc.N_NEG); L, A, B, H = rc.objective(r, arm, lam); L.backward(); g = sl.grad.detach()
    s2 = torch.empty_like(s1)
    for w in range(n):
        sw = window_score(m, head, frd, idx, mk, w); sw.backward(g[w:w + 1]); s2[w] = sw.detach()[0]
    return {"s1": s1, "s2": s2, "g": g, "r": r.detach(), "L": float(L.detach()), "A": float(A.detach()), "B": float(B.detach()), "H": float(H.detach())}


def update_record(u: int, r: dict, seg: torch.Tensor, lam: float, arm: str) -> dict:
    g, sg, rr = r["g"].double().cpu().numpy(), seg.cpu().numpy(), r["r"].double().cpu().numpy(); neg = rr[1:]
    return {"u": int(u), "n_occ": int(len(g)), "n_pos_occ": int((sg == 0).sum()), "n_neg_occ": int((sg > 0).sum()), "loss": r["L"], "A": r["A"], "B": r["B"], "H": r["H"], "r_pos": float(rr[0]),
            "r_neg": [float(x) for x in neg], "r_neg_min": float(neg.min()), "r_neg_max": float(neg.max()), "fp_indicators": [bool(x > 0) for x in neg], "fp_count": int((neg > 0).sum()),
            "hinge_active": [bool(1.0 + x > 0) for x in neg], "lambda_before": float(lam), "gsum_pos": float(np.abs(g[sg == 0]).sum()), "gsum_neg": float(np.abs(g[sg > 0]).sum()),
            "zero_grad_neg_windows": int((g[sg > 0] == 0).sum()), "replay_max_abs": float((r["s2"] - r["s1"]).abs().max()), "arm": arm}


def update(m, head, opt, b, arm: str, lam: float, device, on_grads=None, dual: bool = True) -> tuple[dict, float]:
    """One real update: two-pass whole-membership gradient of one positive and three protected runs, one clip, one AdamW
    step, then (N only) the dual step on the detached pre-step H."""
    u, rows, v, seg, fr, idx, mk = b; u = int(u); t31.sched(opt, u, SCH); opt.zero_grad(set_to_none=True)
    frd, idx, mk, v, seg = fr.to(device, non_blocking=True), idx.to(device), mk.to(device), v.to(device), seg.to(device)
    r = two_pass(m, head, frd, idx, mk, v, seg, arm, lam); rec = update_record(u, r, seg, lam, arm)
    rec.update({f"g_{k}": sq_norm([p.grad for p in ps]) for k, ps in groups_of(m, head).items()})
    if on_grads is not None:
        on_grads(u)
    rec["grad_norm"] = float(torch.nn.utils.clip_grad_norm_(trainable(m, head), CLIP)); opt.step()
    lam2 = rc.dual_update(lam, rec["H"]) if (arm == "N" and dual) else float(lam)
    rec.update({"lambda_after": lam2, "cap_hit": bool(arm == "N" and dual and (lam2 in (rc.LAMBDA_MIN, rc.LAMBDA_MAX)) and lam2 != lam + rc.DUAL_STEP * (rec["H"] - rc.ALPHA)), "pid": os.getpid()})
    return rec, lam2


def run_updates(m, head, opt, F_: Fold, us, arm: str, lam: float, device, workers: int, log: list, hook=None, on_grads=None, dual: bool = True) -> tuple[float, float]:
    gsec = 0.0
    for b in loader(F_, us, workers):
        torch.cuda.synchronize(); a = time.time(); rec, lam = update(m, head, opt, b, arm, lam, device, on_grads, dual); torch.cuda.synchronize()
        rec["gpu_s"] = time.time() - a; rec["s_per_occ"] = rec["gpu_s"] / max(1, rec["n_occ"]); rec["device"] = torch.cuda.get_device_name(0); gsec += rec["gpu_s"]; log.append(rec)
        if hook is not None:
            hook(rec["u"], rec)
    return gsec, lam


def state_of(m, head, opt, lam: float) -> dict:
    return {"model": {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}, "head": {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}, "opt": opt.state_dict(),
            "lambda": float(lam), "cpu_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all()}


def load_state(m, head, opt, st: dict) -> float:
    m.load_state_dict(st["model"]); head.load_state_dict(st["head"]); opt.load_state_dict(st["opt"]); torch.set_rng_state(st["cpu_rng"]); torch.cuda.set_rng_state_all(st["cuda_rng"]); return float(st["lambda"])


def flat_state(m, head) -> dict:
    return {**{f"m.{k}": v.detach().cpu() for k, v in m.state_dict().items()}, **{f"h.{k}": v.detach().cpu() for k, v in head.state_dict().items()}}


@torch.no_grad()
def grid_scores(m, head, seqs: list[dict], W: dict, fp, rows: np.ndarray, device, workers: int, acc: dict, tag: str = "") -> np.ndarray:
    """No-grad scores of a frozen (model, head) at the given fold-local rows (decoded in chunks of 100 sequences); NaN elsewhere."""
    S = np.full(len(W["seq"]), np.nan, np.float32); done = 0
    for c0 in range(0, len(seqs), 100):
        rr_ = rows[(W["seq"][rows] >= c0) & (W["seq"][rows] < c0 + 100)]
        if not len(rr_):
            continue
        a = time.time(); fr, idx = t37.chunk_frames(seqs, {"seq": W["seq"][rr_]}, W["s"][rr_], fp, workers); acc["decode_seconds"] += time.time() - a
        frd, idxd, mkd = torch.from_numpy(fr).to(device), torch.from_numpy(idx).to(device), torch.from_numpy(W["marker"][rr_].astype(np.int64)).to(device)
        torch.cuda.synchronize(); a = time.time(); S[rr_] = torch.cat([window_score(m, head, frd, idxd, mkd, w) for w in range(len(rr_))]).float().cpu().numpy(); torch.cuda.synchronize(); acc["gpu_seconds"] += time.time() - a
        done += len(rr_); del fr, frd
        if tag:
            print(f"{tag}: {done}/{len(rows)} rows; GPU {acc['gpu_seconds']:.0f} s", flush=True)
    if done != len(rows) or not np.isfinite(S[rows]).all():
        raise SystemExit("a row score is missing or not finite")
    return S


def full_fit(m, head, F_: Fold, device, workers: int, tag: str) -> dict:
    """Fit only: every eligible run of the fit fold scored with its original membership (positives, protected negatives and
    the excluded other-class runs); constraint feasibility, actual harm and positive recovery. Never used to select or retune."""
    acc = {"gpu_seconds": 0.0, "decode_seconds": 0.0}; S = grid_scores(m, head, F_.seqs, F_.W, F_.fp, F_.need, device, workers, acc, tag); r = rc.run_weighted_mean(F_.pk, S, F_.off, F_.el); L = F_.L
    fe = rc.feasibility(r, L["negatives"], L["positives"]); ro = r[L["other"]]
    return {**fe, "other_eligible": {"runs": int(len(ro)), "r_positive": int((ro > 0).sum())}, "rows_scored": int(len(F_.need)), "rows_all_minus_2": bool((S[F_.need] == -2.0).all()), **acc,
            "r_all_eligible_positive": int((r[F_.el] > 0).sum()), "score_cache": S[F_.need].astype(np.float32)}


def drift(m, head, s: int, device) -> dict:
    """Per-group drift from the pinned A08 and the initial head; tensors that did not move at all."""
    a08 = torch.load(ROOT / f"models_frozen/video_within_view_v1/all_s{s}/final.pt", map_location="cpu", weights_only=True); cur = {k: v.detach().cpu() for k, v in m.state_dict().items()}
    h0 = rc.ScalarHead().state_dict(); hc = {k: v.detach().cpu() for k, v in head.state_dict().items()}; out = {}
    for name, keys, ref, now_ in (("encoder_trunk", [k for k in cur if k.startswith("encoder.") and not k.startswith("encoder.head.") and cur[k].is_floating_point()], a08, cur),
                                  ("marker_stem", ["marker_stem.weight"], a08, cur), ("projection", ["encoder.head.1.weight", "encoder.head.1.bias"], a08, cur), ("scalar_head", ["fc.weight", "fc.bias"], h0, hc)):
        d = sum(float((now_[k].double() - ref[k].double()).pow(2).sum()) for k in keys) ** 0.5; n0 = sum(float(ref[k].double().pow(2).sum()) for k in keys) ** 0.5
        out[name] = {"l2": d, "relative": d / max(n0, 1e-12), "max_abs": float(max((now_[k] - ref[k]).abs().max() for k in keys))}
    out["classifier_unchanged"] = all(torch.equal(cur[k], a08[k]) for k in cur if k.startswith(FROZEN)); out["head_now"] = {"bias": float(hc["fc.bias"][0]), "weight_l2": float(hc["fc.weight"].norm())}
    out["unchanged_tensors"] = sorted(k for k in cur if cur[k].is_floating_point() and torch.equal(cur[k], a08[k])); return out


# ----------------------------------------------------------------------------------------------- train
def identity(ctx: Ctx, arm: str, s: int, f: int, psha: str) -> str:
    h = hashlib.sha256()
    for p in (Path(rc.__file__), Path(sp.__file__), Path(tm.__file__), Path(__file__)):
        h.update(p.read_bytes())
    spl = json.loads((ctx.a31.out / "split.json").read_text())
    h.update(json.dumps({"arm": arm, "seed": s, "fold": f, "a08": t31.A08[s], "pools": spl["pool_sha256"], "runs": PINS["runs"][s][str(f)], "plan_sha256": psha, "config": sha256_file(ROOT / ctx.config)}, sort_keys=True).encode())
    return h.hexdigest()


def train_one(ctx: Ctx, arm: str, s: int, F_: Fold, ident: str, device, workers: int) -> dict:
    key = f"{arm}_s{s}_f{F_.f}"; rd = ctx.out / "train" / key; rd.mkdir(parents=True, exist_ok=True); rp = rd / "resume.pt"
    m, head = build(ctx, s, F_.f, device); opt = make_opt(m, head); init_digest = {"model": tet.state_digest(m), "head": tet.state_digest(head)}; start, log, resumes, lam = 0, [], [], rc.LAMBDA0
    if rp.exists():
        st = torch.load(rp, map_location="cpu", weights_only=False)
        if st["identity"] != ident:
            raise SystemExit(f"{key}: the resume file belongs to another identity")
        lam = load_state(m, head, opt, st); start, log, init_digest, ff0 = st["update"], st["log"], st["init_digest"], st["full_fit_0"]; resumes.append({"from_update": start, "lambda": lam, "at": now()})
    else:
        ff0 = full_fit(m, head, F_, device, workers, f"[{key}] full fit at update 0")
    prog = ctx.out / f"progress_s{s}.json"; t0 = time.time(); torch.cuda.reset_peak_memory_stats()

    def hook(u: int, rec: dict) -> None:
        if (u + 1) % LOG_EVERY == 0:
            last = log[-LOG_EVERY:]; mean = lambda k: float(np.mean([x[k] for x in last]))  # noqa: E731
            print(f"[{key}] update {u + 1}/{U}: loss {mean('loss'):.4f} A {mean('A'):.4f} B {mean('B'):.4f} H {mean('H'):.4f} r_pos {mean('r_pos'):+.2f} r_neg_max {mean('r_neg_max'):+.2f} fp {sum(x['fp_count'] for x in last)} "
                  f"lam {last[-1]['lambda_after']:.3f} g_trunk {mean('g_encoder_trunk'):.3g} grad {mean('grad_norm'):.3f} occ {sum(x['n_occ'] for x in last)}  {time.time() - t0:.0f} s", flush=True)
            write_json(prog, {"seed": s, "model": key, "update": u + 1, "total": U, "updated_at": now(), "pid": os.getpid(), "device": torch.cuda.get_device_name(0), "lambda": last[-1]["lambda_after"]})
        if (u + 1) % SAVE_EVERY == 0 and u + 1 < U:
            tmp = rp.with_suffix(".tmp"); torch.save({**state_of(m, head, opt, rec["lambda_after"]), "update": u + 1, "log": log, "identity": ident, "init_digest": init_digest, "full_fit_0": ff0}, tmp); tmp.replace(rp)

    gsec, lam = run_updates(m, head, opt, F_, range(start, U), arm, lam, device, workers, log, hook)
    steps = [int(st_["step"]) for st_ in opt.state_dict()["state"].values()]
    if len(log) != U or [x["u"] for x in log] != list(range(U)) or not steps or set(steps) != {U} or len(steps) != len(trainable(m, head)):
        raise SystemExit(f"{key}: {len(log)} logged updates, optimiser steps {sorted(set(steps))} over {len(steps)} tensors")
    ff1 = full_fit(m, head, F_, device, workers, f"[{key}] full fit at update 600")
    torch.save({k: v.detach().cpu().clone() for k, v in m.state_dict().items()}, rd / "final.pt"); torch.save({k: v.detach().cpu().clone() for k, v in head.state_dict().items()}, rd / "head.pt")
    (rd / "updates.jsonl").write_text("".join(json.dumps(x) + "\n" for x in log)); np.savez(rd / "full_fit_scores.npz", rows=F_.need, update_0=ff0.pop("score_cache"), update_600=ff1.pop("score_cache"))
    keys = ("loss", "A", "B", "H", "r_pos", "r_neg_min", "r_neg_max", "fp_count", "lambda_after", "gsum_pos", "gsum_neg", "zero_grad_neg_windows", "g_encoder_trunk", "g_marker_stem", "g_projection", "g_scalar_head", "grad_norm", "replay_max_abs", "s_per_occ")
    man = {"arm": arm, "seed": s, "fit_fold": F_.f, "updates": len(log), "optimizer_steps": U, "identity": ident, "plan_sha256": F_.psha, "a08_sha256": t31.A08[s], "init_digest": init_digest, "lambda_final": lam,
           "lambda_cap_hits": int(sum(x["cap_hit"] for x in log)), "lambda_max_reached": float(max(x["lambda_after"] for x in log)), "lambda_min_reached": float(min(x["lambda_after"] for x in log)),
           "final_sha256": sha256_file(rd / "final.pt"), "head_sha256": sha256_file(rd / "head.pt"), "updates_sha256": sha256_file(rd / "updates.jsonl"), "full_fit_scores_sha256": sha256_file(rd / "full_fit_scores.npz"),
           "full_fit": {"update_0": ff0, "update_600": ff1}, "lists": lists_report(F_), "exposure": rc.exposure(F_.pk, F_.pl), "drift": drift(m, head, s, device),
           "occurrences_trained": int(sum(x["n_occ"] for x in log)), "false_promotions_in_minibatches": int(sum(x["fp_count"] for x in log)), "replay_max_abs_max": float(max(x["replay_max_abs"] for x in log)), "resumes": resumes,
           "gpu_seconds_this_process": gsec, "seconds_this_process": time.time() - t0, "peak_gpu_gb": torch.cuda.max_memory_allocated() / 1e9, "pid": os.getpid(), "device": torch.cuda.get_device_name(0),
           "loss": {k: {"first_60": float(np.mean([x[k] for x in log[:WARM]])), "last_60": float(np.mean([x[k] for x in log[-WARM:]]))} for k in keys}}
    write_json(rd / "run_manifest.json", man); return man


def train(ctx: Ctx, s: int, device, workers: int) -> None:
    for f in FOLDS:
        F_ = Fold(ctx, s, f); ex = EXPECTED_LISTS.get(s, {}).get(f); ep = EXPECTED_PLAN.get(s, {}).get(f)
        if ex and {k: len(F_.L[k]) for k in ex} != ex:
            raise SystemExit(f"fit fold {f}: training lists differ from the asserted counts")
        if ep and F_.psha != ep["sha256"]:
            raise SystemExit(f"fit fold {f}: plan sha256 differs from the asserted value")
        for arm in ARMS:
            key = f"{arm}_s{s}_f{f}"; rd = ctx.out / "train" / key; ident = identity(ctx, arm, s, f, F_.psha); mp = rd / "run_manifest.json"
            if mp.exists() and json.loads(mp.read_text())["identity"] == ident and sha256_file(rd / "final.pt") == json.loads(mp.read_text())["final_sha256"]:
                print(f"[{key}] finished earlier", flush=True); continue
            t0, started, ok, note = time.time(), now(), False, ""
            try:
                man = train_one(ctx, arm, s, F_, ident, device, workers); torch.cuda.empty_cache(); ok = True; note = f"final {man['final_sha256'][:12]} head {man['head_sha256'][:12]} lambda {man['lambda_final']:.4f}"
                if sha256_file(rd / "final.pt") == man["final_sha256"] and sha256_file(rd / "head.pt") == man["head_sha256"] and (rd / "resume.pt").exists():
                    (rd / "resume.pt").unlink()                                  # regenerable; the final weights are verified
                print(json.dumps({k: man[k] for k in ("arm", "seed", "fit_fold", "updates", "final_sha256", "head_sha256", "lambda_final", "loss")}), flush=True)
                print(json.dumps({"full_fit": {k: {kk: vv for kk, vv in v.items() if kk not in ("r_negative", "r_positive")} for k, v in man["full_fit"].items()}}), flush=True)
            except BaseException as e:  # noqa: BLE001 - recorded, then raised again
                note = f"{type(e).__name__}: {str(e)[:200]}"; raise
            finally:
                ledger_append(ctx.ledger, f"train {key} (GPU job wall time incl. full-fit scoring)", "gpu", started, time.time() - t0, ok, note)
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
def plan_stage(ctx: Ctx, s: int) -> dict:
    """(CPU, metadata only) lists / plans / exposure of both fit folds against the asserted counts and plan sha256."""
    out = {}
    for f in FOLDS:
        F_ = Fold(ctx, s, f); ex = EXPECTED_LISTS.get(s, {}).get(f); ep = EXPECTED_PLAN.get(s, {}).get(f); e = rc.exposure(F_.pk, F_.pl)
        out[str(f)] = {"counts": {k: int(len(v)) for k, v in F_.L.items()}, "expected": ex, "counts_equal_expected": ex is None or {k: len(F_.L[k]) for k in ex} == ex, "plan_sha256": F_.psha, "expected_plan": ep,
                       "plan_sha_equal": ep is None or F_.psha == ep["sha256"], "exposure": e, "exposure_equal_expected": ep is None or (e["occurrences_total"], e["occurrences_max_update"]) == (ep["occurrences_total"], ep["occurrences_max_update"]),
                       "all_positives_drawn": e["distinct_positives"] == len(F_.L["positives"]), "all_negatives_drawn": e["distinct_negatives"] == len(F_.L["negatives"]), "eligible_rows": int(len(F_.need))}
    out["ok"] = all(out[str(f)][k] for f in FOLDS for k in ("counts_equal_expected", "plan_sha_equal", "exposure_equal_expected", "all_positives_drawn", "all_negatives_drawn")); return out


def list_checks(ctx: Ctx, s: int) -> dict:
    """(1 / 2) Lists, membership (independent repeated-sample loop), plans (explicit RNG), exposure, run alignment, isolation."""
    out = {}; spl = json.loads((ctx.a31.out / "split.json").read_text()); g = np.random.default_rng([rc.BASE, rc.ROUND, 7])
    for f in FOLDS:
        F_ = Fold(ctx, s, f); pk, seqs, W = F_.pk, F_.seqs, F_.W; L = F_.L; near = t27.nearest_rows(W, seqs); nW = len(W["seq"]); ex = EXPECTED_LISTS.get(s, {}).get(f); ep = EXPECTED_PLAN.get(s, {}).get(f)
        order_ok = all(np.all(np.diff(L[k]) > 0) for k in L) and set(L["eligible"]) == set(L["positives"]) | set(L["negatives"]) | set(L["other"]) and not (set(L["positives"]) & set(L["negatives"]))
        mem = wsum = inside = unique = True; maxdev = 0.0; sc = g.normal(size=nW)
        for i in range(len(pk["labels"])):
            rows, w = rc.members(pk, i, F_.off); er, ew = rc.explicit_membership(near[int(pk["seq"][i])] - F_.off, int(pk["a"][i]), int(pk["b"][i]))
            mem &= bool(np.array_equal(rows, er) and np.abs(w - ew).max() <= TOL["membership"]); wsum &= abs(w.sum() - 1) <= TOL["membership"]; inside &= bool(rows.min() >= 0 and rows.max() < nW); unique &= len(np.unique(rows)) == len(rows)
            if F_.el[i]:
                maxdev = max(maxdev, abs(float(np.dot(w, sc[rows])) - rc.run_weighted_mean_explicit(pk, sc, F_.off, i, near[int(pk["seq"][i])])))
        al = True
        for qi, q in enumerate(seqs):
            ks = np.flatnonzero(pk["seq"] == qi); lab = F_.base_labels[sum(len(x["labels"]) for x in seqs[:qi]):][:len(q["labels"])]
            al &= [tuple(int(v) for v in x) for x in ir.runs(lab)] == [(int(pk["a"][k]), int(pk["b"][k]), int(pk["base"][k])) for k in ks] and bool(np.all(np.diff(ks) == 1))
        pl, perm = F_.pl, rc.negative_permutation(L["negatives"], s, f); explicit = all(int(pl[u, 0]) == int(np.random.default_rng([rc.BASE, rc.ROUND, s, f, u]).choice(L["positives"])) and pl[u, 1:].tolist() == perm[(3 * u + np.arange(3)) % len(perm)].tolist() for u in range(U))
        e = rc.exposure(pk, pl); rec_f = {str(r[0]) for q in seqs for r in q["refs"]}; other = t31.load_pool(ctx.a31, 1 - f); rec_o = {str(r[0]) for q in other for r in q["refs"]}
        out[str(f)] = {"counts": {k: int(len(v)) for k, v in L.items()}, "expected": ex, "counts_equal_expected": ex is None or {k: len(L[k]) for k in ex} == ex, "lists_in_P_order_and_disjoint": bool(order_ok),
                       "eligible_is_base_A4C": bool(np.array_equal(L["eligible"], np.flatnonzero(pk["base"] == A4C))), "positives_are_S1": bool(np.array_equal(L["positives"], np.flatnonzero((pk["base"] == A4C) & (pk["labels"] == A5C)))),
                       "membership_equals_repeated_sample_loop": bool(mem), "weights_sum_1": bool(wsum), "rows_inside_fold": bool(inside), "rows_unique_per_run": bool(unique), "weighted_mean_vs_explicit_mean_max_abs": maxdev,
                       "run_alignment_with_base_sequences": bool(al), "P_runs": int(len(pk["labels"])), "grid": nW, "eligible_rows": int(len(F_.need)),
                       "recordings_in_other_pool": len(rec_f & rec_o), "sessions_in_other_pool": len({session_of(x) for x in rec_f} & {session_of(x) for x in rec_o}),
                       "plan": {"updates": int(len(pl)), "shape": list(pl.shape), "dtype": str(pl.dtype), "sha256": F_.psha, "expected": ep, "sha_equal_expected": ep is None or F_.psha == ep["sha256"], "repeat_equal": bool(np.array_equal(pl, rc.plan(L["positives"], L["negatives"], s, f, U))),
                                "explicit_rng_equal": bool(explicit), "positives_in_list": bool(set(pl[:, 0].tolist()) <= set(L["positives"].tolist())), "negatives_in_list": bool(set(pl[:, 1:].ravel().tolist()) <= set(L["negatives"].tolist())),
                                **e, "all_positives_drawn": e["distinct_positives"] == len(L["positives"]), "all_negatives_drawn": e["distinct_negatives"] == len(L["negatives"]),
                                "exposure_equal_expected": ep is None or (e["occurrences_total"], e["occurrences_max_update"]) == (ep["occurrences_total"], ep["occurrences_max_update"]),
                                "selection_inputs": "the fit pool's Base P-run labels, reference majority labels and request map (no model output of this round, no held-out pool, no session matching); one plan for both arms"},
                       "lists_report": lists_report(F_)}
    out["isolation"] = spl["isolation"]; out["pins"] = {"pools_equal": spl["pool_sha256"] == PINS["pools"], "split_sha256": sha256_file(ctx.a31.out / "split.json"), "split_equal": sha256_file(ctx.a31.out / "split.json") == PINS["split"],
                                                       "runs_equal": all(sha256_file(ctx.a31.out / "data" / f"runs_s{s}_f{f}.npz") == PINS["runs"][s][str(f)] for f in FOLDS),
                                                       "outputs_equal": all(sha256_file(ctx.a31.out / f"outputs_s{s}_f{f}.npz") == PINS["outputs"][s][str(f)] for f in FOLDS), "a08_equal": sha256_file(ROOT / f"models_frozen/video_within_view_v1/all_s{s}/final.pt") == t31.A08[s]}
    return out


def frame_checks(F0: Fold) -> dict:
    """(1) Update 0's occurrences: dataset frames = an independent per-ref decode; markers = A07 marker slots; rows / weights = the plan."""
    item = Updates(F0, [0])[0]; u, rows, v, seg, fr, idx, mk = item; rows = rows.numpy(); memo, same = {}, True
    for w, r in enumerate(rows):
        q = F0.seqs[int(F0.W["seq"][r])]
        for k, p in enumerate(F0.W["s"][r]):
            ref = tuple(q["refs"][int(p)])
            if ref not in memo:
                vv, fi, gg = ref; pth = F0.fp(vv); memo[ref] = jv.load_frame_u8(pth[min(int(fi), len(pth) - 1)], round(float(gg), 6))
            same &= bool(np.array_equal(memo[ref], fr[idx[w, k]].numpy()))
    exp_mk = np.asarray([tm.marker_slot(F0.W["s"][r], int(F0.W["centre"][r])) for r in rows]); er, ev, es = F0.occ(0)
    return {"pixels_equal_independent_decode": same, "occurrences": int(len(rows)), "unique_frames": int(len(fr)), "markers_equal_W": bool(np.array_equal(mk.numpy(), F0.W["marker"][rows])),
            "markers_equal_marker_slot": bool(np.array_equal(mk.numpy(), exp_mk)), "rows_weights_slots_equal_plan": bool(np.array_equal(rows, er) and np.allclose(v.numpy(), ev, atol=1e-7) and np.array_equal(seg.numpy(), es)),
            "repeat_equal": bool(torch.equal(Updates(F0, [0])[0][4], fr))}


def check_head(device, seed: int = 4444) -> rc.ScalarHead:
    """A check-only head with a nonzero weight (N(0, 0.05^2) from its own generator), so the trunk receives gradient."""
    h = rc.ScalarHead(); g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        h.fc.weight.copy_(torch.randn(1, rc.EMBED, generator=g) * 0.05)
    return h.to(device).eval()


def init_checks(ctx: Ctx, s: int, F0: Fold, device) -> dict:
    """(3) Init: A08 copy bitwise, identical arm start, 513 head parameters at 0 / -2, every first-pass score -2, online fit
    logits = A31 cache (no held-out pixel), patched = original forward, Init = Base from the cached q."""
    a08 = torch.load(ROOT / f"models_frozen/video_within_view_v1/all_s{s}/final.pt", map_location="cpu", weights_only=True); res = {}
    m, head = build(ctx, s, 0, device); m2, head2 = build(ctx, s, 0, device)
    res["init_state_equals_A08"] = all(torch.equal(v.cpu(), a08[k]) for k, v in m.state_dict().items()) and sorted(m.state_dict()) == sorted(a08)
    res["identical_arm_start"] = tet.state_digest(m) == tet.state_digest(m2) and tet.state_digest(head) == tet.state_digest(head2)
    res["patched_blocks"] = [i for i, bl in enumerate(m.encoder.blocks) if isinstance(getattr(bl.pool_skip, "pool", None), sp.GatherMaxPool3d)]
    res["head"] = {"parameters": int(sum(p.numel() for p in head.parameters())), "weight_all_zero": bool((head.fc.weight == 0).all()), "bias": float(head.fc.bias[0]), "fold1_head_equal_values": tet.state_digest(build(ctx, s, 1, device)[1]) == tet.state_digest(head)}
    item = Updates(F0, [0])[0]; u, rows, v, seg, fr, idx, mk = item; frd, idxd, mkd = fr.to(device), idx.to(device), mk.to(device); rows = rows.numpy(); n = len(rows)
    with torch.no_grad():
        s0 = torch.cat([window_score(m, head, frd, idxd, mkd, w) for w in range(n)])
    res["init_scores_all_minus_2"] = {"occurrences": int(n), "all_exactly_minus_2": bool((s0 == -2.0).all()), "max_abs_dev": float((s0 + 2).abs().max())}
    O = t31.outputs(ctx.a31, s, 0); k = min(n, 16)
    with torch.no_grad():
        z = torch.cat([m(tm.to_rgb(frd[idxd[w:w + 1]]), tm.marker_planes(mkd[w:w + 1])) for w in range(k)]); e = m.embed(tm.to_rgb(frd[idxd[:1]]), tm.marker_planes(mkd[:1]))
    res["init_online_fit_logits_vs_A31_cache_max_abs"] = float(np.abs(z.cpu().numpy() - O["logits"][rows[:k]]).max()); res["embed_shape"] = list(e.shape); res["classifier_of_embed_equals_forward"] = bool(torch.equal(m.classifier(e), z[:1]))
    mo, _ = build(ctx, s, 0, device, patched=False); hc = check_head(device)
    with torch.no_grad():
        sp_ = torch.cat([window_score(m, hc, frd, idxd, mkd, w) for w in range(k)]); so = torch.cat([window_score(mo, hc, frd, idxd, mkd, w) for w in range(k)])
        zo = torch.cat([mo(tm.to_rgb(frd[idxd[w:w + 1]]), tm.marker_planes(mkd[w:w + 1])) for w in range(k)])
    res["patched_equals_original_forward_bitwise"] = bool(torch.equal(sp_, so) and torch.equal(zo, z)); res["no_hook_left"] = not m.encoder.conv_proj._forward_hooks
    del m, m2, mo; torch.cuda.empty_cache(); res["Base_from_cached_q"] = t40.init_base(ctx, s); return res


def formula_checks() -> dict:
    """(4) Independent float64 run means / A / B / H / gradients for C and N (variable memberships, unequal weights, kinks at
    1 + r = 0, huge scores) and the dual recurrence (clipping at 0 / 100, r = -1 / 0)."""
    g = np.random.default_rng([rc.BASE, rc.ROUND, 99]); out = {}
    cases = {"1x1x1x1": ([1, 1, 1, 1], 2.0), "3x2x5x1": ([3, 2, 5, 1], 2.0), "40x17x9x30": ([40, 17, 9, 30], 3.0), "129_total": ([60, 23, 23, 23], 3.0), "large_margins": ([4, 3, 2, 5], 1e3)}
    for name, (n, sc) in cases.items():
        seg = np.concatenate([np.full(k, i) for i, k in enumerate(n)]); s = g.normal(size=int(sum(n))) * sc; v = np.concatenate([g.dirichlet(np.ones(k)) for k in n])
        for arm in ARMS:
            for lam in (0.0, 1.0, 37.5):
                st = torch.tensor(s, requires_grad=True); r = rc.run_means(st, torch.tensor(v), torch.tensor(seg), 4); L, A, B, H = rc.objective(r, arm, lam); L.backward(); ref = rc.reference(s, v, seg, arm, lam)
                out[f"{name}_{arm}_lam{lam}"] = {"r_max_abs": float(np.abs(r.detach().numpy() - ref["r"]).max()), "loss_abs": abs(float(L) - ref["loss"]), "A_abs": abs(float(A) - ref["A"]), "B_abs": abs(float(B) - ref["B"]), "H_abs": abs(float(H) - ref["H"]),
                                                  "grad_max_abs": float(np.abs(st.grad.numpy() - ref["g"]).max()), "finite": bool(torch.isfinite(L) and torch.isfinite(st.grad).all()),
                                                  "inactive_negative_windows_zero_grad": bool(all((st.grad.numpy()[seg == k] == 0).all() for k in range(1, 4) if 1.0 + ref["r"][k] <= 0)) if arm == "N" else None,
                                                  "active_negatives": int(sum(1.0 + ref["r"][k] > 0 for k in range(1, 4)))}
    # kinks: r = -1 (ReLU derivative 0), r = 0 (decision = no edit; hinge active), huge
    r = torch.tensor([0.0, -1.0, 0.0, -1e3], requires_grad=True); L, A, B, H = rc.objective(r, "N", 2.0); L.backward()
    out["kinks_N"] = {"H": float(H), "H_expected": 1.0 / 3, "grad": r.grad.tolist(), "grad_expected": [-0.5, 0.0, 2.0 / 3, 0.0], "loss_finite": bool(torch.isfinite(L))}
    r2 = torch.tensor([1e3, 1e3, -1e3, 5.0], requires_grad=True); L2, A2, B2, H2 = rc.objective(r2, "C", 0.0); L2.backward()
    out["huge_C"] = {"finite": bool(torch.isfinite(L2) and torch.isfinite(r2.grad).all()), "A": float(A2)}
    # dual recurrence
    seq = [1.0]; Hs = [0.0, 0.01, 0.51, 1.5, 0.0, 400.0, 0.0, 0.0]
    for h in Hs:
        seq.append(rc.dual_update(seq[-1], h))
    exp = [1.0, 0.999, 0.999, 1.049, 1.198, 1.197, 41.196, 41.195, 41.194]
    out["dual"] = {"trajectory": seq, "expected": exp, "max_abs": float(max(abs(a - b) for a, b in zip(seq, exp))), "clip_low": rc.dual_update(0.0, 0.0) == 0.0 and rc.dual_update(0.005, 0.0) == 0.0,
                   "clip_high": rc.dual_update(100.0, 5.0) == 100.0 and rc.dual_update(99.9999, 1.0) == 100.0, "feasible_no_change_alpha": rc.dual_update(3.0, rc.ALPHA) == 3.0,
                   "constants": {"alpha": rc.ALPHA, "lambda0": rc.LAMBDA0, "step": rc.DUAL_STEP, "range": [rc.LAMBDA_MIN, rc.LAMBDA_MAX]}}
    return out


def grads(m, head) -> dict:
    return {**{f"m.{n}": (torch.zeros_like(p) if p.grad is None else p.grad.detach().clone()) for n, p in m.named_parameters() if p.requires_grad},
            **{f"h.{n}": (torch.zeros_like(p) if p.grad is None else p.grad.detach().clone()) for n, p in head.named_parameters()}}


def short_runs(F_: Fold) -> list[int]:
    """The first positive and the first three protected negatives in P order with 2-6 unique rows, at least one with unequal weights."""
    size = lambda i: int(F_.pk["offsets"][i + 1] - F_.pk["offsets"][i])  # noqa: E731
    uneq = lambda i: len(set(np.round(rc.members(F_.pk, i, F_.off)[1], 12).tolist())) > 1  # noqa: E731
    p = next(i for i in F_.L["positives"] if 2 <= size(i) <= 6); ns = [i for i in F_.L["negatives"] if 2 <= size(i) <= 6]
    first_uneq = next(i for i in ns if uneq(i)); rest = [i for i in ns if i != first_uneq][:2]; return [int(p), int(first_uneq), *[int(x) for x in rest]]


def zero_all(m, head) -> None:
    m.zero_grad(set_to_none=True); head.zero_grad(set_to_none=True)


def gradient_checks(ctx: Ctx, s: int, F_: Fold, device) -> dict:
    """(5) Two-pass production FP32 gradient vs one retained-graph backward (check-only nonzero head weight), C and N; pass-1 vs
    replayed scores; patched vs original MaxPool3d backward; batched vs microbatch FP32 (diagnostic only)."""
    runs = short_runs(F_); rows, v, seg = rc.occurrences(F_.pk, runs, F_.off); fr, idx, mk = F_.frames(rows); frd, idx, mk = fr.to(device), idx.to(device), mk.to(device); vt, st = torch.from_numpy(v).float().to(device), torch.from_numpy(seg).to(device)
    n = len(rows); out = {"runs_rows": [int(F_.pk["offsets"][i + 1] - F_.pk["offsets"][i]) for i in runs], "occurrences": int(n)}; m, _ = build(ctx, s, F_.f, device); hc = check_head(device); lam = 1.0
    for arm in ARMS:
        zero_all(m, hc); r = two_pass(m, hc, frd, idx, mk, vt, st, arm, lam); ga = grads(m, hc); zero_all(m, hc)
        sr = torch.cat([window_score(m, hc, frd, idx, mk, w) for w in range(n)]); L = rc.objective(rc.run_means(sr, vt, st, 4), arm, lam)[0]; L.backward(); gb = grads(m, hc); zero_all(m, hc)
        nz = {k: sq_norm([ga[f"{'h' if k == 'scalar_head' else 'm'}.{nm}"] for nm in names]) for k, names in group_names(m, hc).items()}
        out[f"two_pass_vs_retained_{arm}"] = {**rel(ga, gb), "pass1_vs_retained_scores_max_abs": float((r["s1"] - sr.detach()).abs().max()), "pass1_vs_replay_max_abs": float((r["s1"] - r["s2"]).abs().max()), "loss_abs": abs(r["L"] - float(L)),
                                              "group_grad_norms": nz, "all_groups_nonzero": all(x > 0 for x in nz.values()), "H": r["H"], "active_negatives": int(sum(1.0 + float(x) > 0 for x in r["r"][1:])),
                                              "zero_grad_negative_windows": int((r["g"][st > 0] == 0).sum().item()), "status": "gated (production FP32 two-pass gradient)"}
        if arm == "C":
            gb_c = gb
    with torch.no_grad():
        sb = rc.score(m, hc, tm.to_rgb(frd[idx]), tm.marker_planes(mk))
    out["batched_vs_microbatch_scores_fp32"] = {"max_abs": float((sb - r["s1"]).abs().max()), "status": "diagnostic round-off, not a gate"}
    del m; torch.cuda.empty_cache()
    mo, _ = build(ctx, s, F_.f, device, patched=False); torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        sr = torch.cat([window_score(mo, hc, frd, idx, mk, w) for w in range(n)]); rc.objective(rc.run_means(sr, vt, st, 4), "C", lam)[0].backward(); out["patched_vs_original_maxpool_backward"] = rel(gb_c, grads(mo, hc))
    finally:
        torch.use_deterministic_algorithms(True, warn_only=False)
    del mo; torch.cuda.empty_cache(); out["warn_only_restored_false"] = not torch.is_deterministic_algorithms_warn_only_enabled(); return out


def group_names(m, head) -> dict:
    ids = {id(p): n for n, p in m.named_parameters()}; hid = {id(p): n for n, p in head.named_parameters()}; G = groups_of(m, head)
    return {k: [ids.get(id(p)) or hid[id(p)] for p in ps] for k, ps in G.items()}


def mech_run(ctx: Ctx, s: int, F_: Fold, arm: str, us, device, workers: int, state: dict | None = None, track: bool = False, objective_override: str | None = None, dual: bool = True):
    """Check-only training path of an arm; `objective_override` ('C') keeps N's code path but forms C's loss (dual off)."""
    m, head = build(ctx, s, F_.f, device); opt = make_opt(m, head); lam = rc.LAMBDA0
    if state is not None:
        lam = load_state(m, head, opt, state)
    log, seen, gmax = [], {}, {}
    names = {id(p): f"m.{n}" for n, p in m.named_parameters()} | {id(p): f"h.{n}" for n, p in head.named_parameters()}

    def on_grads(u: int) -> None:
        seen[u] = {names[id(p)] for p in trainable(m, head) if p.grad is not None and float(p.grad.abs().sum()) > 0}
        if u >= 1:
            for p in trainable(m, head):
                gmax[names[id(p)]] = max(gmax.get(names[id(p)], 0.0), 0.0 if p.grad is None else float(p.grad.abs().max()))

    g, lam = run_updates(m, head, opt, F_, us, objective_override or arm, lam, device, workers, log, on_grads=on_grads if track else None, dual=dual and objective_override is None)
    return m, head, opt, log, g, seen, gmax, lam


def training_checks(ctx: Ctx, s: int, F_: Fold, device, workers: int) -> dict:
    """(6) Binding, update 0 zero / later genuine gradients, movement, repeat, forced C objective on N's path, N vs C, 3 + 3 resume, timing."""
    us = range(0, 6); res = {}; m, head = build(ctx, s, F_.f, device); opt = make_opt(m, head); init = {k: v.clone() for k, v in flat_state(m, head).items()}
    ids = sorted(id(p) for g in opt.param_groups for p in g["params"]); names = {id(p): n for n, p in m.named_parameters()} | {id(p): f"head.{n}" for n, p in head.named_parameters()}
    res["binding"] = {"binds_exactly_trainable": ids == sorted(id(p) for p in trainable(m, head)) and len(ids) == len(set(ids)),
                      "trainable_equals_requires_grad": {id(p) for p in trainable(m, head)} == {id(p) for p in list(m.parameters()) + list(head.parameters()) if p.requires_grad},
                      "frozen": sorted(n for n, p in m.named_parameters() if not p.requires_grad), "frozen_only_classifier": all(n.startswith(FROZEN) for n, p in m.named_parameters() if not p.requires_grad)
                      and all(not p.requires_grad for n, p in m.named_parameters() if n.startswith(FROZEN)), "eval_mode": not m.training and not head.training and not any(x.training for x in list(m.modules()) + list(head.modules())),
                      "groups": {g["name"]: {"tensors": len(g["params"]), "parameters": int(sum(p.numel() for p in g["params"])), "lr": g["base"], "weight_decay": g["weight_decay"]} for g in opt.param_groups},
                      "projection_group": [names[id(p)] for p in opt.param_groups[2]["params"]], "marker_group": [names[id(p)] for p in opt.param_groups[1]["params"]], "head_group": [names[id(p)] for p in opt.param_groups[3]["params"]],
                      "head_parameters": int(sum(p.numel() for p in head.parameters())), "frozen_parameters": int(sum(p.numel() for p in m.parameters() if not p.requires_grad)), "betas_eps": [list(opt.defaults["betas"]), opt.defaults["eps"]],
                      "lr_update0": {g["name"]: g["base"] * lr_factor(0, WARM, U) for g in opt.param_groups}, "lr_update59_peak": {g["name"]: g["base"] * lr_factor(WARM - 1, WARM, U) for g in opt.param_groups},
                      "lr_update599": {g["name"]: g["base"] * lr_factor(U - 1, WARM, U) for g in opt.param_groups}}
    del m, head, opt; torch.cuda.empty_cache()
    L = lambda a, b: max(max(abs(x["loss"] - y["loss"]), abs(x["A"] - y["A"]), abs(x["B"] - y["B"]), abs(x["H"] - y["H"])) for x, y in zip(a[3], b[3]))  # noqa: E731
    P = lambda a, b: params_max_diff(flat_state(a[0], a[1]), flat_state(b[0], b[1]))  # noqa: E731
    ca = mech_run(ctx, s, F_, "C", us, device, workers, track=True); fin = flat_state(ca[0], ca[1]); lg = ca[3]; gk = ("g_encoder_trunk", "g_marker_stem", "g_projection", "g_scalar_head")
    res["update0"] = {k: lg[0][k] for k in (*gk, "n_occ", "loss", "A", "B", "H", "r_pos", "r_neg")}
    res["updates_1_5"] = [{k: x[k] for k in ("u", *gk, "grad_norm", "n_occ", "loss", "A", "B", "H", "r_pos", "r_neg_max", "fp_count", "replay_max_abs")} for x in lg[1:]]
    tr = {f"m.{n}" for n, p in ca[0].named_parameters() if p.requires_grad} | {f"h.{n}" for n, _ in ca[1].named_parameters()}; nz15 = set().union(*[ca[5][u] for u in range(1, 6)])
    grp = {"encoder_trunk": lambda k: k.startswith("m.encoder.") and not k.startswith("m.encoder.head."), "marker_stem": lambda k: k == "m.marker_stem.weight", "projection": lambda k: k.startswith("m.encoder.head."), "scalar_head": lambda k: k.startswith("h.fc.")}
    res["real_update_C_6"] = {"moved": {g_: {"tensors_moved": int(sum(1 for k in fin if f_(k) and not torch.equal(fin[k], init[k]))), "tensors": int(sum(1 for k in fin if f_(k))),
                                             "max_abs": float(max((fin[k] - init[k]).abs().max() for k in fin if f_(k)))} for g_, f_ in grp.items()},
                              "unchanged_tensors": sorted(k for k in fin if torch.equal(fin[k], init[k])), "optimizer_steps": sorted({int(v["step"]) for v in ca[2].state_dict()["state"].values()}),
                              "optimizer_state_tensors": len(ca[2].state_dict()["state"]), "zero_gradient_trainable_tensors_updates_1_5": sorted(tr - nz15), "update0_nonzero_gradient_tensors": sorted(ca[5][0])}
    gm = ca[6]; tmax = max(v for k, v in gm.items() if grp["encoder_trunk"](k)); res["real_update_C_6"]["trunk_max_abs_grad_updates_1_5"] = tmax
    res["real_update_C_6"]["unchanged_trainable"] = {k: {"max_abs_grad_updates_1_5": gm[k], "relative_to_trunk_max": gm[k] / tmax, "structural_reason": KEY_NORM_BIAS if k.endswith(".attn.pool_k.norm_act.0.bias") else None}
                                                     for k in res["real_update_C_6"]["unchanged_tensors"] if k in tr}
    cb = mech_run(ctx, s, F_, "C", us, device, workers); res["repeat_6_C"] = {"loss_max_abs": L(ca, cb), "params_max_abs": P(ca, cb)}; del cb; torch.cuda.empty_cache()
    na = mech_run(ctx, s, F_, "N", us, device, workers); res["N_differs_from_C"] = {"params_max_abs": P(na, ca), "loss_first_updates": [[x["loss"] for x in na[3][:3]], [x["loss"] for x in ca[3][:3]]], "lambda_after_6": na[7],
                                                                                   "lambda_trajectory": [x["lambda_after"] for x in na[3]], "H_trajectory": [x["H"] for x in na[3]], "fp_counts": [x["fp_count"] for x in na[3]]}
    fc = mech_run(ctx, s, F_, "N", us, device, workers, objective_override="C")          # N's path with C's objective, dual off (check only)
    res["forced_C_objective_N_vs_C"] = {"loss_max_abs": L(fc, ca), "params_max_abs": P(fc, ca), "lambda_unchanged": fc[7] == rc.LAMBDA0, "formal_arms_unchanged": rc.ARMS == ("C", "N")}; del fc, ca; torch.cuda.empty_cache()
    n3 = mech_run(ctx, s, F_, "N", range(0, 3), device, workers); p = ctx.out / "check_resume.pt"; torch.save(state_of(n3[0], n3[1], n3[2], n3[7]), p); log3 = n3[3]; del n3
    st = torch.load(p, map_location="cpu", weights_only=False); nr = mech_run(ctx, s, F_, "N", range(3, 6), device, workers, st); p.unlink()
    res["resume_3_plus_3_N"] = {"loss_max_abs": max(max(abs(x["loss"] - y["loss"]), abs(x["H"] - y["H"])) for x, y in zip(na[3], log3 + nr[3])), "params_max_abs": P(na, nr),
                                "lambda_max_abs": max(abs(x["lambda_after"] - y["lambda_after"]) for x, y in zip(na[3], log3 + nr[3])), "lambda_restored": float(st["lambda"]) == log3[-1]["lambda_after"]}
    res["timing"] = {"updates_0_5": {"occurrences": int(sum(x["n_occ"] for x in na[3])), "gpu_seconds": na[4], "seconds_per_occurrence_gpu": na[4] / sum(x["n_occ"] for x in na[3])}}; del na, nr; torch.cuda.empty_cache()
    per = np.asarray([sum(int(F_.pk["offsets"][i + 1] - F_.pk["offsets"][i]) for i in row) for row in F_.pl]); ul, ush = int(per.argmax()), int(per.argmin())
    torch.cuda.reset_peak_memory_stats(); lo = mech_run(ctx, s, F_, "N", [ul], device, 0); sh = mech_run(ctx, s, F_, "N", [ush], device, 0)
    res["timing"].update({"longest_update": {"u": ul, "occurrences": int(per[ul]), "gpu_seconds": lo[4], "seconds_per_occurrence": lo[4] / per[ul]}, "shortest_update": {"u": ush, "occurrences": int(per[ush]), "gpu_seconds": sh[4]},
                          "peak_gpu_gb_training": torch.cuda.max_memory_allocated() / 1e9}); del lo, sh; torch.cuda.empty_cache()
    t0 = time.time(); w12 = mech_run(ctx, s, F_, "C", range(6, 18), device, workers); wall = time.time() - t0; nw12 = int(sum(x["n_occ"] for x in w12[3]))
    res["timing"]["updates_6_17_wall"] = {"occurrences": nw12, "gpu_seconds": w12[4], "wall_seconds_incl_build_loader": wall, "seconds_per_occurrence_wall": wall / nw12, "seconds_per_occurrence_gpu": w12[4] / nw12}; del w12
    torch.cuda.empty_cache(); return res


def one_way_checks(ctx: Ctx, s: int) -> dict:
    """(7) Synthetic edit / merge invariants incl. ties at 0 / -1; eligible counts from cached P runs; run alignment; Init = Base;
    label-free decision; weighted-mean deployment = explicit sample mean on random scores."""
    out = {}; base = np.asarray([0, 1, 2, 3, 4, 2, 2, 2, 2]); r = np.asarray([5.0, 5.0, 1.0, 5.0, 5.0, 0.0, -1.0, 1e-9, -1e-9])
    out["synthetic_rule"] = {"result": rc.one_way(base, r).tolist(), "expected": [0, 1, 3, 3, 4, 2, 2, 3, 2], "input_unchanged": base.tolist() == [0, 1, 2, 3, 4, 2, 2, 2, 2]}
    seq = np.asarray([0, 0, 2, 2, 3, 3, 2, 2, 2, 1]); pk = {"seq": np.zeros(5, int), "a": np.asarray([0, 2, 4, 6, 9]), "b": np.asarray([2, 4, 6, 9, 10]), "base": np.asarray([0, 2, 3, 2, 1])}
    ed = rc.edit_sequences({"q": seq}, pk, [{"id": "q"}], np.asarray([0, 3, 3, 2, 1]))["q"]
    out["synthetic_merge"] = {"edited": ed.tolist(), "expected": [0, 0, 3, 3, 3, 3, 2, 2, 2, 1], "runs_after": [list(x) for x in ir.runs(ed)], "boundaries_before_after": [int((np.diff(seq) != 0).sum()), int((np.diff(ed) != 0).sum())]}
    X = t40.full_q(ctx, s); a8 = rsh.a08_ctx(); data = t31.vpr.training_data(ctx.a31.a15.a10.a07p); g = np.random.default_rng([rc.BASE, rc.ROUND, 7])
    for h in FOLDS:
        D = d32.fold_data(ctx.d, s, h); D["M"] = d32.set_masks(D); pk = D["pk"]; off = X["offset"][h]; W = t31.pool_W(D["seqs"]); nW = len(W["seq"]); el = pk["base"] == A4C; need = eligible_rows(pk, el); near = t27.nearest_rows(W, D["seqs"])
        al = True
        for qi, q in enumerate(D["seqs"]):
            ks = np.flatnonzero(pk["seq"] == qi); al &= [tuple(int(v) for v in x) for x in ir.runs(D["base_preds"][q["id"]])] == [(int(pk["a"][k]), int(pk["b"][k]), int(pk["base"][k])) for k in ks] and bool(np.all(np.diff(ks) == 1))
        sc = np.full(nW, -2.0); r = rc.run_weighted_mean(pk, sc, off, el); p0 = rc.one_way(pk["base"], r); P = {"seqs": D["seqs"], "n_of": dict(data.n), "label_of": dict(data.label)}; ev = bc.evalset(P, list(range(len(D["seqs"]))), f"heldout_f{h}")
        t0_ = tables_pred(p0, D, a8.cfg, P, ev); tb = tables(X["q"], D, off, a8.cfg, P, ev); js = lambda x: json.loads(json.dumps(x, default=float))  # noqa: E731
        same = all(js(t0_[k]) == js(tb[k]) for k in ("runs", "changes", "correct_A5C_kept", "S1_fixed", "S2_kept", "sequence_row", "counts")) and bool(np.array_equal(t0_["pred"], tb["pred"]))
        rnd = g.normal(size=nW).astype(np.float32); rr_ = rc.run_weighted_mean(pk, rnd, off, el); pr = rc.one_way(pk["base"], rr_); pk2 = {**pk, "labels": np.zeros_like(pk["labels"])}; pr2 = rc.one_way(pk2["base"], rc.run_weighted_mean(pk2, rnd, off, pk2["base"] == A4C))
        edr = rc.edit_sequences(D["base_preds"], pk, D["seqs"], pr); mdev = max(abs(rr_[i] - rc.run_weighted_mean_explicit(pk, rnd, off, int(i), near[int(pk["seq"][i])])) for i in np.flatnonzero(el))
        out[f"heldout_f{h}"] = {"eligible_runs": int(el.sum()), "eligible_unique_rows": int(len(need)), "full_grid": int(nW), "rows_inside_fold": bool(need.min() >= off and need.max() < off + nW),
                                "expected": EXPECTED_ELIGIBLE.get(s, {}).get(h), "run_alignment": al, "init_r_all_minus_2": bool(np.all(r[el] == -2.0)), "init_pred_equals_base": bool(np.array_equal(p0, pk["base"])),
                                "init_sequences_equal_base": all(np.array_equal(t0_["preds"][q["id"]], D["base_preds"][q["id"]]) for q in D["seqs"]), "init_tables_equal_Base_tables": same,
                                "random_scores_non_A4C_unchanged": bool(np.array_equal(pr[~el], pk["base"][~el])), "random_scores_changes_only_2_to_3": bool(set(zip(pk["base"][pr != pk["base"]].tolist(), pr[pr != pk["base"]].tolist())) <= {(A4C, A5C)}),
                                "random_scores_changed": int((pr != pk["base"]).sum()), "label_free_decision": bool(np.array_equal(pr, pr2)), "weighted_mean_vs_explicit_sample_mean_max_abs": float(mdev),
                                "random_sequences_change_only_changed_runs": all(np.array_equal(edr[q["id"]][int(pk["a"][k]):int(pk["b"][k])], np.full(int(pk["b"][k] - pk["a"][k]), pr[k])) for k in range(len(pr)) for q in [D["seqs"][int(pk["seq"][k])]])}
    return out


def scoring_timing(ctx: Ctx, s: int, device, workers: int) -> dict:
    """Fit rows (first 6 pool sequences): decode once, two frozen models (no graph), the no-grad scoring rate."""
    F_ = Fold(ctx, s, 0); rows = np.flatnonzero(F_.W["seq"] < 6)[:64]; t0 = time.time(); fr, idx = t37.chunk_frames(F_.seqs, {"seq": F_.W["seq"][rows]}, F_.W["s"][rows], F_.fp, workers); td = time.time() - t0
    ms = [build(ctx, s, 0, device, patched=False) for _ in range(2)]; frd, idxd, mkd = torch.from_numpy(fr).to(device), torch.from_numpy(idx).to(device), torch.from_numpy(F_.W["marker"][rows].astype(np.int64)).to(device)
    torch.cuda.synchronize(); t1 = time.time()
    with torch.no_grad():
        for m, h in ms:
            for w in range(len(rows)):
                window_score(m, h, frd, idxd, mkd, w)
    torch.cuda.synchronize(); gs = time.time() - t1; del ms; torch.cuda.empty_cache()
    return {"seconds_per_row_two_models": (gs + td) / len(rows), "seconds_per_row_one_model_incl_decode": (gs / 2 + td) / len(rows), "gpu_ms_per_row_one_model": 1e3 * gs / 2 / len(rows), "decode_ms_per_row": 1e3 * td / len(rows), "rows": int(len(rows))}


def check(ctx: Ctx, device, workers: int) -> str:
    s = 0; t_start = time.time(); res = {"config_sha256": sha256_file(ROOT / ctx.config), "tolerances": TOL, "limits": LIMITS, "contract": rc.CONTRACT, "expected_lists": EXPECTED_LISTS, "expected_plan": EXPECTED_PLAN, "expected_eligible": EXPECTED_ELIGIBLE,
                                         "numerics": {"tf32_matmul": torch.backends.cuda.matmul.allow_tf32, "tf32_cudnn": torch.backends.cudnn.allow_tf32, "cudnn_deterministic": torch.backends.cudnn.deterministic,
                                                      "cudnn_benchmark": torch.backends.cudnn.benchmark, "deterministic": torch.are_deterministic_algorithms_enabled(), "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled()}}
    res["lists"] = list_checks(ctx, s)
    if not all(res["lists"][str(f)]["counts_equal_expected"] and res["lists"][str(f)]["plan"]["sha_equal_expected"] and res["lists"][str(f)]["plan"]["exposure_equal_expected"] for f in FOLDS):
        publish(ctx, "checks.json", {**res, "ok": False, "stop": "training lists / plan differ from the asserted counts / sha256", "finished": now()}); raise SystemExit("lists / plan differ from the asserted values: engineering stop before pixels")
    res["one_way"] = one_way_checks(ctx, s)
    F0 = Fold(ctx, s, 0); res["frames"] = frame_checks(F0); res["init"] = init_checks(ctx, s, F0, device); res["formula"] = formula_checks()
    res["gradients"] = gradient_checks(ctx, s, F0, device); res["training"] = training_checks(ctx, s, F0, device, workers)
    res["a08_files_unchanged"] = all(sha256_file(ROOT / f"models_frozen/video_within_view_v1/all_s{k}/final.pt") == t31.A08[k] for k in (0, 2)); tm_ = scoring_timing(ctx, s, device, workers); T = res["training"]["timing"]
    spw = max(T["longest_update"]["seconds_per_occurrence"], T["updates_0_5"]["seconds_per_occurrence_gpu"], T["updates_6_17_wall"]["seconds_per_occurrence_wall"]); ovh = max(0.0, T["shortest_update"]["gpu_seconds"] - T["shortest_update"]["occurrences"] * spw)
    occ = {f: res["lists"][str(f)]["plan"]["occurrences_total"] for f in FOLDS}; fit_rows = {f: res["lists"][str(f)]["eligible_rows"] for f in FOLDS}; spent = t31.vsh.ledger_hours(ctx.ledger, "gpu") + (time.time() - t_start) / 3600
    train_h = sum(2 * (occ[f] * spw + U * ovh + 2 * fit_rows[f] * tm_["seconds_per_row_one_model_incl_decode"]) for f in FOLDS) * 1.15 / 3600
    nrows = sum(v["rows"] for v in EXPECTED_ELIGIBLE[s].values()); read_h = nrows * tm_["seconds_per_row_two_models"] * 1.15 / 3600; proj = spent + train_h + read_h + 0.4
    res["projection"] = {"spent_gpu_h_incl_this_check": spent, "train_4x600_incl_full_fit_h": train_h, "readout_eligible_rows_h": read_h, "reserve_h": 0.4, "source0_total_h": proj, "source2_if_triggered_h_approx": train_h + read_h,
                         "cap_h": ctx.v["budget"]["gpu_hours_max"], "seconds_per_occurrence_used": spw, "per_update_overhead_s": ovh, "plan_occurrences": occ, "full_fit_rows": fit_rows, "eligible_rows": nrows, **tm_}
    npar = res["training"]["binding"]["groups"]; ntr = sum(g["parameters"] for g in npar.values()); nall = ntr + res["training"]["binding"]["frozen_parameters"]
    res["disk"] = {"free_gb": shutil.disk_usage(ctx.out).free / 1e9, "per_model_final_gb": 4 * nall / 1e9, "resume_gb": (4 * nall + 8 * ntr) / 1e9, "min_free_gb": ctx.v["budget"]["min_free_disk_gb"]}
    gs = gates(res); res["enforced_gates"] = {k: bool(v) for k, v in sorted(gs.items())}; res["required_gates_missing"] = sorted(REQUIRED_GATES - set(gs))
    res["loaded_code_sha256"] = t38.loaded_code(); res["failed"] = sorted(k for k, v in gs.items() if not v) + res["required_gates_missing"]; res["ok"] = not res["failed"]; res["finished"] = now(); publish(ctx, "checks.json", res)
    print(json.dumps({k: v for k, v in res.items() if k not in ("loaded_code_sha256", "lists")}, indent=1, default=str)[:14000], flush=True)
    if not res["ok"]:
        raise SystemExit(f"checks failed: {res['failed']}")
    return f"{spw:.3f} s per occurrence; source-0 projection {proj:.2f} GPU h"


def gates(r: dict) -> dict:
    """Every enforced check gate, one dictionary item each; a missing required gate fails the check."""
    Ls, O, Fr, I, Fm, G, T, Pr, N = r["lists"], r["one_way"], r["frames"], r["init"], r["formula"], r["gradients"], r["training"], r["projection"], r["numerics"]
    lim = lambda d: d["loss_max_abs"] <= LIMITS["loss"] and d["params_max_abs"] <= LIMITS["params"]  # noqa: E731
    lk = lambda f: Ls[str(f)]  # noqa: E731
    ex = EXPECTED_ELIGIBLE[0]; up = T["updates_1_5"]; RU = T["real_update_C_6"]
    return {"isolation": not any(Ls["isolation"].values()) and all(lk(f)["recordings_in_other_pool"] == 0 and lk(f)["sessions_in_other_pool"] == 0 for f in FOLDS),
            "inputs": all(Ls["pins"][k] for k in ("pools_equal", "split_equal", "runs_equal", "outputs_equal", "a08_equal")),
            "lists": all(lk(f)["counts_equal_expected"] and lk(f)["lists_in_P_order_and_disjoint"] and lk(f)["eligible_is_base_A4C"] and lk(f)["positives_are_S1"] for f in FOLDS),
            "membership": all(lk(f)["membership_equals_repeated_sample_loop"] and lk(f)["weights_sum_1"] and lk(f)["rows_inside_fold"] and lk(f)["rows_unique_per_run"] and lk(f)["weighted_mean_vs_explicit_mean_max_abs"] <= 1e-9
                              and lk(f)["run_alignment_with_base_sequences"] for f in FOLDS),
            "plan": all(lk(f)["plan"]["updates"] == U and lk(f)["plan"]["shape"] == [U, 4] and lk(f)["plan"]["sha_equal_expected"] and lk(f)["plan"]["repeat_equal"] and lk(f)["plan"]["explicit_rng_equal"] and lk(f)["plan"]["positives_in_list"]
                        and lk(f)["plan"]["negatives_in_list"] and lk(f)["plan"]["all_positives_drawn"] and lk(f)["plan"]["all_negatives_drawn"] and lk(f)["plan"]["same_update_duplicate_negatives"] == 0 and lk(f)["plan"]["exposure_equal_expected"] for f in FOLDS),
            "numerics": not N["tf32_matmul"] and not N["tf32_cudnn"] and N["cudnn_deterministic"] and not N["cudnn_benchmark"] and N["deterministic"] and not N["deterministic_warn_only"],
            "pixels": Fr["pixels_equal_independent_decode"] and Fr["repeat_equal"] and Fr["rows_weights_slots_equal_plan"], "markers": Fr["markers_equal_W"] and Fr["markers_equal_marker_slot"],
            "init_state": I["init_state_equals_A08"] and I["identical_arm_start"] and I["patched_blocks"] == [1, 3, 14],
            "init_head": I["head"]["parameters"] == rc.HEAD_PARAMS and I["head"]["weight_all_zero"] and I["head"]["bias"] == rc.INIT_BIAS and I["head"]["fold1_head_equal_values"] and I["embed_shape"] == [1, rc.EMBED],
            "init_score_constant": I["init_scores_all_minus_2"]["all_exactly_minus_2"],
            "init_online_parity": I["init_online_fit_logits_vs_A31_cache_max_abs"] <= TOL["parity"] and I["classifier_of_embed_equals_forward"], "patched_forward": I["patched_equals_original_forward_bitwise"] and I["no_hook_left"],
            "base_from_cache": all(all(v[k] for k in ("runs_equal_base", "sequences_equal_base", "rows_equal_A32_base")) for v in I["Base_from_cached_q"].values()),
            "formula": all(v["r_max_abs"] <= TOL["formula"] and v["loss_abs"] <= TOL["formula"] and v["A_abs"] <= TOL["formula"] and v["B_abs"] <= TOL["formula"] and v["H_abs"] <= TOL["formula"] and v["grad_max_abs"] <= TOL["formula"] and v["finite"]
                           and v["inactive_negative_windows_zero_grad"] is not False for k, v in Fm.items() if k not in ("kinks_N", "huge_C", "dual"))
            and Fm["kinks_N"]["H"] == Fm["kinks_N"]["H_expected"] and max(abs(a - b) for a, b in zip(Fm["kinks_N"]["grad"], Fm["kinks_N"]["grad_expected"])) <= 1e-12 and Fm["kinks_N"]["loss_finite"] and Fm["huge_C"]["finite"],
            "dual": Fm["dual"]["max_abs"] <= 1e-12 and Fm["dual"]["clip_low"] and Fm["dual"]["clip_high"] and Fm["dual"]["feasible_no_change_alpha"] and Fm["dual"]["constants"] == {"alpha": 0.01, "lambda0": 1.0, "step": 0.1, "range": [0.0, 100.0]},
            "two_pass_fp32": all(G[f"two_pass_vs_retained_{a}"]["relative"] <= TOL["grad_rel"] and G[f"two_pass_vs_retained_{a}"]["all_groups_nonzero"] and G[f"two_pass_vs_retained_{a}"]["loss_abs"] <= 1e-5 for a in ARMS),
            "replay_scores": all(G[f"two_pass_vs_retained_{a}"]["pass1_vs_replay_max_abs"] <= TOL["replay_abs"] and G[f"two_pass_vs_retained_{a}"]["pass1_vs_retained_scores_max_abs"] <= TOL["replay_abs"] for a in ARMS) and all(x["replay_max_abs"] <= TOL["replay_abs"] for x in up),
            "zero_loss_negatives": G["two_pass_vs_retained_N"]["zero_grad_negative_windows"] == 0 if G["two_pass_vs_retained_N"]["active_negatives"] == 3 else G["two_pass_vs_retained_N"]["zero_grad_negative_windows"] > 0,
            "skip_pool_parity": G["patched_vs_original_maxpool_backward"]["relative"] <= TOL["grad_rel"] and G["warn_only_restored_false"],
            "binding": T["binding"]["binds_exactly_trainable"] and T["binding"]["trainable_equals_requires_grad"] and T["binding"]["frozen_only_classifier"] and T["binding"]["eval_mode"] and T["binding"]["head_parameters"] == rc.HEAD_PARAMS
            and T["binding"]["groups"]["scalar_head"]["parameters"] == rc.HEAD_PARAMS and T["binding"]["marker_group"] == ["marker_stem.weight"] and T["binding"]["projection_group"] == ["encoder.head.1.weight", "encoder.head.1.bias"]
            and set(T["binding"]["frozen"]) == {"classifier.weight", "classifier.bias"} and {k: v["lr"] for k, v in T["binding"]["groups"].items()} == LR,
            "first_step_zero": T["update0"]["g_encoder_trunk"] == 0 and T["update0"]["g_marker_stem"] == 0 and T["update0"]["g_projection"] == 0 and T["update0"]["g_scalar_head"] > 0,
            "genuine_gradients": all(all(np.isfinite(x[k]) and x[k] > 0 for k in ("g_encoder_trunk", "g_marker_stem", "g_projection", "g_scalar_head")) for x in up),
            "real_update": RU["optimizer_steps"] == [6] and all(v["tensors_moved"] > 0 for v in RU["moved"].values())
            and all(k.startswith("m.classifier.") or k in RU["zero_gradient_trainable_tensors_updates_1_5"]
                    or (k in RU["unchanged_trainable"] and RU["unchanged_trainable"][k]["structural_reason"] is not None and RU["unchanged_trainable"][k]["relative_to_trunk_max"] <= TOL["roundoff_grad_rel"]) for k in RU["unchanged_tensors"]),
            "repeat": lim(T["repeat_6_C"]), "forced_C_objective": lim(T["forced_C_objective_N_vs_C"]) and T["forced_C_objective_N_vs_C"]["lambda_unchanged"] and T["forced_C_objective_N_vs_C"]["formal_arms_unchanged"],
            "resume": lim(T["resume_3_plus_3_N"]) and T["resume_3_plus_3_N"]["lambda_max_abs"] <= 1e-12 and T["resume_3_plus_3_N"]["lambda_restored"],
            "N_not_C": T["N_differs_from_C"]["params_max_abs"] > 0,
            "one_way_synthetic": O["synthetic_rule"]["result"] == O["synthetic_rule"]["expected"] and O["synthetic_rule"]["input_unchanged"] and O["synthetic_merge"]["edited"] == O["synthetic_merge"]["expected"]
            and O["synthetic_merge"]["boundaries_before_after"] == [4, 3],
            "eligible_counts": all(O[f"heldout_f{h}"]["eligible_runs"] == ex[h]["runs"] and O[f"heldout_f{h}"]["eligible_unique_rows"] == ex[h]["rows"] and O[f"heldout_f{h}"]["full_grid"] == ex[h]["grid"]
                                   and O[f"heldout_f{h}"]["rows_inside_fold"] for h in FOLDS),
            "run_alignment": all(O[f"heldout_f{h}"]["run_alignment"] for h in FOLDS),
            "init_equals_base": all(O[f"heldout_f{h}"][k] for h in FOLDS for k in ("init_r_all_minus_2", "init_pred_equals_base", "init_sequences_equal_base", "init_tables_equal_Base_tables")),
            "readout_label_free": all(O[f"heldout_f{h}"][k] for h in FOLDS for k in ("label_free_decision", "random_scores_non_A4C_unchanged", "random_scores_changes_only_2_to_3", "random_sequences_change_only_changed_runs")),
            "deploy_mean": all(O[f"heldout_f{h}"]["weighted_mean_vs_explicit_sample_mean_max_abs"] <= 1e-5 for h in FOLDS),
            "a08_files": r["a08_files_unchanged"], "memory": T["timing"]["peak_gpu_gb_training"] <= TOL["peak_gpu_gb"], "projection": Pr["source0_total_h"] <= Pr["cap_h"], "disk": r["disk"]["free_gb"] >= r["disk"]["min_free_gb"]}


# ----------------------------------------------------------------------------------------------- readout
def load_model(ctx: Ctx, arm: str, s: int, f: int, device):
    """The committed final weights in the original A08 module (original nn.MaxPool3d; no backward at readout) and head."""
    key = f"{arm}_s{s}_f{f}"; rd = ctx.out / "train" / key; man = json.loads((rd / "run_manifest.json").read_text()); com = json.loads((ctx.out / "weights_committed.json").read_text()).get(key, {})
    if sha256_file(rd / "final.pt") != man["final_sha256"] or sha256_file(rd / "head.pt") != man["head_sha256"] or com != {"final": man["final_sha256"], "head": man["head_sha256"]}:
        raise SystemExit(f"{key}: weights not committed or changed")
    m = ctx.a31.a08(s, device); m.load_state_dict(torch.load(rd / "final.pt", map_location="cpu", weights_only=True), strict=True); head = rc.ScalarHead().to(device)
    head.load_state_dict(torch.load(rd / "head.pt", map_location="cpu", weights_only=True), strict=True); return m.eval(), head.eval()


def paired(y: np.ndarray, b: np.ndarray, pc: np.ndarray, pn: np.ndarray) -> dict:
    """Private N vs C overlap on the same runs: common / C-only / N-only fixes and harms, A4C-specific harms, label disagreements."""
    wr = b != y; fx = {a: (wr & (p == y)) for a, p in (("C", pc), ("N", pn))}; br = {a: (~wr & (p != y)) for a, p in (("C", pc), ("N", pn))}; h4 = {a: ((b == y) & (y == A4C) & (p != y)) for a, p in (("C", pc), ("N", pn))}
    s1 = {a: ((y == A5C) & (b == A4C) & (p == A5C)) for a, p in (("C", pc), ("N", pn))}
    ov = lambda d: {"common": int((d["C"] & d["N"]).sum()), "C_only": int((d["C"] & ~d["N"]).sum()), "N_only": int((~d["C"] & d["N"]).sum())}  # noqa: E731
    return {"fixes": ov(fx), "S1_fixes": ov(s1), "harms": ov(br), "correct_A4C_harms": ov(h4), "run_label_disagreements": int((pc != pn).sum()), "runs": int(len(y))}


def readout(ctx: Ctx, s: int, device, workers: int, dry: bool = False) -> str:
    name = f"readout_s{s}.json"
    if (ctx.out / name).exists() and not dry:
        raise SystemExit("this seed's readout exists; it runs once")
    X = t40.full_q(ctx, s); a8 = rsh.a08_ctx(); data = t31.vpr.training_data(ctx.a31.a15.a10.a07p); rec = {"seed": s, "cells": {}}; acc = {m_: [] for m_ in ("Base", *ARMS)}; ys, bs = [], []
    cov = {a: [] for a in ARMS}; s1recs = {a: [] for a in ARMS}; old = json.loads((ROOT / "docs/results/2026-10-25_a5c_source_external_evidence/readout.json").read_text())["sequences"]; tim = {"gpu_seconds": 0.0, "decode_seconds": 0.0}
    a43p = ROOT / "docs/results/2026-10-27_a5c_witness_mil/readout_s0.json"; a43 = json.loads(a43p.read_text()) if (s == 0 and a43p.exists()) else None; cache = {}; nrows = 0
    for h in FOLDS:
        f = 1 - h; D = d32.fold_data(ctx.d, s, h); D["M"] = d32.set_masks(D); off = X["offset"][h]; pk = D["pk"]; P = {"seqs": D["seqs"], "n_of": dict(data.n), "label_of": dict(data.label)}; ev = bc.evalset(P, list(range(len(D["seqs"]))), f"heldout_f{h}")
        el = pk["base"] == A4C; need = eligible_rows(pk, el) - off; W = t31.pool_W(D["seqs"]); nW = len(W["seq"]); nrows += len(need)
        if s in EXPECTED_ELIGIBLE and (int(el.sum()), len(need), nW) != tuple(EXPECTED_ELIGIBLE[s][h][k] for k in ("runs", "rows", "grid")):
            raise SystemExit(f"held-out fold {h}: eligible runs / rows differ from the asserted counts")
        if dry:
            S = {a: np.full(nW, -2.0, np.float32) for a in ARMS}
        else:
            fp = t28.frame_paths(D["seqs"]); S = {}
            for a in ARMS:
                m, head = load_model(ctx, a, s, f, device); S[a] = grid_scores(m, head, D["seqs"], W, fp, need, device, workers, tim, f"held-out fold {h} {a} (fit {f})"); del m, head; torch.cuda.empty_cache()
        cell = {}; t = tables(X["q"], D, off, a8.cfg, P, ev); p = t.pop("pred"); acc["Base"].append(p); cell["Base"] = t; cell["Base"]["boundaries"] = t38.boundaries(X["q"], D, off); cell["Base"].update(a5c_extras(D, p)[0]); preds_arm = {}
        for arm in ARMS:
            r = rc.run_weighted_mean(pk, S[arm], off, el); p = rc.one_way(pk["base"], r); t = tables_pred(p, D, a8.cfg, P, ev); preds = t.pop("preds"); p = t.pop("pred"); acc[arm].append(p); cell[arm] = t; preds_arm[arm] = p
            ch = p != pk["base"]; cell[arm]["structural"] = {"eligible_runs": int(el.sum()), "eligible_rows": int(len(need)), "changed_runs": int(ch.sum()), "non_A4C_runs_unchanged": bool(np.array_equal(p[~el], pk["base"][~el])),
                                                         "changes_only_A4C_to_A5C": bool(set(zip(pk["base"][ch].tolist(), p[ch].tolist())) <= {(A4C, A5C)}), "changed_by_true_class": {str(c): int((ch & (pk["labels"] == c)).sum()) for c in range(5)}}
            cell[arm]["eligible_scores"] = score_summary(r, pk["labels"], el); cell[arm]["boundaries"] = bounds(D, preds)
            cell[arm]["coverage"], sets = t37.coverage(D, pk["labels"], pk["base"], p); cov[arm].append(sets); ex, recs = a5c_extras(D, p); cell[arm].update(ex); s1recs[arm].append(recs)
            cache[f"scores_{arm}_heldout_f{h}"] = S[arm][need]; cache[f"r_{arm}_heldout_f{h}"] = r[el]; cache[f"run_pred_{arm}_heldout_f{h}"] = p
        cache[f"rows_heldout_f{h}"] = need; cache[f"eligible_runs_heldout_f{h}"] = np.flatnonzero(el); cache[f"run_pred_Base_heldout_f{h}"] = acc["Base"][-1]; cache[f"run_label_heldout_f{h}"] = pk["labels"]
        cache[f"offsets_heldout_f{h}"] = pk["offsets"]; cache[f"rows_all_heldout_f{h}"] = pk["rows"] - off; cache[f"weights_all_heldout_f{h}"] = pk["weights"]
        s1 = np.flatnonzero(D["M"]["S1"]); cell["S1_denominator"] = {"runs": int(len(s1)), **d32.concentration([(D, int(i)) for i in s1], "true_src", None)}
        ob = old[f"s{s}_heldout_f{h}"]["A08 Base"]; cell["Base_reproduced"] = {"A32_row": cell["Base"]["sequence_row"] == ob["sequence_row"] and cell["Base"]["counts"] == ob["counts"]}
        if a43 is not None:
            js = lambda x: json.loads(json.dumps(x, default=float))  # noqa: E731
            cell["Base_reproduced"]["A43_Base_cell"] = all(js(cell["Base"][k]) == a43["cells"][f"heldout_f{h}"]["Base"][k] for k in a43["cells"][f"heldout_f{h}"]["Base"] if k in cell["Base"])
        cell["paired_N_vs_C"] = paired(pk["labels"], pk["base"], preds_arm["C"], preds_arm["N"])
        row = lambda m_: cell[m_]["sequence_row"]  # noqa: E731
        cell["comparisons"] = {f"{a} - {bb}": {"net_fixed": cell[a]["runs"]["all"]["net_fixed"] - (cell[bb]["runs"]["all"]["net_fixed"] if bb != "Base" else 0), "S1_fixed": cell[a]["S1_fixed"]["fixed"] - cell[bb]["S1_fixed"]["fixed"],
                                               "correct_A5C_kept": cell[a]["correct_A5C_kept"]["kept"] - cell[bb]["correct_A5C_kept"]["kept"], "correct_A4C_kept": cell[a]["correct_A4C_kept"]["kept"] - cell[bb]["correct_A4C_kept"]["kept"],
                                               **{k: row(a)[k] - row(bb)[k] for k in ("segment_f1_50_multi", "native_macro_f1", "boundary_recall_multi", "boundary_precision_multi")},
                                               **{k: cell[a]["counts"][k] - cell[bb]["counts"][k] for k in ("fp", "fp_short", "fn", "fn_short")}} for a, bb in (("N", "C"), ("N", "Base"), ("C", "Base"))}
        if not dry:
            mans = {arm: json.loads((ctx.out / "train" / f"{arm}_s{s}_f{f}" / "run_manifest.json").read_text()) for arm in ARMS}
            cell["fit_only"] = {arm: {k: mans[arm][k] for k in ("lambda_final", "lambda_cap_hits", "lambda_max_reached", "full_fit", "drift", "loss", "occurrences_trained", "false_promotions_in_minibatches", "replay_max_abs_max", "lists", "exposure")} for arm in ARMS}
        rec["cells"][f"heldout_f{h}"] = cell; ys.append(pk["labels"]); bs.append(pk["base"])
    y, b = np.concatenate(ys), np.concatenate(bs); wr = b != y; rec["oof"] = {}; pcn = {m_: np.concatenate(acc[m_]) for m_ in ARMS}
    for m_ in ARMS:
        p = pcn[m_]; fx, br = int((wr & (p == y)).sum()), int((~wr & (p != y)).sum()); un = {k: set().union(*[c[k] for c in cov[m_]]) for k in ("fixed", "broken")}
        rec["oof"][m_] = {"runs": int(len(y)), "fixed": fx, "broken": br, "net_fixed": fx - br, **{f"{k}_recordings": len(v) for k, v in un.items()}, **{f"{k}_sessions": len({session_of(x) for x in v}) for k, v in un.items()},
                          "S1_fixed_true_class_recordings": len(set().union(*s1recs[m_])), "broken_correct_A5C": int(((b == y) & (y == A5C) & (p != y)).sum()), "broken_correct_A4C": int(((b == y) & (y == A4C) & (p != y)).sum()),
                          "fixed_true_A5C_read_A4C": int(((y == A5C) & (b == A4C) & (p == A5C)).sum()), "changed_runs": int((p != b).sum()), "non_A4C_runs_unchanged": bool(np.array_equal(p[b != A4C], b[b != A4C]))}
    rec["oof"]["paired_N_vs_C"] = paired(y, b, pcn["C"], pcn["N"])
    if a43 is not None:
        rec["a43_comparison"] = {"denominators": "same source-0 folds, S1 103, correct A4C 1,413, correct A5C 1,150",
                                 **{arm43: {"S1_fixed_by_fold": [a43["cells"][f"heldout_f{h}"][arm43]["S1_fixed"]["fixed"] for h in FOLDS], "correct_A4C_kept_by_fold": [a43["cells"][f"heldout_f{h}"][arm43]["correct_A4C_kept"]["kept"] for h in FOLDS],
                                            "net_by_fold": [a43["cells"][f"heldout_f{h}"][arm43]["runs"]["all"]["net_fixed"] for h in FOLDS], "segment_f1_by_fold": [a43["cells"][f"heldout_f{h}"][arm43]["sequence_row"]["segment_f1_50_multi"] for h in FOLDS],
                                            "missed_short_oof": sum(a43["cells"][f"heldout_f{h}"][arm43]["counts"]["fn_short"] for h in FOLDS), "oof": {k: a43["oof"][arm43][k] for k in ("fixed", "broken", "net_fixed", "broken_correct_A4C")}} for arm43 in ("D", "M")}}
    rec["rule"], rec["values"], rec["summary"] = rule_of(rec, s1recs, dry)
    if dry:
        print(json.dumps({"dry_run_init": True, "Base_reproduced": {k: v["Base_reproduced"] for k, v in rec["cells"].items()}, "summary": rec["summary"], "oof": rec["oof"], "rule": rec["rule"]}, indent=1)); return "dry run (Init); nothing published"
    cp = ctx.out / f"readout_cache_s{s}.npz"; np.savez(cp, **cache); pr = json.loads((ctx.a31.out / "cache" / f"prefix_s{s}.json").read_text())
    rec["private_cache"] = {"path": f"{ctx.v['outputs']['root']}/readout_cache_s{s}.npz (server only)", "sha256": sha256_file(cp), "arrays": sorted(cache), "rows": "fold-local pool_W rows of the eligible runs (sorted unique); full membership offsets / rows / weights for the recount",
                            "full_fit_caches": {f"{arm}_s{s}_f{f}": json.loads((ctx.out / "train" / f"{arm}_s{s}_f{f}" / "run_manifest.json").read_text())["full_fit_scores_sha256"] for f in FOLDS for arm in ARMS}}
    rec["deployment_cost"] = {"forwards": "Base's one 224 forward per window plus one repair-policy forward (an A08-size encoder + 513-parameter head) per unique row of a Base-A4C run only", "parameters": "A08 34.6M (Base) + 34.6M + 513 (repair policy)",
                              "eligible_rows": nrows, "gpu_ms_per_row_policy": 1e3 * tim["gpu_seconds"] / (2 * nrows), "cpu_decode_ms_per_row": 1e3 * tim["decode_seconds"] / (2 * nrows),
                              "gpu_ms_per_window_A31_base_pass": 1e3 * pr["gpu_seconds"] / sum(pr["rows"].values()), "readout_gpu_seconds": tim["gpu_seconds"], "readout_decode_seconds": tim["decode_seconds"], "note": "two policies (C, N) per eligible row, each decoded in its own pass; microbatch 1"}
    rec["written"] = now(); publish(ctx, name, rec)
    print(json.dumps({"summary": rec["summary"], "oof": rec["oof"], "rule": rec["rule"], "values": rec["values"], "Base_reproduced": {k: v["Base_reproduced"] for k, v in rec["cells"].items()}}, indent=1)); return f"absolute gate C {rec['rule']['C']['passed']} N {rec['rule']['N']['passed']}; constraint contribution {rec['rule']['constraint_contribution']['met']}"


def rule_of(rec: dict, s1recs: dict, dry: bool = False) -> tuple[dict, dict, dict]:
    C_ = lambda h: rec["cells"][f"heldout_f{h}"]  # noqa: E731
    tot = lambda m_, k, sub: sum(C_(h)[m_][k][sub] for h in FOLDS)  # noqa: E731
    d_ = lambda a, bb, h, k: C_(h)[a]["sequence_row"][k] - C_(h)[bb]["sequence_row"][k]  # noqa: E731
    net = lambda a, h: C_(h)[a]["runs"]["all"]["net_fixed"]  # noqa: E731
    harm4 = lambda a, h: C_(h)[a]["correct_A4C_kept"]["runs"] - C_(h)[a]["correct_A4C_kept"]["kept"]  # noqa: E731
    n1, n4 = tot("Base", "S1_fixed", "runs"), tot("Base", "correct_A4C_kept", "runs"); out = {}
    for arm in ARMS:
        s1 = tot(arm, "S1_fixed", "fixed"); rf = [len(r) for r in s1recs[arm]]; rall = len(set().union(*s1recs[arm])); k4 = tot(arm, "correct_A4C_kept", "kept"); dF = [d_(arm, "Base", h, "segment_f1_50_multi") for h in FOLDS]
        g = {"S1_fixed": f"{s1}/{n1}", "S1_needed": ceil_pct(10, n1), "S1_ok": s1 >= ceil_pct(10, n1), "S1_recordings_total": rall, "S1_recordings_by_fold": rf, "S1_runs_by_fold": [C_(h)[arm]["S1_fixed"]["fixed"] for h in FOLDS],
             "S1_sources_ok": rall >= 3 and all(C_(h)[arm]["S1_fixed"]["fixed"] >= 1 and len(s1recs[arm][i]) >= 1 for i, h in enumerate(FOLDS)),
             "non_A4C_invariant_ok": all(C_(h)[arm]["structural"]["non_A4C_runs_unchanged"] for h in FOLDS), "correct_A4C_kept": f"{k4}/{n4}", "A4C_needed": ceil_pct(99, n4), "A4C_ok": k4 >= ceil_pct(99, n4),
             "correct_A4C_harms_by_fold": [harm4(arm, h) for h in FOLDS], "net_by_fold": [net(arm, h) for h in FOLDS], "net_each_fold_ge_0": all(net(arm, h) >= 0 for h in FOLDS), "net_oof": rec["oof"][arm]["net_fixed"], "net_oof_ge_5": rec["oof"][arm]["net_fixed"] >= 5,
             "dF_by_fold": dF, "dF_each_fold_ge_0": all(x >= 0 for x in dF), "dF_mean": float(np.mean(dF)), "dF_mean_ge_0.001": float(np.mean(dF)) >= 0.001,
             "native_each_fold_ge_-0.003": all(d_(arm, "Base", h, "native_macro_f1") >= -0.003 for h in FOLDS), "boundary_recall_each_fold_ge_-0.005": all(d_(arm, "Base", h, "boundary_recall_multi") >= -0.005 for h in FOLDS),
             "missed_short": tot(arm, "counts", "fn_short"), "missed_short_Base": tot("Base", "counts", "fn_short"), "missed_short_le_Base_plus_3": tot(arm, "counts", "fn_short") <= tot("Base", "counts", "fn_short") + 3,
             "changed_runs_oof": rec["oof"][arm]["changed_runs"], "not_a_no_op": rec["oof"][arm]["changed_runs"] > 0}
        g["package_passed"] = all(v for v in g.values() if isinstance(v, bool)); g["passed"] = g["package_passed"]; out[arm] = g
    # N's absolute gate = the common package gate AND full-fit H <= alpha in BOTH fit models (the source-2 trigger uses this combined boolean)
    Hs = {f"N_f{1 - h}": C_(h)["fit_only"]["N"]["full_fit"]["update_600"]["H_full_fit"] for h in FOLDS} if not dry else {}
    out["N"]["full_fit_H_600"] = Hs; out["N"]["full_fit_feasible_both"] = (all(v <= rc.ALPHA for v in Hs.values()) if Hs else None)
    out["N"]["passed"] = bool(out["N"]["package_passed"] and out["N"]["full_fit_feasible_both"] is True); out["N"]["constraint_qualified_success"] = out["N"]["passed"]
    if not dry:
        out["N"]["full_fit_false_promotions_600"] = {f"N_f{1 - h}": C_(h)["fit_only"]["N"]["full_fit"]["update_600"]["false_promotions"] for h in FOLDS}
        out["C"]["full_fit_H_600"] = {f"C_f{1 - h}": C_(h)["fit_only"]["C"]["full_fit"]["update_600"]["H_full_fit"] for h in FOLDS}
    hC, hN = [harm4("C", h) for h in FOLDS], [harm4("N", h) for h in FOLDS]; s1C, s1N = [C_(h)["C"]["S1_fixed"]["fixed"] for h in FOLDS], [C_(h)["N"]["S1_fixed"]["fixed"] for h in FOLDS]
    cc = {"S1_C_by_fold": s1C, "S1_N_by_fold": s1N, "S1_N_retains_80pct_each_fold": all(n >= -(-8 * c // 10) for c, n in zip(s1C, s1N)), "harms_C_by_fold": hC, "harms_N_by_fold": hN, "harms_N_le_C_each_fold": all(n <= c for c, n in zip(hC, hN)),
          "harms_oof": [sum(hC), sum(hN)], "harms_N_half_and_5_fewer_oof": sum(hN) <= 0.5 * sum(hC) and sum(hC) - sum(hN) >= 5, "net_N_minus_C_oof": rec["oof"]["N"]["net_fixed"] - rec["oof"]["C"]["net_fixed"],
          "net_N_minus_C_oof_ge_5": rec["oof"]["N"]["net_fixed"] - rec["oof"]["C"]["net_fixed"] >= 5, "F1_N_ge_C_each_fold": all(d_("N", "C", h, "segment_f1_50_multi") >= 0 for h in FOLDS), "N_absolute_passed": out["N"]["passed"]}
    cc["met"] = all(v for v in cc.values() if isinstance(v, bool)); out["constraint_contribution"] = cc
    out["source2_by_rule"] = out["C"]["passed"] or out["N"]["passed"]; out["note"] = "source 2 additionally requires the remaining shared budget (checked by the owner before launch)"
    values = {"N_minus_C_net": [net("N", h) - net("C", h) for h in FOLDS], "C_minus_Base_net": [net("C", h) for h in FOLDS], "N_minus_Base_net": [net("N", h) for h in FOLDS],
              "N_minus_C_dF": [d_("N", "C", h, "segment_f1_50_multi") for h in FOLDS], "C_minus_Base_dF": [d_("C", "Base", h, "segment_f1_50_multi") for h in FOLDS], "N_minus_Base_dF": [d_("N", "Base", h, "segment_f1_50_multi") for h in FOLDS],
              "native": {a: [d_(a, "Base", h, "native_macro_f1") for h in FOLDS] for a in ARMS}, "boundary_recall": {a: [d_(a, "Base", h, "boundary_recall_multi") for h in FOLDS] for a in ARMS},
              "missed_short": {m_: tot(m_, "counts", "fn_short") for m_ in ("Base", *ARMS)}, "paired_oof": rec["oof"]["paired_N_vs_C"]}
    summary = {f"heldout_f{h}": {m_: {"net": net(m_, h) if m_ != "Base" else 0, "F1": C_(h)[m_]["sequence_row"]["segment_f1_50_multi"], "S1": C_(h)[m_]["S1_fixed"]["fixed"], "A5C_kept": C_(h)[m_]["correct_A5C_kept"]["kept"],
                                       "A4C_kept": C_(h)[m_]["correct_A4C_kept"]["kept"]} for m_ in ("Base", *ARMS)} for h in FOLDS}
    return out, values, summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True); ap.add_argument("--stage", choices=("plan", "check", "train", "freeze", "readout"), required=True); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6); ap.add_argument("--commit", default=""); ap.add_argument("--manifest-sha", default=""); ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(); a.set = []; ctx = Ctx(load_cfg(a), a.config); ctx.out.mkdir(parents=True, exist_ok=True)
    if a.stage == "plan":
        def run_plan():
            r = plan_stage(ctx, a.seed); write_json(ctx.out / f"plan_s{a.seed}.json", {**r, "finished": now()}); print(json.dumps(r, indent=1, default=str)); return f"ok {r['ok']}; sha f0 {r['0']['plan_sha256'][:12]} f1 {r['1']['plan_sha256'][:12]}"
        return staged(ctx, f"plan s{a.seed} (CPU, metadata only)", "cpu", run_plan)
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
