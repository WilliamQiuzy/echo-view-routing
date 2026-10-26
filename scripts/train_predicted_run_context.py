#!/usr/bin/env python
"""A38 (configs/experiments/video_predicted_run_context_v1.yaml): the frozen Base's predicted run limits A37's wider window.

  --stage audit    --seed s: independent mapping loop and index statistics of C against W (no reference labels, no model) (CPU)
  --stage check    online e224 / eW parity, explicit C stacking, Init = Base, forced eC = eW, head mechanics, timing and disk (GPU)
  --stage extract  --seed s: eC of every request window, eW reused where C's refs equal W's (source 2: eW first) (GPU)
  --stage train    --seed s: residual heads W_f0, C_f0, W_f1, C_f1 on A31's uniform plans (CPU, single thread)
  --stage freeze   --seed s --commit c --manifest-sha h: record that weights_s{s}.json (sha h) was committed in c
  --stage readout  --seed s: after the heads are committed, one held-out readout (+ the source-2 rule for seed 0) (CPU)
The A08 + B3 path supplies q, e224, Base, its runs and the predicted labels the mapping reads; nothing here changes it.
"""
from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")     # A27's CPU-only trainer import would otherwise hide the GPU

import argparse  # noqa: E402
import hashlib  # noqa: E402
import inspect  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
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
import scripts.report_raw_video_context as rr  # noqa: E402
import scripts.report_video_switch_head as rsh  # noqa: E402
import scripts.train_ensemble_teacher as tet  # noqa: E402
import scripts.train_interval_visual_adaptation as t28  # noqa: E402
import scripts.train_native_spatial_detail as t36  # noqa: E402
import scripts.train_native_window_recognition as t29  # noqa: E402
import scripts.train_session_correction as t31  # noqa: E402
import scripts.train_wider_temporal_evidence as t37  # noqa: E402
from scripts._common import load_cfg  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import echo_routing.temporal.interval_recognition as ir  # noqa: E402
import echo_routing.temporal.joint_visual as jv  # noqa: E402
import echo_routing.temporal.predicted_run_context as prc  # noqa: E402
import echo_routing.temporal.wider_temporal as wt  # noqa: E402
import echo_routing.vit.target_mvit as tm  # noqa: E402
from echo_routing.evaluate.complementarity import sha256_file  # noqa: E402
from echo_routing.evaluate.failure_units import session_of  # noqa: E402
from echo_routing.evaluate.window_target import window_frames  # noqa: E402

A31_CONFIG, A32_CONFIG, A37_CONFIG = t36.A31_CONFIG, t36.A32_CONFIG, "configs/experiments/video_wider_temporal_evidence_v1.yaml"
FOLDS, ARMS, U, WARM, CHUNK_SEQS, BATCH = (0, 1), ("W", "C"), 1350, 135, 150, 32
TOL = {"parity": 1e-4, "explicit": 1e-5, "reuse": 1e-4}
RUNS_SHA = {0: {0: "7b28ac6c224f594ef7bf49d6307186807d91bd8577dbc192e8cece07bea7452c", 1: "98ce5c572aa29576a098685c1d28220684e755cbea537b6739020c0387ddc026"},
            2: {0: "91d04d4f255b7e892343b102c736f1b69c8126a69f2e0df1f5c23f1c60d8860f", 1: "ef76e587a0db2c7b18dcbdaee5631497963ad8eb9988a929e4526d5949259ef9"}}
A37_W = {0: {"final": "9fd523a95251c90dd26de0e72b3a48f427f943f79b71772ba4f772c1fcb98fd1", "updates": "a93d299d926f6c0347459ce42f2e79dbb6e182a40f48bbfc0a09c7e470955c72"},
         1: {"final": "4a1c6c66dd4a9aec3a191a00c97efc2e4a02e5d017f6b6bc351253fea0a29899", "updates": "dddbbdc83e7db759d6341c1ce12e8803bead718f781a29d1632858e2f1576ae6"}}
PIXELS = "A08 frame rule: numerically ordered JPEG, RGB, gamma LUT rounded to 6 decimals, short side 256 then centre crop 224; A08 marker planes; 16 slots"
now, write_json, ledger_append, step, fit_monitor, tables = t31.now, t31.write_json, t31.ledger_append, t36.step, t36.fit_monitor, t36.tables
if (t36.U, t36.WARM) != (U, WARM):
    raise SystemExit("A36's step schedule differs from A37's")


class Ctx:
    def __init__(self, cfg: dict, config: str) -> None:
        self.config, self.v = config, cfg["predicted_run_context"]; o = self.v["outputs"]
        self.out, self.results, self.ledger = ROOT / o["root"], ROOT / o["results_dir"], ROOT / self.v["budget"]["ledger"]
        self.a31 = t31.Ctx(load_cfg(SimpleNamespace(config=A31_CONFIG, set=[])), A31_CONFIG); self.d = d32.Ctx(load_cfg(SimpleNamespace(config=A32_CONFIG, set=[])), A32_CONFIG)
        self.a37 = t37.Ctx(load_cfg(SimpleNamespace(config=A37_CONFIG, set=[])), A37_CONFIG)


def staged(ctx: Ctx, job: str, kind: str, fn) -> None:
    t36.staged(ctx, job, kind, fn)


def publish(ctx: Ctx, name: str, rec: dict) -> None:
    t36.publish(ctx, name, rec)


# ----------------------------------------------------------------------------------------------- windows and features
def base_labels(ctx: Ctx, s: int, f: int) -> np.ndarray:
    """The seed's frozen A08 + B3 per-sample prediction of a pool (pool order); only this array of the run file is read."""
    p = ctx.a31.out / "data" / f"runs_s{s}_f{f}.npz"
    if sha256_file(p) != RUNS_SHA[s][f]:
        raise SystemExit(f"runs_s{s}_f{f}.npz differs from the configuration")
    with np.load(p) as z:
        return z["base_labels"]


def fold_windows(ctx: Ctx, s: int, f: int) -> tuple[list, dict, dict]:
    seqs = t31.load_pool(ctx.a31, f); W = t31.pool_W(seqs); M = prc.pool_limited(seqs, W, base_labels(ctx, s, f))
    if not np.array_equal(M["W"], wt.pool_positions(seqs, W)):
        raise SystemExit("W differs from A37's mapping")
    return seqs, W, M


def positions_sha(W: dict, M: dict) -> str:
    return hashlib.sha256(W["seq"].tobytes() + W["centre"].tobytes() + M["W"].tobytes() + M["C"].tobytes()).hexdigest()


def forward_rows(m, seqs: list, W: dict, pos: np.ndarray, rows: np.ndarray, out: np.ndarray, fp, workers: int, device, acc: dict) -> None:
    """eW / eC of the given window rows at positions pos[rows], in batches of BATCH consecutive rows (A37's frame path)."""
    if len(rows) == 0:
        return
    a = time.time(); fr, idx = t37.chunk_frames(seqs, {"seq": W["seq"][rows]}, pos[rows], fp, workers); acc["decode_seconds"] += time.time() - a
    for b in range(0, len(rows), BATCH):
        ks = np.arange(b, min(b + BATCH, len(rows))); torch.cuda.synchronize(); a = time.time(); e, _ = t37.embed(m, fr, idx[ks], W["marker"][rows[ks]], device); torch.cuda.synchronize()
        acc["gpu_seconds"] += time.time() - a; out[rows[ks]] = e.cpu().numpy()


def feature_key(ctx: Ctx, s: int) -> str:
    """Only what the features depend on: mapping / frame / model code, extraction functions, weights, pools, Base runs, positions, batching, pixels."""
    h = hashlib.sha256()
    for p in (Path(prc.__file__), Path(wt.__file__), Path(tm.__file__), Path(jv.__file__)):
        h.update(p.read_bytes())
    for fn in (base_labels, fold_windows, forward_rows, extract, t37.chunk_frames, t37.embed, rr.decode_frames, rr.RefFrames, rr.half_windows, t31.pool_W, window_frames):
        h.update(inspect.getsource(fn).encode())
    sp = json.loads((ctx.a31.out / "split.json").read_text()); pos = {str(f): positions_sha(*fold_windows(ctx, s, f)[1:]) for f in FOLDS}
    h.update(json.dumps({"a08": t31.A08[s], "pools": sp["pool_sha256"], "runs": RUNS_SHA[s], "positions": pos, "chunk": CHUNK_SEQS, "batch": BATCH, "pixels": PIXELS,
                         "eW_source": "A37 record (weights_s0.json)" if s == 0 else "this round"}, sort_keys=True).encode())
    return h.hexdigest()


def a37_eW(ctx: Ctx) -> torch.Tensor:
    """Source 0: A37's eW, accepted only with the feature record committed with A37's heads and A37's own key."""
    return t37.feats(ctx.a37, 0, committed=True)["W"]


def extract(ctx: Ctx, s: int, device, workers: int) -> str:
    m = ctx.a31.a08(s, device).eval(); d0 = tet.state_digest(m); fdir = ctx.out / "feats"; fdir.mkdir(parents=True, exist_ok=True); acc = {"gpu_seconds": 0.0, "decode_seconds": 0.0}; rec = {}
    src = a37_eW(ctx).numpy() if s == 0 else None; off = 0; prog = ctx.out / f"progress_s{s}.json"; total = 0
    for f in FOLDS:
        seqs, W, M = fold_windows(ctx, s, f); n = len(W["seq"]); fp = t28.frame_paths(seqs); same = prc.same_refs(seqs, W["seq"], M["C"], M["W"]); total += n
        EW = src[off:off + n].copy() if s == 0 else np.zeros((n, 512), np.float32); EC = np.zeros((n, 512), np.float32); new = 0
        for c0 in range(0, len(seqs), CHUNK_SEQS):
            inch = (W["seq"] >= c0) & (W["seq"] < c0 + CHUNK_SEQS)
            if s != 0:
                forward_rows(m, seqs, W, M["W"], np.flatnonzero(inch), EW, fp, workers, device, acc)
            rows = np.flatnonzero(inch & ~same); forward_rows(m, seqs, W, M["C"], rows, EC, fp, workers, device, acc); new += len(rows)
            done = int(np.flatnonzero(inch).max()) + 1
            print(f"seed {s} fold {f}: {done}/{n} windows; new C forwards {new}; GPU {acc['gpu_seconds']:.0f} s, decode {acc['decode_seconds']:.0f} s; device {device}", flush=True)
            write_json(prog, {"seed": s, "fold": f, "fold_windows_done": done, "fold_windows": n, "folds": len(FOLDS), "new_C_forwards_fold": new, "updated_at": now(), **acc})
        EC[same] = EW[same]
        np.save(fdir / f"eC_s{s}_f{f}.npy", EC); rec[str(f)] = {"windows": n, "C_reused_eW": int(same.sum()), "C_new_forwards": new, "C_sha256": sha256_file(fdir / f"eC_s{s}_f{f}.npy"), "positions_sha256": positions_sha(W, M)}
        if s != 0:
            np.save(fdir / f"eW_s{s}_f{f}.npy", EW); rec[str(f)]["W_sha256"] = sha256_file(fdir / f"eW_s{s}_f{f}.npy"); rec[str(f)]["W_new_forwards"] = n
        off += n
    if tet.state_digest(m) != d0:
        raise SystemExit("the encoder changed during extraction")
    if s == 0 and not np.array_equal(a37_eW(ctx).numpy(), src):
        raise SystemExit("A37's eW changed during extraction")
    rec.update({"seed": s, "key": feature_key(ctx, s), "a08": t31.A08[s], "runs_sha256": RUNS_SHA[s], "eW_source": "A37 runs/video_wider_temporal_evidence_v1/feats (record committed in f55c1ed)" if s == 0 else "this round",
                "time": now(), **acc}); write_json(fdir / f"record_s{s}.json", rec)
    return f"{sum(rec[str(f)]['C_new_forwards'] for f in FOLDS)} new C forwards of {total} windows"


def feats(ctx: Ctx, s: int, committed: bool = False) -> dict:
    """committed: the record must equal the one committed with the heads (weights_s{s}.json); the key must match in any case."""
    rec = json.loads((ctx.out / "feats" / f"record_s{s}.json").read_text())
    if committed and json.loads((ctx.results / f"weights_s{s}.json").read_text())["feature_record"] != rec:
        raise SystemExit("the feature record differs from the one committed with the heads")
    if rec["key"] != feature_key(ctx, s):
        raise SystemExit("the feature cache belongs to other feature code")
    out = {"q": [], "e224": [], "C": [], "offset": {}}; o = 0
    for f in FOLDS:
        O = t31.outputs(ctx.a31, s, f); out["q"].append(O["logits"]); out["e224"].append(O["emb"]); out["offset"][f] = o; o += len(O["logits"])
        if int(O["offset"]) != out["offset"][f]:
            raise SystemExit("fold offsets differ from A31's")
        for arm in ("C",) if s == 0 else ("C", "W"):
            p = ctx.out / "feats" / f"e{arm}_s{s}_f{f}.npy"
            if sha256_file(p) != rec[str(f)][f"{arm}_sha256"]:
                raise SystemExit("a feature file differs from its record")
            out.setdefault(arm, []).append(np.load(p))
    X = {k: (torch.from_numpy(np.concatenate(v)) if isinstance(v, list) else v) for k, v in out.items()}
    if s == 0:
        X["W"] = a37_eW(ctx)
    return X


# ----------------------------------------------------------------------------------------------- audit (index only)
def mapping_loop(seqs: list, W: dict, M: dict, bl: np.ndarray, P: dict) -> dict:
    """Independent per-window recomputation of the Base run, W, C and the marker; (a, b) must also be one of the Base P runs."""
    bad = 0; cases = Counter(); starts = np.cumsum([0, *[len(q["refs"]) for q in seqs]]); prun = set(zip(P["seq"].tolist(), P["a"].tolist(), P["b"].tolist()))
    for k in range(len(W["seq"])):
        i, c = int(W["seq"][k]), int(W["centre"][k]); n = len(seqs[i]["refs"]); base = bl[starts[i]:starts[i + 1]]; s = window_frames(c, n, 16, 16)
        a, b = c, c + 1
        while a > 0 and base[a - 1] == base[c]:
            a -= 1
        while b < n and base[b] == base[c]:
            b += 1
        w = [min(max(c + 2 * (int(x) - c), 0), n - 1) for x in s]; cl = [min(max(x, a), b - 1) for x in w]; mk = tm.marker_slot(s, c); limited = any(x < a or x > b - 1 for x in w)
        ok = np.array_equal(s, W["s"][k]) and w == M["W"][k].tolist() and cl == M["C"][k].tolist() and (a, b) == (int(M["a"][k]), int(M["b"][k])) and mk == int(W["marker"][k]) and cl[mk] == c
        ok &= all(x <= y for x, y in zip(cl, cl[1:])) and 0 <= a <= min(cl) and max(cl) <= b - 1 <= n - 1 and (i, a, b) in prun and (limited or cl == w); bad += not ok
        cases["limited"] += limited; cases["unlimited_with_C_ne_W"] += (not limited) and cl != w; cases["N_lt_16"] += n < 16; cases["run_lt_16"] += b - a < 16
        cases["centre_at_run_start"] += c == a; cases["centre_at_run_end"] += c == b - 1; cases["one_sample_run"] += b - a == 1
    return {"windows": int(len(W["seq"])), "mismatches": bad, "cases": {k: int(v) for k, v in cases.items()}}


def audit(ctx: Ctx, s: int) -> str:
    rec = {"seed": s, "folds": {}, "note": "index statistics only: positions, lengths and the frozen Base's predicted labels; no reference label, S1 / harm set or model output"}
    for f in FOLDS:
        seqs, W, M = fold_windows(ctx, s, f); bl = base_labels(ctx, s, f)
        with np.load(ctx.a31.out / "data" / f"runs_s{s}_f{f}.npz") as z:
            P = {k: z[f"P_{k}"] for k in ("seq", "a", "b")}
        loop = mapping_loop(seqs, W, M, bl, P); S, Wp, Cp = W["s"], M["W"], M["C"]; span = lambda x: x.max(1) - x.min(1)  # noqa: E731
        diff_pos = (Cp != Wp).any(1); same = prc.same_refs(seqs, W["seq"], Cp, Wp); recs_all = {r[0] for q in seqs for r in q["refs"]}
        wrefs = wt.window_refs(seqs, W["seq"], Wp); crefs = wt.window_refs(seqs, W["seq"], Cp); recs_lim = {r[0] for k in np.flatnonzero(diff_pos) for r in wrefs[k]}
        rl = (M["b"] - M["a"]).astype(float); q = lambda x: [float(v) for v in np.percentile(x, [5, 25, 50, 75, 95])]  # noqa: E731
        raw = W["centre"][:, None] + 2 * (S - W["centre"][:, None]); nlen = np.asarray([len(x["refs"]) for x in seqs])[W["seq"]][:, None]
        rec["folds"][str(f)] = {
            "requests": int(len(W["seq"])), "sequences": len(seqs), "native_sequences": int(sum(x["kind"] == "native" for x in seqs)), "recordings": len(recs_all), "sessions": len({session_of(v) for v in recs_all}),
            "mapping_loop": loop, "windows_C_ne_W_positions": int(diff_pos.sum()), "windows_C_ne_W_refs": int((~same).sum()), "windows_C_ne_W_share": float(diff_pos.mean()),
            "limited_windows_recordings": len(recs_lim), "limited_windows_sessions": len({session_of(v) for v in recs_lim}),
            "limited_windows_by_kind": {k: int(sum(diff_pos[W["seq"] == i].sum() for i, x in enumerate(seqs) if x["kind"] == k)) for k in ("multi", "native")},
            "W_slots_changed_by_run_limit_share": float((Cp != Wp).mean()), "W_slots_clamped_at_sequence_edge_share": float(((raw < 0) | (raw > nlen - 1)).mean()),
            "span_samples": {"S_mean": float(span(S).mean()), "W_mean": float(span(Wp).mean()), "C_mean": float(span(Cp).mean()), "C_quantiles_5_25_50_75_95": q(span(Cp)), "C_mean_on_limited_windows": float(span(Cp)[diff_pos].mean()) if diff_pos.any() else None,
                             "W_mean_on_limited_windows": float(span(Wp)[diff_pos].mean()) if diff_pos.any() else None, "C_span_below_S_span_share": float((span(Cp) < span(S)).mean())},
            "distinct_refs_per_window": {"S_mean": float(np.mean([len(set(r)) for r in wt.window_refs(seqs, W["seq"], S)])), "W_mean": float(np.mean([len(set(r)) for r in wrefs])), "C_mean": float(np.mean([len(set(r)) for r in crefs]))},
            "windows_with_repeated_positions": {"W": float(np.mean([len(set(r)) < 16 for r in Wp.tolist()])), "C": float(np.mean([len(set(r)) < 16 for r in Cp.tolist()]))},
            "repeated_slot_share": {"W": float(1 - np.mean([len(set(r)) for r in Wp.tolist()]) / 16), "C": float(1 - np.mean([len(set(r)) for r in Cp.tolist()]) / 16)},
            "base_run_length_at_centre_quantiles_5_25_50_75_95": q(rl), "C_equals_S_windows": int((Cp == S).all(1).sum())}
    rec["realised"] = all(v["windows_C_ne_W_positions"] > 0 for v in rec["folds"].values()); rec["ok"] = all(v["mapping_loop"]["mismatches"] == 0 and v["mapping_loop"]["cases"].get("unlimited_with_C_ne_W", 0) == 0 for v in rec["folds"].values())
    rec["finished"] = now(); publish(ctx, f"audit_s{s}.json", rec); print(json.dumps(rec, indent=1), flush=True)
    if not rec["ok"]:
        raise SystemExit("the independent mapping loop disagrees")
    if not rec["realised"]:
        raise SystemExit("C equals W in every window: the input condition is not realised")
    return f"C != W in {rec['folds']['0']['windows_C_ne_W_positions']} / {rec['folds']['1']['windows_C_ne_W_positions']} windows"


# ----------------------------------------------------------------------------------------------- check
def loaded_code() -> dict:
    """sha256 of every loaded repository module file (scripts / echo_routing / third_party), to compare with the committed code."""
    out = {}
    for m in list(sys.modules.values()):
        p = Path(getattr(m, "__file__", None) or "").resolve()
        if p.is_file() and p.is_relative_to(ROOT) and p.relative_to(ROOT).parts[0] in ("scripts", "echo_routing", "third_party"):
            out[str(p.relative_to(ROOT))] = sha256_file(p)
    return out


def check(ctx: Ctx, device, workers: int) -> str:
    s = 0; au = json.loads((ctx.out / f"audit_s{s}.json").read_text())
    if not (au["ok"] and au["realised"]):
        raise SystemExit("the mapping audit has not passed")
    res = {"config_sha256": sha256_file(ROOT / ctx.config), "tolerances": TOL}; a08 = ctx.a31.a08(s, device).eval(); d0 = tet.state_digest(a08)
    seqs, W, M = fold_windows(ctx, s, 0); fp = t28.frame_paths(seqs); O = t31.outputs(ctx.a31, s, 0); eW = a37_eW(ctx); n0 = len(W["seq"]); same = prc.same_refs(seqs, W["seq"], M["C"], M["W"])
    g_ = np.random.default_rng([20261026, 38, 1]); pick = np.sort(g_.choice(np.flatnonzero(~same), 32, replace=False)); keep = np.sort(g_.choice(np.flatnonzero(same), 16, replace=False))
    sub = lambda ks: {"seq": W["seq"][ks]}  # noqa: E731
    fr_s, idx_s = t37.chunk_frames(seqs, sub(pick), W["s"][pick], fp, 0); e_s, z_s = t37.embed(a08, fr_s, idx_s, W["marker"][pick], device)
    fr_c, idx_c = t37.chunk_frames(seqs, sub(pick), M["C"][pick], fp, 0); e_c, _ = t37.embed(a08, fr_c, idx_c, W["marker"][pick], device)
    stack = np.stack([np.stack([jv.load_frame_u8(fp(v)[min(int(fi), len(fp(v)) - 1)], round(float(g), 6)) for v, fi, g in row]) for row in wt.window_refs(seqs, W["seq"][pick], M["C"][pick])])
    with torch.no_grad():
        e_x = a08.embed(tm.to_rgb(torch.from_numpy(stack).to(device)), tm.marker_planes(torch.from_numpy(W["marker"][pick]).long().to(device))).float()
    fr_k, idx_k = t37.chunk_frames(seqs, sub(keep), M["C"][keep], fp, 0); e_k, _ = t37.embed(a08, fr_k, idx_k, W["marker"][keep], device)
    res["features"] = {"S_online_vs_cache_logits_max_abs": float(np.abs(z_s.cpu().numpy() - O["logits"][pick]).max()), "S_online_vs_cache_emb_max_abs": float(np.abs(e_s.cpu().numpy() - O["emb"][pick]).max()),
                       "C_explicit_stack_input_equal": bool(np.array_equal(stack, fr_c[idx_c])), "C_explicit_stack_emb_max_abs": float((e_c - e_x).abs().max()),
                       "C_eq_W_online_vs_reused_eW_max_abs": float(np.abs(e_k.cpu().numpy() - eW[:n0][keep].numpy()).max()), "C_vs_W_emb_max_abs_on_limited_windows": float(np.abs(e_c.cpu().numpy() - eW[:n0][pick].numpy()).max()),
                       "picked": {"limited": int(len(pick)), "unlimited": int(len(keep))}, "tf32": {"matmul": torch.backends.cuda.matmul.allow_tf32, "cudnn": torch.backends.cudnn.allow_tf32}}
    res["weights_unchanged"] = tet.state_digest(a08) == d0
    # Init = Base, forced eC = eW, gradients, binding, resume (eC placeholder = eW: no new features needed)
    X, offs = {"q": [], "e224": []}, {}
    for f in FOLDS:
        Of = t31.outputs(ctx.a31, s, f); X["q"].append(Of["logits"]); X["e224"].append(Of["emb"]); offs[f] = int(Of["offset"])
    X = {k: torch.from_numpy(np.concatenate(v)) for k, v in X.items()}; X["W"] = eW; X["C"] = eW.clone(); a8 = rsh.a08_ctx(); data = t31.vpr.training_data(ctx.a31.a15.a10.a07p); init = {}
    old = json.loads((ROOT / "docs/results/2026-10-25_a5c_source_external_evidence/readout.json").read_text())["sequences"]
    for f in FOLDS:
        g = prc.head(f); D = d32.fold_data(ctx.d, s, f); pk = D["pk"]; off = offs[f]; Wf = t31.pool_W(D["seqs"])
        with torch.no_grad():
            z = g(X["q"], X["e224"], X["C"])
        pp = t29.pooled_probs(z, pk, np.arange(len(pk["labels"]))); ps = rr.per_sample(Wf, z[off:off + len(Wf["seq"])].numpy(), D["seqs"]); preds = {q["id"]: ir.relabel(D["base_preds"][q["id"]], ps[q["id"]]) for q in D["seqs"]}
        P = {"seqs": D["seqs"], "n_of": dict(data.n), "label_of": dict(data.label)}; sr = t31.seq_rows(a8.cfg, P, preds, bc.evalset(P, list(range(len(D["seqs"]))), f"heldout_f{f}")); ob = old[f"s{s}_heldout_f{f}"]["A08 Base"]
        init[str(f)] = {"z_equals_q": bool(torch.equal(z, X["q"])), "runs_equal_base": bool(np.array_equal(pp.argmax(1), pk["base"])), "sequences_equal_base": all(np.array_equal(preds[q["id"]], D["base_preds"][q["id"]]) for q in D["seqs"]),
                        "rows_equal_A32_base": bool(sr["row"] == ob["sequence_row"] and rsh.counts_of(D["seqs"], sr) == ob["counts"]), "P_runs": int(len(pk["labels"]))}
    res["init_equals_base"] = init; R = t31.runs(ctx.a31, s, 0); torch.set_num_threads(1)
    gw = prc.head(0); ow = t31.opt_of([("g", list(gw.parameters()), 1e-3)]); gc = prc.head(0); oc = t31.opt_of([("g", list(gc.parameters()), 1e-3)])
    res["identical_init"] = all(torch.equal(a, b) for a, b in zip(gw.state_dict().values(), gc.state_dict().values())) and all(torch.equal(a, b) for a, b in zip(gw.state_dict().values(), wt.head(0).state_dict().values()))
    lw = [step(gw, ow, X, "W", R, u) for u in range(3)]; lc = [step(gc, oc, X, "C", R, u) for u in range(3)]
    res["forced_equal"] = {"loss_max_abs": max(abs(a["loss"] - b["loss"]) for a, b in zip(lw, lc)), "params_max_abs": float(max((a - b).abs().max() for a, b in zip(gw.state_dict().values(), gc.state_dict().values())))}
    g1 = prc.head(0); o1 = t31.opt_of([("g", list(g1.parameters()), 1e-3)]); t31.sched(o1, 0, {"updates": U, "warmup": WARM}); rows, w, ix, y = t29.batch_rows(R, R["bt"][0], R["bu"][0])
    t29.pooled_ce(g1(X["q"][rows], X["e224"][rows], X["C"][rows]), w, ix, y).mean().backward(); res["gradients"] = {"fc1_grad_abs_sum_step1": float(g1.fc1.weight.grad.abs().sum()), "fc2_grad_abs_sum_step1": float(g1.fc2.weight.grad.abs().sum())}
    res["optimizer_binds_exactly"] = sorted(id(p) for grp in oc.param_groups for p in grp["params"]) == sorted(id(p) for p in gc.parameters())
    ga = prc.head(0); oa = t31.opt_of([("g", list(ga.parameters()), 1e-3)]); [step(ga, oa, X, "C", R, u) for u in range(6)]
    gb = prc.head(0); ob_ = t31.opt_of([("g", list(gb.parameters()), 1e-3)]); [step(gb, ob_, X, "C", R, u) for u in range(3)]; sv = ({k: v.clone() for k, v in gb.state_dict().items()}, ob_.state_dict())
    gr = prc.head(0); orr = t31.opt_of([("g", list(gr.parameters()), 1e-3)]); gr.load_state_dict(sv[0]); orr.load_state_dict(sv[1]); [step(gr, orr, X, "C", R, u) for u in range(3, 6)]
    res["resume_3_plus_3_vs_6_max_abs"] = float(max((a - b).abs().max() for a, b in zip(ga.state_dict().values(), gr.state_dict().values())))
    sp = json.loads((ctx.a31.out / "split.json").read_text()); res["isolation"] = sp["isolation"]
    res["sources_unchanged"] = {"A08_digest": res["weights_unchanged"], "A31_outputs_and_runs": "verified by sha on load (outputs / runs files)", "A37_eW_record_and_key": "verified by A37's feats(committed=True)",
                                "A37_W_heads": all(sha256_file(ctx.a37.out / "train" / f"W_s0_f{f}" / "final.pt") == A37_W[f]["final"] for f in FOLDS)}
    # timing and disk: new C forwards on the first sequences of fold 0
    ch = np.flatnonzero((W["seq"] < 40) & ~same); tmp = np.zeros((n0, 512), np.float32); acc = {"gpu_seconds": 0.0, "decode_seconds": 0.0}
    forward_rows(a08, seqs, W, M["C"], ch, tmp, fp, workers, device, acc); per = (acc["gpu_seconds"] + acc["decode_seconds"]) / max(len(ch), 1)
    nw = sum(v["requests"] for v in au["folds"].values()); nc = sum(v["windows_C_ne_W_refs"] for v in au["folds"].values())
    res["timing"] = {"windows": int(len(ch)), **acc, "seconds_per_window": per, "projected_hours_seed0": nc * per / 3600 * 1.15, "projected_hours_seed2_upper": (nw + nc) * per / 3600 * 1.3, "peak_gpu_gb": torch.cuda.max_memory_allocated() / 1e9}
    res["disk"] = {"projected_feature_gb_seed0": nw * 512 * 4 / 1e9, "projected_feature_gb_seed2": 2 * nw * 512 * 4 / 1e9, "free_gb": shutil.disk_usage(ctx.out).free / 1e9}
    res["loaded_code_sha256"] = loaded_code(); F_ = res["features"]
    res["ok"] = bool(F_["S_online_vs_cache_logits_max_abs"] <= TOL["parity"] and F_["S_online_vs_cache_emb_max_abs"] <= TOL["parity"] and F_["C_explicit_stack_input_equal"] and F_["C_explicit_stack_emb_max_abs"] <= TOL["explicit"]
                     and F_["C_eq_W_online_vs_reused_eW_max_abs"] <= TOL["reuse"] and F_["C_vs_W_emb_max_abs_on_limited_windows"] > 0 and not F_["tf32"]["matmul"] and not F_["tf32"]["cudnn"] and res["weights_unchanged"]
                     and all(all(v[k] for k in ("z_equals_q", "runs_equal_base", "sequences_equal_base", "rows_equal_A32_base")) for v in init.values()) and res["identical_init"]
                     and res["forced_equal"]["loss_max_abs"] == 0 and res["forced_equal"]["params_max_abs"] == 0 and res["gradients"]["fc1_grad_abs_sum_step1"] == 0 and res["gradients"]["fc2_grad_abs_sum_step1"] > 0
                     and res["optimizer_binds_exactly"] and res["resume_3_plus_3_vs_6_max_abs"] == 0 and not any(sp["isolation"].values()) and res["sources_unchanged"]["A37_W_heads"]
                     and res["timing"]["projected_hours_seed0"] + res["timing"]["projected_hours_seed2_upper"] <= 1.5 and res["disk"]["projected_feature_gb_seed0"] + res["disk"]["projected_feature_gb_seed2"] < 1.0 and res["disk"]["free_gb"] > 10)
    res["finished"] = now(); publish(ctx, "checks.json", res); print(json.dumps({k: v for k, v in res.items() if k != "loaded_code_sha256"}, indent=1, default=str)[:6000], flush=True)
    if not res["ok"]:
        raise SystemExit("a check failed")
    return f"{per * 1e3:.1f} ms per new window, {res['timing']['projected_hours_seed0']:.2f} h seed 0"


# ----------------------------------------------------------------------------------------------- train
def train(ctx: Ctx, s: int) -> None:
    torch.set_num_threads(1); X = feats(ctx, s)
    for f in FOLDS:
        R = t31.runs(ctx.a31, s, f)
        for arm in ARMS:
            t0, started, ok, note = time.time(), now(), False, ""
            try:
                rd = ctx.out / "train" / f"{arm}_s{s}_f{f}"; rd.mkdir(parents=True, exist_ok=True); g = prc.head(f); o = t31.opt_of([("g", list(g.parameters()), 1e-3)]); before = fit_monitor(g, X, arm, R); log = []
                for u in range(U):
                    log.append(step(g, o, X, arm, R, u))
                    if (u + 1) % 270 == 0:
                        print(f"[{arm}_s{s}_f{f}] update {u + 1}/{U}: loss {np.mean([x['loss'] for x in log[-270:]]):.4f}  {time.time() - t0:.1f} s", flush=True)
                after = fit_monitor(g, X, arm, R); torch.save({k: v.detach().clone() for k, v in g.state_dict().items()}, rd / "final.pt"); (rd / "updates.jsonl").write_text("".join(json.dumps(x) + "\n" for x in log))
                man = {"arm": arm, "seed": s, "fit_fold": f, "updates": len(log), "final_sha256": sha256_file(rd / "final.pt"), "updates_sha256": sha256_file(rd / "updates.jsonl"), "fit_before": before, "fit_after": after, "seconds": time.time() - t0}
                if arm == "W" and s == 0:
                    old = torch.load(ctx.a37.out / "train" / f"W_s0_f{f}" / "final.pt", map_location="cpu", weights_only=True)
                    man["reproduces_A37"] = {"final_sha256": man["final_sha256"] == A37_W[f]["final"], "updates_sha256": man["updates_sha256"] == A37_W[f]["updates"], "tensors_equal": all(torch.equal(old[k], v) for k, v in g.state_dict().items())}
                    if not all(man["reproduces_A37"].values()):
                        write_json(rd / "run_manifest.json", man); raise SystemExit(f"W_s0_f{f} does not reproduce A37's head: {man['reproduces_A37']}")
                write_json(rd / "run_manifest.json", man); print(json.dumps({k: man[k] for k in ("arm", "seed", "fit_fold", "updates", "final_sha256")} | ({"reproduces_A37": man["reproduces_A37"]} if "reproduces_A37" in man else {})), flush=True); ok = True
            except BaseException as e:  # noqa: BLE001 - recorded, then raised again
                note = f"{type(e).__name__}: {str(e)[:200]}"; raise
            finally:
                ledger_append(ctx.ledger, f"train {arm}_s{s}_f{f} (CPU)", "cpu", started, time.time() - t0, ok, note)
    man = {f"{arm}_s{s}_f{f}": json.loads((ctx.out / "train" / f"{arm}_s{s}_f{f}" / "run_manifest.json").read_text()) for f in FOLDS for arm in ARMS}
    publish(ctx, f"weights_s{s}.json", {"seed": s, "heads": man, "feature_record": json.loads((ctx.out / "feats" / f"record_s{s}.json").read_text()), "config_sha256": sha256_file(ROOT / ctx.config), "written": now()})


def freeze(ctx: Ctx, s: int, commit: str, manifest_sha: str) -> str:
    """After weights_s{s}.json is committed locally (sha manifest_sha in commit), pin the four head sha256 for the readout."""
    p = ctx.results / f"weights_s{s}.json"
    if sha256_file(p) != manifest_sha or sha256_file(ctx.out / f"weights_s{s}.json") != manifest_sha:
        raise SystemExit("the committed weight manifest differs from the server's")
    heads = json.loads(p.read_text())["heads"]; com = ctx.out / "weights_committed.json"; cur = json.loads(com.read_text()) if com.exists() else {}
    for k, man in heads.items():
        if sha256_file(ctx.out / "train" / k / "final.pt") != man["final_sha256"]:
            raise SystemExit(f"{k}: weights differ from the manifest")
        cur[k] = man["final_sha256"]
    cur[f"commit_s{s}"] = commit; write_json(com, cur); return f"{len(heads)} heads pinned at {commit}"


# ----------------------------------------------------------------------------------------------- readout
def load_head(ctx: Ctx, arm: str, s: int, f: int):
    rd = ctx.out / "train" / f"{arm}_s{s}_f{f}"; man = json.loads((rd / "run_manifest.json").read_text()); com = json.loads((ctx.out / "weights_committed.json").read_text())
    if sha256_file(rd / "final.pt") != man["final_sha256"] or com.get(f"{arm}_s{s}_f{f}") != man["final_sha256"]:
        raise SystemExit(f"{arm}_s{s}_f{f}: weights not committed or changed")
    g = prc.head(f); g.load_state_dict(torch.load(rd / "final.pt", map_location="cpu", weights_only=True)); return g.eval()


def boundaries(z: torch.Tensor, D: dict, off: int) -> dict:
    """Base run boundaries inside the held-out sequences, and those the relabel + merge keeps (the initial cuts never move)."""
    W = t31.pool_W(D["seqs"]); ps = rr.per_sample(W, z[off:off + len(W["seq"])].numpy(), D["seqs"]); nb = nf = 0
    for q in D["seqs"]:
        b = D["base_preds"][q["id"]]; p = ir.relabel(b, ps[q["id"]]); nb += int((np.diff(b) != 0).sum()); nf += int((np.diff(p) != 0).sum())
    return {"base_boundaries": nb, "final_boundaries": nf, "deleted_by_merge": nb - nf}


def readout(ctx: Ctx, s: int) -> str:
    name = f"readout_s{s}.json"
    if (ctx.out / name).exists():
        raise SystemExit("this seed's readout exists; it runs once")
    X = feats(ctx, s, committed=True); a8 = rsh.a08_ctx(); data = t31.vpr.training_data(ctx.a31.a15.a10.a07p); rec = {"seed": s, "cells": {}}; acc = {m: [] for m in ("Base", *ARMS)}; ys, bs = [], []; cov = {a: [] for a in ARMS}
    a37 = json.loads((ROOT / "docs/results/2026-10-26_wider_temporal_evidence/readout_s0.json").read_text())["cells"] if s == 0 else None
    for h in FOLDS:
        f = 1 - h; D = d32.fold_data(ctx.d, s, h); D["M"] = d32.set_masks(D); off = X["offset"][h]; P = {"seqs": D["seqs"], "n_of": dict(data.n), "label_of": dict(data.label)}; ev = bc.evalset(P, list(range(len(D["seqs"]))), f"heldout_f{h}")
        cell = {}
        with torch.no_grad():
            for arm in ARMS:
                z = load_head(ctx, arm, s, f)(X["q"], X["e224"], X[arm]); t = tables(z, D, off, a8.cfg, P, ev); p = t.pop("pred"); acc[arm].append(p); cell[arm] = t
                cell[arm]["coverage"], sets = t37.coverage(D, D["pk"]["labels"], D["pk"]["base"], p); cov[arm].append(sets); cell[arm]["boundaries"] = boundaries(z, D, off)
            t = tables(X["q"], D, off, a8.cfg, P, ev); acc["Base"].append(t.pop("pred")); cell["Base"] = t; cell["Base"]["boundaries"] = boundaries(X["q"], D, off)
        mans = {arm: json.loads((ctx.out / "train" / f"{arm}_s{s}_f{f}" / "run_manifest.json").read_text()) for arm in ARMS}; cell["fit"] = {arm: {k: mans[arm][k] for k in ("fit_before", "fit_after")} for arm in ARMS}
        row = lambda m_: cell[m_]["sequence_row"]  # noqa: E731
        cell["comparisons"] = {f"{a} - {bb}": {"net_fixed": cell[a]["runs"]["all"]["net_fixed"] - (cell[bb]["runs"]["all"]["net_fixed"] if bb != "Base" else 0), **{k: row(a)[k] - row(bb)[k] for k in ("segment_f1_50_multi", "native_macro_f1", "boundary_recall_multi", "boundary_precision_multi")},
                                                 **{k: cell[a]["counts"][k] - cell[bb]["counts"][k] for k in ("fp", "fp_short", "fn", "fn_short")}} for a, bb in (("C", "W"), ("C", "Base"), ("W", "Base"))}
        if a37 is not None:
            old, js = a37[f"heldout_f{h}"], lambda x: json.loads(json.dumps(x, default=float))  # noqa: E731
            cell["A37_reproduced"] = {m_: all(js(cell[m_][k]) == old[m_][k] for k in old[m_]) for m_ in ("Base", "W")}
        rec["cells"][f"heldout_f{h}"] = cell; ys.append(D["pk"]["labels"]); bs.append(D["pk"]["base"])
    y, b = np.concatenate(ys), np.concatenate(bs); wr = b != y; rec["oof"] = {}
    for m_ in ARMS:
        p = np.concatenate(acc[m_]); fx, br = int((wr & (p == y)).sum()), int((~wr & (p != y)).sum()); un = {k: set().union(*[c[k] for c in cov[m_]]) for k in ("fixed", "broken")}
        rec["oof"][m_] = {"runs": int(len(y)), "fixed": fx, "broken": br, "net_fixed": fx - br, **{f"{k}_recordings": len(v) for k, v in un.items()}, **{f"{k}_sessions": len({session_of(x) for x in v}) for k, v in un.items()}}
    c = rec["cells"]; cf = lambda m_, k: [c[f"heldout_f{h}"][m_]["sequence_row"][k] for h in FOLDS]  # noqa: E731
    dF = lambda a, bb, h: c[f"heldout_f{h}"][a]["sequence_row"]["segment_f1_50_multi"] - c[f"heldout_f{h}"][bb]["sequence_row"]["segment_f1_50_multi"]  # noqa: E731
    net = lambda a, h: c[f"heldout_f{h}"][a]["runs"]["all"]["net_fixed"]  # noqa: E731
    s1 = sum(c[f"heldout_f{h}"]["C"]["S1_fixed"]["fixed"] for h in FOLDS); n1 = sum(c[f"heldout_f{h}"]["C"]["S1_fixed"]["runs"] for h in FOLDS)
    k5 = sum(c[f"heldout_f{h}"]["C"]["correct_A5C_kept"]["kept"] for h in FOLDS); n5 = sum(c[f"heldout_f{h}"]["C"]["correct_A5C_kept"]["runs"] for h in FOLDS)
    rule = {"matched_net_each_fold_ge_5": all(net("C", h) - net("W", h) >= 5 for h in FOLDS), "matched_dF_each_fold_ge_0": all(dF("C", "W", h) >= 0 for h in FOLDS), "matched_dF_mean_ge_0.003": float(np.mean([dF("C", "W", h) for h in FOLDS])) >= 0.003,
            "useful_net_each_fold_ge_0": all(net("C", h) >= 0 for h in FOLDS), "useful_dF_each_fold_ge_0": all(dF("C", "Base", h) >= 0 for h in FOLDS), "useful_dF_mean_ge_0.003": float(np.mean([dF("C", "Base", h) for h in FOLDS])) >= 0.003,
            "S1_recovered": f"{s1}/{n1}", "S1_needed": math.ceil(0.10 * n1), "S1_ok": s1 >= math.ceil(0.10 * n1), "correct_A5C_kept": f"{k5}/{n5}", "A5C_needed": math.ceil(0.98 * n5), "A5C_ok": k5 >= math.ceil(0.98 * n5),
            "native_each_fold_ge_-0.003": all(c[f"heldout_f{h}"]["C"]["sequence_row"]["native_macro_f1"] - c[f"heldout_f{h}"]["Base"]["sequence_row"]["native_macro_f1"] >= -0.003 for h in FOLDS),
            "missed_short_le_base_plus_3": sum(c[f"heldout_f{h}"]["C"]["counts"]["fn_short"] for h in FOLDS) <= sum(c[f"heldout_f{h}"]["Base"]["counts"]["fn_short"] for h in FOLDS) + 3,
            "boundary_recall_each_fold_ge_-0.005": all(c[f"heldout_f{h}"]["C"]["sequence_row"]["boundary_recall_multi"] - c[f"heldout_f{h}"]["Base"]["sequence_row"]["boundary_recall_multi"] >= -0.005 for h in FOLDS)}
    rule["met"] = all(v for k, v in rule.items() if isinstance(v, bool)); rec["seed2_rule"] = rule
    rec["values"] = {"C_minus_W_net": [net("C", h) - net("W", h) for h in FOLDS], "C_minus_W_dF": [dF("C", "W", h) for h in FOLDS], "C_minus_Base_net": [net("C", h) for h in FOLDS], "C_minus_Base_dF": [dF("C", "Base", h) for h in FOLDS],
                     "C_minus_Base_native": [a - bb for a, bb in zip(cf("C", "native_macro_f1"), cf("Base", "native_macro_f1"))], "C_minus_Base_boundary_recall": [a - bb for a, bb in zip(cf("C", "boundary_recall_multi"), cf("Base", "boundary_recall_multi"))],
                     "missed_short": {m_: sum(c[f"heldout_f{h}"][m_]["counts"]["fn_short"] for h in FOLDS) for m_ in ("Base", *ARMS)}}
    rec["summary"] = {f"heldout_f{h}": {m_: {"net": net(m_, h) if m_ != "Base" else 0, "F1": c[f"heldout_f{h}"][m_]["sequence_row"]["segment_f1_50_multi"]} for m_ in ("Base", *ARMS)} for h in FOLDS}
    fr = json.loads((ctx.out / "feats" / f"record_s{s}.json").read_text()); pr = json.loads((ctx.a31.out / "cache" / f"prefix_s{s}.json").read_text()); nn = sum(fr[str(f)]["C_new_forwards"] + fr[str(f)].get("W_new_forwards", 0) for f in FOLDS)
    rec["deployment_cost"] = {"forwards_C": "Base 224 forward + one auxiliary 224 forward on C for every window (same weights); the cache reuse of eW is a training-time saving only", "parameters": "A08 34.6M shared + 32,965 head",
                              "gpu_ms_per_window_224_A31_pass": 1e3 * pr["gpu_seconds"] / sum(pr["rows"].values()), "gpu_ms_per_new_forward_this_round": 1e3 * fr["gpu_seconds"] / max(nn, 1), "cpu_decode_ms_per_new_forward": 1e3 * fr["decode_seconds"] / max(nn, 1)}
    publish(ctx, name, rec); print(json.dumps({"summary": rec["summary"], "oof": rec["oof"], "rule": rule, "values": rec["values"], "A37_reproduced": {h: rec["cells"][h].get("A37_reproduced") for h in rec["cells"]}}, indent=1)); return f"rule met: {rule['met']}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True); ap.add_argument("--stage", choices=("audit", "check", "extract", "train", "freeze", "readout"), required=True); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6); ap.add_argument("--commit", default=""); ap.add_argument("--manifest-sha", default="")
    a = ap.parse_args(); a.set = []; ctx = Ctx(load_cfg(a), a.config); ctx.out.mkdir(parents=True, exist_ok=True)
    if a.stage == "train":
        return train(ctx, a.seed)
    if a.stage in ("audit", "readout", "freeze"):
        fn = {"audit": lambda: audit(ctx, a.seed), "readout": lambda: readout(ctx, a.seed), "freeze": lambda: freeze(ctx, a.seed, a.commit, a.manifest_sha)}[a.stage]
        return staged(ctx, f"{a.stage} s{a.seed} (CPU)", "cpu", fn)
    t31.vsh.determinism(); device = torch.device("cuda")
    staged(ctx, f"{a.stage}{' s' + str(a.seed) if a.stage == 'extract' else ''} (GPU job wall time)", "gpu", {"check": lambda: check(ctx, device, a.workers), "extract": lambda: extract(ctx, a.seed, device, a.workers)}[a.stage])


if __name__ == "__main__":
    main()
