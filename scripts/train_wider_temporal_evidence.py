#!/usr/bin/env python
"""A37 (configs/experiments/video_wider_temporal_evidence_v1.yaml): wider temporal evidence recognition screen.

  --stage audit    index statistics of the W windows against S (no labels, no model) (CPU)
  --stage check    mapping, parity, explicit-stack, forced W = S, Init = Base, head mechanics, timing and disk (GPU)
  --stage extract  --seed s: frozen A08 512-d eW of every request window of both pools (GPU)
  --stage train    --seed s: residual heads S_f0, W_f0, S_f1, W_f1 on A31's uniform plans (CPU, single thread)
  --stage readout  --seed s: after the heads are committed, one held-out readout (+ the source-2 rule for seed 0) (CPU)
The A08 + B3 path supplies q, e224 (= the S arm's auxiliary input), Base and the runs; nothing here changes it.
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
from scripts._common import load_cfg  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import echo_routing.temporal.interval_recognition as ir  # noqa: E402
import echo_routing.temporal.joint_visual as jv  # noqa: E402
import echo_routing.temporal.wider_temporal as wt  # noqa: E402
import echo_routing.vit.target_mvit as tm  # noqa: E402
from echo_routing.evaluate.complementarity import sha256_file  # noqa: E402
from echo_routing.evaluate.failure_units import session_of  # noqa: E402
from echo_routing.evaluate.window_target import window_frames  # noqa: E402

A31_CONFIG, A32_CONFIG = t36.A31_CONFIG, t36.A32_CONFIG
FOLDS, ARMS, U, WARM, CHUNK_SEQS, BATCH = (0, 1), ("S", "W"), 1350, 135, 150, 32
TOL = {"parity": 1e-4, "explicit": 1e-5, "forced_w_eq_s": 1e-4, "batch_split": 1e-4}
now, write_json, ledger_append, step, fit_monitor, tables = t31.now, t31.write_json, t31.ledger_append, t36.step, t36.fit_monitor, t36.tables


class Ctx:
    def __init__(self, cfg: dict, config: str) -> None:
        self.config, self.v = config, cfg["wider_temporal_evidence"]; o = self.v["outputs"]
        self.out, self.results, self.ledger = ROOT / o["root"], ROOT / o["results_dir"], ROOT / self.v["budget"]["ledger"]
        self.a31 = t31.Ctx(load_cfg(SimpleNamespace(config=A31_CONFIG, set=[])), A31_CONFIG); self.d = d32.Ctx(load_cfg(SimpleNamespace(config=A32_CONFIG, set=[])), A32_CONFIG)


def staged(ctx: Ctx, job: str, kind: str, fn) -> None:
    t36.staged(ctx, job, kind, fn)


def publish(ctx: Ctx, name: str, rec: dict) -> None:
    t36.publish(ctx, name, rec)


# ----------------------------------------------------------------------------------------------- windows and features
def fold_windows(ctx: Ctx, f: int, factor: int = wt.FACTOR) -> tuple[list, dict, np.ndarray]:
    seqs = t31.load_pool(ctx.a31, f); W = t31.pool_W(seqs); return seqs, W, wt.pool_positions(seqs, W, factor)


def chunk_frames(ch: list, Wc: dict, pos: np.ndarray, fp, workers: int) -> tuple[np.ndarray, np.ndarray]:
    """Decoded 224 frames of the refs the chunk's windows use, and each window's 16 indices into them."""
    wrefs = wt.window_refs(ch, Wc["seq"], pos); uniq = sorted({r for row in wrefs for r in row}); at = {r: i for i, r in enumerate(uniq)}
    return rr.decode_frames(uniq, fp, workers), np.asarray([[at[r] for r in row] for row in wrefs], dtype=np.int64)


def embed(m, frames: np.ndarray, idx: np.ndarray, marker: np.ndarray, device) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        e = m.embed(tm.to_rgb(torch.from_numpy(frames[idx]).to(device)), tm.marker_planes(torch.from_numpy(marker).long().to(device))); return e.float(), m.classifier(e).float()


def feature_key(ctx: Ctx, s: int) -> str:
    """Only what the features depend on: mapping / frame / model code, the extraction functions, weights, pools, batching."""
    h = hashlib.sha256()
    for p in (Path(wt.__file__), Path(tm.__file__), Path(jv.__file__)):
        h.update(p.read_bytes())
    for fn in (fold_windows, chunk_frames, embed, extract, rr.decode_frames, rr.RefFrames, rr.half_windows, t31.pool_W, window_frames):
        h.update(inspect.getsource(fn).encode())
    sp = json.loads((ctx.a31.out / "split.json").read_text()); h.update(json.dumps({"a08": t31.A08[s], "pools": sp["pool_sha256"], "chunk": CHUNK_SEQS, "batch": BATCH}, sort_keys=True).encode())
    return h.hexdigest()


def extract(ctx: Ctx, s: int, device, workers: int) -> str:
    m = ctx.a31.a08(s, device).eval(); d0 = tet.state_digest(m); fdir = ctx.out / "feats"; fdir.mkdir(parents=True, exist_ok=True); acc = {"gpu_seconds": 0.0, "decode_seconds": 0.0}; rec = {}
    for f in FOLDS:
        seqs, W, pos = fold_windows(ctx, f); n = len(W["seq"]); fp = t28.frame_paths(seqs); E = np.zeros((n, 512), np.float32); row = 0
        for c0 in range(0, len(seqs), CHUNK_SEQS):
            ch = seqs[c0:c0 + CHUNK_SEQS]; Wc = t31.pool_W(ch); pc = wt.pool_positions(ch, Wc); a = time.time(); fr, idx = chunk_frames(ch, Wc, pc, fp, workers); acc["decode_seconds"] += time.time() - a
            if not (np.array_equal(Wc["seq"] + c0, W["seq"][row:row + len(Wc["seq"])]) and np.array_equal(Wc["centre"], W["centre"][row:row + len(Wc["seq"])]) and np.array_equal(pc, pos[row:row + len(Wc["seq"])])):
                raise SystemExit("chunk windows differ from the pool's")
            for b in range(0, len(Wc["seq"]), BATCH):
                ks = np.arange(b, min(b + BATCH, len(Wc["seq"]))); torch.cuda.synchronize(); a = time.time(); e, _ = embed(m, fr, idx[ks], Wc["marker"][ks], device); torch.cuda.synchronize(); acc["gpu_seconds"] += time.time() - a
                E[row + ks] = e.cpu().numpy()
            row += len(Wc["seq"]); del fr
            print(f"seed {s} fold {f}: {row}/{n} W windows; GPU {acc['gpu_seconds']:.0f} s, decode {acc['decode_seconds']:.0f} s; device {device}", flush=True)
        np.save(fdir / f"eW_s{s}_f{f}.npy", E)
        rec[str(f)] = {"windows": n, "W_sha256": sha256_file(fdir / f"eW_s{s}_f{f}.npy"), "positions_sha256": hashlib.sha256(W["seq"].tobytes() + W["centre"].tobytes() + pos.tobytes()).hexdigest()}
    if tet.state_digest(m) != d0:
        raise SystemExit("the encoder changed during extraction")
    rec.update({"seed": s, "key": feature_key(ctx, s), "a08": t31.A08[s], "factor": wt.FACTOR, "time": now(), **acc}); write_json(fdir / f"record_s{s}.json", rec); return f"{sum(rec[str(f)]['windows'] for f in FOLDS)} W windows"


def feats(ctx: Ctx, s: int, committed: bool = False) -> dict:
    """committed: the record must equal the one committed with the heads (weights_s{s}.json); else it must match the key."""
    rec = json.loads((ctx.out / "feats" / f"record_s{s}.json").read_text())
    if committed and json.loads((ctx.results / f"weights_s{s}.json").read_text())["feature_record"] != rec:
        raise SystemExit("the feature record differs from the one committed with the heads")
    if rec["key"] != feature_key(ctx, s):
        raise SystemExit("the feature cache belongs to other feature code")
    out = {"q": [], "e224": [], "W": [], "offset": {}}; o = 0
    for f in FOLDS:
        O = t31.outputs(ctx.a31, s, f); out["q"].append(O["logits"]); out["e224"].append(O["emb"]); out["offset"][f] = o; o += len(O["logits"])
        if int(O["offset"]) != out["offset"][f]:
            raise SystemExit("fold offsets differ from A31's")
        p = ctx.out / "feats" / f"eW_s{s}_f{f}.npy"
        if sha256_file(p) != rec[str(f)]["W_sha256"]:
            raise SystemExit("a feature file differs from its record")
        out["W"].append(np.load(p))
    X = {k: (torch.from_numpy(np.concatenate(v)) if isinstance(v, list) else v) for k, v in out.items()}; X["S"] = X["e224"]; return X


# ----------------------------------------------------------------------------------------------- audit (index only)
def audit(ctx: Ctx) -> str:
    rec = {"folds": {}}
    for f in FOLDS:
        seqs, W, pos = fold_windows(ctx, f); n = np.asarray([len(q["refs"]) for q in seqs])[W["seq"]]; S = W["s"]
        span_s, span_w = S.max(1) - S.min(1), pos.max(1) - pos.min(1); raw = W["centre"][:, None] + 2 * (S - W["centre"][:, None])
        sref, wref = wt.window_refs(seqs, W["seq"], S), wt.window_refs(seqs, W["seq"], pos)
        new = np.asarray([sum(r not in sa for r in b) for sa, b in ((set(a), b) for a, b in zip(sref, wref))]); outside = ((pos < S.min(1, keepdims=True)) | (pos > S.max(1, keepdims=True))).sum(1)
        recs = {r[0] for q in seqs for r in q["refs"]}
        rec["folds"][str(f)] = {
            "requests": int(len(W["seq"])), "sequences": len(seqs), "native_sequences": sum(q["kind"] == "native" for q in seqs), "recordings": len(recs), "sessions": len({session_of(v) for v in recs}),
            "span_samples": {"S_mean": float(span_s.mean()), "W_mean": float(span_w.mean()), "W_quantiles_5_25_50_75_95": [float(x) for x in np.percentile(span_w, [5, 25, 50, 75, 95])], "W_full_30": float((span_w == 30).mean()), "W_eq_S": float((span_w == span_s).mean())},
            "distinct_refs_per_window": {"S_mean": float(np.mean([len(set(r)) for r in sref])), "W_mean": float(np.mean([len(set(r)) for r in wref]))},
            "W_slots_not_in_S_refs_share": float(new.sum() / new.size / 16), "W_slots_outside_S_range_share": float(outside.sum() / (16 * len(outside))), "windows_with_new_refs": float((new > 0).mean()),
            "clamped_slots_share": float(((raw < 0) | (raw > n[:, None] - 1)).mean()), "windows_with_repeated_W_positions": float(np.mean([len(set(r)) < 16 for r in pos.tolist()])),
            "windows_identical_to_S": int(sum(a == b for a, b in zip(sref, wref))),
            "sequences_N_lt_16": int(sum(len(q["refs"]) < 16 for q in seqs)), "sequences_N_lt_31": int(sum(len(q["refs"]) < 31 for q in seqs)), "requests_in_N_lt_31": int((n < 31).sum())}
    rec["realised"] = all(v["windows_identical_to_S"] < v["requests"] for v in rec["folds"].values()); rec["finished"] = now(); publish(ctx, "audit.json", rec); print(json.dumps(rec, indent=1), flush=True)
    if not rec["realised"]:
        raise SystemExit("W equals S in every window: the input condition is not realised")
    return f"W new-ref share {rec['folds']['0']['W_slots_not_in_S_refs_share']:.3f} / {rec['folds']['1']['W_slots_not_in_S_refs_share']:.3f}"


# ----------------------------------------------------------------------------------------------- check
def mapping_check(ctx: Ctx) -> dict:
    out = {}
    for f in FOLDS:
        seqs, W, pos = fold_windows(ctx, f); bad = 0; cases = Counter()
        for k in range(len(W["seq"])):
            q = seqs[int(W["seq"][k])]; n, c = len(q["refs"]), int(W["centre"][k]); s = window_frames(c, n, 16, 16); loop = [min(max(c + 2 * (int(x) - c), 0), n - 1) for x in s]
            ok = np.array_equal(s, W["s"][k]) and loop == pos[k].tolist() and tm.marker_slot(s, c) == int(W["marker"][k]) and tuple(q["refs"][loop[int(W["marker"][k])]]) == tuple(q["refs"][c])
            ok &= all(a <= b for a, b in zip(loop, loop[1:])) and 0 <= min(loop) and max(loop) <= n - 1; bad += not ok
            src = [tuple(q["refs"][p]) for p in loop]
            cases["N_lt_16"] += n < 16; cases["N_eq_16"] += n == 16; cases["N_ge_31"] += n >= 31; cases["first_centre"] += c == 0; cases["last_centre"] += c == n - 1
            cases["repeated_positions"] += len(set(loop)) < 16; cases["crosses_a_join"] += len({r[0] for r in src}) > 1; cases["one_source_two_gammas"] += any(len({r[2] for r in src if r[0] == v}) > 1 for v in {r[0] for r in src})
        out[str(f)] = {"windows": int(len(W["seq"])), "mismatches": bad, "cases": dict(cases)}
    return out


def check(ctx: Ctx, device, workers: int) -> str:
    res = {"config_sha256": sha256_file(ROOT / ctx.config), "tolerances": TOL}; s = 0; a08 = ctx.a31.a08(s, device).eval(); d0 = tet.state_digest(a08)
    res["mapping"] = mapping_check(ctx); seqs, W, pos = fold_windows(ctx, 0); fp = t28.frame_paths(seqs); O = t31.outputs(ctx.a31, s, 0)
    pick = np.sort(np.random.default_rng([20261026, 37, 1]).choice(len(W["seq"]), 32, replace=False)); sub = lambda D, ks: {k: v[ks] for k, v in D.items()}  # noqa: E731
    # online S parity, forced W = S through the W path, explicit stacking, repeat, batch split
    fr_s, idx_s = chunk_frames(seqs, sub(W, pick), W["s"][pick], fp, 0); e_s, z_s = embed(a08, fr_s, idx_s, W["marker"][pick], device)
    s_as_w = wt.pool_positions(seqs, W, factor=1)[pick]; fr_f, idx_f = chunk_frames(seqs, sub(W, pick), s_as_w, fp, 0); e_f, _ = embed(a08, fr_f, idx_f, W["marker"][pick], device)
    fr_w, idx_w = chunk_frames(seqs, sub(W, pick), pos[pick], fp, 0); e_w, _ = embed(a08, fr_w, idx_w, W["marker"][pick], device)
    stack = np.stack([np.stack([jv.load_frame_u8(fp(v)[min(int(fi), len(fp(v)) - 1)], round(float(g), 6)) for v, fi, g in row]) for row in wt.window_refs(seqs, W["seq"][pick], pos[pick])])
    with torch.no_grad():
        e_x = a08.embed(tm.to_rgb(torch.from_numpy(stack).to(device)), tm.marker_planes(torch.from_numpy(W["marker"][pick]).long().to(device))).float()
    e_w2, _ = embed(a08, fr_w, idx_w, W["marker"][pick], device); e_w3 = torch.cat([embed(a08, fr_w, idx_w[:16], W["marker"][pick][:16], device)[0], embed(a08, fr_w, idx_w[16:], W["marker"][pick][16:], device)[0]])
    res["features"] = {"S_online_vs_cache_logits_max_abs": float(np.abs(z_s.cpu().numpy() - O["logits"][pick]).max()), "S_online_vs_cache_emb_max_abs": float(np.abs(e_s.cpu().numpy() - O["emb"][pick]).max()),
                       "forced_W_eq_S_vs_cache_emb_max_abs": float(np.abs(e_f.cpu().numpy() - O["emb"][pick]).max()), "explicit_stack_input_equal": bool(np.array_equal(stack, fr_w[idx_w])),
                       "explicit_stack_emb_max_abs": float((e_w - e_x).abs().max()), "repeat_max_abs": float((e_w - e_w2).abs().max()), "batch_split_max_abs": float((e_w - e_w3).abs().max()),
                       "W_vs_S_emb_max_abs": float((e_w - e_s).abs().max()), "W_windows_differing_from_S": int(sum(not np.array_equal(a, b) for a, b in zip(pos[pick], W["s"][pick]))),
                       "tf32": {"matmul": torch.backends.cuda.matmul.allow_tf32, "cudnn": torch.backends.cudnn.allow_tf32}}
    res["weights_unchanged"] = tet.state_digest(a08) == d0
    # Init = Base, forced eW = e224, gradients, binding, resume (eW placeholder = e224: no new features needed)
    X, offs = {"q": [], "e224": []}, {}
    for f in FOLDS:
        Of = t31.outputs(ctx.a31, s, f); X["q"].append(Of["logits"]); X["e224"].append(Of["emb"]); offs[f] = int(Of["offset"])
    X = {k: torch.from_numpy(np.concatenate(v)) for k, v in X.items()}; X["S"] = X["e224"]; X["W"] = X["e224"].clone(); a8 = rsh.a08_ctx(); data = t31.vpr.training_data(ctx.a31.a15.a10.a07p); init = {}
    old = json.loads((ROOT / "docs/results/2026-10-25_a5c_source_external_evidence/readout.json").read_text())["sequences"]
    for f in FOLDS:
        g = wt.head(f); D = d32.fold_data(ctx.d, s, f); pk = D["pk"]; off = offs[f]; Wf = t31.pool_W(D["seqs"])
        with torch.no_grad():
            z = g(X["q"], X["e224"], X["W"])
        pp = t29.pooled_probs(z, pk, np.arange(len(pk["labels"]))); ps = rr.per_sample(Wf, z[off:off + len(Wf["seq"])].numpy(), D["seqs"]); preds = {q["id"]: ir.relabel(D["base_preds"][q["id"]], ps[q["id"]]) for q in D["seqs"]}
        P = {"seqs": D["seqs"], "n_of": dict(data.n), "label_of": dict(data.label)}; sr = t31.seq_rows(a8.cfg, P, preds, bc.evalset(P, list(range(len(D["seqs"]))), f"heldout_f{f}")); ob = old[f"s{s}_heldout_f{f}"]["A08 Base"]
        init[str(f)] = {"z_equals_q": bool(torch.equal(z, X["q"])), "runs_equal_base": bool(np.array_equal(pp.argmax(1), pk["base"])), "sequences_equal_base": all(np.array_equal(preds[q["id"]], D["base_preds"][q["id"]]) for q in D["seqs"]),
                        "rows_equal_A32_base": bool(sr["row"] == ob["sequence_row"] and rsh.counts_of(D["seqs"], sr) == ob["counts"]), "P_runs": int(len(pk["labels"])), "base_wrong": int((pk["base"] != pk["labels"]).sum())}
    res["init_equals_base"] = init; R = t31.runs(ctx.a31, s, 0); torch.set_num_threads(1)
    gs = wt.head(0); os_ = t31.opt_of([("g", list(gs.parameters()), 1e-3)]); gw = wt.head(0); ow = t31.opt_of([("g", list(gw.parameters()), 1e-3)])
    res["identical_init"] = all(torch.equal(a, b) for a, b in zip(gs.state_dict().values(), gw.state_dict().values()))
    ls = [step(gs, os_, X, "S", R, u) for u in range(3)]; lw = [step(gw, ow, X, "W", R, u) for u in range(3)]
    res["forced_equal"] = {"loss_max_abs": max(abs(a["loss"] - b["loss"]) for a, b in zip(ls, lw)), "params_max_abs": float(max((a - b).abs().max() for a, b in zip(gs.state_dict().values(), gw.state_dict().values())))}
    g1 = wt.head(0); o1 = t31.opt_of([("g", list(g1.parameters()), 1e-3)]); t31.sched(o1, 0, {"updates": U, "warmup": WARM}); rows, w, ix, y = t29.batch_rows(R, R["bt"][0], R["bu"][0])
    t29.pooled_ce(g1(X["q"][rows], X["e224"][rows], X["W"][rows]), w, ix, y).mean().backward(); gr = {"fc1_grad_abs_sum_step1": float(g1.fc1.weight.grad.abs().sum()), "fc2_grad_abs_sum_step1": float(g1.fc2.weight.grad.abs().sum())}
    g2 = wt.head(0); o2 = t31.opt_of([("g", list(g2.parameters()), 1e-3)]); step(g2, o2, X, "W", R, 0); o2.zero_grad(); rows, w, ix, y = t29.batch_rows(R, R["bt"][1], R["bu"][1])
    t29.pooled_ce(g2(X["q"][rows], X["e224"][rows], X["W"][rows]), w, ix, y).mean().backward(); gr["fc1_grad_abs_sum_step2"] = float(g2.fc1.weight.grad.abs().sum()); res["gradients"] = gr
    res["optimizer_binds_exactly"] = sorted(id(p) for grp in ow.param_groups for p in grp["params"]) == sorted(id(p) for p in gw.parameters())
    ga = wt.head(0); oa = t31.opt_of([("g", list(ga.parameters()), 1e-3)]); [step(ga, oa, X, "W", R, u) for u in range(6)]
    gb = wt.head(0); ob_ = t31.opt_of([("g", list(gb.parameters()), 1e-3)]); [step(gb, ob_, X, "W", R, u) for u in range(3)]; sv = ({k: v.clone() for k, v in gb.state_dict().items()}, ob_.state_dict())
    gc = wt.head(0); oc = t31.opt_of([("g", list(gc.parameters()), 1e-3)]); gc.load_state_dict(sv[0]); oc.load_state_dict(sv[1]); [step(gc, oc, X, "W", R, u) for u in range(3, 6)]
    res["resume_3_plus_3_vs_6_max_abs"] = float(max((a - b).abs().max() for a, b in zip(ga.state_dict().values(), gc.state_dict().values())))
    sp = json.loads((ctx.a31.out / "split.json").read_text()); res["isolation"] = sp["isolation"]; res["windows"] = {str(f): int(len(t31.pool_W(t31.load_pool(ctx.a31, f))["seq"])) for f in FOLDS}
    # timing and disk on the first sequences of fold 0
    ch = seqs[:12]; Wt = t31.pool_W(ch); pt = wt.pool_positions(ch, Wt); torch.cuda.synchronize(); t0 = time.time(); frt, idt = chunk_frames(ch, Wt, pt, fp, workers); td = time.time() - t0; nt = len(Wt["seq"]); t1 = time.time()
    for b in range(0, nt, BATCH):
        embed(a08, frt, idt[b:b + BATCH], Wt["marker"][b:b + BATCH], device)
    torch.cuda.synchronize(); tg = time.time() - t1; nw = sum(res["windows"].values()); per = (td + tg) / nt
    res["timing"] = {"windows": nt, "decode_s": td, "gpu_s": tg, "seconds_per_window": per, "projected_hours_per_seed": nw * per / 3600 * 1.15, "peak_gpu_gb": torch.cuda.max_memory_allocated() / 1e9}
    res["disk"] = {"projected_feature_gb_per_seed": nw * 512 * 4 / 1e9, "free_gb": shutil.disk_usage(ctx.out).free / 1e9}
    F_ = res["features"]
    res["ok"] = bool(all(v["mismatches"] == 0 for v in res["mapping"].values()) and F_["S_online_vs_cache_logits_max_abs"] <= TOL["parity"] and F_["S_online_vs_cache_emb_max_abs"] <= TOL["parity"]
                     and F_["forced_W_eq_S_vs_cache_emb_max_abs"] <= TOL["forced_w_eq_s"] and F_["explicit_stack_input_equal"] and F_["explicit_stack_emb_max_abs"] <= TOL["explicit"] and F_["repeat_max_abs"] == 0
                     and F_["batch_split_max_abs"] <= TOL["batch_split"] and F_["W_vs_S_emb_max_abs"] > 0 and not F_["tf32"]["matmul"] and not F_["tf32"]["cudnn"] and res["weights_unchanged"]
                     and all(all(v[k] for k in ("z_equals_q", "runs_equal_base", "sequences_equal_base", "rows_equal_A32_base")) for v in init.values()) and res["identical_init"]
                     and res["forced_equal"]["loss_max_abs"] == 0 and res["forced_equal"]["params_max_abs"] == 0 and gr["fc1_grad_abs_sum_step1"] == 0 and gr["fc2_grad_abs_sum_step1"] > 0 and gr["fc1_grad_abs_sum_step2"] > 0
                     and res["optimizer_binds_exactly"] and res["resume_3_plus_3_vs_6_max_abs"] == 0 and not any(sp["isolation"].values()) and res["timing"]["projected_hours_per_seed"] <= 0.8 and res["disk"]["projected_feature_gb_per_seed"] < 1.0)
    res["finished"] = now(); publish(ctx, "checks.json", res); print(json.dumps(res, indent=1, default=str)[:6000], flush=True)
    if not res["ok"]:
        raise SystemExit("a check failed")
    return f"{per * 1e3:.1f} ms/window, {res['timing']['projected_hours_per_seed']:.2f} h/seed"


# ----------------------------------------------------------------------------------------------- train
def train(ctx: Ctx, s: int) -> None:
    torch.set_num_threads(1); X = feats(ctx, s)
    for f in FOLDS:
        R = t31.runs(ctx.a31, s, f)
        for arm in ARMS:
            t0, started, ok, note = time.time(), now(), False, ""
            try:
                rd = ctx.out / "train" / f"{arm}_s{s}_f{f}"; rd.mkdir(parents=True, exist_ok=True); g = wt.head(f); o = t31.opt_of([("g", list(g.parameters()), 1e-3)]); before = fit_monitor(g, X, arm, R); log = []
                for u in range(U):
                    log.append(step(g, o, X, arm, R, u))
                    if (u + 1) % 270 == 0:
                        print(f"[{arm}_s{s}_f{f}] update {u + 1}/{U}: loss {np.mean([x['loss'] for x in log[-270:]]):.4f}  {time.time() - t0:.1f} s", flush=True)
                after = fit_monitor(g, X, arm, R); torch.save({k: v.detach().clone() for k, v in g.state_dict().items()}, rd / "final.pt"); (rd / "updates.jsonl").write_text("".join(json.dumps(x) + "\n" for x in log))
                man = {"arm": arm, "seed": s, "fit_fold": f, "updates": len(log), "final_sha256": sha256_file(rd / "final.pt"), "fit_before": before, "fit_after": after, "seconds": time.time() - t0}
                write_json(rd / "run_manifest.json", man); print(json.dumps({k: man[k] for k in ("arm", "seed", "fit_fold", "updates", "final_sha256")}), flush=True); ok = True
            except BaseException as e:  # noqa: BLE001 - recorded, then raised again
                note = f"{type(e).__name__}: {str(e)[:200]}"; raise
            finally:
                ledger_append(ctx.ledger, f"train {arm}_s{s}_f{f} (CPU)", "cpu", started, time.time() - t0, ok, note)


# ----------------------------------------------------------------------------------------------- readout
def load_head(ctx: Ctx, arm: str, s: int, f: int):
    rd = ctx.out / "train" / f"{arm}_s{s}_f{f}"; man = json.loads((rd / "run_manifest.json").read_text()); com = json.loads((ctx.out / "weights_committed.json").read_text())
    if sha256_file(rd / "final.pt") != man["final_sha256"] or com.get(f"{arm}_s{s}_f{f}") != man["final_sha256"]:
        raise SystemExit(f"{arm}_s{s}_f{f}: weights not committed or changed")
    g = wt.head(f); g.load_state_dict(torch.load(rd / "final.pt", map_location="cpu", weights_only=True)); return g.eval()


def coverage(D: dict, y: np.ndarray, b: np.ndarray, p: np.ndarray) -> tuple[dict, dict]:
    """Recordings / sessions of fixed and broken runs; a run counts all its source recordings."""
    wr = b != y; out, sets = {}, {}
    for k, m in (("fixed", wr & (p == y)), ("broken", ~wr & (p != y))):
        sets[k] = {v for i in np.flatnonzero(m) for v in D["src"][i]}; out[k] = {"runs": int(m.sum()), "recordings": len(sets[k]), "sessions": len({session_of(v) for v in sets[k]})}
    return out, sets


def context_description(D: dict, W: dict, pos: np.ndarray, off: int) -> dict:
    """Reference-based description after the weights are frozen: label mix of S and W windows, all windows and S1 runs' windows."""
    lab = {i: np.asarray(q["labels"]) for i, q in enumerate(D["seqs"])}; cl = np.asarray([lab[int(i)][int(c)] for i, c in zip(W["seq"], W["centre"])])
    other = lambda P_: np.asarray([(lab[int(i)][row] != c).mean() for i, row, c in zip(W["seq"], P_, cl)])  # noqa: E731
    oS, oW = other(W["s"]), other(pos); new = np.asarray([len(set(b.tolist()) - set(a.tolist())) / 16 for a, b in zip(W["s"], pos)]); pk = D["pk"]; M = D["M"]
    s1_rows = np.unique(np.concatenate([pk["rows"][pk["offsets"][i]:pk["offsets"][i + 1]] for i in np.flatnonzero(M["S1"])]) - off) if M["S1"].any() else np.array([], int)
    part = lambda ix: {"windows": int(len(ix)), "other_label_share_S": float(oS[ix].mean()), "other_label_share_W": float(oW[ix].mean()), "new_position_share": float(new[ix].mean()),  # noqa: E731
                       "windows_W_adds_other_label": float((oW[ix] > oS[ix]).mean()), "windows_W_pure_where_S_mixed": float(((oW[ix] == 0) & (oS[ix] > 0)).mean())}
    return {"all_windows": part(np.arange(len(cl))), "S1_run_windows": part(s1_rows), "note": "reference labels describe the input only; not a rule, selection, route or anatomy"}


def readout(ctx: Ctx, s: int) -> str:
    name = f"readout_s{s}.json"
    if (ctx.out / name).exists():
        raise SystemExit("this seed's readout exists; it runs once")
    X = feats(ctx, s, committed=True); a8 = rsh.a08_ctx(); data = t31.vpr.training_data(ctx.a31.a15.a10.a07p); rec = {"seed": s, "cells": {}}; acc = {m: [] for m in ("Base", "S", "W")}; ys, bs = [], []; cov = {a: [] for a in ARMS}
    for h in FOLDS:
        f = 1 - h; D = d32.fold_data(ctx.d, s, h); D["M"] = d32.set_masks(D); off = X["offset"][h]; P = {"seqs": D["seqs"], "n_of": dict(data.n), "label_of": dict(data.label)}; ev = bc.evalset(P, list(range(len(D["seqs"]))), f"heldout_f{h}")
        cell = {}
        with torch.no_grad():
            for arm in ARMS:
                t = tables(load_head(ctx, arm, s, f)(X["q"], X["e224"], X[arm]), D, off, a8.cfg, P, ev); p = t.pop("pred"); acc[arm].append(p); cell[arm] = t
                cell[arm]["coverage"], sets = coverage(D, D["pk"]["labels"], D["pk"]["base"], p); cov[arm].append(sets)
            t = tables(X["q"], D, off, a8.cfg, P, ev); acc["Base"].append(t.pop("pred")); cell["Base"] = t
        mans = {arm: json.loads((ctx.out / "train" / f"{arm}_s{s}_f{f}" / "run_manifest.json").read_text()) for arm in ARMS}; cell["fit"] = {arm: {k: mans[arm][k] for k in ("fit_before", "fit_after")} for arm in ARMS}
        row = lambda m_: cell[m_]["sequence_row"]  # noqa: E731
        cell["comparisons"] = {f"{a} - {bb}": {"net_fixed": cell[a]["runs"]["all"]["net_fixed"] - (cell[bb]["runs"]["all"]["net_fixed"] if bb != "Base" else 0), **{k: row(a)[k] - row(bb)[k] for k in ("segment_f1_50_multi", "native_macro_f1", "boundary_recall_multi", "boundary_precision_multi")},
                                                 **{k: cell[a]["counts"][k] - cell[bb]["counts"][k] for k in ("fp", "fp_short", "fn", "fn_short")}} for a, bb in (("W", "S"), ("W", "Base"), ("S", "Base"))}
        seqs_h, W_h, pos_h = fold_windows(ctx, h); cell["context_description"] = context_description(D, W_h, pos_h, off)
        rec["cells"][f"heldout_f{h}"] = cell; ys.append(D["pk"]["labels"]); bs.append(D["pk"]["base"])
    y, b = np.concatenate(ys), np.concatenate(bs); wr = b != y; rec["oof"] = {}
    for m_ in ARMS:
        p = np.concatenate(acc[m_]); fx, br = int((wr & (p == y)).sum()), int((~wr & (p != y)).sum()); un = {k: set().union(*[c[k] for c in cov[m_]]) for k in ("fixed", "broken")}
        rec["oof"][m_] = {"runs": int(len(y)), "fixed": fx, "broken": br, "net_fixed": fx - br, **{f"{k}_recordings": len(v) for k, v in un.items()}, **{f"{k}_sessions": len({session_of(x) for x in v}) for k, v in un.items()}}
    c = rec["cells"]; dF = lambda a, bb, h: c[f"heldout_f{h}"][a]["sequence_row"]["segment_f1_50_multi"] - c[f"heldout_f{h}"][bb]["sequence_row"]["segment_f1_50_multi"]  # noqa: E731
    net = lambda a, h: c[f"heldout_f{h}"][a]["runs"]["all"]["net_fixed"]  # noqa: E731
    s1 = sum(c[f"heldout_f{h}"]["W"]["S1_fixed"]["fixed"] for h in FOLDS); n1 = sum(c[f"heldout_f{h}"]["W"]["S1_fixed"]["runs"] for h in FOLDS)
    k5 = sum(c[f"heldout_f{h}"]["W"]["correct_A5C_kept"]["kept"] for h in FOLDS); n5 = sum(c[f"heldout_f{h}"]["W"]["correct_A5C_kept"]["runs"] for h in FOLDS)
    rule = {"wider_net_each_fold_ge_5": all(net("W", h) - net("S", h) >= 5 for h in FOLDS), "wider_dF_mean_ge_0.003": float(np.mean([dF("W", "S", h) for h in FOLDS])) >= 0.003, "wider_dF_each_fold_ge_0": all(dF("W", "S", h) >= 0 for h in FOLDS),
            "useful_net_each_fold_ge_0": all(net("W", h) >= 0 for h in FOLDS), "useful_dF_each_fold_ge_0": all(dF("W", "Base", h) >= 0 for h in FOLDS), "useful_dF_mean_ge_0.003": float(np.mean([dF("W", "Base", h) for h in FOLDS])) >= 0.003,
            "S1_recovered": f"{s1}/{n1}", "S1_ok": s1 >= math.ceil(0.10 * n1), "correct_A5C_kept": f"{k5}/{n5}", "A5C_ok": k5 >= math.ceil(0.98 * n5),
            "native_each_fold_ge_-0.003": all(c[f"heldout_f{h}"]["W"]["sequence_row"]["native_macro_f1"] - c[f"heldout_f{h}"]["Base"]["sequence_row"]["native_macro_f1"] >= -0.003 for h in FOLDS),
            "missed_short_le_base_plus_3": sum(c[f"heldout_f{h}"]["W"]["counts"]["fn_short"] for h in FOLDS) <= sum(c[f"heldout_f{h}"]["Base"]["counts"]["fn_short"] for h in FOLDS) + 3,
            "boundary_recall_each_fold_ge_-0.005": all(c[f"heldout_f{h}"]["W"]["sequence_row"]["boundary_recall_multi"] - c[f"heldout_f{h}"]["Base"]["sequence_row"]["boundary_recall_multi"] >= -0.005 for h in FOLDS)}
    rule["met"] = all(v for k, v in rule.items() if isinstance(v, bool)); rec["seed2_rule"] = rule
    rec["summary"] = {f"heldout_f{h}": {m_: {"net": net(m_, h) if m_ != "Base" else 0, "F1": c[f"heldout_f{h}"][m_]["sequence_row"]["segment_f1_50_multi"]} for m_ in ("Base", "S", "W")} for h in FOLDS}
    fr = json.loads((ctx.out / "feats" / f"record_s{s}.json").read_text()); pr = json.loads((ctx.a31.out / "cache" / f"prefix_s{s}.json").read_text()); nw = sum(fr[str(f)]["windows"] for f in FOLDS)
    rec["deployment_cost"] = {"forwards_W": "Base 224 forward + one auxiliary 224 forward on W (same weights)", "forwards_S": "Base 224 forward; the auxiliary input equals e224 (cache reuse; a matched deployment would cost one more identical forward)",
                              "parameters": "A08 34.6M shared + 32,965 head", "gpu_ms_per_window_224_A31_pass": 1e3 * pr["gpu_seconds"] / sum(pr["rows"].values()), "gpu_ms_per_window_W": 1e3 * fr["gpu_seconds"] / nw,
                              "cpu_decode_ms_per_window_W": 1e3 * fr["decode_seconds"] / nw, "note": "the 224 pass also wrote A31's prefix cache; the W figure is the frozen encoder only"}
    publish(ctx, name, rec); print(json.dumps({"summary": rec["summary"], "oof": rec["oof"], "rule": rule}, indent=1)); return f"rule met: {rule['met']}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True); ap.add_argument("--stage", choices=("audit", "check", "extract", "train", "readout"), required=True); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args(); a.set = []; ctx = Ctx(load_cfg(a), a.config); ctx.out.mkdir(parents=True, exist_ok=True)
    if a.stage == "train":
        return train(ctx, a.seed)
    if a.stage in ("audit", "readout"):
        return staged(ctx, f"{a.stage}{' s' + str(a.seed) if a.stage == 'readout' else ''} (CPU)", "cpu", lambda: audit(ctx) if a.stage == "audit" else readout(ctx, a.seed))
    t31.vsh.determinism(); device = torch.device("cuda")
    staged(ctx, f"{a.stage}{' s' + str(a.seed) if a.stage == 'extract' else ''} (GPU job wall time)", "gpu", {"check": lambda: check(ctx, device, a.workers), "extract": lambda: extract(ctx, a.seed, device, a.workers)}[a.stage])


if __name__ == "__main__":
    main()
