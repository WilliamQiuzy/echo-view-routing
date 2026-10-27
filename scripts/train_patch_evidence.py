#!/usr/bin/env python
"""A39 (configs/experiments/video_patch_evidence_v1.yaml): projected final-token channel max (P) against mean (G).

  --stage check    hook / cache parity, token shape and CLS, pooling / projection recomputation, repeat / batch split,
                   Init = Base, forced eP = eG, head mechanics, timing and disk (GPU)
  --stage extract  --seed s: one original A08 forward per request window gives eG and eP (q / e224 compared with A31) (GPU)
  --stage train    --seed s: residual heads G_f0, P_f0, G_f1, P_f1 on A31's uniform plans (CPU, single thread)
  --stage freeze   --seed s --commit c --manifest-sha h: record that weights_s{s}.json (sha h) was committed in c
  --stage readout  --seed s: after the heads are committed, one held-out readout (+ the source-2 rule for seed 0) (CPU)
The A08 + B3 path supplies q, e224, Base and its runs; nothing here changes it.
"""
from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")     # A27's CPU-only trainer import would otherwise hide the GPU

import argparse  # noqa: E402
import hashlib  # noqa: E402
import inspect  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
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
import scripts.train_predicted_run_context as t38  # noqa: E402
import scripts.train_session_correction as t31  # noqa: E402
import scripts.train_wider_temporal_evidence as t37  # noqa: E402
from scripts._common import load_cfg  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import echo_routing.temporal.interval_recognition as ir  # noqa: E402
import echo_routing.temporal.joint_visual as jv  # noqa: E402
import echo_routing.temporal.token_pool_evidence as tpe  # noqa: E402
import echo_routing.temporal.wider_temporal as wt  # noqa: E402
import echo_routing.vit.target_mvit as tm  # noqa: E402
import echo_routing.vit.target_patch_readout as tp  # noqa: E402
from echo_routing.evaluate.complementarity import sha256_file  # noqa: E402
from echo_routing.evaluate.failure_units import session_of  # noqa: E402
from echo_routing.evaluate.window_target import window_frames  # noqa: E402

A31_CONFIG, A32_CONFIG = t36.A31_CONFIG, t36.A32_CONFIG
FOLDS, ARMS, U, WARM, CHUNK_SEQS, BATCH = (0, 1), tpe.ARMS, 1350, 135, 150, 32
A4C, A5C = d32.A4C, d32.A5C
TOL = {"parity": 1e-4, "cls": 1e-5, "explicit": 1e-4, "loop_mean": 1e-5, "batch_split": 1e-4}
PIXELS = t38.PIXELS
now, write_json, ledger_append, step, fit_monitor, tables = t31.now, t31.write_json, t31.ledger_append, t36.step, t36.fit_monitor, t36.tables
if (t36.U, t36.WARM) != (U, WARM):
    raise SystemExit("A36's step schedule differs from the configuration")


class Ctx:
    def __init__(self, cfg: dict, config: str) -> None:
        self.config, self.v = config, cfg["patch_evidence"]; o = self.v["outputs"]
        self.out, self.results, self.ledger = ROOT / o["root"], ROOT / o["results_dir"], ROOT / self.v["budget"]["ledger"]
        self.a31 = t31.Ctx(load_cfg(SimpleNamespace(config=A31_CONFIG, set=[])), A31_CONFIG); self.d = d32.Ctx(load_cfg(SimpleNamespace(config=A32_CONFIG, set=[])), A32_CONFIG)


def staged(ctx: Ctx, job: str, kind: str, fn) -> None:
    t36.staged(ctx, job, kind, fn)


def publish(ctx: Ctx, name: str, rec: dict) -> None:
    t36.publish(ctx, name, rec)


# ----------------------------------------------------------------------------------------------- windows and features
def fold_windows(ctx: Ctx, f: int) -> tuple[list, dict]:
    seqs = t31.load_pool(ctx.a31, f); return seqs, t31.pool_W(seqs)


def positions_sha(W: dict) -> str:
    return hashlib.sha256(W["seq"].tobytes() + W["centre"].tobytes() + W["s"].tobytes() + W["marker"].tobytes()).hexdigest()


def token_batch(m, frames: np.ndarray, idx: np.ndarray, marker: np.ndarray, device, stats: dict | None = None) -> dict:
    """q, e224, eG, eP of one batch through tpe's capture / projection / pooling; stats: label-free argmax description."""
    with torch.no_grad():
        rgb, mk = tm.to_rgb(torch.from_numpy(frames[idx]).to(device)), tm.marker_planes(torch.from_numpy(marker).long().to(device))
        e, u = tpe.embed_with_tokens(m, rgb, mk); v = tpe.project_tokens(m.encoder.head, u); d = tpe.pool(v)
        if stats is not None:
            am = v.argmax(1); stats["distinct_argmax_tokens"].extend(int(torch.unique(r).numel()) for r in am)
            stats["argmax_time_index_counts"] += np.bincount((am // (tpe.GRID[1] * tpe.GRID[2])).flatten().cpu().numpy(), minlength=tpe.GRID[0])
        return {"e224": e.float(), "q": m.classifier(e).float(), "G": d["G"].float(), "P": d["P"].float()}


def feature_key(ctx: Ctx, s: int) -> str:
    """Only what the features depend on: model / frame code, token capture / projection / pooling, extraction, weights, pools, requests, batching, pixels."""
    h = hashlib.sha256()
    for p in (Path(tm.__file__), Path(jv.__file__)):
        h.update(p.read_bytes())
    for fn in (tpe.embed_with_tokens, tpe.project_tokens, tpe.pool, token_batch, fold_windows, extract, t37.chunk_frames, rr.decode_frames, rr.RefFrames, rr.half_windows, t31.pool_W, window_frames):
        h.update(inspect.getsource(fn).encode())
    sp = json.loads((ctx.a31.out / "split.json").read_text()); pos = {str(f): positions_sha(fold_windows(ctx, f)[1]) for f in FOLDS}
    h.update(json.dumps({"a08": t31.A08[s], "pools": sp["pool_sha256"], "requests": pos, "chunk": CHUNK_SEQS, "batch": BATCH, "pixels": PIXELS, "tokens": tpe.CONTRACT, "grid": list(tpe.GRID)}, sort_keys=True).encode())
    return h.hexdigest()


def extract(ctx: Ctx, s: int, device, workers: int) -> str:
    m = ctx.a31.a08(s, device).eval(); d0 = tet.state_digest(m); fdir = ctx.out / "feats"; fdir.mkdir(parents=True, exist_ok=True); acc = {"gpu_seconds": 0.0, "decode_seconds": 0.0}; rec = {}
    prog = ctx.out / f"progress_s{s}.json"; st = {"distinct_argmax_tokens": [], "argmax_time_index_counts": np.zeros(tpe.GRID[0], np.int64)}; norms = {k: [] for k in ("e224", "G", "P", "P_minus_G")}
    for f in FOLDS:
        seqs, W = fold_windows(ctx, f); n = len(W["seq"]); fp = t28.frame_paths(seqs); O = t31.outputs(ctx.a31, s, f); E = {a: np.zeros((n, 512), np.float32) for a in ARMS}; dq = de = 0.0; finite = True
        for c0 in range(0, len(seqs), CHUNK_SEQS):
            rows = np.flatnonzero((W["seq"] >= c0) & (W["seq"] < c0 + CHUNK_SEQS))
            a = time.time(); fr, idx = t37.chunk_frames(seqs, {"seq": W["seq"][rows]}, W["s"][rows], fp, workers); acc["decode_seconds"] += time.time() - a
            for b in range(0, len(rows), BATCH):
                ks = np.arange(b, min(b + BATCH, len(rows))); r = rows[ks]; torch.cuda.synchronize(); a = time.time(); d = token_batch(m, fr, idx[ks], W["marker"][r], device, st); torch.cuda.synchronize(); acc["gpu_seconds"] += time.time() - a
                d = {k: v.cpu().numpy() for k, v in d.items()}; E["G"][r] = d["G"]; E["P"][r] = d["P"]; finite &= all(bool(np.isfinite(v).all()) for v in d.values())
                dq = max(dq, float(np.abs(d["q"] - O["logits"][r]).max())); de = max(de, float(np.abs(d["e224"] - O["emb"][r]).max()))
                for k, v in (("e224", d["e224"]), ("G", d["G"]), ("P", d["P"]), ("P_minus_G", d["P"] - d["G"])):
                    norms[k].append(np.linalg.norm(v, axis=1))
            del fr; done = int(rows.max()) + 1
            print(f"seed {s} fold {f}: {done}/{n} windows; GPU {acc['gpu_seconds']:.0f} s, decode {acc['decode_seconds']:.0f} s; q / e224 vs A31 {dq:.1e} / {de:.1e}; device {device}", flush=True)
            write_json(prog, {"seed": s, "fold": f, "fold_windows_done": done, "fold_windows": n, "folds": len(FOLDS), "updated_at": now(), **acc})
        if not (finite and dq <= TOL["parity"] and de <= TOL["parity"]):
            raise SystemExit(f"fold {f}: online q / e224 differ from A31's cache ({dq}, {de}) or a value is not finite")
        for arm in ARMS:
            np.save(fdir / f"e{arm}_s{s}_f{f}.npy", E[arm])
        rec[str(f)] = {"windows": n, **{f"{arm}_sha256": sha256_file(fdir / f"e{arm}_s{s}_f{f}.npy") for arm in ARMS}, "requests_sha256": positions_sha(W), "q_vs_A31_max_abs": dq, "e224_vs_A31_max_abs": de, "finite": finite}
    if tet.state_digest(m) != d0 or m.encoder.norm._forward_hooks or m.encoder.conv_proj._forward_hooks:
        raise SystemExit("the encoder changed during extraction or a hook was left")
    dt = np.asarray(st["distinct_argmax_tokens"]); q5 = lambda x: [float(v) for v in np.percentile(x, [5, 25, 50, 75, 95])]  # noqa: E731
    rec.update({"seed": s, "key": feature_key(ctx, s), "a08": t31.A08[s], "contract": tpe.CONTRACT, "time": now(), **acc,
                "description": {"distinct_argmax_tokens_of_512_channels_quantiles_5_25_50_75_95": q5(dt), "argmax_time_index_share": (st["argmax_time_index_counts"] / st["argmax_time_index_counts"].sum()).tolist(),
                                "l2_norm_mean": {k: float(np.concatenate(v).mean()) for k, v in norms.items()},
                                "note": "label-free; a channel maximum's token index is not a location, structure or phase"}})
    write_json(fdir / f"record_s{s}.json", rec); return f"{sum(rec[str(f)]['windows'] for f in FOLDS)} windows; q / e224 vs A31 max {max(rec[str(f)]['q_vs_A31_max_abs'] for f in FOLDS):.1e} / {max(rec[str(f)]['e224_vs_A31_max_abs'] for f in FOLDS):.1e}"


def feats(ctx: Ctx, s: int, committed: bool = False) -> dict:
    """committed: the record must equal the one committed with the heads (weights_s{s}.json); the key must match in any case."""
    rec = json.loads((ctx.out / "feats" / f"record_s{s}.json").read_text())
    if committed and json.loads((ctx.results / f"weights_s{s}.json").read_text())["feature_record"] != rec:
        raise SystemExit("the feature record differs from the one committed with the heads")
    if rec["key"] != feature_key(ctx, s):
        raise SystemExit("the feature cache belongs to other feature code")
    out = {"q": [], "e224": [], **{a: [] for a in ARMS}, "offset": {}}; o = 0
    for f in FOLDS:
        O = t31.outputs(ctx.a31, s, f); out["q"].append(O["logits"]); out["e224"].append(O["emb"]); out["offset"][f] = o; o += len(O["logits"])
        if int(O["offset"]) != out["offset"][f]:
            raise SystemExit("fold offsets differ from A31's")
        for arm in ARMS:
            p = ctx.out / "feats" / f"e{arm}_s{s}_f{f}.npy"
            if sha256_file(p) != rec[str(f)][f"{arm}_sha256"]:
                raise SystemExit("a feature file differs from its record")
            out[arm].append(np.load(p))
    return {k: (torch.from_numpy(np.concatenate(v)) if isinstance(v, list) else v) for k, v in out.items()}


# ----------------------------------------------------------------------------------------------- check
def token_checks(a08, fr: np.ndarray, idx: np.ndarray, marker: np.ndarray, O: dict, pick: np.ndarray, device) -> dict:
    with torch.no_grad():
        rgb, mk = tm.to_rgb(torch.from_numpy(fr[idx]).to(device)), tm.marker_planes(torch.from_numpy(marker).long().to(device))
        e0 = a08.embed(rgb, mk); q0 = a08.classifier(e0); d = tpe.descriptors(a08, rgb, mk); e, u = tpe.embed_with_tokens(a08, rgb, mk); v = tpe.project_tokens(a08.encoder.head, u)
        tb = token_batch(a08, fr, idx, marker, device); rep = tpe.descriptors(a08, rgb, mk); h = len(pick) // 2
        sp = [tpe.descriptors(a08, rgb[:h], mk[:h]), tpe.descriptors(a08, rgb[h:], mk[h:])]; cls = a08.encoder.head(u[:, 0])
    lin = a08.encoder.head[-1]; W64, b64, u64 = lin.weight.double().cpu(), lin.bias.double().cpu(), u.double().cpu(); v64 = u64[:, 1:] @ W64.T + b64; vc = v.cpu()
    mx = torch.stack([_loop_max(vc[i]) for i in range(len(vc))]); mn = torch.stack([_loop_mean(vc[i]) for i in range(len(vc))]); gm = u64[:, 1:].mean(1) @ W64.T + b64
    split = {k: torch.cat([sp[0][k], sp[1][k]]) for k in d}
    return {"hooked_equals_plain": {"e224": bool(torch.equal(d["e224"], e0.float())), "q": bool(torch.equal(d["q"], q0.float())), "embed_with_tokens_e": bool(torch.equal(e, e0))},
            "online_vs_cache_logits_max_abs": float(np.abs(q0.float().cpu().numpy() - O["logits"][pick]).max()), "online_vs_cache_emb_max_abs": float(np.abs(e0.float().cpu().numpy() - O["emb"][pick]).max()),
            "U_shape": list(u.shape), "grid_from_network": list(tp.token_grid(a08.encoder)[0]), "V_shape": list(v.shape), "head_is_dropout_linear": [type(x).__name__ for x in a08.encoder.head],
            "cls_token_through_head_vs_e224_max_abs": float((cls - e).abs().max()), "V_vs_explicit_float64_matmul_max_abs": float((vc.double() - v64).abs().max()),
            "eG_vs_projected_token_mean_float64_max_abs": float((d["G"].cpu().double() - gm).abs().max()), "eP_vs_loop_max_equal": bool(torch.equal(d["P"].cpu(), mx)), "eG_vs_loop_mean_max_abs": float((d["G"].cpu() - mn).abs().max()),
            "token_batch_equals_descriptors": all(bool(torch.equal(tb[k], d[k])) for k in d), "repeat_max_abs": float(max((rep[k] - d[k]).abs().max() for k in d)),
            "batch_split_max_abs": float(max((split[k] - d[k]).abs().max() for k in d)), "finite": all(bool(torch.isfinite(x).all()) for x in (*d.values(), u)),
            "eP_minus_eG_max_abs": float((d["P"] - d["G"]).abs().max()), "eP_ge_eG": bool((d["P"] >= d["G"] - 1e-6).all()), "hooks_left": bool(a08.encoder.norm._forward_hooks or a08.encoder.conv_proj._forward_hooks),
            "tf32": {"matmul": torch.backends.cuda.matmul.allow_tf32, "cudnn": torch.backends.cudnn.allow_tf32}, "eval": not a08.training and not a08.encoder.head.training}


def _loop_max(v: torch.Tensor) -> torch.Tensor:
    out = v[0].clone()
    for t in range(1, len(v)):
        out = torch.maximum(out, v[t])
    return out


def _loop_mean(v: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(v[0], dtype=torch.float64)
    for t in range(len(v)):
        out += v[t].double()
    return (out / len(v)).float()


def synthetic() -> dict:
    r = torch.randn(4, 392, 512, generator=torch.Generator().manual_seed(39)); same = r[:, :1].expand(4, 392, 512); d, ds = tpe.pool(r), tpe.pool(same)
    return {"max_ge_mean": bool((d["P"] >= d["G"]).all()), "identical_tokens_P_eq_G": bool(torch.allclose(ds["P"], ds["G"])), "identical_tokens_P_eq_token": bool(torch.equal(ds["P"], r[:, 0]))}


def check(ctx: Ctx, device, workers: int) -> str:
    s = 0; res = {"config_sha256": sha256_file(ROOT / ctx.config), "tolerances": TOL, "contract": tpe.CONTRACT}; a08 = ctx.a31.a08(s, device).eval(); d0 = tet.state_digest(a08)
    seqs, W = fold_windows(ctx, 0); fp = t28.frame_paths(seqs); O = t31.outputs(ctx.a31, s, 0); n0 = len(W["seq"])
    pick = np.sort(np.random.default_rng([20261026, 39, 1]).choice(n0, 32, replace=False)); fr, idx = t37.chunk_frames(seqs, {"seq": W["seq"][pick]}, W["s"][pick], fp, 0)
    stack = np.stack([np.stack([jv.load_frame_u8(fp(v)[min(int(fi), len(fp(v)) - 1)], round(float(g), 6)) for v, fi, g in row]) for row in wt.window_refs(seqs, W["seq"][pick], W["s"][pick])])
    res["tokens"] = token_checks(a08, fr, idx, W["marker"][pick], O, pick, device); res["tokens"]["frames_equal_explicit_stack"] = bool(np.array_equal(stack, fr[idx])); res["synthetic"] = synthetic()
    res["weights_unchanged"] = tet.state_digest(a08) == d0
    # Init = Base, forced eP = eG, gradients, binding, resume (placeholders eG = eP = e224: no new features needed)
    X, offs = {"q": [], "e224": []}, {}
    for f in FOLDS:
        Of = t31.outputs(ctx.a31, s, f); X["q"].append(Of["logits"]); X["e224"].append(Of["emb"]); offs[f] = int(Of["offset"])
    X = {k: torch.from_numpy(np.concatenate(v)) for k, v in X.items()}; X["G"] = X["e224"].clone(); X["P"] = X["e224"].clone(); a8 = rsh.a08_ctx(); data = t31.vpr.training_data(ctx.a31.a15.a10.a07p); init = {}
    old = json.loads((ROOT / "docs/results/2026-10-25_a5c_source_external_evidence/readout.json").read_text())["sequences"]
    for f in FOLDS:
        g = tpe.head(f); D = d32.fold_data(ctx.d, s, f); pk = D["pk"]; off = offs[f]; Wf = t31.pool_W(D["seqs"])
        with torch.no_grad():
            z = g(X["q"], X["e224"], X["P"])
        pp = t29.pooled_probs(z, pk, np.arange(len(pk["labels"]))); ps = rr.per_sample(Wf, z[off:off + len(Wf["seq"])].numpy(), D["seqs"]); preds = {q["id"]: ir.relabel(D["base_preds"][q["id"]], ps[q["id"]]) for q in D["seqs"]}
        P = {"seqs": D["seqs"], "n_of": dict(data.n), "label_of": dict(data.label)}; sr = t31.seq_rows(a8.cfg, P, preds, bc.evalset(P, list(range(len(D["seqs"]))), f"heldout_f{f}")); ob = old[f"s{s}_heldout_f{f}"]["A08 Base"]
        init[str(f)] = {"z_equals_q": bool(torch.equal(z, X["q"])), "runs_equal_base": bool(np.array_equal(pp.argmax(1), pk["base"])), "sequences_equal_base": all(np.array_equal(preds[q["id"]], D["base_preds"][q["id"]]) for q in D["seqs"]),
                        "rows_equal_A32_base": bool(sr["row"] == ob["sequence_row"] and rsh.counts_of(D["seqs"], sr) == ob["counts"]), "P_runs": int(len(pk["labels"]))}
    res["init_equals_base"] = init; R = t31.runs(ctx.a31, s, 0); torch.set_num_threads(1)
    gg = tpe.head(0); og = t31.opt_of([("g", list(gg.parameters()), 1e-3)]); gp = tpe.head(0); op = t31.opt_of([("g", list(gp.parameters()), 1e-3)])
    res["identical_init"] = all(torch.equal(a, b) for a, b in zip(gg.state_dict().values(), gp.state_dict().values()))
    lg = [step(gg, og, X, "G", R, u) for u in range(3)]; lp = [step(gp, op, X, "P", R, u) for u in range(3)]
    res["forced_equal"] = {"loss_max_abs": max(abs(a["loss"] - b["loss"]) for a, b in zip(lg, lp)), "params_max_abs": float(max((a - b).abs().max() for a, b in zip(gg.state_dict().values(), gp.state_dict().values())))}
    g1 = tpe.head(0); o1 = t31.opt_of([("g", list(g1.parameters()), 1e-3)]); t31.sched(o1, 0, {"updates": U, "warmup": WARM}); rows, w, ix, y = t29.batch_rows(R, R["bt"][0], R["bu"][0])
    t29.pooled_ce(g1(X["q"][rows], X["e224"][rows], X["P"][rows]), w, ix, y).mean().backward(); res["gradients"] = {"fc1_grad_abs_sum_step1": float(g1.fc1.weight.grad.abs().sum()), "fc2_grad_abs_sum_step1": float(g1.fc2.weight.grad.abs().sum())}
    res["optimizer_binds_exactly"] = sorted(id(p) for grp in op.param_groups for p in grp["params"]) == sorted(id(p) for p in gp.parameters())
    ga = tpe.head(0); oa = t31.opt_of([("g", list(ga.parameters()), 1e-3)]); [step(ga, oa, X, "P", R, u) for u in range(6)]
    gb = tpe.head(0); ob_ = t31.opt_of([("g", list(gb.parameters()), 1e-3)]); [step(gb, ob_, X, "P", R, u) for u in range(3)]; sv = ({k: v.clone() for k, v in gb.state_dict().items()}, ob_.state_dict())
    gr = tpe.head(0); orr = t31.opt_of([("g", list(gr.parameters()), 1e-3)]); gr.load_state_dict(sv[0]); orr.load_state_dict(sv[1]); [step(gr, orr, X, "P", R, u) for u in range(3, 6)]
    res["resume_3_plus_3_vs_6_max_abs"] = float(max((a - b).abs().max() for a, b in zip(ga.state_dict().values(), gr.state_dict().values())))
    sp = json.loads((ctx.a31.out / "split.json").read_text()); res["isolation"] = sp["isolation"]; res["windows"] = {str(f): int(len(fold_windows(ctx, f)[1]["seq"])) for f in FOLDS}
    # timing and disk on the first sequences of fold 0: plain embed against the token path on the same batches
    rows = np.flatnonzero(W["seq"] < 40); t0 = time.time(); frt, idt = t37.chunk_frames(seqs, {"seq": W["seq"][rows]}, W["s"][rows], fp, workers); td = time.time() - t0; tg = {"plain": 0.0, "tokens": 0.0}
    for b in range(0, len(rows), BATCH):
        ks = np.arange(b, min(b + BATCH, len(rows))); mk = W["marker"][rows[ks]]; torch.cuda.synchronize(); a = time.time(); t37.embed(a08, frt, idt[ks], mk, device); torch.cuda.synchronize(); tg["plain"] += time.time() - a
        a = time.time(); token_batch(a08, frt, idt[ks], mk, device); torch.cuda.synchronize(); tg["tokens"] += time.time() - a
    nw = sum(res["windows"].values()); per = (td + tg["tokens"]) / len(rows)
    res["timing"] = {"windows": int(len(rows)), "decode_s": td, "gpu_s_plain_embed": tg["plain"], "gpu_s_token_path": tg["tokens"], "extra_ms_per_window_hook_projection_pooling": 1e3 * (tg["tokens"] - tg["plain"]) / len(rows),
                     "seconds_per_window": per, "projected_hours_per_seed": nw * per / 3600 * 1.15, "peak_gpu_gb": torch.cuda.max_memory_allocated() / 1e9}
    res["disk"] = {"projected_feature_gb_per_seed": 2 * nw * 512 * 4 / 1e9, "free_gb": shutil.disk_usage(ctx.out).free / 1e9}
    res["loaded_code_sha256"] = t38.loaded_code(); T = res["tokens"]
    res["ok"] = bool(all(T["hooked_equals_plain"].values()) and T["online_vs_cache_logits_max_abs"] <= TOL["parity"] and T["online_vs_cache_emb_max_abs"] <= TOL["parity"] and T["frames_equal_explicit_stack"]
                     and T["U_shape"] == [32, tpe.N_TOKENS, 768] and T["grid_from_network"] == list(tpe.GRID) and T["V_shape"] == [32, tpe.N_TOKENS - 1, 512] and T["head_is_dropout_linear"] == ["Dropout", "Linear"]
                     and T["cls_token_through_head_vs_e224_max_abs"] <= TOL["cls"] and T["V_vs_explicit_float64_matmul_max_abs"] <= TOL["explicit"] and T["eG_vs_projected_token_mean_float64_max_abs"] <= TOL["explicit"]
                     and T["eP_vs_loop_max_equal"] and T["eG_vs_loop_mean_max_abs"] <= TOL["loop_mean"] and T["token_batch_equals_descriptors"] and T["repeat_max_abs"] == 0 and T["batch_split_max_abs"] <= TOL["batch_split"]
                     and T["finite"] and T["eP_minus_eG_max_abs"] > 0 and T["eP_ge_eG"] and not T["hooks_left"] and not T["tf32"]["matmul"] and not T["tf32"]["cudnn"] and T["eval"] and all(res["synthetic"].values())
                     and res["weights_unchanged"] and all(all(v[k] for k in ("z_equals_q", "runs_equal_base", "sequences_equal_base", "rows_equal_A32_base")) for v in init.values()) and res["identical_init"]
                     and res["forced_equal"]["loss_max_abs"] == 0 and res["forced_equal"]["params_max_abs"] == 0 and res["gradients"]["fc1_grad_abs_sum_step1"] == 0 and res["gradients"]["fc2_grad_abs_sum_step1"] > 0
                     and res["optimizer_binds_exactly"] and res["resume_3_plus_3_vs_6_max_abs"] == 0 and not any(sp["isolation"].values())
                     and 2 * res["timing"]["projected_hours_per_seed"] <= 1.5 and 2 * res["disk"]["projected_feature_gb_per_seed"] < 1.0 and res["disk"]["free_gb"] > 10)
    res["finished"] = now(); publish(ctx, "checks.json", res); print(json.dumps({k: v for k, v in res.items() if k != "loaded_code_sha256"}, indent=1, default=str)[:7000], flush=True)
    if not res["ok"]:
        raise SystemExit("a check failed")
    return f"{per * 1e3:.1f} ms per window, {res['timing']['projected_hours_per_seed']:.2f} h per seed"


# ----------------------------------------------------------------------------------------------- train
def train(ctx: Ctx, s: int) -> None:
    torch.set_num_threads(1); X = feats(ctx, s)
    for f in FOLDS:
        R = t31.runs(ctx.a31, s, f)
        for arm in ARMS:
            t0, started, ok, note = time.time(), now(), False, ""
            try:
                rd = ctx.out / "train" / f"{arm}_s{s}_f{f}"; rd.mkdir(parents=True, exist_ok=True); g = tpe.head(f); o = t31.opt_of([("g", list(g.parameters()), 1e-3)]); before = fit_monitor(g, X, arm, R); log = []
                for u in range(U):
                    log.append(step(g, o, X, arm, R, u))
                    if (u + 1) % 270 == 0:
                        print(f"[{arm}_s{s}_f{f}] update {u + 1}/{U}: loss {np.mean([x['loss'] for x in log[-270:]]):.4f}  {time.time() - t0:.1f} s", flush=True)
                after = fit_monitor(g, X, arm, R); torch.save({k: v.detach().clone() for k, v in g.state_dict().items()}, rd / "final.pt"); (rd / "updates.jsonl").write_text("".join(json.dumps(x) + "\n" for x in log))
                man = {"arm": arm, "seed": s, "fit_fold": f, "updates": len(log), "final_sha256": sha256_file(rd / "final.pt"), "updates_sha256": sha256_file(rd / "updates.jsonl"), "fit_before": before, "fit_after": after, "seconds": time.time() - t0}
                write_json(rd / "run_manifest.json", man); print(json.dumps({k: man[k] for k in ("arm", "seed", "fit_fold", "updates", "final_sha256")}), flush=True); ok = True
            except BaseException as e:  # noqa: BLE001 - recorded, then raised again
                note = f"{type(e).__name__}: {str(e)[:200]}"; raise
            finally:
                ledger_append(ctx.ledger, f"train {arm}_s{s}_f{f} (CPU)", "cpu", started, time.time() - t0, ok, note)
    man = {f"{arm}_s{s}_f{f}": json.loads((ctx.out / "train" / f"{arm}_s{s}_f{f}" / "run_manifest.json").read_text()) for f in FOLDS for arm in ARMS}
    publish(ctx, f"weights_s{s}.json", {"seed": s, "heads": man, "feature_record": json.loads((ctx.out / "feats" / f"record_s{s}.json").read_text()), "config_sha256": sha256_file(ROOT / ctx.config), "written": now()})


def freeze(ctx: Ctx, s: int, commit: str, manifest_sha: str) -> str:
    return t38.freeze(ctx, s, commit, manifest_sha)


# ----------------------------------------------------------------------------------------------- readout
def load_head(ctx: Ctx, arm: str, s: int, f: int):
    rd = ctx.out / "train" / f"{arm}_s{s}_f{f}"; man = json.loads((rd / "run_manifest.json").read_text()); com = json.loads((ctx.out / "weights_committed.json").read_text())
    if sha256_file(rd / "final.pt") != man["final_sha256"] or com.get(f"{arm}_s{s}_f{f}") != man["final_sha256"]:
        raise SystemExit(f"{arm}_s{s}_f{f}: weights not committed or changed")
    g = tpe.head(f); g.load_state_dict(torch.load(rd / "final.pt", map_location="cpu", weights_only=True)); return g.eval()


def a5c_extras(D: dict, pred: np.ndarray) -> tuple[dict, set]:
    """Correct A4C kept, confusion of all true A5C / A4C runs, and the A5C-coded sources of the S1 runs this prediction fixes."""
    y, b, M = D["pk"]["labels"], D["pk"]["base"], D["M"]; ok4 = (y == A4C) & (b == A4C); fx = np.flatnonzero(M["S1"] & (pred == A5C)); recs = {v for i in fx for v in D["true_src"][i]}
    conf = {name: {"runs": int((y == k).sum()), "recall": float((pred[y == k] == k).mean()), "pred_counts": {str(c): int(((y == k) & (pred == c)).sum()) for c in range(5)}} for name, k in (("true_A5C", A5C), ("true_A4C", A4C))}
    return {"correct_A4C_kept": {"runs": int(ok4.sum()), "kept": int((ok4 & (pred == A4C)).sum())}, "confusion": conf,
            "S1_fixed_sources": {"runs": int(len(fx)), "true_class_recordings": len(recs), "sessions": len({session_of(v) for v in recs}), "concentration": d32.concentration([(D, int(i)) for i in fx], "true_src", None)}}, recs


def ceil_pct(pct: int, n: int) -> int:
    return -(-pct * n // 100)


def readout(ctx: Ctx, s: int) -> str:
    name = f"readout_s{s}.json"
    if (ctx.out / name).exists():
        raise SystemExit("this seed's readout exists; it runs once")
    X = feats(ctx, s, committed=True); a8 = rsh.a08_ctx(); data = t31.vpr.training_data(ctx.a31.a15.a10.a07p); rec = {"seed": s, "cells": {}}; acc = {m: [] for m in ("Base", *ARMS)}; ys, bs = [], []
    cov = {a: [] for a in ARMS}; s1recs = {a: [] for a in ARMS}; old = json.loads((ROOT / "docs/results/2026-10-25_a5c_source_external_evidence/readout.json").read_text())["sequences"]
    a38 = json.loads((ROOT / "docs/results/2026-10-26_predicted_run_context/readout_s0.json").read_text())["cells"] if s == 0 else None; s1_all = []
    for h in FOLDS:
        f = 1 - h; D = d32.fold_data(ctx.d, s, h); D["M"] = d32.set_masks(D); off = X["offset"][h]; P = {"seqs": D["seqs"], "n_of": dict(data.n), "label_of": dict(data.label)}; ev = bc.evalset(P, list(range(len(D["seqs"]))), f"heldout_f{h}")
        cell = {}
        with torch.no_grad():
            for arm in ARMS:
                z = load_head(ctx, arm, s, f)(X["q"], X["e224"], X[arm]); t = tables(z, D, off, a8.cfg, P, ev); p = t.pop("pred"); acc[arm].append(p); cell[arm] = t
                cell[arm]["coverage"], sets = t37.coverage(D, D["pk"]["labels"], D["pk"]["base"], p); cov[arm].append(sets); cell[arm]["boundaries"] = t38.boundaries(z, D, off)
                ex, recs = a5c_extras(D, p); cell[arm].update(ex); s1recs[arm].append(recs)
            t = tables(X["q"], D, off, a8.cfg, P, ev); p = t.pop("pred"); acc["Base"].append(p); cell["Base"] = t; cell["Base"]["boundaries"] = t38.boundaries(X["q"], D, off); cell["Base"].update(a5c_extras(D, p)[0])
        s1 = np.flatnonzero(D["M"]["S1"]); s1_all.append(D); cell["S1_denominator"] = {"runs": int(len(s1)), **d32.concentration([(D, int(i)) for i in s1], "true_src", None)}
        ob = old[f"s{s}_heldout_f{h}"]["A08 Base"]; cell["Base_reproduced"] = {"A32_row": cell["Base"]["sequence_row"] == ob["sequence_row"] and cell["Base"]["counts"] == ob["counts"]}
        if a38 is not None:
            js = lambda x: json.loads(json.dumps(x, default=float))  # noqa: E731
            cell["Base_reproduced"]["A38_Base_cell"] = all(js(cell["Base"][k]) == a38[f"heldout_f{h}"]["Base"][k] for k in a38[f"heldout_f{h}"]["Base"])
        mans = {arm: json.loads((ctx.out / "train" / f"{arm}_s{s}_f{f}" / "run_manifest.json").read_text()) for arm in ARMS}; cell["fit"] = {arm: {k: mans[arm][k] for k in ("fit_before", "fit_after")} for arm in ARMS}
        row = lambda m_: cell[m_]["sequence_row"]  # noqa: E731
        cell["comparisons"] = {f"{a} - {bb}": {"net_fixed": cell[a]["runs"]["all"]["net_fixed"] - (cell[bb]["runs"]["all"]["net_fixed"] if bb != "Base" else 0), "S1_fixed": cell[a]["S1_fixed"]["fixed"] - cell[bb]["S1_fixed"]["fixed"],
                                               "correct_A5C_kept": cell[a]["correct_A5C_kept"]["kept"] - cell[bb]["correct_A5C_kept"]["kept"], "correct_A4C_kept": cell[a]["correct_A4C_kept"]["kept"] - cell[bb]["correct_A4C_kept"]["kept"],
                                               **{k: row(a)[k] - row(bb)[k] for k in ("segment_f1_50_multi", "native_macro_f1", "boundary_recall_multi", "boundary_precision_multi")},
                                               **{k: cell[a]["counts"][k] - cell[bb]["counts"][k] for k in ("fp", "fp_short", "fn", "fn_short")}} for a, bb in (("P", "G"), ("P", "Base"), ("G", "Base"))}
        rec["cells"][f"heldout_f{h}"] = cell; ys.append(D["pk"]["labels"]); bs.append(D["pk"]["base"])
    y, b = np.concatenate(ys), np.concatenate(bs); wr = b != y; rec["oof"] = {}
    for m_ in ARMS:
        p = np.concatenate(acc[m_]); fx, br = int((wr & (p == y)).sum()), int((~wr & (p != y)).sum()); un = {k: set().union(*[c[k] for c in cov[m_]]) for k in ("fixed", "broken")}
        rec["oof"][m_] = {"runs": int(len(y)), "fixed": fx, "broken": br, "net_fixed": fx - br, **{f"{k}_recordings": len(v) for k, v in un.items()}, **{f"{k}_sessions": len({session_of(x) for x in v}) for k, v in un.items()},
                          "S1_fixed_true_class_recordings": len(set().union(*s1recs[m_]))}
    c = rec["cells"]; C = lambda h: c[f"heldout_f{h}"]  # noqa: E731
    tot = lambda m_, k, sub: sum(C(h)[m_][k][sub] for h in FOLDS)  # noqa: E731
    d_ = lambda a, bb, h, k: C(h)[a]["sequence_row"][k] - C(h)[bb]["sequence_row"][k]  # noqa: E731
    net = lambda a, h: C(h)[a]["runs"]["all"]["net_fixed"]  # noqa: E731
    n1, n5, n4 = tot("P", "S1_fixed", "runs"), tot("P", "correct_A5C_kept", "runs"), tot("P", "correct_A4C_kept", "runs"); s1P, s1G = tot("P", "S1_fixed", "fixed"), tot("G", "S1_fixed", "fixed")
    k5P, k5G, k4P, k4G = tot("P", "correct_A5C_kept", "kept"), tot("G", "correct_A5C_kept", "kept"), tot("P", "correct_A4C_kept", "kept"), tot("G", "correct_A4C_kept", "kept")
    recP = [len(r) for r in s1recs["P"]]; recP_all = len(set().union(*s1recs["P"]))
    rule = {"target_each_fold_P_minus_G_S1_ge_1": all(C(h)["P"]["S1_fixed"]["fixed"] - C(h)["G"]["S1_fixed"]["fixed"] >= 1 for h in FOLDS), "target_sum_P_minus_G_S1_ge_5": s1P - s1G >= 5,
            "S1_P": f"{s1P}/{n1}", "S1_G": f"{s1G}/{n1}", "S1_needed": ceil_pct(10, n1), "signal_S1_runs_ok": s1P >= ceil_pct(10, n1),
            "S1_P_true_class_recordings_by_fold": recP, "S1_P_true_class_recordings_total": recP_all, "signal_recordings_ok": recP_all >= 3 and all(r >= 1 for r in recP),
            "correct_A5C_P": f"{k5P}/{n5}", "correct_A5C_G": f"{k5G}/{n5}", "A5C_needed": ceil_pct(98, n5), "retain_A5C_ok": k5P >= ceil_pct(98, n5) and k5P >= k5G - 3,
            "correct_A4C_P": f"{k4P}/{n4}", "correct_A4C_G": f"{k4G}/{n4}", "A4C_needed": ceil_pct(98, n4), "retain_A4C_ok": k4P >= ceil_pct(98, n4) and k4P >= k4G - 3,
            "net_P_minus_Base_each_fold_ge_0": all(net("P", h) >= 0 for h in FOLDS), "net_P_minus_G_each_fold_ge_0": all(net("P", h) - net("G", h) >= 0 for h in FOLDS),
            "dF_P_minus_Base_each_fold_ge_0": all(d_("P", "Base", h, "segment_f1_50_multi") >= 0 for h in FOLDS), "dF_P_minus_G_each_fold_ge_0": all(d_("P", "G", h, "segment_f1_50_multi") >= 0 for h in FOLDS),
            "native_P_minus_Base_each_fold_ge_-0.003": all(d_("P", "Base", h, "native_macro_f1") >= -0.003 for h in FOLDS), "boundary_recall_P_minus_Base_each_fold_ge_-0.005": all(d_("P", "Base", h, "boundary_recall_multi") >= -0.005 for h in FOLDS),
            "missed_short_P_le_Base_plus_3": tot("P", "counts", "fn_short") <= tot("Base", "counts", "fn_short") + 3}
    rule["met"] = all(v for v in rule.values() if isinstance(v, bool)); rec["seed2_rule"] = rule
    rec["values"] = {"P_minus_G_S1": [C(h)["P"]["S1_fixed"]["fixed"] - C(h)["G"]["S1_fixed"]["fixed"] for h in FOLDS], "P_minus_G_net": [net("P", h) - net("G", h) for h in FOLDS], "P_minus_Base_net": [net("P", h) for h in FOLDS],
                     "P_minus_G_dF": [d_("P", "G", h, "segment_f1_50_multi") for h in FOLDS], "P_minus_Base_dF": [d_("P", "Base", h, "segment_f1_50_multi") for h in FOLDS],
                     "P_minus_Base_native": [d_("P", "Base", h, "native_macro_f1") for h in FOLDS], "P_minus_Base_boundary_recall": [d_("P", "Base", h, "boundary_recall_multi") for h in FOLDS],
                     "missed_short": {m_: tot(m_, "counts", "fn_short") for m_ in ("Base", *ARMS)}}
    rec["summary"] = {f"heldout_f{h}": {m_: {"net": net(m_, h) if m_ != "Base" else 0, "F1": C(h)[m_]["sequence_row"]["segment_f1_50_multi"], "S1": C(h)[m_]["S1_fixed"]["fixed"], "A5C_kept": C(h)[m_]["correct_A5C_kept"]["kept"],
                                                  "A4C_kept": C(h)[m_]["correct_A4C_kept"]["kept"]} for m_ in ("Base", *ARMS)} for h in FOLDS}
    fr = json.loads((ctx.out / "feats" / f"record_s{s}.json").read_text()); pr = json.loads((ctx.a31.out / "cache" / f"prefix_s{s}.json").read_text()); nw = sum(fr[str(f)]["windows"] for f in FOLDS)
    ck = json.loads((ctx.out / "checks.json").read_text())["timing"]; g = load_head(ctx, "P", s, 0); torch.set_num_threads(1); t0 = time.time()
    with torch.no_grad():
        g(X["q"], X["e224"], X["P"])
    rec["deployment_cost"] = {"forwards": "one Base 224 forward per window; eG / eP come from its final tokens (no second encoder pass)", "parameters": "A08 34.6M shared + 32,965 head",
                              "gpu_ms_per_window_224_A31_pass": 1e3 * pr["gpu_seconds"] / sum(pr["rows"].values()), "gpu_ms_per_window_token_path_this_round": 1e3 * fr["gpu_seconds"] / nw, "cpu_decode_ms_per_window": 1e3 * fr["decode_seconds"] / nw,
                              "check_extra_ms_per_window_hook_projection_pooling": ck["extra_ms_per_window_hook_projection_pooling"], "head_cpu_ms_per_window": 1e3 * (time.time() - t0) / len(X["q"]),
                              "note": "the A31 pass also wrote its prefix cache; the check's extra time is token path minus plain embed on the same batches (both include the forward)"}
    publish(ctx, name, rec); print(json.dumps({"summary": rec["summary"], "oof": rec["oof"], "rule": rule, "values": rec["values"], "Base_reproduced": {h: rec["cells"][h]["Base_reproduced"] for h in rec["cells"]}}, indent=1)); return f"rule met: {rule['met']}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True); ap.add_argument("--stage", choices=("check", "extract", "train", "freeze", "readout"), required=True); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6); ap.add_argument("--commit", default=""); ap.add_argument("--manifest-sha", default="")
    a = ap.parse_args(); a.set = []; ctx = Ctx(load_cfg(a), a.config); ctx.out.mkdir(parents=True, exist_ok=True)
    if a.stage == "train":
        return train(ctx, a.seed)
    if a.stage in ("readout", "freeze"):
        fn = {"readout": lambda: readout(ctx, a.seed), "freeze": lambda: freeze(ctx, a.seed, a.commit, a.manifest_sha)}[a.stage]
        return staged(ctx, f"{a.stage} s{a.seed} (CPU)", "cpu", fn)
    t31.vsh.determinism(); device = torch.device("cuda")
    staged(ctx, f"{a.stage}{' s' + str(a.seed) if a.stage == 'extract' else ''} (GPU job wall time)", "gpu", {"check": lambda: check(ctx, device, a.workers), "extract": lambda: extract(ctx, a.seed, device, a.workers)}[a.stage])


if __name__ == "__main__":
    main()
