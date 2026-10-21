#!/usr/bin/env python
"""A19 (configs/experiments/video_b3_coverage_v1.yaml): frozen window logits of nine video models on A15's pool, B3
choices on the small-fit and large-fit pools, and one evaluation of the committed choices on the check pool and the
development halves. No model is trained.

  --probe                     GPU: one chunk of the seven uncached models; budget projection
  --infer                     GPU: chunked frozen inference of the seven models over the pool (A08 s0/s2 reused from A15)
  --check                     pool, cache parity (small GPU batch), the selector against old committed choices, states
  --choose                    CPU: 18 choices (9 models x small/large fit); writes choices.json
  --evaluate --choices-sha S  CPU: the committed choices on the check pool and both development halves, once
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
import scripts.report_center_supervision as rcs  # noqa: E402
import scripts.report_raw_video_context as rr  # noqa: E402
import scripts.report_router_data as rrd  # noqa: E402
import scripts.report_router_trajectory as rrt  # noqa: E402
import scripts.report_segment_objective as rso  # noqa: E402
import scripts.report_video_switch_head as rsh  # noqa: E402
import scripts.report_video_within_view as rw  # noqa: E402
import scripts.segment_objective_tables as st  # noqa: E402
import scripts.train_ensemble_teacher as tet  # noqa: E402
import scripts.train_router_data as trd  # noqa: E402
import scripts.train_router_trajectory as trt  # noqa: E402
import scripts.train_video_switch_head as vsh  # noqa: E402
import scripts.trained_token_readout as ttr  # noqa: E402
import scripts.prepare_video_sequence_objective as vpr  # noqa: E402
from scripts._common import load_cfg  # noqa: E402
from scripts.calibrate_simple_decoders import grid as simple_grid  # noqa: E402
from scripts.ev9v_complementarity import decode  # noqa: E402
from scripts.prepare_fold_output_correction import assert_no_ids  # noqa: E402
from scripts.train_joint_visual_temporal import FramePaths  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

import echo_routing.evaluate.dev_metrics as dm  # noqa: E402
from echo_routing.evaluate.complementarity import select_setting, sha256_file  # noqa: E402
from echo_routing.evaluate.failure_units import session_of  # noqa: E402
from echo_routing.temporal.dev_data import EvalSet, Seq  # noqa: E402

A15_CONFIG = "configs/experiments/video_router_data_v1.yaml"
A17_CONFIG = "configs/experiments/video_router_trajectory_v1.yaml"
KEYS = [f"{g}_s{s}" for g in ("A08", "R", "E") for s in (0, 1, 2)]
REUSED = {"A08_s0": 0, "A08_s2": 2}                                  # A15's cached windows of these two models
CHUNK, HALVES, POOLS = 150, ("selection", "calibration"), ("small", "large")
now, write_json, ledger_append, ledger_hours = tet.now, tet.write_json, tet.ledger_append, tet.ledger_hours


class Ctx:
    def __init__(self, cfg: dict, config: str) -> None:
        self.cfg, self.config, self.vb = cfg, config, cfg["video_b3_coverage"]
        self.a15 = trd.Ctx(load_cfg(SimpleNamespace(config=A15_CONFIG, set=[])), A15_CONFIG)
        o = self.vb["outputs"]; self.out, self.results = ROOT / o["root"], ROOT / o["results_dir"]; self.ledger = ROOT / self.vb["budget"]["ledger"]

    def sha(self, key: str) -> str:
        g, s = key.split("_s"); return self.vb["models"][g][int(s)]

    def weights(self, key: str) -> Path:
        g, s = key.split("_s")
        if g == "A08":
            return ROOT / "models_frozen/video_within_view_v1" / f"all_s{s}" / "final.pt"
        return ROOT / ("models_frozen/video_switch_head_v1" if g == "R" and s in ("0", "2") else "models_frozen/video_ensemble_teacher_v1") / key / "final.pt"


def load_model(ctx: Ctx, key: str, device):
    p = ctx.weights(key)
    if sha256_file(p) != ctx.sha(key):
        raise SystemExit(f"{key}: the weights differ from their pin")
    st_ = torch.load(p, map_location="cpu", weights_only=True); st_ = st_["model"] if isinstance(st_, dict) and "model" in st_ and isinstance(st_["model"], dict) else st_
    m = vsh.make_model(st_, device).eval()
    for q in m.parameters():
        q.requires_grad_(False)
    return m


# ----------------------------------------------------------------------------------------------- pool
def pool(ctx: Ctx) -> dict:
    P = trd.pools(ctx.a15); prep = json.loads((ctx.a15.out / "preparation.json").read_text())
    if P["digest"] != prep["pool_digest"] or sha256_file(ctx.a15.out / "pool_manifest.json") != "5ab124d48f48f826fd0b52542cf8cf1aa7f97e0f6b82327358406cf574a14050":
        raise SystemExit("the rebuilt pool differs from A15's pinned pool")
    seqs = P["seqs"]; refs = rw.half_refs(seqs); at = {r: i for i, r in enumerate(refs)}; W = rr.half_windows(seqs, at, 5)
    return {**P, "W": {k: W[k] for k in ("seq", "centre", "marker")}, "n_refs": len(refs)}


def members(P: dict, name: str) -> list[int]:
    g, sm = P["group"], P["small"]
    sel = {"large": g == "fit", "small": (g == "fit") & sm, "check": g == "check"}[name]
    return [int(i) for i in np.flatnonzero(sel)]


def evalset(P: dict, idx: list[int], name: str) -> EvalSet:
    n_of, label_of = P["n_of"], P["label_of"]; natives, multi = [], []
    for k, i in enumerate(idx):
        q = P["seqs"][i]
        if q["kind"] == "native":
            natives.append(Seq(q["id"], True, None, np.asarray(q["labels"], np.int64), np.zeros(0, np.int64)))
        else:
            d = {"recipe_id": q["id"], "cell": "multi", "fragments": [{"video_id": f[0], "label": int(f[1]), "variant": f[2], "start": int(f[3]), "end": int(f[4])} for f in q["frags"]]}
            s = ttr.meta_stream(d, n_of, label_of, f"{name} stream {k}")
            if not np.array_equal(s.labels, np.asarray(q["labels"])):
                raise SystemExit(f"{name} stream {k}: the stream labels differ from the sequence labels")
            multi.append(s)
    return EvalSet(name, tuple(natives), (), tuple(multi))


def evaluate_pool(cfg: dict, preds: dict, ev: EvalSet) -> tuple[dict, dict]:
    """dev_metrics.evaluate_predictions on a pool without pair streams: the original native and stream tables, and empty
    typed pair tables (pair metrics undefined and unused)."""
    hz, tol = float(cfg["protocol"]["hz"]), float(cfg["selection"]["boundary_tolerance_s"]) * float(cfg["protocol"]["hz"])
    nat = dm.native_table(preds, ev.natives); ms, mf = dm.stream_tables(preds, ev.multi, tol)
    ps = pd.DataFrame({c: pd.Series(dtype="int64") for c in ms.columns}); pf = pd.DataFrame({c: pd.Series(dtype=mf[c].dtype) for c in mf.columns})
    with np.errstate(invalid="ignore", divide="ignore"):
        m = dm.summarize(nat, ms, mf, ps, pf, hz)
    m = {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in m.items()}
    return m, {"native": nat, "multi_seg": ms, "multi_frag": mf, "pairs_seg": ps, "pairs_frag": pf}


def choose_b3(cfg: dict, probs: dict, ev, evaluate_fn) -> dict:
    """report_center_supervision.b3_choose's procedure (grid, decode, select_setting), without its diagnostic fallback."""
    settings = [p for m, p in simple_grid(cfg, [probs[s.seq_id] for s in ev.natives]) if m == "b3"]; recs = []
    for order, params in enumerate(settings):
        m, _ = evaluate_fn(cfg, decode(params, probs), ev); recs.append({"order": order, "params": params, **{k: m[k] for k in rcs.FLAT}})
    sel = select_setting(recs, 0.980, 0.95)
    return {"status": sel["status"], "params": sel["chosen"]["params"] if sel["chosen"] else None, "order": sel["chosen"]["order"] if sel["chosen"] else None,
            "n_feasible": sel["n_feasible"], "n_settings": len(recs), "grid": recs}


# ----------------------------------------------------------------------------------------------- inference (GPU)
def infer_pool(ctx: Ctx, keys: list[str], workers: int, probe: bool = False) -> None:
    t0, started, ok, note, gsec, dsec = time.time(), now(), False, "", 0.0, 0.0
    try:
        P = pool(ctx); seqs = P["seqs"]; device = torch.device("cuda"); tet.determinism(); M = {k: load_model(ctx, k, device) for k in keys}; d0 = {k: tet.state_digest(m) for k, m in M.items()}
        vids = {r[0] for q in seqs for r in q["refs"]}
        if vids - set(prv.allowlist(ctx.a15.a10.rv, ROOT)):
            raise SystemExit("a pool recording lies outside the training allowlist")
        fp = FramePaths(prv.frame_dirs(ctx.a15.a10.paths, ctx.a15.a10.rv, vids)); acc = {k: {"seq": [], "centre": [], "logits": []} for k in keys}; nwin = 0
        chunks = list(range(0, CHUNK if probe else len(seqs), CHUNK)); (ctx.out / "logits").mkdir(parents=True, exist_ok=True)
        for c0 in chunks:
            chunk = seqs[c0:c0 + CHUNK]; refs = rw.half_refs(chunk); at = {r: i for i, r in enumerate(refs)}; W = rr.half_windows(chunk, at, 5)
            orders = np.broadcast_to(np.arange(16, dtype=np.int64), W["s"].shape).copy(); a = time.time(); frames = rr.decode_frames(refs, fp, workers); dsec += time.time() - a
            for k, m in M.items():
                lg, _, sec = rsh.infer(m, W, orders, frames, device); gsec += sec; acc[k]["seq"].append(W["seq"] + c0); acc[k]["centre"].append(W["centre"]); acc[k]["logits"].append(lg.astype(np.float32))
            del frames; nwin += len(W["seq"])
            prog = {"stage": "frozen inference of the seven uncached models (eval, no_grad; no training)", "chunk": c0 // CHUNK + 1, "of": -(-len(seqs) // CHUNK), "windows_done": nwin,
                    "windows_total": int(len(P["W"]["seq"])), "models": keys, "gpu_s": round(gsec, 1), "decode_s": round(dsec, 1), "elapsed_s": round(time.time() - t0, 1)}
            write_json(ctx.out / "progress.json", prog); print(json.dumps(prog), flush=True)
        if any(tet.state_digest(m) != d0[k] for k, m in M.items()):
            raise SystemExit("a frozen model's state changed")
        if probe:
            per = gsec / nwin; proj = {"chunk_windows": nwin, "gpu_seconds_chunk": gsec, "decode_seconds_chunk": dsec, "projected_gpu_hours": per * len(P["W"]["seq"]) / 3600,
                                       "projected_decode_worker_cpu_hours": dsec / nwin * len(P["W"]["seq"]) * workers / 3600, "caps": ctx.vb["budget"], "time": now()}
            write_json(ctx.out / "probe.json", proj); print(json.dumps(proj, indent=1), flush=True)
        else:
            for k in keys:
                seq, cen = np.concatenate(acc[k]["seq"]), np.concatenate(acc[k]["centre"])
                if not (np.array_equal(seq, P["W"]["seq"]) and np.array_equal(cen, P["W"]["centre"])):
                    raise SystemExit(f"{k}: the inferred windows differ from the pool's window plan")
                np.savez(ctx.out / "logits" / f"{k}.npz", seq=seq, centre=cen, logits=np.concatenate(acc[k]["logits"]), weights_sha256=ctx.sha(k), pool_digest=P["digest"])
            write_json(ctx.out / "logits" / "index.json", {k: sha256_file(ctx.out / "logits" / f"{k}.npz") for k in keys})
        ok = True; note = f"{nwin} windows x {len(keys)} models"
    except BaseException as e:  # noqa: BLE001 - recorded, then raised again
        note = f"{type(e).__name__}: {str(e)[:200]}"; raise
    finally:
        total = time.time() - t0; g = min(gsec, total)
        ledger_append(ctx.ledger, f"{'probe' if probe else 'pool inference'} (frozen, GPU inference seconds)", "gpu", started, g, ok, note)
        ledger_append(ctx.ledger, f"{'probe' if probe else 'pool inference'} (CPU: main process outside GPU + {workers - 1} more decode workers)", "cpu", started, (total - g) + dsec * (workers - 1), ok, note)


def pool_logits(ctx: Ctx, P: dict, key: str) -> np.ndarray:
    if key in REUSED:
        s = REUSED[key]; prep = json.loads((ctx.a15.out / "preparation.json").read_text())["seeds"][str(s)]
        with np.load(ctx.a15.out / "features" / f"windows_s{s}.npz") as z:
            if not (np.array_equal(z["seq"], P["W"]["seq"]) and np.array_equal(z["centre"], P["W"]["centre"]) and str(z["start_sha256"]) == ctx.sha(key) and str(z["path_digest"]) == prep["path_digest"]):
                raise SystemExit(f"{key}: A15's cache does not match the pool's windows or the model identity")
            return z["logits"].astype(np.float32)
    idx = json.loads((ctx.out / "logits" / "index.json").read_text()); p = ctx.out / "logits" / f"{key}.npz"
    if sha256_file(p) != idx[key]:
        raise SystemExit(f"{key}: the pool logits differ from their index")
    with np.load(p) as z:
        if not (np.array_equal(z["seq"], P["W"]["seq"]) and np.array_equal(z["centre"], P["W"]["centre"]) and str(z["weights_sha256"]) == ctx.sha(key)):
            raise SystemExit(f"{key}: the pool logits do not match the pool's windows or the model identity")
        return z["logits"].astype(np.float32)


def pool_probs(ctx: Ctx, P: dict, key: str) -> dict:
    lg = pool_logits(ctx, P, key); seqs = [{"id": q["id"], "labels": q["labels"]} for q in P["seqs"]]; return rr.per_sample(P["W"], lg, seqs)


def with_index(ctx: Ctx, P: dict) -> dict:
    data = vpr.training_data(ctx.a15.a10.a07p); P["n_of"], P["label_of"] = dict(data.n), dict(data.label); return P


# ----------------------------------------------------------------------------------------------- choices (CPU)
def choose(ctx: Ctx) -> None:
    t0, started, ok, note = time.time(), now(), False, ""
    try:
        a8 = rsh.a08_ctx(); cfg = a8.cfg; P = with_index(ctx, pool(ctx)); out = {}; ids = {q["id"] for q in P["seqs"] if q["kind"] == "native"}
        evs = {name: evalset(P, members(P, name), name) for name in POOLS}
        for key in KEYS:
            probs = pool_probs(ctx, P, key)
            for name in POOLS:
                idx = members(P, name); sub = {P["seqs"][i]["id"]: probs[P["seqs"][i]["id"]] for i in idx}; ch = choose_b3(cfg, sub, evs[name], evaluate_pool)
                ch["pool"] = {"name": name, "native": len(evs[name].natives), "multi": len(evs[name].multi), "ids_sha256": hashlib.sha256(json.dumps(sorted(sub)).encode()).hexdigest()}
                out[f"{key}|{name}"] = ch; print(f"{key} {name}: {ch['status']} {ch['params']} ({ch['n_feasible']} feasible of {ch['n_settings']})", flush=True)
        rec = {"stage": "choose", "config_sha256": sha256_file(ROOT / ctx.config), "pool_digest": P["digest"], "models": {k: ctx.sha(k) for k in KEYS}, "choices": out,
               "grid": {"source": "A08 configuration decode_grids.b3", "b3": cfg["decode_grids"]["b3"]}, "time": now()}
        flat = json.loads(json.dumps(rec, default=str)); assert_no_ids(flat, ids | {r[0] for q in P["seqs"] for r in q["refs"]}, "choices")
        for p in (ctx.out / "choices.json", ctx.results / "choices.json"):
            p.parent.mkdir(parents=True, exist_ok=True); write_json(p, flat)
        ok = True
    except BaseException as e:  # noqa: BLE001 - recorded, then raised again
        note = f"{type(e).__name__}: {str(e)[:200]}"; raise
    finally:
        ledger_append(ctx.ledger, "choose 18 B3 settings (CPU)", "cpu", started, time.time() - t0, ok, note)


# ----------------------------------------------------------------------------------------------- development outputs
def dev_probs(ctx: Ctx, a8, h: str) -> dict:
    """Saved development logits of all nine models (identity-checked) as per-sample probabilities."""
    ev, seqs = a8.half(h, rso.DevAccess({h})); refs = rw.half_refs(seqs); at = {r: i for i, r in enumerate(refs)}; W = rr.half_windows(seqs, at, 5)
    orders = np.broadcast_to(np.arange(16, dtype=np.int64), W["s"].shape).copy(); a8sum = json.loads((ROOT / "docs/results/2026-10-16_video_within_view/summary.json").read_text())
    pins = rsh.a08_output_pins(); out = {"ev": ev, "seqs": seqs, "probs": {}}
    for key in KEYS:
        g, s = key.split("_s"); s = int(s)
        if g == "A08":
            apath = ROOT / "runs/video_within_view_v1/evaluation" / h / f"all_s{s}.npz"
            if sha256_file(apath) != pins[h][s]:
                raise SystemExit(f"{key} {h}: saved outputs differ from their pin")
            lg, _ = rr.cached_logits(apath, a8sum["runs"][f"all_s{s}"]["identity"], W, orders, seqs)
        else:
            a10 = g == "R" and s in (0, 2); root = ROOT / ("runs/video_switch_head_v1" if a10 else "runs/video_ensemble_teacher_v1")
            man = json.loads((root / "train" / key / "run_manifest.json").read_text())
            if man["final_sha256"] != ctx.sha(key):
                raise SystemExit(f"{key}: the run record differs from the pinned weights")
            lg = rret_logits(root / "evaluation" / h / f"{key}.npz", man["identity"], W)
        if lg is None:
            raise SystemExit(f"{key} {h}: the saved outputs do not match this half's windows")
        out["probs"][key] = rr.per_sample(W, lg.astype(np.float32), seqs)
    return out


def rret_logits(path: Path, ident: str, W: dict):
    with np.load(path) as z:
        return z["logits"].astype(np.float32) if (str(z["identity"]) == ident and np.array_equal(z["seq"], W["seq"]) and np.array_equal(z["centre"], W["centre"])) else None


def old_params(key: str) -> dict:
    g, s = key.split("_s"); s = int(s)
    if g == "A08":
        return json.loads((ROOT / "docs/results/2026-10-16_video_within_view/choices.json").read_text())["choices"][f"all_s{s}"]["params"]
    if g == "R" and s in (0, 2):
        return json.loads((ROOT / "runs/video_switch_head_v1/evaluation/screen/choices.json").read_text())["choices"][f"R_s{s}|b3"]["params"]
    f = "choices_seed1.json" if s == 1 else "choices.json"
    return json.loads((ROOT / "runs/video_ensemble_teacher_v1/evaluation" / f).read_text())["choices"][f"{g} s{s}|b3"]["params"]


# ----------------------------------------------------------------------------------------------- checks
def check(ctx: Ctx, workers: int) -> None:
    t0, started, ok, note, gsec = time.time(), now(), False, "", 0.0; out = {}
    try:
        a8 = rsh.a08_ctx(); cfg = a8.cfg; P = with_index(ctx, pool(ctx)); seqs = P["seqs"]
        out["pool"] = {"sequences": len(seqs), "windows": int(len(P["W"]["seq"])), "large": len(members(P, "large")), "small": len(members(P, "small")), "check": len(members(P, "check")),
                       "small_inside_large": set(members(P, "small")) <= set(members(P, "large")), "counts": P["counts"]}
        fit_v = {r[0] for i in members(P, "large") for r in seqs[i]["refs"]}; chk_v = {r[0] for i in members(P, "check") for r in seqs[i]["refs"]}
        dev_clips = {c for h in HALVES for c in a8.proto()["validation"][h]["clips"]}
        out["disjoint"] = {"fit_check_recordings": not (fit_v & chk_v), "fit_check_sessions": not ({session_of(v) for v in fit_v} & {session_of(v) for v in chk_v}),
                           "no_development_clip": not ((fit_v | chk_v) & dev_clips)}
        evs = {name: evalset(P, members(P, name), name) for name in ("small", "large", "check")}; out["evalsets"] = {n: {"native": len(e.natives), "multi": len(e.multi)} for n, e in evs.items()}
        device = torch.device("cuda"); tet.determinism(); rng = np.random.default_rng([20261021, 19]); par = {}
        for key, s in REUSED.items():
            lg = pool_logits(ctx, P, key); pick = np.sort(rng.choice(len(lg), size=32, replace=False)); m = load_model(ctx, key, device); d0 = tet.state_digest(m)
            wi = [int(i) for i in pick]; qset = sorted({int(P["W"]["seq"][i]) for i in wi}); sub = [seqs[i] for i in qset]; refs = rw.half_refs(sub); at = {r: i for i, r in enumerate(refs)}
            Ws = rr.half_windows(sub, at, 5); loc = {g: k for k, g in enumerate(qset)}
            rows = [int(np.flatnonzero((Ws["seq"] == loc[int(P["W"]["seq"][i])]) & (Ws["centre"] == int(P["W"]["centre"][i])))[0]) for i in wi]
            fp = FramePaths(prv.frame_dirs(ctx.a15.a10.paths, ctx.a15.a10.rv, {r[0] for r in refs})); frames = rr.decode_frames(refs, fp, workers)
            Wp = {"ref": Ws["ref"][rows], "marker": Ws["marker"][rows], "s": Ws["s"][rows]}; orders = np.broadcast_to(np.arange(16, dtype=np.int64), Wp["s"].shape).copy()
            a = time.time(); lgo, _, sec = rsh.infer(m, Wp, orders, frames, device); gsec += sec
            par[key] = {**tet.vsh.vso.parity(lgo, lg[pick], 1e-4), "state_unchanged": tet.state_digest(m) == d0}; del m, frames; torch.cuda.empty_cache()
        out["reused_cache_parity"] = par
        rep = {}; D = dev_probs(ctx, a8, "calibration")                 # old inputs only: re-derives the old committed choices
        for key in KEYS:
            got = choose_b3(cfg, D["probs"][key], D["ev"], rcs.evaluate); want = old_params(key)
            rep[key] = {"reproduces_old_choice": got["params"] == want, "n_feasible": got["n_feasible"]}
        out["selector_reproduces_old_choices"] = rep
        out["grid"] = {"settings": 40, "b3": cfg["decode_grids"]["b3"]}
        out["budget"] = {"gpu_spent": ledger_hours(ctx.ledger, "gpu"), "cpu_spent": ledger_hours(ctx.ledger, "cpu")}
        out["ok"] = bool(out["pool"]["small_inside_large"] and all(out["disjoint"].values()) and all(v["ok"] and v["state_unchanged"] for v in par.values()) and all(v["reproduces_old_choice"] for v in rep.values()))
        ok = out["ok"]
        for p in (ctx.out / "checks.json", ctx.results / "checks.json"):
            p.parent.mkdir(parents=True, exist_ok=True); write_json(p, json.loads(json.dumps(out, default=str)))
        print(json.dumps(out, indent=1, default=str), flush=True)
        if not ok:
            raise SystemExit("a check failed")
    except BaseException as e:  # noqa: BLE001 - recorded, then raised again
        note = f"{type(e).__name__}: {str(e)[:200]}"; raise
    finally:
        total = time.time() - t0; ledger_append(ctx.ledger, "check (GPU parity seconds)", "gpu", started, min(gsec, total), ok, note)
        ledger_append(ctx.ledger, "check (CPU)", "cpu", started, total - min(gsec, total), ok, note)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True); ap.add_argument("--workers", type=int, default=6); ap.add_argument("--choices-sha")
    g = ap.add_mutually_exclusive_group(required=True)
    for f in ("probe", "infer", "check", "choose", "evaluate"):
        g.add_argument(f"--{f}", action="store_true")
    a = ap.parse_args(); a.set = []; ctx = Ctx(load_cfg(a), a.config); todo = [k for k in KEYS if k not in REUSED]
    if a.probe:
        return infer_pool(ctx, todo, a.workers, probe=True)
    if a.infer:
        return infer_pool(ctx, todo, a.workers)
    if a.check:
        return check(ctx, a.workers)
    if a.choose:
        return choose(ctx)
    import scripts.b3_coverage_evaluate as ev_
    ev_.evaluate(ctx, a.choices_sha)


if __name__ == "__main__":
    main()
