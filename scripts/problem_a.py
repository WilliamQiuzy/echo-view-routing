#!/usr/bin/env python
"""Problem A (docs/research_plan.md): extra cuts at a fixed recall of different-label joins, for the phase method,
its two same-budget controls and the existing temporal baselines. Exploratory: the validation and test streams have
already informed research decisions.

--stage periods  For every validation/test stream (native clips, two-fragment and 4-8-fragment streams rebuilt with the
                 ladder's recipes from cache_v2), estimate the cardiac period from the stream's own 30-fps frames only
                 (echo_routing/realvideo/phase_change.local_period) and save <out>/periods/<stream_id>.npz.
--stage eval     Decode every stream with every method over its pre-fixed grid. On validation, choose the setting with
                 the fewest extra cuts per minute in native clips among those whose different-label join recall is at
                 least 0.95 (0.5 s tolerance); freeze it and report test. Writes <out>/grid.csv (trade-off curves),
                 <out>/operating_points.json and <out>/summary.json.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts._common import base_parser, load_cfg  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

from echo_routing.compose.banks import multi_bank, native_bank, pair_bank, render_bank  # noqa: E402
from echo_routing.config import load_paths  # noqa: E402
from echo_routing.features.cache import load_video_features, read_index, variant_dir  # noqa: E402
from echo_routing.ingest.frames import list_frame_paths  # noqa: E402
from echo_routing.realvideo.phase_change import (change_score, cut_counts, decode_cuts, local_period,  # noqa: E402
                                                 percentile_table, to_percentile)
from echo_routing.realvideo.screening import patient_of  # noqa: E402
from echo_routing.temporal.baselines.b1_smoothing import hysteresis  # noqa: E402
from echo_routing.temporal.baselines.registry import decode_with_prob  # noqa: E402
from echo_routing.temporal.segments import boundaries_from_labels  # noqa: E402
from echo_routing.temporal.windowed import frame_indices_30fps, stream_plan  # noqa: E402

FPS, HZ, TOL = 30.0, 10.0, 5          # tolerance 5 samples = 0.5 s
TARGET_RECALL = 0.95
GRIDS = {
    "B1 smoothing": [("min_run", m) for m in (1, 3, 5, 7, 9, 11, 15, 21)],
    "B2 HMM": [("stay", s) for s in (0.9, 0.95, 0.98, 0.99, 0.995, 0.998, 0.999, 0.9995)],
    "B3 change points": [("threshold", t) for t in (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5)],
    "B4 MS-TCN": [("min_run", m) for m in (1, 3, 5, 7, 9, 11, 15, 21)],
}
PCT_GRID = (0.9, 0.95, 0.97, 0.98, 0.99, 0.993, 0.995, 0.997, 0.998, 0.999)
K_GRID = (1, 2)
SCRAMBLE_SEEDS = (0, 1, 2)


# ---------------------------------------------------------------- streams
def build_streams(paths, cfg, ckpt: str, recognizer: str) -> tuple[dict[str, list], object]:
    idx = read_index(variant_dir(paths.cache_root, ckpt, "orig"))
    cache: dict = {}

    def loader(vid: str, variant: str):
        if (vid, variant) not in cache:
            cache[(vid, variant)] = load_video_features(variant_dir(paths.cache_root, ckpt, variant) / f"{vid}.npz")
        return cache[(vid, variant)]

    out = {}
    for split in ("validation", "test"):
        for kind, bank in (("native", native_bank(idx, split)), ("pairs", pair_bank(idx, split, cfg)), ("multi", multi_bank(idx, split, cfg))):
            streams = render_bank(bank, loader)
            if recognizer == "echoprime":
                d = paths.cache_root / "echoprime_windowed" / ckpt
                streams = [_with_prob(s, np.load(d / f"{s.recipe.recipe_id}.npz")["prob"]) for s in streams]
            out[f"{split}/{kind}"] = streams
    return out, loader


def _with_prob(s, prob: np.ndarray):
    from dataclasses import replace
    return replace(s, prob=np.asarray(prob, dtype=float))


# ---------------------------------------------------------------- stage 1: periods from the stream's own frames
_PATHS: dict[str, list] = {}


def _grey(images_root: str, vid: str, fi: int, gamma: float) -> np.ndarray:
    if vid not in _PATHS:
        _PATHS[vid] = list_frame_paths(Path(images_root) / vid)
    p = _PATHS[vid][min(fi, len(_PATHS[vid]) - 1)]
    with Image.open(p) as im:
        g = im.convert("L")
        if abs(gamma - 1.0) > 1e-9:
            g = g.point([int(255 * ((i / 255.0) ** gamma) + 0.5) for i in range(256)])
        return np.asarray(g.reduce(2), dtype=np.float32)


def _period_job(args: tuple[str, str, list[tuple[str, int, float]], str]) -> str:
    sid, images_root, frames, out = args
    grey = np.stack([_grey(images_root, v, fi, g) for v, fi, g in frames])
    per_frames = local_period(grey, FPS)
    n = len(frames) // 3
    per_samples = per_frames[::3][:n] / 3.0
    np.savez_compressed(Path(out) / f"{sid}.npz", period_frames=per_frames.astype(np.float32), period_samples=per_samples.astype(np.float32))
    return sid


def stage_periods(paths, streams: dict, loader, out: Path, workers: int) -> None:
    (out / "periods").mkdir(parents=True, exist_ok=True)
    jobs = []
    for group in streams.values():
        for s in group:
            sid = s.recipe.recipe_id
            if (out / "periods" / f"{sid}.npz").is_file():
                continue
            refs = stream_plan(s.recipe, loader)
            frames = frame_indices_30fps(refs, np.arange(len(refs)))
            jobs.append((sid, str(paths.ev9v_images), frames, str(out / "periods")))
    print(f"period jobs: {len(jobs)}", flush=True)
    with Pool(workers) as pool:
        for k, _ in enumerate(pool.imap_unordered(_period_job, jobs, chunksize=4), 1):
            if k % 200 == 0:
                print(f"  {k}/{len(jobs)}", flush=True)


# ---------------------------------------------------------------- stage 2: decoding and accounting
def baseline_labels(method: str, key: str, value, s, cfg: dict) -> np.ndarray:
    sid = s.recipe.recipe_id
    if method == "B4 MS-TCN":
        _, p = decode_with_prob("b4", s.prob, cfg, s.feat, sid)
        return hysteresis(p.argmax(1), int(value))
    c = copy.deepcopy(cfg)
    code = {"B1 smoothing": "b1", "B2 HMM": "b2", "B3 change points": "b3"}[method]
    c["decode"][code][key] = value
    return decode_with_prob(code, s.prob, c, s.feat, sid)[0]


def accumulate(rows: list[dict]) -> dict:
    t = pd.DataFrame(rows)
    out = {}
    for name, g in t.groupby("set"):
        minutes = g["n_samples"].sum() / HZ / 60.0
        out[name] = {"recall": g["matched"].sum() / max(1, g["n_diff"].sum()), "n_diff": int(g["n_diff"].sum()),
                     "extra_per_min": g["extra_internal"].sum() / minutes, "minutes": minutes,
                     "same_join_rate": g["same_cuts"].sum() / max(1, g["n_same"].sum()),
                     "double_cut_rate": g["extra_near"].sum() / max(1, g["n_diff"].sum()),
                     "loc_median_s": float(np.median(np.concatenate([np.array(e) for e in g["loc"]]) / HZ)) if g["matched"].sum() else None}
    con = t[t["set"] != "native"]
    out["constructed"] = {"recall": con["matched"].sum() / max(1, con["n_diff"].sum()), "n_diff": int(con["n_diff"].sum()),
                          "extra_per_min": con["extra_internal"].sum() / (con["n_samples"].sum() / HZ / 60.0)}
    return out


def row_of(sid: str, kind: str, labels: np.ndarray, s, avail: np.ndarray | None) -> dict:
    c = cut_counts(labels, s.joins, s.semantic_boundaries, TOL)
    r = {"sid": sid, "set": kind, "n_diff": c.n_diff_joins, "matched": c.matched, "n_same": c.n_same_joins,
         "same_cuts": c.same_join_cuts, "extra_internal": c.extra_internal, "extra_near": c.extra_near_join,
         "loc": list(c.loc_error_samples), "n_samples": c.n_samples}
    if avail is not None:  # where the internal extra cuts and the joins fall: period available or fallback
        cuts = boundaries_from_labels(labels); far = [x for x in cuts if all(abs(x - j) > TOL for j in s.joins)]
        r.update(extra_avail=sum(1 for x in far if avail[x]), extra_fallback=sum(1 for x in far if not avail[x]),
                 samples_avail=int(avail.sum()))
    return r


def main() -> None:
    ap = base_parser("Problem A")
    ap.add_argument("--stage", choices=["periods", "eval"], required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt-hash", default="79f4a41af6e3"); ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--recognizer", choices=["b0", "echoprime"], default="b0")
    ap.add_argument("--candidates", default="runs/_demo/realvideo/screen_ev9v.csv", help="round-1 screening table (frozen 34 sustained candidates)")
    args = ap.parse_args(); cfg = load_cfg(args); paths = load_paths(); out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    cfg.setdefault("mstcn_official", {})["pred_dir"] = str(paths.cache_root / "mstcn_official" / args.ckpt_hash)
    streams, loader = build_streams(paths, cfg, args.ckpt_hash, args.recognizer)
    print({k: len(v) for k, v in streams.items()}, flush=True)
    if args.stage == "periods":
        stage_periods(paths, streams, loader, out, args.workers); return

    per = {s.recipe.recipe_id: np.load(out / "periods" / f"{s.recipe.recipe_id}.npz")["period_samples"].astype(float)
           for g in streams.values() for s in g}
    methods = {m: g for m, g in GRIDS.items() if not (args.recognizer == "echoprime" and m == "B4 MS-TCN")}
    rows = []  # one row per (method, setting, split, stream)

    def add(method: str, setting: str, split: str, kind: str, s, labels, avail=None) -> None:
        rows.append({"method": method, "setting": setting, "split": split, **row_of(s.recipe.recipe_id, kind, labels, s, avail)})

    for method, grid in methods.items():
        for key, value in grid:
            for name, group in streams.items():
                split, kind = name.split("/")
                for s in group:
                    add(method, f"{key}={value}", split, kind, s, baseline_labels(method, key, value, s, cfg))
        print("done", method, flush=True)

    variants = [("phase", 0), ("pool", 0)] + [("scramble", sd) for sd in SCRAMBLE_SEEDS]
    for mode, seed in variants:
        for k in K_GRID:
            scores = {s.recipe.recipe_id: change_score(s.prob, per[s.recipe.recipe_id], mode, k, seed) for g in streams.values() for s in g}
            ref = [scores[s.recipe.recipe_id] for s in streams["validation/native"]]
            tab_a = percentile_table(np.concatenate([sc[av] for sc, av in ref])); tab_f = percentile_table(np.concatenate([sc[~av] for sc, av in ref]))
            pct = {sid: np.where(av, to_percentile(sc, tab_a), to_percentile(sc, tab_f)) for sid, (sc, av) in scores.items()}
            name_m = f"{mode}" if mode != "scramble" else f"scramble s{seed}"
            for thr in PCT_GRID:
                for name, group in streams.items():
                    split, kind = name.split("/")
                    for s in group:
                        sid = s.recipe.recipe_id
                        add(name_m, f"K={k} pct={thr}", split, kind, s, decode_cuts(s.prob, pct[sid], thr), scores[sid][1])
            print("done", name_m, "K", k, flush=True)

    df = pd.DataFrame(rows)
    df.to_pickle(out / f"rows_{args.recognizer}.pkl")
    curves = []
    for (method, setting, split), g in df.groupby(["method", "setting", "split"]):
        a = accumulate(g.to_dict("records"))
        curves.append({"method": method, "setting": setting, "split": split, "recall": a["constructed"]["recall"],
                       "extra_native_per_min": a["native"]["extra_per_min"], "extra_constructed_per_min": a["constructed"]["extra_per_min"],
                       "same_join_rate_pairs": a.get("pairs", {}).get("same_join_rate"), "same_join_rate_multi": a.get("multi", {}).get("same_join_rate"),
                       "double_cut_rate": a.get("multi", {}).get("double_cut_rate"), "loc_median_s_multi": a.get("multi", {}).get("loc_median_s")})
    curves = pd.DataFrame(curves); curves.to_csv(out / f"grid_{args.recognizer}.csv", index=False)

    ops = {}
    for method, g in curves[curves["split"] == "validation"].groupby("method"):
        ok = g[g["recall"] >= TARGET_RECALL]
        if len(ok):
            pick = ok.sort_values(["extra_native_per_min", "recall"], ascending=[True, False]).iloc[0]; met = True
        else:
            pick = g.sort_values("recall", ascending=False).iloc[0]; met = False
        test = curves[(curves["method"] == method) & (curves["setting"] == pick["setting"]) & (curves["split"] == "test")].iloc[0]
        ops[method] = {"setting": pick["setting"], "target_met_on_validation": met, "validation": pick.drop(["method", "setting", "split"]).to_dict(),
                       "test": test.drop(["method", "setting", "split"]).to_dict()}
    (out / f"operating_points_{args.recognizer}.json").write_text(json.dumps(ops, indent=1, default=float))
    summarize(df, ops, streams, per, out, args)


def _boot_rate(units: pd.DataFrame, num: str, den: str, cluster: str, n_boot: int, rng) -> np.ndarray:
    groups = units[cluster].unique(); g = units.groupby(cluster)[[num, den]].sum()
    draws = []
    for _ in range(n_boot):
        b = g.loc[rng.choice(groups, size=len(groups), replace=True)]
        draws.append(b[num].sum() / b[den].sum())
    return np.array(draws)


def summarize(df: pd.DataFrame, ops: dict, streams: dict, per: dict, out: Path, args) -> None:
    rng = np.random.default_rng(0)
    test = df[df["split"] == "test"]
    chosen = {m: test[(test["method"] == m) & (test["setting"] == o["setting"])].copy() for m, o in ops.items()}
    for m, t in chosen.items():
        t["minutes"] = t["n_samples"] / HZ / 60.0
        t["group"] = [patient_of(sid) if kind == "native" else sid for sid, kind in zip(t["sid"], t["set"])]
    res: dict = {"recognizer": args.recognizer, "operating_points": {m: o["setting"] for m, o in ops.items()}}
    # coverage of the period on the actual inputs
    cov = {}
    for name, group in streams.items():
        tot = sum(len(per[s.recipe.recipe_id]) for s in group); av = sum(int(np.isfinite(per[s.recipe.recipe_id]).sum()) for s in group)
        cov[name] = round(av / tot, 3)
    res["coverage"] = cov
    # paired, cluster-resampled differences on test natives (presumed patient groups) and constructed recall (streams)
    base = "phase"
    diffs = {}
    for other in [m for m in chosen if m != base]:
        a, b = chosen[base], chosen[other]
        na, nb = a[a["set"] == "native"].set_index("sid"), b[b["set"] == "native"].set_index("sid")
        units = pd.DataFrame({"group": na["group"], "ea": na["extra_internal"], "eb": nb.loc[na.index, "extra_internal"], "min": na["minutes"]})
        gsum = units.groupby("group")[["ea", "eb", "min"]].sum(); groups = gsum.index.to_numpy(); d = []
        for _ in range(2000):
            bs = gsum.loc[rng.choice(groups, size=len(groups), replace=True)]
            d.append(bs["ea"].sum() / bs["min"].sum() - bs["eb"].sum() / bs["min"].sum())
        ca, cb = a[a["set"] != "native"].set_index("sid"), b[b["set"] != "native"].set_index("sid")
        su = pd.DataFrame({"ma": ca["matched"], "mb": cb.loc[ca.index, "matched"], "n": ca["n_diff"]})
        r = []
        for _ in range(2000):
            bs = su.iloc[rng.integers(0, len(su), len(su))]
            r.append(bs["ma"].sum() / bs["n"].sum() - bs["mb"].sum() / bs["n"].sum())
        diffs[f"{base} minus {other}"] = {"extra_native_per_min": [round(float(np.mean(d)), 3), round(float(np.percentile(d, 2.5)), 3), round(float(np.percentile(d, 97.5)), 3)],
                                          "recall_constructed": [round(float(np.mean(r)), 3), round(float(np.percentile(r, 2.5)), 3), round(float(np.percentile(r, 97.5)), 3)]}
    res["differences_test"] = diffs
    # available vs fallback for the phase-family methods on test natives
    sub = {}
    for m, t in chosen.items():
        if "extra_avail" in t and t["extra_avail"].notna().any():
            nat = t[t["set"] == "native"]
            ma = nat["samples_avail"].sum() / HZ / 60.0; mf = (nat["n_samples"].sum() - nat["samples_avail"].sum()) / HZ / 60.0
            sub[m] = {"extra_per_min_available": round(float(nat["extra_avail"].sum() / max(ma, 1e-9)), 3),
                      "extra_per_min_fallback": round(float(nat["extra_fallback"].sum() / max(mf, 1e-9)), 3),
                      "minutes_available": round(float(ma), 2), "minutes_fallback": round(float(mf), 2)}
    res["available_vs_fallback_test_natives"] = sub
    # sensitivity: without the frozen 34 sustained candidates
    cand = pd.read_csv(args.candidates)
    sus = set(cand[cand["stratum"].isin(["sustained", "mixed"]) & ~cand["discovery"]]["video_id"])
    (out / "frozen_sustained_candidates.json").write_text(json.dumps(sorted(sus)))
    ids_with = {s.recipe.recipe_id for g in streams.values() for s in g if any(f.video_id in sus for f in s.recipe.fragments)}
    sens = {}
    for m, t in chosen.items():
        k = t[~t["sid"].isin(ids_with)]
        a = accumulate(k.to_dict("records"))
        sens[m] = {"recall": round(float(a["constructed"]["recall"]), 3), "extra_native_per_min": round(float(a["native"]["extra_per_min"]), 3)}
    res["sensitivity_without_34"] = {"n_candidates": len(sus), "streams_dropped": len(ids_with), "test": sens}
    (out / f"summary_{args.recognizer}.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
