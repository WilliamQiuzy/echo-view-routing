#!/usr/bin/env python
"""Component audit of the pooled two-window detector against B3 (docs/research_plan.md, audit round; fixed B0).

Every configuration uses B3's decoding pipeline (echo_routing/realvideo/two_window.py) and changes one component
relative to the previous one:
  A0  B3 as configured: JS score, 0.5 s windows, B3's raw threshold grid (8 values)
  A1a JS, 0.5 s, thresholds at 10 fixed quantiles of validation native-clip scores (re-parameterisation only)
  A1b JS, fixed windows 0.5 / 0.9 / 1.8 s, quantile thresholds                         (window choice)
  A2  TV, fixed windows 0.5 / 0.9 / 1.8 s, quantile thresholds   (score; this is the period-free detector)
  A3  TV, windows K x period, K = 0.5 / 1 / 2 (0.9 s fallback), one quantile map    (period sets the window)
  A4  TV, windows K x period, quantiles taken separately with and without a period  (two-group calibration)
A1b-A4 have the same budget (3 x 10 settings). Operating point: validation recall of different-label joins >= 0.95
within 0.5 s, fewest extra cuts per minute in native clips; frozen for test. Exploratory (the data informed earlier
decisions). Writes <out>/grid.csv, operating_points.json, summary.json and summary.md.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts._common import base_parser, load_cfg  # noqa: E402
from scripts.problem_a import accumulate, build_streams, row_of  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from echo_routing.config import load_paths  # noqa: E402
from echo_routing.realvideo.phase_change import percentile_table, to_percentile  # noqa: E402
from echo_routing.realvideo.screening import patient_of  # noqa: E402
from echo_routing.realvideo.two_window import detect_labels, quantile_thresholds, two_window_score, window_lengths  # noqa: E402

HZ, TARGET = 10.0, 0.95
QUANTILES = (0.9, 0.95, 0.97, 0.98, 0.99, 0.993, 0.995, 0.997, 0.998, 0.999)
B3_RAW = (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5)
FIXED = (5, 9, 18)            # samples per side: 0.5, 0.9, 1.8 s
KS = (0.5, 1.0, 2.0)
CONFIGS = {
    "A0 B3 as configured": {"score": "js", "windows": [("fixed", 5)], "thresholds": "raw"},
    "A1a JS 0.5 s, quantile grid": {"score": "js", "windows": [("fixed", 5)], "thresholds": "quantile"},
    "A1b JS fixed windows": {"score": "js", "windows": [("fixed", w) for w in FIXED], "thresholds": "quantile"},
    "A2 TV fixed windows": {"score": "tv", "windows": [("fixed", w) for w in FIXED], "thresholds": "quantile"},
    "A3 TV period windows, one map": {"score": "tv", "windows": [("period", k) for k in KS], "thresholds": "quantile"},
    "A4 TV period windows, two maps": {"score": "tv", "windows": [("period", k) for k in KS], "thresholds": "two_group"},
}
CONTRASTS = [("A1a JS 0.5 s, quantile grid", "A0 B3 as configured"), ("A1b JS fixed windows", "A1a JS 0.5 s, quantile grid"),
             ("A2 TV fixed windows", "A1b JS fixed windows"), ("A3 TV period windows, one map", "A2 TV fixed windows"),
             ("A4 TV period windows, two maps", "A3 TV period windows, one map"), ("A2 TV fixed windows", "B4 MS-TCN (problem A)")]


def run_config(name: str, spec: dict, streams: dict, per: dict) -> tuple[list[dict], float]:
    rows, t0 = [], time.perf_counter()
    for kind_w, val in spec["windows"]:
        lens, scores, avails = {}, {}, {}
        for g in streams.values():
            for s in g:
                sid = s.recipe.recipe_id
                L, av = (window_lengths(len(s.prob), fixed=val) if kind_w == "fixed"
                         else window_lengths(len(s.prob), period_samples=per[sid], k=val))
                lens[sid], avails[sid] = L, av
                scores[sid] = two_window_score(s.prob, L, spec["score"])
        ref = [sid for sid in (s.recipe.recipe_id for s in streams["validation/native"])]
        if spec["thresholds"] == "raw":
            levels = [(f"thr={t}", t) for t in B3_RAW]; used = scores
        elif spec["thresholds"] == "quantile":
            th = quantile_thresholds(np.concatenate([scores[sid] for sid in ref]), QUANTILES)
            levels = [(f"q={q}", t) for q, t in zip(QUANTILES, th)]; used = scores
        else:  # separate quantile maps for samples with and without a period
            ta = percentile_table(np.concatenate([scores[sid][avails[sid]] for sid in ref]))
            tf = percentile_table(np.concatenate([scores[sid][~avails[sid]] for sid in ref]))
            used = {sid: np.where(avails[sid], to_percentile(sc, ta), to_percentile(sc, tf)) for sid, sc in scores.items()}
            levels = [(f"q={q}", q) for q in QUANTILES]
        for tag, thr in levels:
            for gname, g in streams.items():
                split, kind = gname.split("/")
                for s in g:
                    sid = s.recipe.recipe_id
                    labels = detect_labels(s.prob, used[sid], thr, lens[sid])
                    r = row_of(sid, kind, labels, s, avails[sid] if kind_w == "period" else None)
                    rows.append({"method": name, "setting": f"{kind_w}={val} {tag}", "split": split, **r})
    return rows, time.perf_counter() - t0


def curves_of(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (m, setting, split), g in df.groupby(["method", "setting", "split"]):
        a = accumulate(g.to_dict("records"))
        out.append({"method": m, "setting": setting, "split": split, "recall": a["constructed"]["recall"],
                    "extra_native_per_min": a["native"]["extra_per_min"], "extra_constructed_per_min": a["constructed"]["extra_per_min"],
                    "same_join_rate_pairs": a.get("pairs", {}).get("same_join_rate"), "same_join_rate_multi": a.get("multi", {}).get("same_join_rate"),
                    "double_cut_rate": a.get("multi", {}).get("double_cut_rate"), "loc_median_s_multi": a.get("multi", {}).get("loc_median_s")})
    return pd.DataFrame(out)


def pick(curves: pd.DataFrame) -> dict:
    ops = {}
    for m, g in curves[curves["split"] == "validation"].groupby("method"):
        ok = g[g["recall"] >= TARGET]; met = bool(len(ok))
        p = (ok.sort_values(["extra_native_per_min", "recall"], ascending=[True, False]) if met else g.sort_values("recall", ascending=False)).iloc[0]
        t = curves[(curves["method"] == m) & (curves["setting"] == p["setting"]) & (curves["split"] == "test")].iloc[0]
        ops[m] = {"setting": p["setting"], "target_met_on_validation": met, "validation": p.drop(["method", "setting", "split"]).to_dict(),
                  "test": t.drop(["method", "setting", "split"]).to_dict()}
    return ops


def paired(a: pd.DataFrame, b: pd.DataFrame, rng, n_boot: int = 2000) -> dict:
    """Test differences a - b: native extra cuts per minute (presumed patient groups resampled), recall (streams)."""
    na, nb = a[a["set"] == "native"].set_index("sid"), b[b["set"] == "native"].set_index("sid")
    u = pd.DataFrame({"g": [patient_of(s) for s in na.index], "ea": na["extra_internal"], "eb": nb.loc[na.index, "extra_internal"],
                      "m": na["n_samples"] / HZ / 60.0}).groupby("g")[["ea", "eb", "m"]].sum()
    ca, cb = a[a["set"] != "native"].set_index("sid"), b[b["set"] != "native"].set_index("sid")
    v = pd.DataFrame({"ma": ca["matched"], "mb": cb.loc[ca.index, "matched"], "n": ca["n_diff"]})
    de, dr = [], []
    for _ in range(n_boot):
        x = u.iloc[rng.integers(0, len(u), len(u))]; de.append((x["ea"].sum() - x["eb"].sum()) / x["m"].sum())
        y = v.iloc[rng.integers(0, len(v), len(v))]; dr.append((y["ma"].sum() - y["mb"].sum()) / y["n"].sum())
    est_e = (u["ea"].sum() - u["eb"].sum()) / u["m"].sum(); est_r = (v["ma"].sum() - v["mb"].sum()) / v["n"].sum()
    q = lambda d: [round(float(np.percentile(d, 2.5)), 3), round(float(np.percentile(d, 97.5)), 3)]  # noqa: E731
    return {"extra_native_per_min": [round(float(est_e), 3), *q(de)], "recall": [round(float(est_r), 3), *q(dr)]}


def main() -> None:
    ap = base_parser("Pooling component audit"); ap.add_argument("--out", required=True)
    ap.add_argument("--periods", default="runs/_demo/problem_a/periods"); ap.add_argument("--problem-a", default="runs/_demo/problem_a")
    ap.add_argument("--ckpt-hash", default="79f4a41af6e3")
    args = ap.parse_args(); cfg = load_cfg(args); paths = load_paths(); out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    streams, _ = build_streams(paths, cfg, args.ckpt_hash, "b0")
    per = {s.recipe.recipe_id: np.load(Path(args.periods) / f"{s.recipe.recipe_id}.npz")["period_samples"].astype(float)
           for g in streams.values() for s in g}
    rows, cost = [], {}
    for name, spec in CONFIGS.items():
        r, sec = run_config(name, spec, streams, per); rows += r; cost[name] = round(sec, 1)
        print(f"done {name} in {sec:.0f} s", flush=True)
    df = pd.DataFrame(rows)
    pa = pd.read_pickle(Path(args.problem_a) / "rows_b0.pkl")  # reference rows from problem A at its operating points
    pa_ops = json.loads((Path(args.problem_a) / "operating_points_b0.json").read_text())
    refs = {"B4 MS-TCN (problem A)": ("B4 MS-TCN", pa_ops["B4 MS-TCN"]["setting"]), "pool (problem A)": ("pool", pa_ops["pool"]["setting"])}
    for new, (old, setting) in refs.items():
        df = pd.concat([df, pa[(pa["method"] == old) & (pa["setting"] == setting)].assign(method=new)], ignore_index=True)
    curves = curves_of(df); curves.to_csv(out / "grid.csv", index=False)
    ops = pick(curves)
    rng = np.random.default_rng(0); test = df[df["split"] == "test"]
    chosen = {m: test[(test["method"] == m) & (test["setting"] == o["setting"])] for m, o in ops.items()}
    contrasts = {f"{a} minus {b}": paired(chosen[a], chosen[b], rng) for a, b in CONTRASTS}
    loc = {m: round(float(np.mean(np.concatenate([np.array(x) for x in t.loc[t["set"] != "native", "loc"]])) / HZ), 3) for m, t in chosen.items()}
    summary = {"operating_points": ops, "contrasts_test": contrasts, "decode_seconds_all_settings": cost,
               "mean_localisation_s_test": loc, "note": "period estimation for A3/A4 is a separate cost (problem A periods stage)"}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    lines = ["| Config | Setting | Val recall | Val extra/min | Test recall | Test extra/min native | Test extra/min constructed | Same-label cuts, multi | Mean localisation (s) | Decode time, all settings (s) |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for m, o in ops.items():
        v, t = o["validation"], o["test"]
        lines.append(f"| {m} | {o['setting']} | {v['recall']:.3f} | {v['extra_native_per_min']:.2f} | {t['recall']:.3f} | {t['extra_native_per_min']:.2f} | "
                     f"{t['extra_constructed_per_min']:.2f} | {t['same_join_rate_multi']:.3f} | {loc[m]} | {cost.get(m, '')} |")
    lines += ["", "| Test contrast | Extra cuts/min, native [95% CI] | Recall [95% CI] |", "|---|---|---|"]
    for k, v in contrasts.items():
        e, r = v["extra_native_per_min"], v["recall"]
        lines.append(f"| {k} | {e[0]:+.3f} [{e[1]:+.3f}, {e[2]:+.3f}] | {r[0]:+.3f} [{r[1]:+.3f}, {r[2]:+.3f}] |")
    (out / "summary.md").write_text("\n".join(lines) + "\n"); print("\n".join(lines))


if __name__ == "__main__":
    main()
