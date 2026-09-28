#!/usr/bin/env python
"""Paired bootstrap intervals for a ladder run (proposal §7.3).

Reads runs/<run>/fragments/<method>.<bank>.csv (written by scripts/run_baselines.py) and writes
runs/<run>/reports/bootstrap.{json,md}. Pair streams use each cine at most once, so they are resampled by stream;
multi-fragment streams reuse cines across streams, so they are resampled by cine. Every method shares the same draws,
so differences to --reference are paired.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from echo_routing.evaluate.bootstrap import count_cols, resample_weights, summarise, unit_sums  # noqa: E402

BANKS = {"pairs_test": "stream", "multi_test": "video_id"}
SHOW = ["boundary_f1", "false_split_rate", "missed_rate", "risk|pair@0.05",
        "coverage|multi@0.05", "risk|multi@0.05", "coverage|multi@0.01", "risk|multi@0.01",
        "coverage|p2@0.05", "risk|p2@0.05", "coverage|p2@0.01", "risk|p2@0.01"]


def reuse_stats(df: pd.DataFrame) -> dict:
    per = df.groupby("video_id").size()
    return {"fragments": int(len(df)), "streams": int(df["stream"].nunique()), "cines": int(per.size),
            "appearances_mean": round(float(per.mean()), 2), "appearances_max": int(per.max())}


def _cell(e: dict, diff: bool = False) -> str:
    v, ci = (e.get("diff"), e.get("diff_ci95")) if diff else (e["estimate"], e["ci95"])
    if v is None:
        return "—"
    sign = "+" if diff and v >= 0 else ""
    lo, hi = ci
    return f"{sign}{v:.3f} [{lo:.3f}, {hi:.3f}]" if lo is not None else f"{sign}{v:.3f}"


def markdown(bank: str, unit: str, reuse: dict, res: dict, reference: str) -> list[str]:
    cols = [k for k in SHOW if k in next(iter(res.values()))]
    head = "| method | " + " | ".join(cols) + " |"
    sep = "|---|" + "---|" * len(cols)
    lines = [f"## {bank} · unit = {unit}", "",
             f"{reuse['streams']} streams, {reuse['fragments']} fragments from {reuse['cines']} cines "
             f"(mean {reuse['appearances_mean']}, max {reuse['appearances_max']} appearances per cine).", "",
             "Estimate [95% interval]", "", head, sep]
    lines += [f"| {m} | " + " | ".join(_cell(r[k]) for k in cols) + " |" for m, r in res.items()]
    if reference in res:
        lines += ["", f"Paired difference to {reference} [95% interval]", "", head, sep]
        lines += [f"| {m} | " + " | ".join(_cell(r[k], diff=True) for k in cols) + " |" for m, r in res.items() if m != reference]
    return lines + [""]


def main() -> None:
    ap = argparse.ArgumentParser(description="Paired bootstrap intervals for a ladder run")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--methods", default=None, help="comma list; default = every method with fragment tables")
    ap.add_argument("--reference", default="b4")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    run = Path(args.run_dir); fdir = run / "fragments"
    # file names are <method>.<bank>.csv and method ids may contain dots (fusions such as "b4+b7~0.5")
    methods = args.methods.split(",") if args.methods else sorted({p.name.rsplit(".", 2)[0] for p in fdir.glob("*.csv")})
    out = {"run": run.name, "n_boot": args.n_boot, "seed": args.seed, "reference": args.reference, "banks": {}}
    md = [f"# Paired bootstrap · {run.name}", "",
          f"{args.n_boot} draws, seed {args.seed}. Pair streams are resampled by stream (each cine used once); "
          "multi-fragment streams by cine (cines recur across streams). Intervals are descriptive: there are no patient ids.", ""]
    for bank, unit in BANKS.items():
        frames = {m: pd.read_csv(fdir / f"{m}.{bank}.csv") for m in methods if (fdir / f"{m}.{bank}.csv").is_file()}
        if not frames:
            continue
        cols = count_cols(next(iter(frames.values())))
        tables, ids0 = {}, None
        for m, df in frames.items():
            ids, x = unit_sums(df, unit, cols)
            if ids0 is not None and not np.array_equal(ids, ids0):
                raise SystemExit(f"{m}: {bank} units differ from the other methods; not a paired comparison")
            ids0 = ids; tables[m] = x
        weights = resample_weights(len(ids0), args.n_boot, np.random.default_rng(args.seed))
        res = summarise(tables, cols, weights, reference=args.reference)
        reuse = reuse_stats(next(iter(frames.values())))
        out["banks"][bank] = {"unit": unit, "n_units": int(len(ids0)), "reuse": reuse, "methods": res}
        md += markdown(bank, unit, reuse, res, args.reference)
        print(f"{bank}: {len(frames)} methods, {len(ids0)} {unit} units", flush=True)
    (run / "reports").mkdir(exist_ok=True)
    (run / "reports" / "bootstrap.json").write_text(json.dumps(out, indent=1))
    (run / "reports" / "bootstrap.md").write_text("\n".join(md))
    print(f"wrote {run / 'reports' / 'bootstrap.md'}")


if __name__ == "__main__":
    main()
