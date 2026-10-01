#!/usr/bin/env python
"""2 x 2 MS-TCN permutation matrix per model seed (task 3 of the 2026-10-01 round), from scripts/mstcn_matrix.py output.

Cells: training order (true / within-fragment random, one fixed permutation) x test order (true / random, mean over
the ten task-1 seeds). Per model seed the report gives the four cells, the test-order effect inside each training
condition, the training-order effect at each test order, and their interaction, each with a paired 95% bootstrap
interval (same unit draws for every cell: native clips by presumed patient group, pair streams by stream, 4-8
fragment streams by clip). Seeds are summarised by mean and range only; they are not pooled into one test. B2 / B3
rows decode the identical test inputs and give the test-order effect of the simple decoders.
Writes <dir>/matrix.json and <dir>/matrix.md (aggregates only).
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from report_permutation_round import FRAG, FRAG_COLS, NATIVE, prep_natives, ratio_funcs, read, sums, weights  # noqa: E402

BOOT_SEED = 20261001
RUN = re.compile(r"-tr(?P<cond>true|rand)-ps[^-]+-.*-ms(?P<seed>\d+)$")


def cell(df: pd.DataFrame, unit: str, cols: list[str], funcs: dict, units: pd.Index, W: np.ndarray, test: str):
    """(point, replicates) per endpoint for one model and test order; random = mean over its seeds."""
    part = df[df["test_condition"] == test]
    pts, reps = [], []
    for _, g in part.groupby("test_seed"):
        s = sums(g, unit, cols, units)
        with np.errstate(divide="ignore", invalid="ignore"):
            pts.append({k: float(fn(np.moveaxis(s.sum(0), -1, 0))) for k, fn in funcs.items()})
            reps.append({k: fn(np.moveaxis(W @ s, -1, 0)) for k, fn in funcs.items()})
    return ({k: float(np.mean([p[k] for p in pts])) for k in funcs}, {k: np.mean([r[k] for r in reps], axis=0) for k in funcs})


def interval(rep: np.ndarray) -> list[float]:
    return np.nanpercentile(rep, [2.5, 97.5]).tolist()


def seed_block(tabs: dict, runs: dict, simple: list[str]) -> dict:
    """tabs: name -> (df with column 'model', unit, cols, funcs). runs: training condition -> model name."""
    rng = np.random.default_rng(BOOT_SEED); out = {}
    for name, (df, unit, cols, funcs) in tabs.items():
        units = pd.Index(sorted(df[unit].unique())); W = weights(len(units), rng)
        c = {(tc, te): cell(df[df["model"] == runs[tc]], unit, cols, funcs, units, W, te) for tc in runs for te in ("true", "random")}
        for k in funcs:
            pt = {key: v[0][k] for key, v in c.items()}; rp = {key: v[1][k] for key, v in c.items()}
            e = {"cells": {f"train {a} / test {b}": pt[(a, b)] for a, b in pt}}
            for tc in runs:
                e[f"test effect (train {tc})"] = {"value": pt[(tc, "random")] - pt[(tc, "true")], "ci95": interval(rp[(tc, "random")] - rp[(tc, "true")])}
            for te in ("true", "random"):
                e[f"training effect (test {te})"] = {"value": pt[("rand", te)] - pt[("true", te)], "ci95": interval(rp[("rand", te)] - rp[("true", te)])}
            inter = (rp[("rand", "random")] - rp[("rand", "true")]) - (rp[("true", "random")] - rp[("true", "true")])
            e["interaction"] = {"value": (pt[("rand", "random")] - pt[("rand", "true")]) - (pt[("true", "random")] - pt[("true", "true")]),
                                "ci95": interval(inter)}
            for m in simple:
                sp, sr = cell(df[df["model"] == m], unit, cols, funcs, units, W, "true"), cell(df[df["model"] == m], unit, cols, funcs, units, W, "random")
                e[f"test effect ({m})"] = {"value": sr[0][k] - sp[0][k], "ci95": interval(sr[1][k] - sp[1][k])}
                e[f"{m} minus MS-TCN trained in true order (test true)"] = {"value": sp[0][k] - pt[("true", "true")], "ci95": interval(sp[1][k] - rp[("true", "true")])}
                e[f"{m} minus MS-TCN trained in true order (test random)"] = {"value": sr[0][k] - pt[("true", "random")], "ci95": interval(sr[1][k] - rp[("true", "random")])}
            out[f"{name}: {k}"] = e
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="MS-TCN 2x2 permutation matrix report"); ap.add_argument("--dir", default="runs/_demo/mstcn_matrix")
    args = ap.parse_args(); d = Path(args.dir)
    met = read(d / "metrics.csv"); nat = prep_natives(read(d / "natives.csv.gz")); frag = read(d / "fragments.csv.gz")
    nat = nat.rename(columns={"condition": "test_condition", "seed": "test_seed"})
    # paired simple decoders carry model "-": give each its own pseudo-model name per method
    for t in (nat, frag, met):
        t.loc[t["model"] == "-", "model"] = "simple:" + t.loc[t["model"] == "-", "method"]
    simple = sorted(m for m in met["model"].unique() if m.startswith("simple:"))
    runs = sorted(m for m in met["model"].unique() if RUN.search(m))
    by_seed: dict[int, dict[str, str]] = {}
    for r in runs:
        g = RUN.search(r); by_seed.setdefault(int(g["seed"]), {})[g["cond"]] = r
    ncols, nfuncs = ratio_funcs(NATIVE)
    res = {"seeds": {}, "runs": by_seed}
    md = ["# MS-TCN training/test permutation matrix", "",
          "Rows are training orders and columns are test orders. Test random = mean over the ten task-1 seeds. "
          "Intervals are paired 95% bootstrap intervals over the stated units; seeds are summarised, not pooled.", ""]
    for seed, rr in sorted(by_seed.items()):
        if set(rr) != {"true", "rand"}:
            continue
        tabs = {"native": (nat, "group", ncols, nfuncs),
                "pairs": (frag[frag["bank"] == "pairs_test"], "stream", FRAG_COLS, FRAG),
                "multi": (frag[frag["bank"] == "multi_test"], "video_id", FRAG_COLS, FRAG)}
        blk = seed_block(tabs, rr, simple)
        cf1 = met[met["model"].isin(rr.values())].assign(train=lambda x: x["model"].map({v: k for k, v in rr.items()}))
        blk["clip macro-F1"] = {f"train {t} / test {c}": float(g["clip macro-F1"].mean()) for (t, c), g in cf1.groupby(["train", "test_condition"])}
        res["seeds"][seed] = blk
    keys = [k for k in next(iter(res["seeds"].values())) if k != "clip macro-F1"] if res["seeds"] else []
    summary = {}
    for k in keys:
        effs = [e for e in next(iter(res["seeds"].values()))[k] if e != "cells"]
        summary[k] = {e: {"per_seed": {s: res["seeds"][s][k][e]["value"] for s in res["seeds"]},
                          "mean": float(np.mean([res["seeds"][s][k][e]["value"] for s in res["seeds"]])),
                          "range": [float(np.min([res["seeds"][s][k][e]["value"] for s in res["seeds"]])),
                                    float(np.max([res["seeds"][s][k][e]["value"] for s in res["seeds"]]))]} for e in effs}
    res["summary"] = summary
    for k in keys:
        md += [f"## {k}", "", "| effect | " + " | ".join(f"seed {s}" for s in res["seeds"]) + " | mean (range) |",
               "|---|" + "---|" * (len(res["seeds"]) + 1)]
        first = res["seeds"][next(iter(res["seeds"]))][k]
        md.append("| cells (TT / TR / RT / RR) | " + " | ".join(
            " / ".join(f"{res['seeds'][s][k]['cells'][c]:.3f}" for c in ("train true / test true", "train true / test random",
                                                                         "train rand / test true", "train rand / test random"))
            for s in res["seeds"]) + " | |")
        for e in (x for x in first if x != "cells"):
            cells = []
            for s in res["seeds"]:
                v = res["seeds"][s][k][e]; lo, hi = v["ci95"]; cells.append(f"{v['value']:+.3f} [{lo:+.3f}, {hi:+.3f}]")
            sm = summary[k][e]
            md.append(f"| {e} | " + " | ".join(cells) + f" | {sm['mean']:+.3f} ({sm['range'][0]:+.3f} to {sm['range'][1]:+.3f}) |")
        md.append("")
    md += ["## Clip macro-F1 (no interval)", ""] + [f"- seed {s}: " + json.dumps({k: round(v, 3) for k, v in b["clip macro-F1"].items()})
                                                  for s, b in res["seeds"].items()]
    (d / "matrix.json").write_text(json.dumps(res, indent=1, default=float)); (d / "matrix.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
