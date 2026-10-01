#!/usr/bin/env python
"""Per-seed true-order comparison (paper Table 3 / S4 Table), from three runs of scripts/true_order_joint.py that use
the MS-TCN of seed 0, 1 and 2 (trained in true order) as the paired reference.

For every method and endpoint it prints method minus MS-TCN seed s with the 95% interval, and a verdict: "higher" or
"lower in all seeds" when every interval excludes zero on the same side, "no difference in any seed" when every
interval includes zero, and "seed-dependent" otherwise.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

EPS = ["clip macro-F1", "segment F1@50, multi", "segment F1@50, pairs", "boundary F1, pairs", "boundary F1, multi",
       "boundary recall, multi", "same-class cut rate, multi", "native cuts per clip"]
METHODS = ["B1 smoothing", "B2 HMM", "B3 change points, default", "B3 change points, audited", "B4 MS-TCN v2",
           "B5 ASFormer v2", "B6 STFM windowed, numeric-order model", "B7 EchoViewCLIP windowed"]


def verdict(xs: list[tuple[float, list[float]]]) -> str:
    sig = [(lo > 0) - (hi < 0) for _, (lo, hi) in xs]
    if all(v == 1 for v in sig):
        return "higher in all seeds"
    if all(v == -1 for v in sig):
        return "lower in all seeds"
    if all(v == 0 for v in sig):
        return "no difference in any seed"
    return "seed-dependent " + "".join({1: "+", -1: "-", 0: "0"}[v] for v in sig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Per-seed true-order comparison")
    ap.add_argument("--runs", default="runs/_demo/true_order_joint_seed{s}/joint.json"); ap.add_argument("--seeds", default="0,1,2")
    args = ap.parse_args(); seeds = [int(s) for s in args.seeds.split(",")]
    J = {s: json.loads(Path(args.runs.format(s=s)).read_text()) for s in seeds}
    own = {s: f"MS-TCN seed {s} (true-trained)" for s in seeds}
    print("## reference values (true-trained MS-TCN per seed)")
    for e in EPS:
        print(e, " | ".join("s%d %.3f [%.3f, %.3f]" % (s, J[s]["endpoints"][own[s]][e]["value"], *J[s]["endpoints"][own[s]][e]["ci95"]) for s in seeds))
    for e in EPS:
        print(f"\n## {e}: method minus MS-TCN seed s (true-trained, true test)")
        for m in METHODS + ["shuffle-trained, same seed"]:
            xs = []
            for s in seeds:
                name = f"MS-TCN seed {s} (shuffle-trained)" if m.startswith("shuffle") else m
                x = J[s]["endpoints"][name][e]; xs.append((x["diff"], x["diff_ci95"]))
            print(f"{m:40s} " + " | ".join("%+.3f [%+.3f, %+.3f]" % (d, lo, hi) for d, (lo, hi) in xs) + "  -> " + verdict(xs))


if __name__ == "__main__":
    main()
