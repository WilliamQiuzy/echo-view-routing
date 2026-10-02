#!/usr/bin/env python
"""Per-seed true-order comparison (paper Table 3 / S4 Table), from three runs of scripts/true_order_joint.py that use
the MS-TCN of seed 0, 1 and 2 (trained in true order) as the paired reference.

For every method and endpoint it prints method minus MS-TCN seed s with the 95% interval and a category: "consistently
higher" or "consistently lower" when all three intervals exclude zero on the same side, otherwise "no consistent
difference detected" (not evidence of equivalence), with the per-seed pattern (+, -, 0) appended. It also prints the
direct contrasts between the MS-TCN seeds, which are the evidence for saying that a result varies with the training
seed. Every interval is conditional on one trained model, and all seeds share the same test data.
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
        return "consistently higher"
    if all(v == -1 for v in sig):
        return "consistently lower"
    return "no consistent difference detected (" + "".join({1: "+", -1: "-", 0: "0"}[v] for v in sig) + ")"


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
    print("\n## direct contrasts between MS-TCN seeds (true-trained, true test)")
    for ref in seeds:
        for o in seeds:
            if o <= ref:
                continue
            row = J[ref]["endpoints"][own[o]]
            print(f"seed {o} - seed {ref}: " + " | ".join("%s %+.3f [%+.3f, %+.3f]" % (e, row[e]["diff"], *row[e]["diff_ci95"]) for e in EPS))


if __name__ == "__main__":
    main()
