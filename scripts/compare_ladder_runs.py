#!/usr/bin/env python
"""Side-by-side key metrics of two ladder runs (e.g. before / after the 2026-09-30 frame-order fix).

Prints a markdown table per method: test clip macro-F1 and frame accuracy on native clips, fragments per minute,
boundary F1 within 0.5 s on two-fragment and on 4-8-fragment test streams, and routed coverage / error at the 5% target
on 4-8-fragment streams (thresholds from multi_validation, and policy 2).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

METRICS = [
    ("clip macro-F1", "native.test.cine.macro_f1"),
    ("frame acc", "native.test.frame.accuracy"),
    ("frag/min", "native.test.stability.fragments_per_minute_mean"),
    ("bF1 pair", "constructed.boundary.0.5.f1"),
    ("bF1 multi", "constructed_multi.boundary.0.5.f1"),
    ("cov multi@5%", "constructed_multi.routing_by.multi@0.05.coverage"),
    ("err multi@5%", "constructed_multi.routing_by.multi@0.05.risk"),
    ("cov p2@5%", "constructed_multi.routing_by.p2@0.05.coverage"),
    ("err p2@5%", "constructed_multi.routing_by.p2@0.05.risk"),
]


def get(d: dict, dotted: str):
    """Follow a dotted path; keys may themselves contain dots (e.g. 'multi@0.05'), so match greedily."""
    parts, cur = dotted.split("."), d
    i = 0
    while i < len(parts):
        for j in range(len(parts), i, -1):
            k = ".".join(parts[i:j])
            if isinstance(cur, dict) and k in cur:
                cur, i = cur[k], j
                break
        else:
            return None
    return cur


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare two ladder runs")
    ap.add_argument("--old", required=True); ap.add_argument("--new", required=True); ap.add_argument("--methods", required=True)
    args = ap.parse_args()
    print("| method | run | " + " | ".join(n for n, _ in METRICS) + " |")
    print("|---|---|" + "---|" * len(METRICS))
    for m in args.methods.split(","):
        for tag, run in (("old", args.old), ("new", args.new)):
            f = Path(run) / "metrics" / f"{m}.json"
            if not f.is_file():
                continue
            d = json.loads(f.read_text())
            vals = [get(d, k) for _, k in METRICS]
            print(f"| {m} | {tag} | " + " | ".join("" if v is None else f"{v:.3f}" for v in vals) + " |")


if __name__ == "__main__":
    main()
