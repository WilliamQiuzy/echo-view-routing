#!/usr/bin/env python
"""Markdown summary of a problem A run (scripts/problem_a.py): operating points chosen on validation and their test
values, paired differences, period coverage, available-vs-fallback split and the sensitivity analysis."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def fmt(v, nd: int = 3) -> str:
    return "" if v is None else f"{v:.{nd}f}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Problem A summary")
    ap.add_argument("--dir", required=True); ap.add_argument("--recognizer", default="b0")
    args = ap.parse_args(); d = Path(args.dir)
    ops = json.loads((d / f"operating_points_{args.recognizer}.json").read_text())
    s = json.loads((d / f"summary_{args.recognizer}.json").read_text())
    lines = [f"# Problem A, recognizer {args.recognizer}", "",
             "| Method | Setting | Val target met | Val recall | Val extra/min (native) | Test recall | Test extra/min (native) | Test extra/min (constructed) | Same-label join cuts, pairs | Same-label join cuts, multi | Double cuts | Median localisation (s) |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for m, o in ops.items():
        v, t = o["validation"], o["test"]
        lines.append(f"| {m} | {o['setting']} | {'yes' if o['target_met_on_validation'] else 'no'} | {fmt(v['recall'])} | {fmt(v['extra_native_per_min'])} | "
                     f"{fmt(t['recall'])} | {fmt(t['extra_native_per_min'])} | {fmt(t['extra_constructed_per_min'])} | {fmt(t['same_join_rate_pairs'])} | "
                     f"{fmt(t['same_join_rate_multi'])} | {fmt(t['double_cut_rate'])} | {fmt(t['loc_median_s_multi'], 2)} |")
    lines += ["", "Period coverage on the actual inputs (share of samples): " + ", ".join(f"{k} {v}" for k, v in s["coverage"].items()), "",
              "| Test difference (phase minus other) | Extra cuts per minute, native [95% CI, presumed patient groups] | Constructed recall [95% CI, streams] |", "|---|---|---|"]
    for k, v in s["differences_test"].items():
        e, r = v["extra_native_per_min"], v["recall_constructed"]
        lines.append(f"| {k} | {e[0]:+.3f} [{e[1]:+.3f}, {e[2]:+.3f}] | {r[0]:+.3f} [{r[1]:+.3f}, {r[2]:+.3f}] |")
    lines += ["", "| Method (test natives) | Extra/min where a period was available | Extra/min in fallback | Minutes available | Minutes fallback |", "|---|---|---|---|---|"]
    for m, v in s["available_vs_fallback_test_natives"].items():
        lines.append(f"| {m} | {v['extra_per_min_available']} | {v['extra_per_min_fallback']} | {v['minutes_available']} | {v['minutes_fallback']} |")
    sens = s["sensitivity_without_34"]
    lines += ["", f"Without the {sens['n_candidates']} frozen sustained candidates ({sens['streams_dropped']} streams dropped), test:", "",
              "| Method | Recall | Extra/min (native) |", "|---|---|---|"]
    for m, v in sens["test"].items():
        lines.append(f"| {m} | {v['recall']} | {v['extra_native_per_min']} |")
    text = "\n".join(lines) + "\n"
    (d / f"summary_{args.recognizer}.md").write_text(text); print(text)


if __name__ == "__main__":
    main()
