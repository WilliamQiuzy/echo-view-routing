#!/usr/bin/env python
"""Markdown tables for scripts/routing_scores.py results.

The score variant is chosen on validation only (lowest mean validation AURC over all methods); the threshold rule for
risk control is fixed in advance (ucb_class: one threshold per predicted class, 95% upper bound of cine-resampled
validation risk). Test numbers are reported for the chosen pair and, for reference, for every other combination.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

KEY = ["b0", "b3", "b6", "b7", "mvit_echoprime_centre", "b4", "b5", "mstcn_vitfeat", "vit_router"]
NAMES = {"b0": "ResNet-18 (B0)", "b1": "B1 smoothing", "b2": "B2 HMM", "b3": "B3 JS divergence", "b4": "MS-TCN", "b5": "ASFormer",
         "b6": "STFM", "b7": "EchoViewCLIP", "b8": "EchoPrime", "vit_frame": "ViT-S (D1)", "mstcn_vitfeat": "MS-TCN on ViT-S",
         "vit_router": "ViT-Router (D4)", "mvit_echoprime_centre": "Echo-MViT + centre"}
VARIANTS = ["own_p10", "own_mean", "ownT_p10", "frame_p10", "frame_mean", "min_own_frame"]
RULES = ["point", "ucb", "point_class", "ucb_class"]
LADDER = ("own_p10", "point")
RULE = "ucb_class"


def cell(x: dict, ci: bool = True) -> str:
    t = x["test"]
    if t["risk"] is None:
        return f"{t['coverage']:.2f} / —"
    s = f"{t['coverage']:.2f} / {t['risk']:.3f}"
    if ci and t.get("risk_ci"):
        s += f" ({t['risk_ci'][0]:.3f}–{t['risk_ci'][1]:.3f})"
    return s


def main() -> None:
    ap = argparse.ArgumentParser(description="Routing-scores tables")
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()
    r = json.loads((Path(args.run_dir) / "results.json").read_text())
    M = r["methods"]; methods = list(M)
    val_aurc = {v: statistics.mean(M[m]["variants"][v]["aurc_val"] for m in methods) for v in VARIANTS}
    best = min(val_aurc, key=val_aurc.get)
    out = [f"Run `{Path(args.run_dir).name}`: {len(methods)} methods, {r['n_boot']} cine-level bootstrap draws, "
           f"{r['n_cines']['multi_validation']} validation and {r['n_cines']['multi_test']} test cines (4–8 fragment streams).", "",
           f"Score chosen on validation: **{best}** (lowest mean validation AURC). Rule fixed in advance: **{RULE}**.", "",
           "### Ranking quality (AURC, lower is better; mean over all methods)", "",
           "| score | validation AURC | test AURC | test hindsight coverage at 1% |", "|---|---|---|---|"]
    for v in VARIANTS:
        out.append(f"| {v} | {val_aurc[v]:.4f} | {statistics.mean(M[m]['variants'][v]['aurc_test'] for m in methods):.4f} | "
                   f"{statistics.mean(M[m]['variants'][v]['hindsight_coverage_test']['0.01'] for m in methods):.3f} |")
    out += ["", f"### Ladder rule ({LADDER[0]}, {LADDER[1]}) vs {best} + {RULE} on the 4–8 fragment test streams",
            "", "Coverage / actual error (95% interval of the error, resampled by cine).", "",
            "| model | ladder, 5% target | new, 5% target | ladder, 1% target | new, 1% target |", "|---|---|---|---|---|"]
    for m in [k for k in KEY if k in M]:
        lad, new = M[m]["variants"][LADDER[0]]["rules"][LADDER[1]], M[m]["variants"][best]["rules"][RULE]
        out.append(f"| {NAMES.get(m, m)} | {cell(lad['0.05'])} | {cell(new['0.05'])} | {cell(lad['0.01'])} | {cell(new['0.01'])} |")
    out += ["", "### Every score and rule over all methods", "",
            "Methods whose test error stays at or below the target (point estimate), and the median test coverage.", "",
            "| score | rule | 5%: held | 5%: median coverage | 1%: held | 1%: median coverage |", "|---|---|---|---|---|---|"]
    for v in VARIANTS:
        for rule in RULES:
            row = [f"| {v} | {rule}"]
            for t in ("0.05", "0.01"):
                xs = [M[m]["variants"][v]["rules"][rule][t]["test"] for m in methods]
                held = sum(1 for x in xs if x["risk"] is not None and x["risk"] <= float(t))
                row.append(f"{held}/{len(xs)} | {statistics.median(x['coverage'] for x in xs):.2f}")
            out.append(" | ".join(row) + " |")
    out += ["", f"### {best} + {RULE}, all methods", "",
            "| model | T | 5%: coverage / error | 5%: A5C share of error | 1%: coverage / error | pairs 5%: coverage / error |",
            "|---|---|---|---|---|---|"]
    for m in methods:
        e = M[m]["variants"][best]["rules"][RULE]
        a5 = e["0.05"]["a5c_share_of_routed_error"]
        pr = "—" if e["0.05"]["pairs_risk"] is None else f"{e['0.05']['pairs_risk']:.3f}"
        out.append(f"| {NAMES.get(m, m)} | {M[m]['temperature']:.2f} | {cell(e['0.05'])} | {'—' if a5 is None else f'{a5:.2f}'} | "
                   f"{cell(e['0.01'])} | {e['0.05']['pairs_coverage']:.2f} / {pr} |")
    print("\n".join(out))


if __name__ == "__main__":
    main()
