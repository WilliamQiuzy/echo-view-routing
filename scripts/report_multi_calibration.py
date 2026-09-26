#!/usr/bin/env python
"""Tables for the multi-fragment calibration study: routing on 4-8 fragment test streams with thresholds frozen on
the pair validation bank versus the multi-fragment validation bank, with paired-bootstrap intervals.

Reads runs/<run>/metrics/*.json and runs/<run>/reports/bootstrap.json; prints Markdown (redirect into docs/results).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ORDER = ["b0", "b1", "b2", "b3", "b8", "vit_frame", "b6", "b7", "vivit_fe", "vivit_fe_centre", "mvit_k400", "mvit_echoprime",
         "mvit_echoprime_centre", "mvit_echoprime_centre@s1", "mvit_echoprime_centre@s2", "b4", "b5", "mstcn_vitfeat", "vit_router"]
NAMES = {"b0": "ResNet-18 (B0)", "b1": "B1 smoothing", "b2": "B2 HMM", "b3": "B3 JS divergence", "b8": "EchoPrime",
         "vit_frame": "ViT-S (D1)", "b6": "STFM", "b7": "EchoViewCLIP", "vivit_fe": "Factorised ViT (D2)",
         "vivit_fe_centre": "Factorised ViT + centre", "mvit_k400": "MViT Kinetics (D3)", "mvit_echoprime": "Echo-MViT (D3)",
         "mvit_echoprime_centre": "Echo-MViT + centre", "mvit_echoprime_centre@s1": "Echo-MViT + centre s1",
         "mvit_echoprime_centre@s2": "Echo-MViT + centre s2", "b4": "MS-TCN", "b5": "ASFormer",
         "mstcn_vitfeat": "MS-TCN on ViT-S", "vit_router": "ViT-Router (D4)"}


def g(d, *keys):
    for k in keys:
        if not isinstance(d, dict) or d.get(k) is None:
            return None
        d = d[k]
    return d


def v(x, nd=3) -> str:
    return "—" if x is None else f"{x:.{nd}f}"


def vci(e: dict | None, nd=3) -> str:
    if not e or e.get("estimate") is None:
        return "—"
    lo, hi = e["ci95"]
    return f"{e['estimate']:.{nd}f} ({lo:.{nd}f}–{hi:.{nd}f})" if lo is not None else f"{e['estimate']:.{nd}f}"


def dci(e: dict | None, nd=3) -> str:
    if not e or e.get("diff") is None:
        return "—"
    lo, hi = e["diff_ci95"]
    return f"{e['diff']:+.{nd}f} ({lo:+.{nd}f} to {hi:+.{nd}f})"


def main() -> None:
    ap = argparse.ArgumentParser(description="Multi-fragment calibration tables")
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()
    run = Path(args.run_dir)
    met = {p.stem: json.loads(p.read_text()) for p in (run / "metrics").glob("*.json")}
    bs = json.loads((run / "reports" / "bootstrap.json").read_text())["banks"]
    mb, pb = bs["multi_test"], bs["pairs_test"]
    rows = [m for m in ORDER if m in met and m in mb["methods"]]
    r = mb["reuse"]
    out = [f"Run `{run.name}`. Paired bootstrap with {json.loads((run / 'reports' / 'bootstrap.json').read_text())['n_boot']} draws. "
           f"Multi-fragment test streams are resampled by cine ({r['cines']} cines, {r['fragments']} fragments in {r['streams']} "
           f"streams, up to {r['appearances_max']} appearances per cine); pair streams by stream. Values are estimate (95% interval).", ""]
    out += ["### 4–8 fragment test streams: routing under the two threshold sources", "",
            "| model | change F1 ±0.5 s | error, pair τ (5%) | coverage, multi τ (5%) | error, multi τ (5%) | "
            "coverage, multi τ (1%) | error, multi τ (1%) | val coverage at 1% |",
            "|---|---|---|---|---|---|---|---|"]
    for m in rows:
        e, d = mb["methods"][m], met[m]
        unmet = [t for t in ("0.05", "0.01") if g(d, "policy_multi", "by_target", t, "status") == "target_unmet"]
        flag = f" (target unmet on validation: {', '.join(unmet)})" if unmet else ""
        out.append(f"| {NAMES.get(m, m)}{flag} | {vci(e['boundary_f1'])} | {vci(e['risk|pair@0.05'])} | "
                   f"{vci(e['coverage|multi@0.05'])} | {vci(e['risk|multi@0.05'])} | {vci(e['coverage|multi@0.01'])} | "
                   f"{vci(e['risk|multi@0.01'])} | {v(g(d, 'policy_multi', 'by_target', '0.01', 'coverage'))} |")
    ref = bs["multi_test"]["methods"].get("b4") and "b4"
    out += ["", f"### Paired differences to MS-TCN on the 4–8 fragment streams (model minus MS-TCN)", "",
            "| model | change F1 | false-split rate | missed rate | coverage, multi τ (5%) | error, multi τ (5%) | coverage, multi τ (1%) |",
            "|---|---|---|---|---|---|---|"]
    for m in rows:
        if m == ref:
            continue
        e = mb["methods"][m]
        out.append(f"| {NAMES.get(m, m)} | {dci(e['boundary_f1'])} | {dci(e['false_split_rate'])} | {dci(e['missed_rate'])} | "
                   f"{dci(e['coverage|multi@0.05'])} | {dci(e['risk|multi@0.05'])} | {dci(e['coverage|multi@0.01'])} |")
    out += ["", "### Pair test streams (legacy primary setting, pair τ)", "",
            "| model | change F1 ±0.5 s | false-split rate | missed rate | coverage (5%) | error (5%) | coverage (1%) | error (1%) |",
            "|---|---|---|---|---|---|---|---|"]
    for m in rows:
        e = pb["methods"][m]
        out.append(f"| {NAMES.get(m, m)} | {vci(e['boundary_f1'])} | {vci(e['false_split_rate'])} | {vci(e['missed_rate'])} | "
                   f"{vci(e['coverage|pair@0.05'])} | {vci(e['risk|pair@0.05'])} | {vci(e['coverage|pair@0.01'])} | "
                   f"{vci(e['risk|pair@0.01'])} |")
    print("\n".join(out))


if __name__ == "__main__":
    main()
