#!/usr/bin/env python
"""Tables of round-5 step 2 (scripts/round5_evidence_decider.py; configs/experiments/evidence_pooling_v1.yaml).

The success test per rule; per-seed metrics with all clips and without the ten sensitivity clips; paired contrasts
(95% bootstrap intervals, "consistent" only when all three seeds agree); the A5C units and native clips that turn right
and the A4C ones that turn wrong, against the mean rule and against ASRF's default, by units, clips and sessions, with
the share of the two clips that change most. Writes decider.md next to decider.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

HALVES = ("selection", "calibration")
SEEDS = ("0", "1", "2")
RULES = ("default", "mean", "trained_mean", "trained_distribution")


def ci(c: dict) -> str:
    return "; ".join(f"s{k} {100 * v['delta']:+.1f} [{100 * v['ci95'][0]:+.1f}, {100 * v['ci95'][1]:+.1f}]" for k, v in c["per_seed"].items()) + f" -> {c['category']}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument("run_dir")
    a = ap.parse_args(); r = json.loads((Path(a.run_dir) / "decider.json").read_text()); m = r["metrics"]
    md = ["# A4C/A5C decision on E1 ASRF candidate spans (round 5, step 2)", "",
          f"Targets: selection {r['target']['selection']:.4f}, calibration {r['target']['calibration']:.4f}; native floors: selection "
          f"{r['floors']['selection']:.4f}, calibration {r['floors']['calibration']:.4f}.", "", "## Training (training split only)", ""]
    for k, t in r["train"].items():
        md.append(f"- seed {k}: {t['spans']} spans ({t['A5C']} A5C) from {t['clips']} clips; weighted training accuracy "
                  f"trained_mean {t['trained_mean']['train_accuracy']:.3f}, trained_distribution {t['trained_distribution']['train_accuracy']:.3f}")
    for scope, title in (("all", "All clips (main result)"), ("without", "Without the ten sensitivity clips (dependence only)")):
        md += ["", f"## {title}", "", "| rule | half | segment F1@50 s0 / s1 / s2 (mean) | recall s0 / s1 / s2 | native macro-F1 s0 / s1 / s2 |", "|---|---|---|---|---|"]
        for rule in RULES:
            for h in HALVES:
                x = [m[f"{rule}|{scope}|{h}|{k}"] for k in SEEDS]; seg = [v["segment_f1_50_multi"] for v in x]
                md.append(f"| {rule} | {h} | " + " / ".join(f"{v:.3f}" for v in seg) + f" ({sum(seg) / 3:.3f}) | " +
                          " / ".join(f"{v['boundary_recall_multi']:.3f}" for v in x) + " | " + " / ".join(f"{v['native_macro_f1']:.3f}" for v in x) + " |")
    md += ["", "## Success test (all clips)", "", "| rule | half | mean segment F1@50 (target) | min recall | min native F1 (floor) | passes |", "|---|---|---|---|---|---|"]
    for rule, ok in r["success"].items():
        for h in HALVES:
            o = ok[h]
            md.append(f"| {rule} | {h} | {o['mean_segment_f1_50']:.3f} ({o['target']:.3f}) | {o['min_recall']:.3f} | {o['min_native_f1']:.3f} ({o['floor']:.3f}) | "
                      f"{'yes' if o['passes'] else 'no'} |")
    md += ["", "## Paired contrasts of segment F1@50 (points, 95% bootstrap intervals)", ""]
    md += [f"- {k.replace('|', ', ')}: {ci(c)}" for k, c in r["contrasts"].items()]
    md += ["", "## A5C turning right and A4C turning wrong (units = reference segments; clips and sessions deduplicated)", "",
           "| comparison | half | seed | stream units: A5C fixed | stream units: A4C broken | natives: A5C fixed | natives: A4C broken |", "|---|---|---|---|---|---|---|"]
    for key, f in r["flips"].items():
        comp, h, k = key.split("|")
        cell = lambda x: f"{x['n']} ({x['clips']} clips, {x['sessions']} sessions" + ("" if x["top2_clips_share"] is None else f"; top 2 clips {100 * x['top2_clips_share']:.0f}%") + ")"  # noqa: E731
        md.append(f"| {comp} | {h} | {k} | {cell(f['units: A5C fixed'])} | {cell(f['units: A4C broken'])} | {cell(f['natives: A5C fixed'])} | "
                  f"{cell(f['natives: A4C broken'])} |")
    md += ["", "## False pieces: unmatched predicted A5C segments inside A4C units, and A4C inside A5C units", "",
           "| rule | half | A5C inside A4C, s0 / s1 / s2 (clips) | A4C inside A5C, s0 / s1 / s2 (clips) | predicted segments s0 / s1 / s2 |", "|---|---|---|---|---|"]
    for rule in RULES:
        for h in HALVES:
            pc = [r["pieces"][f"{rule}|{h}|{k}"] for k in SEEDS]
            md.append(f"| {rule} | {h} | " + " / ".join(f"{x['A5C pieces inside A4C units']['n']} ({x['A5C pieces inside A4C units']['clips']})" for x in pc) +
                      " | " + " / ".join(f"{x['A4C pieces inside A5C units']['n']} ({x['A4C pieces inside A5C units']['clips']})" for x in pc) + " | " +
                      " / ".join(str(m[f"{rule}|all|{h}|{k}"]["predicted_segments"]) for k in SEEDS) + " |")
    (Path(a.run_dir) / "decider.md").write_text("\n".join(md) + "\n"); print("\n".join(md))


if __name__ == "__main__":
    main()
