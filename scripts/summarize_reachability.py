#!/usr/bin/env python
"""Tables of round-5 step 1 (scripts/round5_reachability.py) and the pre-declared reachability test.

For each base and span type, per seed and half: the recall of the candidate cuts themselves, and the A4C/A5C oracle's
segment F1@50, boundary recall and native macro-F1; then the test of configs/experiments/evidence_pooling_v1.yaml:
three-seed mean segment F1@50 at least the target on both halves, recall >= 0.95 and native macro-F1 at or above the
input's floor in every seed and half. Also the sensitivity tables (all clips against the ten clips removed).
Writes reach.md next to reach.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

HALVES = ("selection", "calibration")
SEEDS = ("ASRF_s0", "ASRF_s1", "ASRF_s2")
MARGIN, RECALL = 0.01, 0.95


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument("run_dir")
    a = ap.parse_args(); r = json.loads((Path(a.run_dir) / "reach.json").read_text())
    target = {"selection": r["E0|selection|B3"]["default"]["segment_f1_50_multi"] + MARGIN,
              "calibration": r["system|calibration|B7_B3"]["default"]["segment_f1_50_multi"] + MARGIN}
    md = ["# Reachability of an A4C/A5C decision (round 5, step 1; diagnostic only)", "",
          f"Targets: selection {target['selection']:.4f}, calibration {target['calibration']:.4f}. Floors: " +
          "; ".join(f"{e} {h} {r[f'{e}|{h}|counts']['floor']:.4f}" for e in ("E0", "E1") for h in HALVES) + ".", "",
          "| base | spans | half | candidate-cut recall per seed | oracle segment F1@50 per seed (mean) | oracle recall per seed | "
          "oracle native F1 per seed | reachable on this half |", "|---|---|---|---|---|---|---|---|"]
    verdict = {}
    for enc in ("E0", "E1"):
        for base, names in (("ASRF", SEEDS), ("B3 (reference)", ("B3",))):
            for span in ("final", "candidate"):
                ok_all = True
                for h in HALVES:
                    rows = [r[f"{enc}|{h}|{n}"] for n in names]; o = [x[f"oracle_{span}"] for x in rows]
                    seg = [x["segment_f1_50_multi"] for x in o]; rec = [x["boundary_recall_multi"] for x in o]
                    nat = [x["native_macro_f1"] for x in o]; floor = r[f"{enc}|{h}|counts"]["floor"]
                    cand = [x["candidate_cut_recall"] if span == "candidate" else x["final_cut_recall"] for x in rows]
                    ok = sum(seg) / len(seg) >= target[h] and min(rec) >= RECALL and min(nat) >= floor; ok_all &= ok
                    md.append(f"| {enc} {base} | {span} | {h} | " + " / ".join(f"{c:.3f}" for c in cand) + " | " +
                              " / ".join(f"{x:.3f}" for x in seg) + f" ({sum(seg) / len(seg):.3f}) | " + " / ".join(f"{x:.3f}" for x in rec) +
                              " | " + " / ".join(f"{x:.3f}" for x in nat) + f" | {'yes' if ok else 'no'} |")
                verdict[f"{enc} {base}, {span} spans"] = ok_all
    md += ["", "**Reachable on both halves (pre-declared test):** " + "; ".join(f"{k}: {'yes' if v else 'no'}" for k, v in verdict.items()), "",
           "## Sensitivity: all clips against the ten clips removed (same list for every system, nothing re-tuned)", "",
           "| system | half | segment F1@50 all -> without | recall all -> without | native F1 all -> without |", "|---|---|---|---|---|"]
    for key, v in r.items():
        if key.endswith("|counts") or "default_without_clips" not in v:
            continue
        enc, h, name = key.split("|"); d, w = v["default"], v["default_without_clips"]
        md.append(f"| {enc} {name} | {h} | {d['segment_f1_50_multi']:.3f} -> {w['segment_f1_50_multi']:.3f} | "
                  f"{d['boundary_recall_multi']:.3f} -> {w['boundary_recall_multi']:.3f} | {d['native_macro_f1']:.3f} -> {w['native_macro_f1']:.3f} |")
    md += ["", "Removed: " + "; ".join(f"{e} {h}: natives {c['natives'][0]}->{c['natives'][1]}, pairs {c['pairs'][0]}->{c['pairs'][1]}, "
                                         f"streams {c['multi'][0]}->{c['multi'][1]}" for e in ("E0",) for h in HALVES for c in [r[f'{e}|{h}|counts']])]
    (Path(a.run_dir) / "reach.md").write_text("\n".join(md) + "\n"); print("\n".join(md))


if __name__ == "__main__":
    main()
