#!/usr/bin/env python
"""Tables of the round-4 neighbour-swap check (scripts/round4_neighbour_swap.py; diagnostic only).

For every system and original outcome, each variant's share of right units (majority of the unit's samples has its
class) and, where the variant keeps different-class neighbours, the share matched by segment F1@50. Draws of the same
unit are averaged first, so every unit weighs the same; units are also counted by source clip.
Writes swap_summary.md next to swap.csv.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

SYSTEMS = ("B3", "ASRF_s0", "ASRF_s1", "ASRF_s2")
VARIANTS = {"target": ("original", "isolated", "same_class_neighbours", "other_clip_neighbours", "other_a5c_only"),
            "control": ("original", "inject_same", "inject_a5c")}
KEY = ["half", "stream", "unit"]


def cell(g: pd.DataFrame, system: str) -> str:
    per_unit = g.groupby(KEY).agg(right=(f"{system}_right", "mean"), matched=(f"{system}_matched", lambda x: x.dropna().astype(float).mean()))
    clips = len({c for s in g.drop_duplicates(KEY)["sources"] for c in s.split(";")})
    txt = f"{100 * per_unit.right.mean():.0f}%"
    return txt + (f" ({100 * per_unit.matched.mean():.0f}% matched)" if per_unit.matched.notna().any() else "") + f", {clips} clips"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument("run_dir")
    a = ap.parse_args(); d = pd.read_csv(Path(a.run_dir) / "swap.csv", dtype={"stream": str})
    md = ["# Neighbour-swap check (round 4; diagnostic only)", "",
          "Targets are E0 A4C units next to an A5C unit; controls are E0 A4C units without an A5C neighbour that every main",
          "system gets right. A cell gives the share of units whose samples mostly keep their class, the share matched by",
          "segment F1@50 where the variant keeps different-class neighbours, and the number of source clips.", ""]
    for kind, variants in VARIANTS.items():
        md += [f"## {kind}s", "", "| system, originally | units | " + " | ".join(variants) + " |", "|---|---:|" + "---|" * len(variants)]
        for s in SYSTEMS:
            for status in ((False, True) if kind == "target" else (True,)):
                g = d[(d.kind == kind) & (d[f"{s}_orig_matched"] == status)]
                n = g.drop_duplicates(KEY).shape[0]
                if n:
                    md.append(f"| {s}, {'right' if status else 'wrong'} | {n} | " + " | ".join(cell(g[g.variant == v], s) for v in variants) + " |")
        md.append("")
    md += ["## Split by how the frozen classifier reads the inserted A5C blocks", "",
           "Draw level: each row is one donor draw. 'read as A5C' means the classifier's mean probability over every",
           "inserted A5C block's own samples picks A5C; otherwise at least one inserted A5C block looks like another class.", "",
           "| kind, variant, system | read as A5C: matched | read as another class: matched |", "|---|---|---|"]
    for kind, variant in (("target", "other_clip_neighbours"), ("target", "other_a5c_only"), ("control", "inject_a5c")):
        g = d[(d.kind == kind) & (d.variant == variant) & (d.a5c_blocks > 0)]
        clean = g.a5c_blocks_read_as_a5c == g.a5c_blocks
        for s in SYSTEMS:
            cells = [f"{100 * x[f'{s}_matched'].astype(float).mean():.0f}% of {len(x)}" if len(x) else "–" for x in (g[clean], g[~clean])]
            md.append(f"| {kind}, {variant}, {s} | " + " | ".join(cells) + " |")
    (Path(a.run_dir) / "swap_summary.md").write_text("\n".join(md)); print("\n".join(md))


if __name__ == "__main__":
    main()
