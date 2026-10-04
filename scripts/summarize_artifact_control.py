#!/usr/bin/env python
"""Markdown tables for the slice-artifact control (configs/experiments/slice_artifact_control_v1.yaml), read from the
per-volume numbers written by scripts/echoslicer_artifact_control.py. Reads numbers only, never images.
python scripts/summarize_artifact_control.py <numbers/artifact_control> [--first-run <numbers/artifact_control_run1>]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

VOLUMES = ("A_0", "B_0", "C_0")
STARTS = ("S1", "S2", "S3", "S4")
CURVE_TILTS = (0.5, 1, 2, 4, 8, 15)


def fmt(x: float | None) -> str:
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:+.2f}"


def pair(summary: dict, start: str, key: str) -> str:
    """The value at the first volume and at the middle of the recording, as 'first / middle' ('not run' for a start
    plane the run lacks)."""
    return " / ".join(fmt(v.get(key)) for k, v in summary["sweeps"].items() if k.split("|")[0] == start) or "not run"


def contrast_table(root: Path) -> list[str]:
    head = ["Volume", "Start", "θ_n swept (°)", "Contrast vs signed tilt", "Fold (°)", "vs obliqueness, below the fold", "vs obliqueness, above the fold",
            "After matching, vs signed tilt", "Nearest-neighbour, vs signed tilt", "Largest blur (px)"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for v in VOLUMES:
        s = json.loads((root / v / "summary.json").read_text())
        for st in STARTS:
            first = next(x for k, x in s["sweeps"].items() if k.startswith(st + "|"))
            th = first["start_theta"]; blur = max(x["sigma_px_max"] for k, x in s["sweeps"].items() if k.startswith(st + "|"))
            lines.append(f"| {v} | {st} | {th - 15:.0f}–{th + 15:.0f} | {pair(s, st, 'rho_sd_raw_vs_tilt_deg')} | "
                         f"{first['fold_deg']:.0f} | {pair(s, st, 'rho_sd_raw_vs_obliqueness_below_fold')} | "
                         f"{pair(s, st, 'rho_sd_raw_vs_obliqueness_above_fold')} | {pair(s, st, 'rho_sd_matched_vs_tilt_deg')} | "
                         f"{pair(s, st, 'rho_sd_nearest_vs_tilt_deg')} | {blur:.2f} |")
    return lines


def sharpness_table(root: Path, first_run: Path | None) -> list[str]:
    head = ["Volume", "Start"] + (["Sharpness vs signed tilt, first run (no margin)"] if first_run else []) + \
           ["Sharpness vs signed tilt", "Relative sharpness vs signed tilt", "Relative sharpness vs obliqueness"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for v in VOLUMES:
        s = json.loads((root / v / "summary.json").read_text())
        f = json.loads((first_run / v / "summary.json").read_text()) if first_run else None
        for st in STARTS:
            cells = [v, st] + ([pair(f, st, "rho_sharpness_raw_vs_tilt_deg")] if f else []) + \
                    [pair(s, st, "rho_sharpness_raw_vs_tilt_deg"), pair(s, st, "rho_sharpness_rel_raw_vs_tilt_deg"),
                     pair(s, st, "rho_sharpness_rel_raw_vs_obliqueness_deg")]
            lines.append("| " + " | ".join(cells) + " |")
    return lines


def side_counts(root: Path, cut: float = -0.5) -> list[str]:
    """How often raw contrast falls clearly with obliqueness (Spearman <= cut, a descriptive cut chosen after the run),
    per one-sided test and per sweep with both sides testable; S1-S3 (one fan) and S4 (another fan) apart."""
    lines = []
    for group in (("S1", "S2", "S3"), ("S4",)):
        one = clear_one = both = clear_both = 0
        for v in VOLUMES:
            for k, x in json.loads((root / v / "summary.json").read_text())["sweeps"].items():
                if k.split("|")[0] not in group:
                    continue
                sides = [x[f"rho_sd_raw_vs_obliqueness_{s}"] for s in ("below_fold", "above_fold") if x[f"rho_sd_raw_vs_obliqueness_{s}"] is not None]
                one += len(sides); clear_one += sum(r <= cut for r in sides)
                if len(sides) == 2:
                    both += 1; clear_both += all(r <= cut for r in sides)
        name = group[0] if len(group) == 1 else f"{group[0]}-{group[-1]}"
        both_text = f", and on both sides in {clear_both} of {both} sweeps where both sides could be tested" if both else \
            ", and no sweep reached both sides of its fold"
        lines.append(f"- {name}: contrast fell clearly with obliqueness (rho <= {cut}) in {clear_one} of {one} one-sided tests{both_text}.")
    return lines


def crossing(sizes: np.ndarray, change: np.ndarray, level: float) -> str:
    """The tilt at which the tilt-change curve first reaches `level` (linear interpolation between measured tilts)."""
    above = np.flatnonzero(change >= level)
    if not len(above):
        return f"> {sizes[-1]:.0f}°"
    i = above[0]
    if i == 0:
        return f"≤ {sizes[0]:.1f}°"
    x = sizes[i - 1] + (level - change[i - 1]) * (sizes[i] - sizes[i - 1]) / (change[i] - change[i - 1])
    return f"{x:.1f}°"


FANS = {"A4C fan (S1–S3)": ("S1", "S2", "S3"), "A2C fan (S4)": ("S4",)}


def curve_table(root: Path) -> list[str]:
    """Per volume and fan: tilt change (both directions, both times, the fan's start planes) and time change at the
    fan's start planes, with the largest time change up to half a cycle."""
    head = ["Volume", "Fan", "Volumes per s"] + [f"Tilt {c}°" for c in CURVE_TILTS] + \
           ["Time 1 volume", "Time 2 volumes", "Largest time change", "1 volume ÷ 0.5° tilt", "Tilt matching 1 volume"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for v in VOLUMES:
        s = json.loads((root / v / "summary.json").read_text())
        tc = pd.read_csv(root / v / "tilt_curves.csv"); sc = pd.read_csv(root / v / "time_curves.csv")
        for fan, starts in FANS.items():
            t, u = tc[tc.start.isin(starts)], sc[sc.start.isin(starts)]
            if t.empty:
                continue
            g = t.assign(size=t.tilt_deg.abs()).groupby("size").change_raw.mean(); h = u.groupby("gap_volumes").change.mean()
            one = float(h.loc[1]); k_max = int(h.idxmax())
            cells = [v, fan, f"{s['fps']:.1f}"] + [f"{g.loc[c]:.1f}" for c in CURVE_TILTS] + \
                    [f"{h.loc[k]:.1f} ({k / s['fps']:.2f} s)" for k in (1, 2)] + [f"{h.max():.1f} ({k_max / s['fps']:.2f} s)"] + \
                    [f"{one / g.loc[0.5]:.1f}", crossing(g.index.to_numpy(float), g.to_numpy(float), one)]
            lines.append("| " + " | ".join(cells) + " |")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", type=Path); ap.add_argument("--first-run", type=Path, default=None)
    a = ap.parse_args()
    for title, lines in (("Contrast (SD) against tilt, Spearman, first volume / middle of the recording", contrast_table(a.root)),
                         ("Contrast against obliqueness, one side of the fold at a time", side_counts(a.root)),
                         ("Sharpness against tilt, Spearman, first volume / middle of the recording", sharpness_table(a.root, a.first_run)),
                         ("Mean absolute image change (grey levels)", curve_table(a.root))):
        print(f"### {title}\n"); print("\n".join(lines)); print()


if __name__ == "__main__":
    main()
