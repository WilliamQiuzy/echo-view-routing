#!/usr/bin/env python
"""Calibration curves of the six L1 + ASRF runs from the round's summary.json (per-epoch records; no recording ids).

  python scripts/plot_l1_asrf_curves.py docs/results/2026-10-07_l1_asrf_integration/summary.json \
      docs/results/2026-10-07_l1_asrf_integration/training_curves.png
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

src = Path(sys.argv[1]); out = Path(sys.argv[2])
runs = json.loads(src.read_text())["runs"]
panels = (("segment_f1_50_multi", "Segment F1@50 (multi streams)", (0.94, 0.985)),
          ("native_macro_f1", "Native macro-F1", (0.965, 0.996)),
          ("boundary_recall_multi", "Boundary recall (multi streams)", (0.94, 0.995)),
          ("native_internal_cuts_per_clip", "Cuts per native clip", (0.0, 0.06)))
colors = {"A": "#4c72b0", "B": "#dd8452"}; styles = {0: "-", 1: "--", 2: ":"}
fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True)
for ax, (key, title, ylim) in zip(axes.flat, panels):
    for name, r in sorted(runs.items()):
        arm, seed = name.split("_s"); seed = int(seed); rec = r["records"]; ch = r["chosen"]["epoch"]
        x = [e["epoch"] for e in rec]; y = [e[key] for e in rec]
        ax.plot(x, y, styles[seed], color=colors[arm], lw=1.2, label=f"{arm} seed {seed}")
        ax.plot([ch], [rec[ch][key]], "o", color=colors[arm], ms=5)
    if key == "native_macro_f1":
        ax.axhline(0.98, color="grey", lw=0.8, ls="-.")
    if key == "boundary_recall_multi":
        ax.axhline(0.95, color="grey", lw=0.8, ls="-.")
    ax.set_title(title, fontsize=10); ax.set_ylim(*ylim); ax.grid(alpha=0.3)
for ax in axes[1]:
    ax.set_xlabel("epoch (375 updates each)")
axes[0, 0].legend(fontsize=8, ncol=2)
fig.suptitle("L1 + ASRF: calibration half after every epoch (dots: chosen epoch; dash-dot: checkpoint-rule floors)", fontsize=11)
fig.tight_layout(); fig.savefig(out, dpi=130)
print(out)
