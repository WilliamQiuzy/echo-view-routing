#!/usr/bin/env python
"""Training-loss curves of the nine fold-output correction runs from their run manifests (per-epoch training records
only; no development data was evaluated during training and no epoch was chosen).

  python scripts/plot_fold_output_correction_curves.py runs/fold_output_correction_v1/train \
      docs/results/2026-10-08_fold_output_correction/training_curves.png
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

src = Path(sys.argv[1]); out = Path(sys.argv[2])
runs = {p.parent.name: json.loads(p.read_text()) for p in sorted(src.glob("*/run_manifest.json"))}
colors = {"T-in": "#4c72b0", "T-out": "#dd8452", "P-out": "#55a868"}; styles = {0: "-", 1: "--", 2: ":"}
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
for name, r in runs.items():
    arm, seed = r["arm"], int(r["seed"]); x = [e["epoch"] for e in r["records"]]; loss = [e["loss"] for e in r["records"]]
    if arm == "P-out":
        axes[2].plot(x, [v["ce"] for v in loss], styles[seed], color=colors[arm], lw=1.2, label=f"P-out seed {seed}")
        continue
    axes[0].plot(x, [v["total"] for v in loss], styles[seed], color=colors[arm], lw=1.2, label=f"{arm} seed {seed}")
    axes[1].plot(x, [v["cls"] for v in loss], styles[seed], color=colors[arm], lw=1.0, label=f"{arm} seed {seed}, classification")
    axes[1].plot(x, [v["boundary"] for v in loss], styles[seed], color=colors[arm], lw=1.0, alpha=0.45)
axes[0].set_title("ASRF total loss", fontsize=10); axes[1].set_title("ASRF classification (dark) and boundary (light) loss", fontsize=10)
axes[2].set_title("P-out class-weighted cross entropy", fontsize=10)
for ax in axes:
    ax.set_yscale("log"); ax.set_xlabel("epoch (375 updates each)"); ax.grid(alpha=0.3, which="both")
axes[0].legend(fontsize=8, ncol=2); axes[2].legend(fontsize=8)
fig.suptitle("Fold-output correction: training losses (per-update means per epoch; the final epoch is kept, nothing is chosen)", fontsize=11)
fig.tight_layout(); fig.savefig(out, dpi=130)
print(out)
