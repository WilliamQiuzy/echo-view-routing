#!/usr/bin/env python
"""Figure of the paired permutation effects (2026-10-01 round) from report_permutation_round.py's effects.json.

Panels: EV9V native internal cuts per clip and 4-8 fragment boundary F1 (random and block orders, B0 decoders, plus
EchoPrime under random order); EchoNet-Pediatric native internal cuts per clip and exam-stream boundary F1 (random
order, descriptive because classification transfer failed). Points are permuted-minus-true effects with 95% intervals.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

LABEL = {"b1": "B1 smoothing", "b2": "B2 HMM", "b3": "B3 default", "b3a": "B3 audited", "b4": "B4 MS-TCN v2"}
CONDS = [("random", "random"), ("block5", "block 5"), ("block10", "block 10"), ("block20", "block 20")]
SHOWN = {"block5": "block 5", "block10": "block 10", "block20": "block 20", "echoprime": "random"}
COLORS = {"random": "#c0392b", "block5": "#2471a3", "block10": "#5dade2", "block20": "#aed6f1", "echoprime": "#7d3c98"}


def panel(ax, rows: list[tuple[str, str, dict]], title: str, xlabel: str) -> None:
    """rows: (method label, condition key, effect entry)."""
    for y, (lab, cond, e) in enumerate(rows):
        lo, hi = e["ci95"]
        ax.errorbar(e["effect"], y, xerr=[[e["effect"] - lo], [hi - e["effect"]]], fmt="o", color=COLORS[cond], ms=4, capsize=2)
    ax.axvline(0, color="0.4", lw=0.8)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([f"{lab} · {SHOWN.get(c, c)}" for lab, c, _ in rows], fontsize=7)
    ax.invert_yaxis(); ax.set_title(title, fontsize=9); ax.set_xlabel(xlabel, fontsize=8); ax.tick_params(axis="x", labelsize=7)


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot permutation effects"); ap.add_argument("--effects", required=True); ap.add_argument("--out", required=True)
    args = ap.parse_args(); d = json.loads(Path(args.effects).read_text())
    ev, ep, pe = d["ev9v/b0"], d["ev9v/echoprime"], d.get("pediatric")
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    for col, (getter, title, xl) in enumerate([
            (lambda blk, m: blk["natives"][m]["cuts per clip"], "EV9V native clips: internal cuts per clip", "permuted − true (cuts per clip)"),
            (lambda blk, m: blk["banks"]["multi_test"][m]["boundary F1"], "EV9V 4–8 fragment streams: boundary F1 at 0.5 s", "permuted − true")]):
        rows = [(LABEL[m], c, getter(ev, m)[c]) for m in ("b1", "b2", "b3", "b3a") for c, _ in CONDS]
        rows += [(f"EchoPrime + {LABEL[m]}", "echoprime", getter(ep, m)["random"]) for m in ("b2", "b3a")]
        panel(axes[0, col], rows, title, xl)
    if pe:
        for col, (getter, title) in enumerate([
                (lambda m: pe["natives"][m]["cuts per clip"], "Pediatric native clips (descriptive): cuts per clip"),
                (lambda m: pe["banks"]["exam"][m]["boundary F1"], "Pediatric exam streams (descriptive): boundary F1")]):
            panel(axes[1, col], [(LABEL[m], "random", getter(m)["random"]) for m in ("b1", "b2", "b3", "b3a", "b4")], title, "random − true")
    handles = [plt.Line2D([], [], marker="o", ls="", color=COLORS[k], label=v) for k, v in
               [("random", "random order"), ("block5", "block 5"), ("block10", "block 10"), ("block20", "block 20"), ("echoprime", "EchoPrime, random order")]]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=8, frameon=False)
    fig.suptitle("Paired permutation effects with 95% bootstrap intervals (10 seeds per permuted condition)", fontsize=10)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97)); fig.savefig(args.out, dpi=150); print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
