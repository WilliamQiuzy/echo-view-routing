#!/usr/bin/env python
"""Figures for the PLOS ONE draft (docs/paper/), from aggregate outputs only (no data access, no training).

Inputs (server paths): runs/_demo/permutation_effects/effects.json, runs/_demo/mstcn_matrix/matrix.json,
runs/_demo/true_order_joint_b5/joint.json, runs/_demo/pediatric/joint_v1/pediatric_joint.json.
Outputs: Fig1.png ... Fig4.png (300 dpi, at most 7.5 in wide) in --out.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "font.family": "DejaVu Sans"})
C = {"true": "#4d4d4d", "random": "#c0392b", "block5": "#1f5f99", "block10": "#4a90c9", "block20": "#9cc6e6",
     "b2": "#c0392b", "b3": "#e67e22", "b3a": "#1f5f99", "b4": "#2e8b57", "b0": "#7f7f7f", "b1": "#8e44ad"}
NAME = {"b0": "B0 frame", "b1": "B1 smoothing", "b2": "HMM", "b3": "B3 default", "b3a": "B3 audited", "b4": "MS-TCN"}


def errbar(ax, x, e, color, marker="o", label=None):
    lo, hi = e["ci95"]; v = e["effect"] if "effect" in e else e["value"]
    ax.errorbar(x, v, yerr=[[v - lo], [hi - v]], fmt=marker, color=color, ms=4, capsize=2, lw=1, label=label)


def fig1(out: Path) -> None:
    """Study design: (A) frame-name order, (B) evaluation units and orders, (C) 2 x 2 training/test design."""
    fig = plt.figure(figsize=(7.5, 4.0))
    # A: name order vs numeric order
    ax = fig.add_axes([0.02, 0.58, 0.46, 0.36]); ax.axis("off"); ax.set_title("A  Frame files read in name order", loc="left")
    true = list(range(12)); name = sorted(true, key=str)
    for row, (lab, seq) in enumerate((("true order", true), ("name order", name))):
        y = 0.62 - row * 0.42
        ax.text(0.0, y + 0.12, lab, fontsize=7.5)
        for i, f in enumerate(seq):
            col = "#d9d9d9" if f == true[i] else "#f4b6b0"
            ax.add_patch(mpatches.Rectangle((0.2 + i * 0.066, y), 0.06, 0.24, color=col, transform=ax.transAxes))
            ax.text(0.23 + i * 0.066, y + 0.08, f"{f}", ha="center", fontsize=6.5, transform=ax.transAxes)
    ax.text(0.0, -0.06, "Files 0.jpg, 1.jpg, ... sorted as text: 0, 1, 10, 11, 2, ... (shaded: displaced)", fontsize=6.5)
    # B: units and within-fragment orders
    ax = fig.add_axes([0.52, 0.58, 0.46, 0.36]); ax.axis("off"); ax.set_title("B  Same samples, different order", loc="left")
    rng = np.random.default_rng(3); frag = np.arange(15)
    orders = {"true": frag, "random": rng.permutation(15),
              "block 5": np.concatenate([frag[5:10], frag[10:15], frag[0:5]]), "reverse": frag[::-1]}
    cmap = plt.get_cmap("viridis")
    for row, (lab, seq) in enumerate(orders.items()):
        y = 0.78 - row * 0.24
        ax.text(0.0, y + 0.04, lab, fontsize=7)
        for i, f in enumerate(seq):
            ax.add_patch(mpatches.Rectangle((0.2 + i * 0.05, y), 0.046, 0.15, color=cmap(f / 14), transform=ax.transAxes))
    ax.text(0.0, -0.10, "Each fragment keeps its frames, probabilities and features;\nonly the order of its samples changes", fontsize=6.5)
    # C: 2 x 2 design
    ax = fig.add_axes([0.02, 0.10, 0.46, 0.34]); ax.axis("off"); ax.set_title("C  MS-TCN training x test order (seeds 0, 1, 2)", loc="left")
    cells = [("train true\ntest true", "deployment"), ("train true\ntest shuffled", "test order only"),
             ("train shuffled\ntest true", "training order only"), ("train shuffled\ntest shuffled", "both shuffled")]
    for k, (t, sub) in enumerate(cells):
        x = 0.05 + (k % 2) * 0.48; y = 0.52 - (k // 2) * 0.46
        ax.add_patch(mpatches.FancyBboxPatch((x, y), 0.42, 0.36, boxstyle="round,pad=0.01", fc="#eef3f8", ec="#4a90c9", transform=ax.transAxes))
        ax.text(x + 0.21, y + 0.21, t, ha="center", va="center", fontsize=7, transform=ax.transAxes)
        ax.text(x + 0.21, y + 0.05, sub, ha="center", fontsize=6.3, color="#555555", transform=ax.transAxes)
    # D: units
    ax = fig.add_axes([0.52, 0.10, 0.46, 0.34]); ax.axis("off"); ax.set_title("D  Evaluation units", loc="left")
    rows = [("native clip", [("A4C", 1.0)]), ("pair stream", [("A4C", 0.5), ("PSAX", 0.5)]),
            ("4-8 fragment stream", [("PLAX", 0.2), ("PLAX", 0.15), ("A5C", 0.25), ("A4C", 0.2), ("SC4C", 0.2)])]
    pal = {"A4C": "#9cc6e6", "PSAX": "#f7c59f", "PLAX": "#b8e0b8", "A5C": "#d7b8e8", "SC4C": "#f4b6b0"}
    for row, (lab, segs) in enumerate(rows):
        y = 0.70 - row * 0.30; x = 0.30
        ax.text(0.0, y + 0.05, lab, fontsize=7)
        for v, w in segs:
            ax.add_patch(mpatches.Rectangle((x, y), w * 0.66, 0.16, fc=pal[v], ec="white", lw=1.5, transform=ax.transAxes))
            ax.text(x + w * 0.33, y + 0.05, v, ha="center", fontsize=6.3, transform=ax.transAxes); x += w * 0.66
    ax.text(0.0, -0.08, "Joins between different views score boundaries;\nsame-view joins and clip interiors score cuts", fontsize=6.3)
    fig.savefig(out / "Fig1.png", dpi=300); plt.close(fig)


def fig2(eff: dict, out: Path) -> None:
    ev = eff["ev9v/b0"]; ep = eff["ev9v/echoprime"]
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.9), gridspec_kw={"width_ratios": [1.15, 1.15, 1]})
    conds = ["random", "block5", "block10", "block20"]
    for ax, (getter, title, ylab) in zip(axes[:2], [
            (lambda b, m: b["natives"][m]["cuts per clip"], "A  Internal cuts per native clip", "shuffled − true"),
            (lambda b, m: b["banks"]["multi_test"][m]["boundary F1"], "B  Boundary F1, 4–8 fragment streams", "shuffled − true")]):
        for i, m in enumerate(["b1", "b2", "b3", "b3a"]):
            for j, c in enumerate(conds):
                errbar(ax, i + (j - 1.5) * 0.17, getter(ev, m)[c], C[c], label=c if i == 0 else None)
        for i, m in enumerate(["b2", "b3a"]):
            errbar(ax, 4.2 + i, getter(ep, m)["random"], C["random"], marker="D", label="EchoPrime, random" if i == 0 else None)
        ax.axhline(0, color="0.5", lw=0.7)
        ax.set_xticks([0, 1, 2, 3, 4.2, 5.2]); ax.set_xticklabels(["B1", "HMM", "B3 def.", "B3 aud.", "EP HMM", "EP B3 aud."], rotation=35, ha="right")
        ax.set_title(title, loc="left"); ax.set_ylabel(ylab)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, ["random", "block 5", "block 10", "block 20", "EchoPrime (EP), random"], loc="lower center", ncol=5, frameon=False, fontsize=7)
    ax = axes[2]; share = ev["run_length_share"]
    bins = ["1", "2", "3-4", "5-9", "10-19", ">=20"]; shades = ["#fde0dd", "#fbb4b9", "#f768a1", "#c51b8a", "#7a0177", "#49006a"]
    for i, c in enumerate(["true", "random", "block5", "block10", "block20"]):
        bottom = 0.0
        for b, col in zip(bins, shades):
            v = share[f"raw|{c}"][b]; ax.bar(i, v, bottom=bottom, color=col, width=0.7, label=b if i == 0 else None); bottom += v
    ax.set_xticks(range(5)); ax.set_xticklabels(["true", "random", "block 5", "block 10", "block 20"], rotation=30)
    ax.set_ylabel("share of B0 deviating samples"); ax.set_title("C  Length of B0 deviation runs", loc="left")
    ax.legend(title="run length", frameon=False, fontsize=6, title_fontsize=6.5, bbox_to_anchor=(1.0, 1.0), loc="upper left")
    fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig(out / "Fig2.png", dpi=300); plt.close(fig)


def fig3(mat: dict, out: Path) -> None:
    seeds = sorted(mat["seeds"], key=int)
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.8), gridspec_kw={"width_ratios": [1.2, 1, 1]})
    ax = axes[0]; key = "native: cuts per clip"
    for i, s in enumerate(seeds):
        cells = mat["seeds"][s][key]["cells"]
        tt, tr = cells["train true / test true"], cells["train true / test random"]
        rt, rr = cells["train rand / test true"], cells["train rand / test random"]
        ax.plot([i - 0.15, i + 0.15], [tt, tr], "-o", color=C["b4"], ms=3.5, lw=1, label="MS-TCN, true-trained" if i == 0 else None)
        ax.plot([i - 0.15, i + 0.15], [rt, rr], "--s", color="#7fbf7b", ms=3.5, lw=1, label="MS-TCN, shuffle-trained" if i == 0 else None)
    b2_true = 0.211; b2_rand = b2_true + mat["seeds"][seeds[0]][key]["test effect (simple:b2)"]["value"]
    ax.plot([2.65, 2.95], [b2_true, b2_rand], "-^", color=C["b2"], ms=4, lw=1, label="HMM (no training)")
    ax.set_xticks(list(range(len(seeds))) + [2.8]); ax.set_xticklabels([f"seed {s}" for s in seeds] + ["HMM"])
    ax.set_ylabel("internal cuts per native clip"); ax.set_title("A  True vs shuffled test (left/right point)", loc="left")
    ax.legend(frameon=False, fontsize=6.3, loc="lower left")
    ax.set_ylim(0.0, 0.27)
    ax = axes[1]
    for j, (m, lab) in enumerate((("simple:b2", "HMM"), ("simple:b3", "B3 default"), ("simple:b3a", "B3 audited"))):
        for i, s in enumerate(seeds):
            e = mat["seeds"][s][key][f"method x test order ({m} vs MS-TCN trained in true order)"]
            errbar(ax, i + (j - 1) * 0.22, e, [C["b2"], C["b3"], C["b3a"]][j], label=lab if i == 0 else None)
    ax.axhline(0, color="0.5", lw=0.7); ax.set_xticks(range(len(seeds))); ax.set_xticklabels([f"seed {s}" for s in seeds])
    ax.set_ylabel("(method − MS-TCN) shuffled − true"); ax.set_title("B  Method x test-order contrast", loc="left")
    ax.set_ylim(-0.33, 0.12); ax.legend(frameon=False, fontsize=6.3, loc="lower right", ncol=1)
    ax = axes[2]
    for i, s in enumerate(seeds):
        errbar(ax, i, mat["seeds"][s][key]["training effect (test true)"], "#7fbf7b")
    ax.axhline(0, color="0.5", lw=0.7); ax.set_xticks(range(len(seeds))); ax.set_xticklabels([f"seed {s}" for s in seeds])
    ax.set_ylabel("shuffle-trained − true-trained"); ax.set_title("C  Shuffled training", loc="left")
    fig.tight_layout(); fig.savefig(out / "Fig3.png", dpi=300); plt.close(fig)


def fig4(joint: dict, ped: dict, out: Path) -> None:
    jn = {"b0": "B0 frame classifier", "b2": "B2 HMM", "b3": "B3 change points, default", "b3a": "B3 change points, audited", "b4": "B4 MS-TCN v2"}
    methods = ["b0", "b2", "b3", "b3a", "b4"]
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.3), sharey=True)
    specs = [(axes[0], "A  EV9V test (4–8 fragment streams; native clips)",
              [("clip macro-F1", lambda m: joint["endpoints"][jn[m]]["clip macro-F1"]),
               ("segment F1@50", lambda m: joint["endpoints"][jn[m]]["segment F1@50, multi"]),
               ("boundary F1", lambda m: joint["endpoints"][jn[m]]["boundary F1, multi"])]),
             (axes[1], "B  EchoNet-Pediatric test (exam streams; native clips)",
              [("clip accuracy", lambda m: ped["pediatric_test_true_order"]["clip accuracy"][m]),
               ("segment F1@50", lambda m: ped["pediatric_test_true_order"]["segment F1@50, exam streams"][m]),
               ("boundary F1", lambda m: ped["pediatric_test_true_order"]["boundary F1, exam streams"][m]),
               ("changes found,\nboth views right", lambda m: ped["pediatric_test_true_order"]["changes found with both views right"][m])])]
    for ax, title, metrics in specs:
        w = 0.8 / len(metrics); greys = ["#bdbdbd", "#6baed6", "#c0392b", "#252525"]
        for j, (lab, g) in enumerate(metrics):
            xs = np.arange(len(methods)) + (j - (len(metrics) - 1) / 2) * w
            vals = [g(m)["value"] for m in methods]; lo = [g(m)["ci95"][0] for m in methods]; hi = [g(m)["ci95"][1] for m in methods]
            ax.bar(xs, vals, width=w, color=greys[j], label=lab,
                   yerr=[np.array(vals) - np.array(lo), np.array(hi) - np.array(vals)], error_kw={"lw": 0.7, "capsize": 1.5})
        ax.set_xticks(range(len(methods))); ax.set_xticklabels([NAME[m] for m in methods], rotation=20)
        ax.set_title(title, loc="left"); ax.set_ylim(0, 1.05)
        ax.legend(frameon=False, fontsize=6.2, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=len(metrics))
    axes[0].set_ylabel("score")
    fig.tight_layout(); fig.subplots_adjust(bottom=0.32); fig.savefig(out / "Fig4.png", dpi=300); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Paper figures"); ap.add_argument("--out", required=True)
    ap.add_argument("--effects", default="runs/_demo/permutation_effects/effects.json")
    ap.add_argument("--matrix", default="runs/_demo/mstcn_matrix/matrix.json")
    ap.add_argument("--joint", default="runs/_demo/true_order_joint_b5/joint.json")
    ap.add_argument("--pediatric", default="runs/_demo/pediatric/joint_v1/pediatric_joint.json")
    args = ap.parse_args(); out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    fig1(out)
    fig2(json.loads(Path(args.effects).read_text()), out)
    fig3(json.loads(Path(args.matrix).read_text()), out)
    fig4(json.loads(Path(args.joint).read_text()), json.loads(Path(args.pediatric).read_text()), out)
    print(f"wrote figures to {out}")


if __name__ == "__main__":
    main()
