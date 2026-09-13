#!/usr/bin/env python
"""Mac-side demo: render the baseline ladder, the four-cell figure and risk-coverage curves from pulled artifacts.

Usage: .venv/bin/python demo/demo_cli.py --run runs/<run_id>   (defaults to the newest baselines run)
Reads only runs/<run>/metrics/*.json and reports/ladder.md — no server access needed during the meeting.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from echo_routing.evaluate.report import bfile_rows, ladder_rows, to_markdown  # noqa: E402


def newest_run(runs_root: Path) -> Path:
    cands = sorted(p for p in runs_root.glob("*-baselines-*") if (p / "metrics").is_dir())
    if not cands:
        raise SystemExit(f"no baselines run under {runs_root}; run remote/pull_results.sh first")
    return cands[-1]


def four_cell_table(metrics: list[dict]) -> str:
    lines = ["### Four-cell design: false splits (same view) vs missed changes (different view), tolerance 0.5 s", "",
             "| method | same_none false-split | same_edit false-split | diff_none missed | diff_edit missed | boundary F1@0.5s |", "|---|---|---|---|---|---|"]
    for m in metrics:
        if m["method_id"] == "b_file":
            continue
        c = m["constructed"]["cells"]; f = lambda k, v: "—" if c.get(k, {}).get(v) is None else f"{c[k][v]:.3f}"
        b = m["constructed"]["boundary"]["0.5"]["f1"]
        lines.append(f"| {m['method_id']} | {f('same_none','false_split_rate')} | {f('same_edit','false_split_rate')} | "
                     f"{f('diff_none','missed_rate')} | {f('diff_edit','missed_rate')} | {b:.3f} |")
    return "\n".join(lines) + "\n"


def plot(metrics: list[dict], out: Path) -> Path | None:
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping figures"); return None
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    cells = ["same_none", "same_edit", "diff_none", "diff_edit"]
    ms = [m for m in metrics if m["method_id"] != "b_file"]
    w = 0.8 / max(len(ms), 1)
    for i, m in enumerate(ms):
        c = m["constructed"]["cells"]
        vals = [c.get(k, {}).get("false_split_rate" if k.startswith("same") else "missed_rate") or 0 for k in cells]
        axes[0].bar([j + i * w for j in range(4)], vals, width=w, label=m["method_id"])
    axes[0].set_xticks([j + 0.4 - w / 2 for j in range(4)]); axes[0].set_xticklabels(["same/none\nfalse split", "same/edit\nfalse split", "diff/none\nmissed", "diff/edit\nmissed"])
    axes[0].set_ylim(0, 1); axes[0].set_title("Four-cell errors (test streams)"); axes[0].legend(fontsize=8)
    for m in ms:
        curve = m["constructed"].get("risk_coverage_curve", [])
        pts = [(p["coverage"], p["risk"]) for p in curve if p["risk"] is not None]
        if pts:
            axes[1].plot([p[0] for p in pts], [p[1] for p in pts], marker=".", label=m["method_id"])
        r = m["constructed"].get("routing")
        if r and r.get("risk") is not None:
            axes[1].scatter([r["coverage"]], [r["risk"]], s=60, zorder=5, edgecolor="k")
    axes[1].axhline(0.05, ls="--", c="gray", lw=1); axes[1].set_xlabel("coverage (accepted duration / all)"); axes[1].set_ylabel("risk (wrong accepted / accepted)")
    axes[1].set_title("Risk–coverage (frozen 5% policy marked)"); axes[1].legend(fontsize=8)
    fig.tight_layout(); out.parent.mkdir(parents=True, exist_ok=True); fig.savefig(out, dpi=140)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--run", default=None); ap.add_argument("--no-fig", action="store_true")
    a = ap.parse_args()
    run = Path(a.run) if a.run else newest_run(REPO / "runs")
    metrics = [json.loads(p.read_text()) for p in sorted((run / "metrics").glob("*.json"))]
    print(f"# Demo — {run.name}\n")
    print(to_markdown(ladder_rows(metrics), "Temporal routing ladder (test; frozen validation-selected 5% policy)"))
    print(to_markdown(bfile_rows(metrics), "B-file operational comparator (native files)"))
    print(four_cell_table(metrics))
    if not a.no_fig:
        fig = plot(metrics, run / "reports" / "demo_figure.png")
        if fig:
            print(f"figure: {fig}")


if __name__ == "__main__":
    main()
