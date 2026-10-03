#!/usr/bin/env python
"""Figures of the round-4 representative cases (cases.csv of scripts/round4_failure_report.py; diagnostic only).

Each figure shows, around one unit of a multi-fragment stream: the E0 frame-level classifier's class probabilities per
10 Hz sample, the reference labels and the outputs of B3, the three ASRF seeds and B7 with B3, the checks of the
inventory (own-sample mean probability, whole-clip classes, the unit alone through B3 and ASRF), and five frames of the
unit with the contrast variant used in the stream. Frames are from EV9V (CC BY 4.0). Writes case_<n>.png and
cases.md into --out.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from echo_routing.compose.recipe_builder import recipe_from_dict  # noqa: E402
from echo_routing.config import load_paths  # noqa: E402
from echo_routing.ingest.frames import list_frame_paths  # noqa: E402
from echo_routing.manifest import load_manifest  # noqa: E402
from echo_routing.temporal.dev_data import FeatureStore  # noqa: E402
from echo_routing.temporal.joint_visual import load_frame_u8  # noqa: E402
from echo_routing.temporal.windowed import stream_plan  # noqa: E402

CLASSES = ["PLAX", "PSAX", "A4C", "A5C", "SC4C"]
COLOURS = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e"]
SYSTEMS = (("B3", "B3"), ("ASRF_s0", "ASRF s0"), ("ASRF_s1", "ASRF s1"), ("ASRF_s2", "ASRF s2"), ("B7_B3", "B7 with B3"))
MARGIN = 30


def strip(ax, rows: list[tuple[str, np.ndarray]], lo: int) -> None:
    cmap = matplotlib.colors.ListedColormap(COLOURS)
    ax.imshow(np.stack([r for _, r in rows]), aspect="auto", cmap=cmap, vmin=-0.5, vmax=4.5, interpolation="nearest",
              extent=(lo - 0.5, lo + rows[0][1].size - 0.5, len(rows) - 0.5, -0.5))
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([n for n, _ in rows], fontsize=8); ax.set_xlabel("sample (10 Hz)")


def figure(case, unit, arrays: dict, refs: list, frames_of, out: Path) -> str:
    h, sid, a, b = case.half, case.stream, int(unit.start), int(unit.end)
    ref = arrays[f"{h}|ref|{sid}"].astype(int); clf = arrays[f"E0|{h}|clf|{sid}"]
    lo, hi = max(0, a - MARGIN), min(ref.size, b + MARGIN); x = np.arange(lo, hi)
    fig = plt.figure(figsize=(13, 9)); gs = fig.add_gridspec(3, 5, height_ratios=[2.2, 1.6, 2.0])
    ax = fig.add_subplot(gs[0, :])
    for k, c in enumerate(CLASSES):
        ax.plot(x, clf[lo:hi, k], color=COLOURS[k], label=c, lw=1.6 if k == unit.label else 1.0)
    ax.axvspan(a - 0.5, b - 0.5, color="0.85", zorder=0)
    for e in np.flatnonzero(np.diff(ref[lo:hi])) + lo + 1:
        ax.axvline(e - 0.5, color="k", lw=0.6, ls="--")
    ax.set_ylim(0, 1); ax.set_xlim(lo - 0.5, hi - 0.5); ax.set_ylabel("E0 classifier probability"); ax.legend(ncol=5, fontsize=8, loc="upper right")
    ax.set_title(f"{case.category}: {sid}, unit {int(case.unit)} ({CLASSES[unit.label]}, {unit.length} samples), {h} half", fontsize=10)
    rows = [("reference", ref[lo:hi])] + [(name, arrays[f"E0|{h}|{s}|labels|{sid}"][lo:hi].astype(int)) for s, name in SYSTEMS]
    strip(fig.add_subplot(gs[1, :]), rows, lo)
    picks = np.linspace(a, b - 1, 5).round().astype(int)
    for i, p in enumerate(picks):
        r = refs[p]; axf = fig.add_subplot(gs[2, i]); axf.imshow(load_frame_u8(frames_of(r.video_id)[r.frame_idx], r.gamma)); axf.axis("off")
        axf.set_title(f"sample {p}: {r.video_id[-12:]} f{r.frame_idx}" + ("" if r.gamma == 1.0 else f" g{r.gamma:g}"), fontsize=7)
    own = clf[a:b].mean(0)
    note = (f"own-sample mean: {CLASSES[int(own.argmax())]} ({own.max():.2f}), {CLASSES[unit.label]} {own[unit.label]:.2f}; whole source clip(s): "
            + ", ".join(CLASSES[int(c)] for c in str(unit.full_clip_classes).split(";")) + f"; alone: B3 {CLASSES[int(unit.iso_B3)]}, ASRF "
            + "/".join(CLASSES[int(getattr(unit, f'iso_ASRF_s{k}'))] for k in range(3)) + "; causes: "
            + ", ".join(f"{name} {getattr(unit, f'{s}_cause')}" for s, name in SYSTEMS if not getattr(unit, f"{s}_matched")))
    fig.text(0.01, 0.005, note, fontsize=8, wrap=True); fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig(out, dpi=110); plt.close(fig)
    return note


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inventory", required=True); ap.add_argument("--cases", required=True); ap.add_argument("--protocol", required=True)
    ap.add_argument("--feature-manifest", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args(); paths = load_paths(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    units = pd.read_csv(Path(a.inventory) / "units.csv", dtype={"stream": str}); units = units[units.enc == "E0"]
    arrays = dict(np.load(Path(a.inventory) / "predictions.npz")); proto = json.loads(Path(a.protocol).read_text())
    store = FeatureStore(paths.cache_root, json.loads(Path(a.feature_manifest).read_text())["encoder"]["cache_key"])
    dirs = load_manifest(paths.manifest_path).set_index("video_id")["frames_dir"]; cache: dict[str, list] = {}
    frames_of = lambda v: cache.setdefault(v, list_frame_paths(Path(dirs[v])))  # noqa: E731
    md = ["# Representative cases (round 4; diagnostic only; frames from EV9V, CC BY 4.0)", ""]
    for n, case in enumerate(pd.read_csv(a.cases, dtype={"stream": str}).itertuples(), 1):
        unit = units[(units.half == case.half) & (units.stream == case.stream) & (units.unit == int(case.unit))].iloc[0]
        recipe = next(recipe_from_dict(r) for r in proto["validation"][case.half]["banks"]["multi"]["recipes"] if r["recipe_id"] == case.stream)
        note = figure(case, unit, arrays, stream_plan(recipe, store.get), frames_of, out / f"case_{n}.png")
        md += [f"## Case {n}: {case.category} ({case.cause})", "", f"![case {n}](case_{n}.png)", "", note, ""]
    (out / "cases.md").write_text("\n".join(md)); print("\n".join(md))


if __name__ == "__main__":
    main()
