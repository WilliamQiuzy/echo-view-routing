#!/usr/bin/env python
"""Shape of the CETUS LV cavity masks, for the supervision inventory (docs/results/2026-10-04_cetus_intake/). Runs in
envs/cetus on the usable patients of data/cetus/verify. Numbers only.

- Elongation and roundness: singular values of the cavity's voxel coordinates. s1/s2 measures elongation along the
  long axis; s2/s3 measures how far the cross-section is from round. A round cross-section leaves the rotation about
  the long axis, which separates the apical views, undetermined by the cavity alone.
- Base shape: cross-sectional area along the long axis in 1 mm bins, from the end nearer the probe (apex) to the far
  end (base). A cavity closed by a flat cut at the base keeps a large area up to its last bins; a rounded end tapers.
  Reported: the area of the last 2 mm against the largest area.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from echo_routing.geometry import cetus  # noqa: E402


def shape(mask: np.ndarray, spacing: np.ndarray, apex: np.ndarray, axis: int) -> dict:
    p = cetus.points_mm(cetus.moved(mask, axis), cetus.spacing_xyz(spacing, axis))
    c = p.mean(0); _, sv, vt = np.linalg.svd(p - c, full_matrices=False); u = vt[0]
    t = (p - c) @ u
    if np.mean((p[t < np.percentile(t, 5)] - apex) @ np.array([0, 0, 1.0])) > np.mean((p[t > np.percentile(t, 95)] - apex) @ np.array([0, 0, 1.0])):
        t = -t                                               # t grows from the apex end (nearer the probe) to the base
    bins = np.floor(t - t.min()).astype(int); area = np.bincount(bins) * np.prod(spacing)
    return {"s1_over_s2": float(sv[0] / sv[1]), "s2_over_s3": float(sv[1] / sv[2]),
            "base_last2mm_over_max_area": float(area[-2:].mean() / area.max()),
            "apex_first2mm_over_max_area": float(area[:2].mean() / area.max())}


def main() -> None:
    rows = {}
    for path in sorted(glob.glob("data/cetus/verify/patient*.json")):
        r = json.loads(Path(path).read_text())
        if not r["usable"]:
            continue
        case = cetus.read_case(Path("data/cetus/raw/dataset") / r["patient"]); out = {}
        for ph in cetus.PHASES:
            s = r["phases"][ph]["sector"]
            out[ph] = shape(case.masks[ph], case.spacing_mm, np.array(s["apex_mm"]), s["beam_axis"])
        rows[r["patient"]] = out
    stats = {}
    for ph in cetus.PHASES:
        for k in ("s1_over_s2", "s2_over_s3", "base_last2mm_over_max_area", "apex_first2mm_over_max_area"):
            v = np.array([x[ph][k] for x in rows.values()])
            stats[f"{ph} {k}"] = [float(v.min()), float(np.median(v)), float(v.max())]
    res = {"patients": len(rows), "min_median_max": stats}
    Path("data/cetus/mask_shape.json").write_text(json.dumps({"summary": res, "per_patient": rows}, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
