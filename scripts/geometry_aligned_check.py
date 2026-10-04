#!/usr/bin/env python
"""Aligned comparison of the slice pipeline with EchoSlicer's own rendering (project lead's review of 2026-10-04).

The first pipeline check (scripts/geometry_pipeline_check.py) compared classifier scores of two different images: the
pipeline's fixed 160 x 150 mm grid and EchoSlicer's own grid, extent, in-plane rotation and flips. Here both render the
same A4C plane under the same conditions, frame 0:
- the same physical sample points: EchoSlicer's own grid (its slice_volume), whose spherical query points are turned
  into probe-frame points and sampled by the pipeline's volume classes;
- the same display: EchoSlicer's in-plane axes, rotation, flips and even-size trimming come with its grid;
- the same grey processing: EchoSlicer's per-frame min-max to uint8, applied to both.
Correctness is judged on pixels and coordinates. Every EchoSlicer sample point must lie on the pipeline's plane
(planes.from_echoslicer), and the two images must agree on shared pixels with a best shift of zero. For STACOM the two
paths are independent: EchoSlicer interpolates a spherical resampling (0.5 mm, 1 degree, as in the verification) and
the pipeline interpolates the Cartesian volume directly. For EchoSlicer's own volumes both use its spherical grid, so
the check covers the coordinate conversions only. Classifier scores of both images are recorded as an auxiliary check.
Writes <out>/aligned_check.json. Runs in EchoSlicer's environment.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from echo_routing.geometry import planes, stacom, volumes  # noqa: E402
from geometry_pipeline_check import best_shift, echoslicer_volume  # noqa: E402


def echoslicer_grey(values: np.ndarray, h: int, w: int) -> np.ndarray:
    """EchoSlicer's math_utils.interpolate for one frame: min-max to uint8 (truncation), then drop an odd last row or
    column. Returns the grey image (its RGB copies are identical)."""
    x = values.reshape(h, w).astype(np.float64); lo, hi = x.min(), x.max()
    u8 = ((x - lo) * (255.0 / (hi - lo))).astype(np.uint8)
    return u8[: h - h % 2, : w - w % 2]


def probe_points(spherical_grid: np.ndarray) -> np.ndarray:
    r, p, t = spherical_grid[:, 0], spherical_grid[:, 1], spherical_grid[:, 2]
    return np.stack([r * np.cos(t) * np.cos(p), r * np.sin(t), r * np.cos(t) * np.sin(p)], 1)


def compare(ve, ex, math_utils, data, grid, cart_points_fn, inside_fn, row) -> dict:
    """EchoSlicer's rendering against the pipeline's at EchoSlicer's sample points (frame 0)."""
    view = ex.ViewParams("A4C", row.d, row.phi, row.theta, bool(row.flipu), bool(row.flipv), row.rotation_angle, row.cm_pix, row.h_cm, row.w_cm)
    pos = math_utils.convert_to_cartesian(data[..., 0], *grid)[0]
    sg, _, (w_px, h_px) = ve.slice_volume(view, pos)
    es = np.asarray(math_utils.interpolate(grid, data, sg, h_px, w_px, first_frame=True))[..., 0]
    pts = probe_points(np.stack(sg, 1))       # slice_volume returns (rho, phi, theta) as flat arrays, rows over height
    ours = echoslicer_grey(cart_points_fn(pts), h_px, w_px)
    inside = inside_fn(pts).reshape(h_px, w_px)[: es.shape[0], : es.shape[1]]
    plane = planes.from_echoslicer(row.d, row.phi, row.theta); n = plane.basis()[2]
    off_plane = float(np.abs(pts @ n - plane.offset_mm).max())
    a, b = es.astype(np.float64), ours.astype(np.float64)
    rgb = lambda g: np.repeat(g[..., None], 3, -1)
    return {"image_shape": list(es.shape), "max_point_distance_to_pipeline_plane_mm": off_plane,
            "shared_fraction": float(inside.mean()), "pearson_r": float(np.corrcoef(a[inside], b[inside])[0, 1]),
            "mean_abs_grey": float(np.abs(a - b)[inside].mean()), "best_shift_px": list(best_shift(a, b, inside)),
            "a4c_prob_echoslicer_image": float(ve._view_probs(rgb(es), "A4C")),
            "a4c_prob_pipeline_image": float(ve._view_probs(rgb(ours), "A4C")), "a4c_prob_recorded": float(row.p)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--echoslicer", required=True); ap.add_argument("--echoslicer-data", required=True)
    ap.add_argument("--echoslicer-volumes", nargs="*", default=["A_0", "B_0", "C_0"])
    ap.add_argument("--stacom-root", required=True); ap.add_argument("--probe-frames", required=True)
    ap.add_argument("--stacom-cases", nargs="*", default=[]); ap.add_argument("--first-check", required=True,
                    help="pipeline_check.json of the first check, whose scores are reported alongside")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(); sys.path.insert(0, a.echoslicer)
    import stacom_verify as sv
    from extractor import extractor as ex, math_utils
    ve = ex.ViewExtractor(); first = json.loads(Path(a.first_check).read_text()); report = {"echoslicer": {}, "stacom": {}}
    es = Path(a.echoslicer_data)
    for v in a.echoslicer_volumes:
        vol = echoslicer_volume(es / "unpacked" / "dataset" / f"{v}.dcm", v); row = pd.read_csv(es / "extracted" / v / "A4C.csv").iloc[0]
        r = compare(ve, ex, math_utils, vol.data, vol.grid, lambda p: vol.sample(p, 0)[0], vol.inside, row)
        r["a4c_prob_pipeline_fixed_grid_first_check"] = first["echoslicer"][v]["a4c_prob_pipeline"]
        report["echoslicer"][v] = r; print(v, json.dumps(r), flush=True)
    pf = json.loads(Path(a.probe_frames).read_text())["cases"]
    for c in a.stacom_cases:
        case = stacom.read_case(Path(a.stacom_root) / c / "3DUS"); sector = stacom.sector_geometry(case.volume, case.spacing)
        beam, el, az = (np.array(x) for x in pf[c]["frame_rows_beam_elevation_azimuth"])
        cart = volumes.CartesianVolume(case.volume, case.spacing, sector.apex_mm, np.stack([beam, el, az]), sector.mask,
                                       np.arange(case.volume.shape[0]) * case.frame_time_ms / 1000.0, c)
        first_frame = stacom.Case(case.volume[:1], case.spacing, case.frame_time_ms, case.dicom_equal_fraction)
        data, bounds = sv.to_spherical(first_frame, sector, el, az)                 # the verification's resampling
        grid = tuple(np.linspace(lo, hi, n) for (lo, hi), n in zip(bounds, data.shape[:3]))
        row = pd.read_csv(pf[c]["views_csv"]).set_index("name").loc["A4C"]
        r = compare(ve, ex, math_utils, data, grid, lambda p: cart.sample(p, 0)[0], cart.inside, row)
        r["a4c_prob_pipeline_fixed_grid_first_check"] = first["stacom"][c]["a4c_prob_pipeline"]
        report["stacom"][c] = r; print(c, json.dumps(r), flush=True)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "aligned_check.json").write_text(json.dumps(report, indent=1, allow_nan=False))


if __name__ == "__main__":
    main()
