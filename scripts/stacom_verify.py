#!/usr/bin/env python
"""STACOM 2011 3D echo: decode, coordinate, time and slice checks for one volunteer. Diagnostic only: nothing is
trained and no slice gets a label.

1. Decode. The per-frame VTK volumes must equal the DICOM voxel for voxel; the frame time comes from the DICOM.
2. Coordinates. The probe sector is fitted to the visible region. The challenge's expert LV landmarks (two observers,
   first frame) must sit on brighter wall than their ring centroids in the cavity, and the true position must beat
   copies shifted by up to 12 mm.
3. Field of view and time. LV depth from the probe, landmark margins to the sector edge, room beyond the base, frames
   per cycle and volumes per second.
4. Slices. The volume is resampled onto a probe-centred spherical grid in EchoSlicer's convention, and EchoSlicer's
   classifier-guided search is run. Its scores check that slices look like standard views; they are not labels. The
   probe axes are not stored in the data, so the A4C search tries the four directions of the azimuth axis. The full
   search then runs in the four frames of the better axis: both directions of it, each right-handed and mirrored. A
   180-degree turn about the beam leaves A4C almost unchanged for the classifier but reverses the side the A5C search
   tilts to, so the anterior side is decided across all volunteers, not per case.
Writes <out>/<case>/summary.json and <out>/<case>/views_<sign>/*.csv; PNGs and videos stay with them on the server.
Runs in EchoSlicer's environment: third_party/3d-echo_run/.venv/bin/python scripts/stacom_verify.py ...
"""
from __future__ import annotations

import argparse
import json
import sys
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from echo_routing.geometry import stacom  # noqa: E402

CASES = ["v1", "v2", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16"]
README_LAST_FRAME = dict(zip(CASES, [13, 15, 10, 14, 13, 16, 13, 13, 12, 14, 12, 23, 17, 12, 12]))   # 3DUS_ff
README_ES_FRAME = dict(zip(CASES, [6, 8, 6, 7, 5, 6, 5, 5, 6, 6, 5, 6, 7, 6, 5]))                    # 3DUS_es
AXES = {"+x": np.array([1.0, 0, 0]), "-x": np.array([-1.0, 0, 0]), "+y": np.array([0, 1.0, 0]), "-y": np.array([0, -1.0, 0])}


def to_spherical(case: stacom.Case, sector: stacom.Sector, e: np.ndarray, a: np.ndarray, step_mm: float = 0.5,
                 step_deg: float = 1.0, pad_deg: float = 3.0) -> tuple[np.ndarray, np.ndarray]:
    """Resample every frame onto (rho, phi, theta) around the apex: x along the beam, y = e (elevation), z = a
    (azimuth), as EchoSlicer's X = (R cos T cos P, R sin T, R cos T sin P). Returns (rho, phi, theta, T) float32 and
    the bounds [[rho], [phi], [theta]] in mm and radians."""
    half = {0: sector.half_angle_x_deg, 1: sector.half_angle_y_deg}
    ha, he = (half[int(np.argmax(np.abs(v[:2])))] + pad_deg for v in (a, e))
    rho = np.arange(0.0, sector.rho_max_mm + step_mm, step_mm)
    phi = np.radians(np.arange(-ha, ha + 1e-9, step_deg)); theta = np.radians(np.arange(-he, he + 1e-9, step_deg))
    R, P, T = np.meshgrid(rho, phi, theta, indexing="ij")
    probe = np.stack([R * np.cos(T) * np.cos(P), R * np.sin(T), R * np.cos(T) * np.sin(P)], -1)
    world = sector.apex_mm + probe @ np.stack([sector.axis, e, a])           # rows: beam, elevation, azimuth
    idx = (world[..., ::-1] / case.spacing[::-1]).reshape(-1, 3).T             # (z, y, x) voxel coordinates
    data = np.stack([ndimage.map_coordinates(f.astype(np.float32), idx, order=1, cval=0.0).reshape(R.shape)
                     for f in case.volume], -1)
    bounds = np.array([[rho[0], rho[-1]], [phi[0], phi[-1]], [theta[0], theta[-1]]])
    return data, bounds


def echoslicer_inputs(data: np.ndarray, bounds: np.ndarray, math_utils) -> tuple[tuple, np.ndarray]:
    grid = tuple(np.linspace(lo, hi, n) for (lo, hi), n in zip(bounds, data.shape[:3]))
    return grid, math_utils.convert_to_cartesian(data[..., 0], *grid)[0]


def view_params(extractor_mod, name: str):
    df = pd.read_csv(Path(extractor_mod.__file__).parent / "view_planes.csv").set_index("Name")
    r = df.loc[name]
    return extractor_mod.ViewParams(name, r.d, r.phi, r.theta, r.flipu, r.flipv, r.rotation_angle, r.cm_pix, r.h_cm, r.w_cm)


def plane_world(view, sector: stacom.Sector, e: np.ndarray, a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """EchoSlicer's plane (normal n = (cos t cos p, sin t, cos t sin p), through d*n) in the volume's mm frame."""
    p, t = np.radians(view.phi), np.radians(view.theta)
    n = np.array([np.cos(t) * np.cos(p), np.sin(t), np.cos(t) * np.sin(p)]); M = np.stack([sector.axis, e, a])
    return n @ M, sector.apex_mm + (view.d * n) @ M


TILTS = np.arange(-30.0, 30.0 + 1e-9, 2.5)


def tilt_coverage(view, sector: stacom.Sector, spacing: np.ndarray, e: np.ndarray, a: np.ndarray) -> dict:
    """Share of the start plane's visible area that stays inside the visible region when the plane's normal elevation
    (theta) is tilted, both ways; positive is the direction EchoSlicer searches for A5C."""
    import echoslicer_controlled as ec                                   # same plane basis as the feasibility study
    M = np.stack([sector.axis, e, a]); g = np.arange(-150.0, 150.0 + 1e-9, 1.0); S, T = np.meshgrid(g, g)
    def inside(theta: float) -> np.ndarray:
        n, u, v = ec.basis(view.phi, theta, view.rotation_angle, view.flipu, view.flipv)
        world = sector.apex_mm + (view.d * n + S[..., None] * u + T[..., None] * v).reshape(-1, 3) @ M
        idx = np.round(world[:, ::-1] / spacing[::-1]).astype(int); ok = np.all((idx >= 0) & (idx < sector.mask.shape), 1)
        out = np.zeros(len(idx), bool); out[ok] = sector.mask[tuple(idx[ok].T)]; return out
    base = inside(view.theta); cov = {float(dt): float(inside(view.theta + dt)[base].mean()) for dt in TILTS}
    reach = {side: float(max([abs(dt) for dt, c in cov.items() if np.sign(dt) == side and c >= 0.8] or [0.0])) for side in (1, -1)}
    return {"coverage_by_tilt_deg": cov, "max_tilt_with_80pct_coverage_deg": {"positive": reach[1], "negative": reach[-1]},
            "start_plane_visible_cm2": float(base.sum() / 100.0)}


def landmark_checks(case: stacom.Case, sector: stacom.Sector, gt_dir: Path) -> dict:
    out = {}
    for obs in ("obs1", "obs2"):
        L = stacom.read_landmarks(gt_dir / "LMKS" / "VTK_COORDINATES" / f"{obs}_groundTruth00.vtk")
        levels = stacom.landmark_levels(L); rank, total = stacom.shift_rank(case.volume[0], case.spacing, L)
        rings = [L[g] for g in levels]; dep = [float(stacom.depth(r.mean(0), sector)[0]) for r in rings]
        m_rank, m_total = stacom.mapping_rank(case.volume[0], case.spacing, L)
        contrast = stacom.landmark_contrast(case.volume[0], case.spacing, L, levels=levels)
        out[obs] = {"landmarks": int(len(L)), "contrast_grey": contrast if np.isfinite(contrast) else None,
                    "shift_rank": rank, "shift_copies": total, "axis_mapping_rank": m_rank, "axis_mappings": m_total,
                    "ring_depth_mm_apical_mid_basal": dep,
                    "ring_radius_mm_apical_mid_basal": [float(np.linalg.norm(r - r.mean(0), axis=1).mean()) for r in rings],
                    "apical_to_basal_ring_mm": float(np.linalg.norm(rings[0].mean(0) - rings[2].mean(0))),
                    "min_margin_to_sector_edge_mm": float(stacom.margin_to_edge(L, sector, case.spacing).min()),
                    "room_beyond_basal_ring_mm": float(sector.rho_max_mm - dep[2])}
        out["_points"] = L if obs == "obs1" else out["_points"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case", required=True, choices=CASES); ap.add_argument("--root", required=True, help="data/stacom2011/unpacked")
    ap.add_argument("--echoslicer", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--family", choices=["x", "y"], default=None,
                    help="force the azimuth axis family of the full runs (default: the better A4C search)")
    a = ap.parse_args(); root = Path(a.root); out = Path(a.out) / a.case; out.mkdir(parents=True, exist_ok=True)
    case = stacom.read_case(root / a.case / "3DUS"); sector = stacom.sector_geometry(case.volume, case.spacing)
    t = case.volume.shape[0]
    summary = {"case": a.case, "decode": {"frames": t, "readme_frames": README_LAST_FRAME[a.case] + 1,
                                          "dicom_vtk_equal_fraction": case.dicom_equal_fraction,
                                          "volume_zyx": list(case.volume.shape[1:]), "spacing_mm_xyz": case.spacing.tolist(),
                                          "fov_mm_xyz": (np.array(case.volume.shape[1:][::-1]) * case.spacing).tolist()},
               "time": {"frame_time_ms": case.frame_time_ms, "volumes_per_s": 1000.0 / case.frame_time_ms,
                        "cycle_s": t * case.frame_time_ms / 1000.0, "readme_es_frame": README_ES_FRAME[a.case]},
               "sector": {"apex_mm_xyz": sector.apex_mm.tolist(), "axis": sector.axis.tolist(),
                          "half_angle_deg_x_y": [sector.half_angle_x_deg, sector.half_angle_y_deg],
                          "rho_max_mm": sector.rho_max_mm, "centre_tilt_deg": sector.centre_tilt_deg,
                          "width_fit_rmse_mm": sector.width_fit_rmse_mm, "width_fit_r2": sector.width_fit_r2,
                          "visible_fraction": float(sector.mask.mean())}}
    lm = landmark_checks(case, sector, root / "cMAC" / "GT" / "3DUS" / a.case); L = lm.pop("_points"); summary["landmarks"] = lm

    sys.path.insert(0, a.echoslicer)
    import PIL.Image
    import torchvision
    from extractor import extractor as ex, math_utils
    ve = ex.ViewExtractor()
    a4c = {}
    for name, az in AXES.items():
        el = np.cross(az, sector.axis)                                # right-handed (beam, elevation, azimuth)
        data, bounds = to_spherical(case, sector, el, az); grid, cart = echoslicer_inputs(data, bounds, math_utils)
        view, prob = ve._find_a4c(view_params(ex, "A4C"), data, grid, cart)
        a4c[name] = float(prob)
    best = max(a4c, key=a4c.get); family = a.family or best[1]; summary["a4c_search_prob_by_azimuth_axis"] = a4c
    summary["family_forced"] = a.family is not None
    summary["azimuth_axis_family"] = family
    runs = {}
    for direction, sign in ((d, s_) for d in ("+", "-") for s_ in (1, -1)):
        az = AXES[direction + family]; el = sign * np.cross(az, sector.axis)
        tag = f"{direction}{family}_{'right' if sign == 1 else 'mirrored'}"
        data, bounds = to_spherical(case, sector, el, az)
        try:
            views, videos, probs = ve.get_videos(data, bounds)
        except ValueError as err:                    # e.g. EchoSlicer's LV segmentation finds no ventricle in a wrong frame
            runs[tag] = {"error": f"{type(err).__name__}: {err}"}; continue
        missing = sorted({"A4C", "A5C", "A2C"} - {v.name for v in views})
        if missing:
            runs[tag] = {"error": f"EchoSlicer returned no {missing}"}; continue
        vdir = out / f"views_{tag}"; vdir.mkdir(exist_ok=True); rows = []
        for v, video, p in zip(views, videos, probs):
            rows.append({**v.__dict__, "p": float(p)})
            PIL.Image.fromarray(np.asarray(video[0])).save(vdir / f"{v.name}_frame0.png")
            torchvision.io.write_video(str(vdir / f"{v.name}.mp4"), torch_video(video), fps=max(1, round(1000 / case.frame_time_ms)))
        pd.DataFrame(rows).to_csv(vdir / "views.csv", index=False)
        by = {r["name"]: r for r in rows}; dist = {}; ring = {}
        centroids = np.stack([L[g].mean(0) for g in stacom.landmark_levels(L)])
        for vname in ("A4C", "A2C"):
            n, p0 = plane_world(SimpleNamespace(**by[vname]), sector, el, az)
            dist[vname] = sorted(float(abs(x)) for x in (L - p0) @ n)
            ring[vname] = [float(abs(x)) for x in (centroids - p0) @ n]      # apical, mid, basal ring centres
        cov = tilt_coverage(SimpleNamespace(**by["A4C"]), sector, case.spacing, el, az)
        a5c_tilt = float(by["A5C"]["theta"] - by["A4C"]["theta"])
        runs[tag] = {"probs": {r["name"]: r["p"] for r in rows}, "mean_prob": float(np.mean([r["p"] for r in rows])),
                     "a5c_tilt_from_a4c_deg": a5c_tilt, "a4c_tilt_coverage": cov,
                     "obs1_landmark_distance_mm_sorted": dist, "obs1_ring_centre_distance_mm": ring}
    summary["echoslicer"] = runs
    scored = {k: r for k, r in runs.items() if "mean_prob" in r}             # None when EchoSlicer failed in every frame
    summary["best_frame"] = max(scored, key=lambda k: scored[k]["mean_prob"]) if scored else None
    (out / "summary.json").write_text(json.dumps(summary, indent=1, allow_nan=False)); print(json.dumps(summary, indent=1))


def torch_video(video):
    import torch
    v = np.asarray(video)
    return torch.from_numpy(v if v.ndim == 4 else np.repeat(v[..., None], 3, -1))


if __name__ == "__main__":
    main()
