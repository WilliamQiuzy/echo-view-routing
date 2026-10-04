#!/usr/bin/env python
"""Slice-artifact control (configs/experiments/slice_artifact_control_v1.yaml). Diagnostic only: no network is trained
and no slice gets a view label.

For each volume, three start planes derived geometrically from EchoSlicer's selected A4C plane (S1 as selected; S2 with
the normal's azimuth rotated by +20 degrees; S3 with the normal's elevation set to 90 degrees, parallel to the volume's
central azimuthal plane), and EchoSlicer's A2C plane as a fourth, different fan (S4, added after the first run), are
tilted from -15 to +15 degrees in 0.5-degree steps at two times (the first volume and the middle of the recording; each
file is a live recording of about 3 s and several heartbeats, not one cycle), on one fixed grid per start plane with raw
intensities. No plane is used as a view label. Every measure is taken inside the evaluation
region: the pixels inside the volume in every frame of the start plane's sweeps, shrunk by a margin so that neither the
Laplacian nor the blur of the matched control reaches a pixel outside that common region (amendment after the first
run, which had no margin). Inside it the script records contrast (SD), sharpness (variance of the Laplacian), relative
sharpness (sharpness / SD^2, descriptive, added after the first run) and mean, raw and after matching every frame's
sharpness to the sweep's lowest (smallest Gaussian blur found by bisection). As a post-hoc check of the interpolation
hypothesis it also records the contrast of a nearest-neighbour rendering of the same planes, in which no pixel averages
several voxels (added after the first run). It tests contrast against the signed tilt
and against obliqueness (the angle to the zero-elevation plane), overall and on each side of the fold nearest the start
plane (theta_n = 90 for S1-S3, 0 for S4), and writes image-change curves against
tilt size (fixed time) and against time gap (fixed plane, non-cyclic pairs), both inside the same evaluation region.
Writes <out>/<volume>/{frames.csv, tilt_curves.csv, time_curves.csv, summary.json} and curves.png.
Runs in EchoSlicer's environment: third_party/3d-echo_run/.venv/bin/python scripts/echoslicer_artifact_control.py ...
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
from scipy import ndimage
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import echoslicer_controlled as ec  # noqa: E402  (same geometry and renderer as the feasibility videos)

TILTS = np.round(np.arange(-15, 15 + 1e-9, 0.5), 1)
CURVE_TILTS = (0.5, 1, 2, 4, 8, 15)
SIGMA_MAX = 4.0                             # upper end of the blur bisection (pixels)
MARGIN = 1 + int(4.0 * SIGMA_MAX + 0.5)     # Laplacian stencil + scipy's Gaussian radius at SIGMA_MAX (truncate 4)
MIN_SIDE = 5                                # frames needed on one side of the fold for a per-side correlation


def start_planes(view: dict, other: dict) -> dict[str, dict]:
    """S1-S3 as declared. S3 keeps S1's offset and azimuth, so its sweep is another window on S1's fan of planes, and S2
    lies only a few degrees from S1. S4 (added after the first run for that reason) starts from EchoSlicer's A2C plane,
    60-76 degrees from S1, used only as a geometric start: a different fan of planes."""
    return {"S1": dict(view), "S2": {**view, "phi": view["phi"] + 20.0}, "S3": {**view, "theta": 90.0}, "S4": dict(other)}


def obliqueness(theta_n: float) -> float:
    """Angle (degrees) between the slice and the central azimuthal plane, whose normal is y: arccos |sin theta_n|.
    Equals |theta_n - 90| for theta_n in [0, 180] and stays correct outside that range."""
    return float(np.degrees(np.arccos(min(1.0, abs(np.sin(np.radians(theta_n)))))))


def read_plane(views: Path, name: str) -> dict:
    view = pd.read_csv(views / f"{name}.csv").iloc[0].to_dict()
    return {**view, "flipu": str(view["flipu"]) == "True", "flipv": str(view["flipv"]) == "True"}


def evaluation_mask(common: np.ndarray, margin: int) -> np.ndarray:
    """The common region shrunk by `margin` pixels in every direction (square neighbourhood). Guarded because scipy
    erodes until nothing changes when iterations < 1."""
    return ndimage.binary_erosion(common, structure=np.ones((3, 3), bool), iterations=margin) if margin > 0 else common


def sharpness(x: np.ndarray, m: np.ndarray) -> float:
    return float(ndimage.laplace(x)[m].var())


def matched(x: np.ndarray, m: np.ndarray, target: float) -> tuple[np.ndarray, float, bool]:
    """The smallest Gaussian blur (bisection on sigma in 0..SIGMA_MAX pixels) that brings the sharpness inside m to target,
    and whether the target was reached (False when even SIGMA_MAX leaves the frame sharper than the target)."""
    if sharpness(x, m) <= target:
        return x, 0.0, True
    lo, hi = 0.0, SIGMA_MAX
    for _ in range(14):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if sharpness(ndimage.gaussian_filter(x, mid), m) > target else (lo, mid)
    y = ndimage.gaussian_filter(x, hi)
    return y, hi, sharpness(y, m) <= target


def sweep(vol, grid, plane: dict, s_vals, t_vals, t: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Every tilt at time t: the linear rendering (as EchoSlicer), the nearest-neighbour rendering (each pixel is one
    voxel's value, no averaging between samples; post-hoc check of the interpolation hypothesis) and the inside mask."""
    shape = (len(t_vals), len(s_vals)); imgs, nearest, masks = [], [], []
    nn = RegularGridInterpolator(grid, vol[..., t].astype(np.float32), method="nearest", bounds_error=False, fill_value=0.0)
    for dt in TILTS:
        pts = ec.spherical_points(plane, plane["theta"] + dt, s_vals, t_vals); img, m = ec.render(vol, grid, pts, shape, t)
        imgs.append(img[0].astype(np.float32)); masks.append(m)
        nearest.append(np.clip(nn(pts), 0, 255).astype(np.uint8).reshape(shape).astype(np.float32))
    return np.stack(imgs), np.stack(nearest), np.stack(masks)


def mad(a: np.ndarray, b: np.ndarray, m: np.ndarray) -> float:
    return float(np.abs(a - b)[m].mean())


def analyse_sweep(imgs: np.ndarray, nearest: np.ndarray, ev: np.ndarray, plane: dict) -> tuple[list[dict], list[dict]]:
    target = min(sharpness(x, ev) for x in imgs); rows, curves = [], []
    mt = [matched(x, ev, target) for x in imgs]
    for k, dt in enumerate(TILTS):
        raw, (mx, sigma, ok) = imgs[k], mt[k]; sd = float(raw[ev].std()); sh = sharpness(raw, ev)
        rows.append({"tilt_deg": float(dt), "theta_n_deg": float(plane["theta"] + dt), "obliqueness_deg": obliqueness(plane["theta"] + dt),
                     "sigma_px": sigma, "matched_ok": ok, "sd_raw": sd, "sd_matched": float(mx[ev].std()), "sd_nearest": float(nearest[k][ev].std()),
                     "mean_raw": float(raw[ev].mean()), "sharpness_raw": sh, "sharpness_matched": sharpness(mx, ev),
                     "sharpness_rel_raw": sh / sd ** 2 if sd > 0 else float("nan")})
    k0 = int(np.flatnonzero(TILTS == 0)[0])
    for c in CURVE_TILTS:
        for sign in (-1, 1):
            k = int(np.flatnonzero(np.isclose(TILTS, sign * c))[0])
            curves.append({"tilt_deg": sign * c, "change_raw": mad(imgs[k], imgs[k0], ev), "change_matched": mad(mt[k][0], mt[k0][0], ev)})
    return rows, curves


def time_curve(vol, grid, plane: dict, s_vals, t_vals, fps: float, ev: np.ndarray) -> list[dict]:
    """Mean absolute change between frames k volumes apart at the start plane, over all non-cyclic pairs (i, i + k), for k
    up to half the recording."""
    frames, _ = ec.render(vol, grid, ec.spherical_points(plane, plane["theta"], s_vals, t_vals), (len(t_vals), len(s_vals)), None)
    frames = frames.astype(np.float32); T = frames.shape[0]
    return [{"gap_volumes": k, "gap_s": k / fps, "pairs": T - k, "change": float(np.mean([mad(frames[i], frames[i + k], ev) for i in range(T - k)]))}
            for k in range(1, T // 2 + 1)]


def volume_rate(dcm) -> float:
    """Volumes per second from the DICOM frame time (0018,1063, ms). The recommended display rate (0008,2144), which
    EchoSlicer uses for its videos, is a rounded display setting and only the fallback."""
    if (0x0018, 0x1063) in dcm:
        return 1000.0 / float(dcm[(0x0018, 0x1063)].value)
    return float(dcm[(0x0008, 0x2144)].value) if (0x0008, 0x2144) in dcm else 10.0


def rho(d: pd.DataFrame, x: str, y: str) -> float | None:
    """Spearman correlation, or None where it is undefined (a missing value or a constant column)."""
    if d[[x, y]].isna().any().any() or d[x].nunique() < 2 or d[y].nunique() < 2:
        return None
    return float(spearmanr(d[x], d[y])[0])


def side_rhos(d: pd.DataFrame, start_theta: float) -> dict:
    """Contrast against obliqueness separately on each side of the fold nearest the start plane (the reading rule's
    'both sides'). Obliqueness turns at every multiple of 90 degrees of theta_n: at 90 for S1-S3 and at 0 for S4. A
    +-15 degree sweep crosses at most one fold."""
    fold = 90.0 * round(start_theta / 90.0); out = {"fold_deg": fold}
    for side, sel in (("below_fold", d.theta_n_deg < fold), ("above_fold", d.theta_n_deg > fold)):
        s = d[sel]; out[f"n_{side}"] = int(len(s))
        for y in ("sd_raw", "sd_matched", "sd_nearest"):
            out[f"rho_{y}_vs_obliqueness_{side}"] = rho(s, "obliqueness_deg", y) if len(s) >= MIN_SIDE else None
    return out


def sweep_summary(d: pd.DataFrame, plane: dict, common: np.ndarray, ev: np.ndarray) -> dict:
    return {"start_theta": plane["theta"], "start_phi": plane["phi"], "common_mask_fraction": float(common.mean()),
            "evaluation_mask_fraction": float(ev.mean()),
            **{f"rho_{y}_vs_{x}": rho(d, x, y) for y in ("sd_raw", "sd_matched", "sd_nearest", "sharpness_raw", "sharpness_rel_raw")
               for x in ("tilt_deg", "obliqueness_deg")},
            **side_rhos(d, plane["theta"]), "sigma_px_max": float(d.sigma_px.max()), "matched_all_ok": bool(d.matched_ok.all())}


def plot(tc: pd.DataFrame, sc: pd.DataFrame, title: str, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for name in tc.start.unique():
        g = tc[tc.start == name].assign(size=lambda z: z.tilt_deg.abs()).groupby("size")[["change_raw", "change_matched"]].mean()
        ax[0].plot(g.index, g.change_raw, "o-", label=f"{name} raw"); ax[0].plot(g.index, g.change_matched, "x--", label=f"{name} matched")
        h = sc[sc.start == name]; ax[1].plot(h.gap_s, h.change, "o-", label=name)
    ax[0].set_xlabel("tilt from the start plane (degrees), fixed time"); ax[0].set_ylabel("mean absolute change (grey levels)"); ax[0].legend(fontsize=7)
    ax[1].set_xlabel("time gap (s), fixed plane"); ax[1].legend(fontsize=7); fig.suptitle(title)
    for x in ax:
        x.set_ylim(bottom=0)
    fig.tight_layout(); fig.savefig(path, dpi=110)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dicom", required=True); ap.add_argument("--views", required=True); ap.add_argument("--echoslicer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--margin", type=int, default=MARGIN, help="pixels removed from the common region's edge (0 reproduces the first run's tilt sweeps)")
    a = ap.parse_args(); sys.path.insert(0, a.echoslicer)
    if a.margin < 0:
        ap.error("--margin must be >= 0")
    from extractor.utils_3d import read_3d
    dcm = pydicom.dcmread(a.dicom); vol, bounds = read_3d(dcm)
    fps = volume_rate(dcm); hr = float(dcm[(0x0018, 0x1088)].value) if (0x0018, 0x1088) in dcm else None
    grid = tuple(np.linspace(lo, hi, n) for (lo, hi), n in zip(bounds, vol.shape[:3]))
    view, other = read_plane(Path(a.views), "A4C"), read_plane(Path(a.views), "A2C")
    out = Path(a.out) / Path(a.dicom).stem; out.mkdir(parents=True, exist_ok=True)
    times = (0, vol.shape[3] // 2); rows, tcurves, scurves = [], [], []
    summary = {"volume": Path(a.dicom).name, "fps": fps, "volumes_in_recording": int(vol.shape[3]), "recording_s": vol.shape[3] / fps,
               "heart_rate_bpm": hr, "beats_in_recording": None if hr is None else vol.shape[3] / fps * hr / 60.0,
               "margin_px": a.margin, "sweeps": {}}
    for name, plane in start_planes(view, other).items():
        s_vals, t_vals = ec.start_grid(plane, *grid); data = {t: sweep(vol, grid, plane, s_vals, t_vals, t) for t in times}
        common = np.logical_and.reduce([m for _, _, ms in data.values() for m in ms]); ev = evaluation_mask(common, a.margin)
        if not ev.any():
            raise SystemExit(f"{name}: the evaluation region is empty (margin {a.margin} px)")
        for t in times:
            r, c = analyse_sweep(data[t][0], data[t][1], ev, plane)
            rows += [{"start": name, "time_index": t, **x} for x in r]; tcurves += [{"start": name, "time_index": t, **x} for x in c]
            summary["sweeps"][f"{name}|t{t}"] = sweep_summary(pd.DataFrame(r), plane, common, ev)
        scurves += [{"start": name, **x} for x in time_curve(vol, grid, plane, s_vals, t_vals, fps, ev)]
    pd.DataFrame(rows).to_csv(out / "frames.csv", index=False); pd.DataFrame(tcurves).to_csv(out / "tilt_curves.csv", index=False)
    text = json.dumps(summary, indent=1, allow_nan=False)    # an undefined correlation is None, never a bare NaN
    pd.DataFrame(scurves).to_csv(out / "time_curves.csv", index=False); (out / "summary.json").write_text(text)
    plot(pd.DataFrame(tcurves), pd.DataFrame(scurves), f"{Path(a.dicom).stem}: image change against tilt and time", out / "curves.png")
    print(text)


if __name__ == "__main__":
    main()
