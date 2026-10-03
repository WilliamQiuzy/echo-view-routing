#!/usr/bin/env python
"""Controlled 2D re-slices of one 3D echocardiogram (feasibility check, 2026-10-03). Not a model and not a real scan.

The start plane is the A4C plane that EchoSlicer's classifier-guided search selects for the volume (its own output
CSV). That selection is a classifier's choice, not a verified label, and no A4C/A5C label is given to any frame here.
Three videos are cut from the volume, all on ONE fixed in-plane grid (the start plane's physical extent and pixel
count, reused for every plane) and with raw intensities (no per-frame rescaling):
- fixed_plane: the start plane at every time point of the volume;
- fixed_time: the first time point while the plane tilts from theta0 to theta0 + 15 degrees in 0.5-degree steps, the
  range in which EchoSlicer searches for A5C (theta0 + 10 to + 35);
- both: the tilt advances by the same total amount while time advances one volume per frame (cycling).
Every frame's plane parameters, time index and inside-volume fraction are saved. Each video is also written with a
common mask: only the pixels inside the volume in every frame of all three videos are kept, so the visible region
cannot reveal the plane. Leakage checks compare, against the plane tilt and against time, the inside-volume fraction
and, within the common mask, sharpness (variance of the Laplacian), mean and spread of intensity.
Writes <out>/<volume>/{fixed_plane,fixed_time,both}{,_masked}.mp4, frames.csv, checks.json and a contact sheet.
Runs in EchoSlicer's environment: third_party/3d-echo_run/.venv/bin/python scripts/echoslicer_controlled.py ...
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

TILT, STEP = 15.0, 0.5


def basis(phi_deg: float, theta_deg: float, rotation_deg: float, flipu: bool, flipv: bool) -> tuple[np.ndarray, ...]:
    """Plane normal and in-plane axes exactly as EchoSlicer's ViewExtractor.slice_volume builds them."""
    phi, theta = np.radians(phi_deg), np.radians(theta_deg)
    n = np.array([np.cos(theta) * np.cos(phi), np.sin(theta), np.cos(theta) * np.sin(phi)]); n /= np.linalg.norm(n)
    a = np.array([0.0, 1.0, 0.0]) if np.allclose(np.abs(n), [1, 0, 0]) else np.array([1.0, 0.0, 0.0])
    u = np.cross(n, a); u /= np.linalg.norm(u); u = -u if flipu else u
    v = np.cross(n, u); v = -v if flipv else v
    c, s = np.cos(np.radians(rotation_deg)), np.sin(np.radians(rotation_deg))
    return n, c * u + s * v, -s * u + c * v


def spherical_points(view: dict, theta_deg: float, s_vals: np.ndarray, t_vals: np.ndarray) -> np.ndarray:
    """(H*W, 3) query points (rho, phi, theta) of a plane on the fixed grid s_vals x t_vals (mm)."""
    n, u, v = basis(view["phi"], theta_deg, view["rotation_angle"], view["flipu"], view["flipv"])
    S, T = np.meshgrid(s_vals, t_vals); P = view["d"] * n + S[..., None] * u + T[..., None] * v; P = P.reshape(-1, 3)
    r = np.linalg.norm(P, axis=1)
    return np.stack([r, np.arctan2(P[:, 2], P[:, 0]), np.arctan2(P[:, 1], np.hypot(P[:, 0], P[:, 2]))], axis=1)


def start_grid(view: dict, rho, phi, theta) -> tuple[np.ndarray, np.ndarray]:
    """The start plane's extent (as EchoSlicer: projections of every volume point) at its pixel size."""
    R, Ph, Th = np.meshgrid(rho, phi, theta, indexing="ij")
    X = np.stack([R * np.cos(Th) * np.cos(Ph), R * np.sin(Th), R * np.cos(Th) * np.sin(Ph)], -1).reshape(-1, 3)
    n, u, v = basis(view["phi"], view["theta"], view["rotation_angle"], view["flipu"], view["flipv"])
    s = (X - view["d"] * n) @ u; t = (X - view["d"] * n) @ v; px = view["cm_pix"] * 10.0
    nx, ny = int((s.max() - s.min()) / px) // 2 * 2, int((t.max() - t.min()) / px) // 2 * 2
    return np.linspace(s.min(), s.max(), nx), np.linspace(t.min(), t.max(), ny)


def render(vol: np.ndarray, grid: tuple, pts: np.ndarray, shape: tuple, t_index: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Linear interpolation (EchoSlicer's interpolator, fill 0 outside) of one time point or all; and the inside mask."""
    vals = vol[..., t_index] if t_index is not None else vol
    img = RegularGridInterpolator(grid, vals.astype(np.float32), bounds_error=False, fill_value=0.0)(pts)
    inside = RegularGridInterpolator(grid, np.ones(vol.shape[:3], np.float32), bounds_error=False, fill_value=0.0)(pts) > 0.999
    img = img.reshape(*shape, -1) if img.ndim == 2 else img.reshape(*shape, 1)
    return np.clip(img, 0, 255).astype(np.uint8).transpose(2, 0, 1), inside.reshape(shape)


def frame_stats(img: np.ndarray, mask: np.ndarray, prev: np.ndarray | None) -> dict:
    """Within the common mask: sharpness, mean, spread, and the mean absolute change from the previous frame."""
    x = img.astype(np.float32); lap = ndimage.laplace(x)
    change = float(np.abs(x - prev.astype(np.float32))[mask].mean()) if prev is not None else float("nan")
    return {"sharpness": float(lap[mask].var()), "mean": float(x[mask].mean()), "std": float(x[mask].std()), "change_from_previous": change}


def default_sizes(view: dict, rho, phi, theta) -> list[dict]:
    """EchoSlicer's default (unscaled) output size for each tilt: its grid spans every volume point's projection."""
    out = []
    for dt in np.arange(0, TILT + 1e-9, 2.5):
        s_vals, t_vals = start_grid({**view, "theta": view["theta"] + dt}, rho, phi, theta)
        out.append({"tilt_deg": float(dt), "width_px": int(len(s_vals)), "height_px": int(len(t_vals))})
    return out


def write_mp4(path: Path, frames: np.ndarray, fps: float) -> None:
    import torch
    import torchvision
    torchvision.io.write_video(str(path), torch.from_numpy(np.repeat(frames[..., None], 3, -1)), fps=max(1, round(fps)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dicom", required=True); ap.add_argument("--views", required=True, help="EchoSlicer output folder (A4C.csv)")
    ap.add_argument("--echoslicer", required=True, help="third_party/3d-echo_run"); ap.add_argument("--out", required=True)
    a = ap.parse_args(); sys.path.insert(0, a.echoslicer)
    from extractor.utils_3d import read_3d   # EchoSlicer's own reader of the Philips 3D DICOM
    dcm = pydicom.dcmread(a.dicom); vol, bounds = read_3d(dcm)
    fps = float(dcm[(0x0008, 0x2144)].value) if (0x0008, 0x2144) in dcm else 10.0
    grid = tuple(np.linspace(lo, hi, n) for (lo, hi), n in zip(bounds, vol.shape[:3]))
    view = pd.read_csv(Path(a.views) / "A4C.csv").iloc[0].to_dict()
    view.update(flipu=str(view["flipu"]) == "True", flipv=str(view["flipv"]) == "True")
    s_vals, t_vals = start_grid(view, *grid); shape = (len(t_vals), len(s_vals)); T = vol.shape[3]
    tilts = np.arange(0, TILT + 1e-9, STEP); out = Path(a.out) / Path(a.dicom).stem; out.mkdir(parents=True, exist_ok=True)
    pts0 = spherical_points(view, view["theta"], s_vals, t_vals)
    plans = {"fixed_plane": [(0.0, t) for t in range(T)], "fixed_time": [(dt, 0) for dt in tilts],
             "both": [(dt, k % T) for k, dt in enumerate(np.linspace(0, TILT, max(T, len(tilts))))]}
    videos, masks, rows = {}, {}, []
    for name, plan in plans.items():
        frames, ms = [], []
        for dt, t in plan:
            img, m = render(vol, grid, pts0 if dt == 0 else spherical_points(view, view["theta"] + dt, s_vals, t_vals), shape, t)
            frames.append(img[0]); ms.append(m)
        videos[name], masks[name] = np.stack(frames), np.stack(ms)
        for k, (dt, t) in enumerate(plan):
            rows.append({"video": name, "frame": k, "time_index": t, "d": view["d"], "phi": view["phi"], "theta": view["theta"] + dt,
                         "tilt_deg": dt, "rotation_angle": view["rotation_angle"], "flipu": view["flipu"], "flipv": view["flipv"],
                         "pixel_mm": float(s_vals[1] - s_vals[0]), "inside_fraction": float(ms[k].mean())})
    common = np.logical_and.reduce([m for ms in masks.values() for m in ms])
    df = pd.DataFrame(rows); stats = []
    for name, v in videos.items():
        write_mp4(out / f"{name}.mp4", v, fps); write_mp4(out / f"{name}_masked.mp4", v * common, fps)
        stats += [frame_stats(f, common, v[i - 1] if i else None) for i, f in enumerate(v)]
    df = pd.concat([df, pd.DataFrame(stats)], axis=1); df.to_csv(out / "frames.csv", index=False)
    checks = {"volume": Path(a.dicom).name, "volume_shape": list(vol.shape), "fps": fps, "frame_shape": list(shape),
              "pixel_mm": float(s_vals[1] - s_vals[0]), "common_mask_fraction": float(common.mean()), "start_plane": view,
              "echoslicer_default_size_vs_tilt": default_sizes(view, *grid)}
    for name in videos:
        # each video is checked against what changes in it: the tilt (fixed_time, both) or time (fixed_plane, where the
        # visible fraction is constant by construction, so its correlation is undefined and stored as null)
        d = df[df.video == name]; by, axis = ("tilt_deg", "tilt") if name != "fixed_plane" else ("time_index", "time")
        rho = {k: spearmanr(d[by], d[k])[0] if d[k].nunique() > 1 else None for k in ("inside_fraction", "sharpness", "mean", "std")}
        checks[name] = {"frames": int(len(d)), "inside_fraction_range": [float(d.inside_fraction.min()), float(d.inside_fraction.max())],
                        **{f"{k}_relative_range": float((d[k].max() - d[k].min()) / d[k].mean()) for k in ("sharpness", "mean", "std")},
                        "median_change_from_previous": float(d.change_from_previous.median()),
                        **{f"spearman_{k}_vs_{axis}": (None if v is None else float(v)) for k, v in rho.items()}}
    (out / "checks.json").write_text(json.dumps(checks, indent=1, default=str))
    sheet = np.concatenate([np.concatenate([videos[n][i] for i in (0, len(videos[n]) // 2, len(videos[n]) - 1)], axis=1) for n in videos], axis=0)
    from PIL import Image
    Image.fromarray(sheet).save(out / "contact_sheet.png"); print(json.dumps(checks, indent=1, default=str))


if __name__ == "__main__":
    main()
