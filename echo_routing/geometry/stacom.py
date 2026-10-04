"""STACOM 2011 motion-tracking data (Tobon-Gomez et al., Med Image Anal 17(6):632-648, 2013): 3D echo reader, probe
sector and landmark checks.

Each volunteer's `3DUS` folder holds one Philips 3D DICOM (`IM_nnnn`: Cartesian voxels and the frame time) and the same
voxels as one VTK file per frame (`VTK/VTKnn.vtk`: STRUCTURED_POINTS, spacing in mm, origin 0). Volumes are returned
as (T, Z, Y, X) uint8 with spacing (x, y, z) in mm, the frame the challenge's expert LV landmarks use
(`cMAC/GT/3DUS/<case>/LMKS/VTK_COORDINATES`). Numbers only; no image of the data leaves the server.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage

VTK_KEY = b"COLOR_SCALARS scalars 1\n"


@dataclass(frozen=True)
class Case:
    volume: np.ndarray            # (T, Z, Y, X) uint8
    spacing: np.ndarray           # (x, y, z) mm
    frame_time_ms: float          # DICOM (0018,1063)
    dicom_equal_fraction: float   # voxels equal between the DICOM and the VTK files (1.0 expected)


@dataclass(frozen=True)
class Sector:
    apex_mm: np.ndarray           # (x, y, z) mm, the probe position
    axis: np.ndarray              # unit vector from the apex along the central beam, (x, y, z)
    half_angle_x_deg: float
    half_angle_y_deg: float
    rho_max_mm: float             # deepest visible voxel from the apex
    centre_tilt_deg: float        # tilt of the sector's centre line against the z axis (0 if the beam runs along z)
    width_fit_rmse_mm: float
    width_fit_r2: float           # the smaller R^2 of the two width fits
    mask: np.ndarray              # (Z, Y, X) bool, visible in any frame


def read_vtk_volume(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """One STRUCTURED_POINTS frame as (Z, Y, X) uint8 and its spacing (x, y, z) in mm."""
    b = Path(path).read_bytes(); i = b.index(VTK_KEY); head = b[:i].decode()
    dims = [int(x) for x in re.search(r"DIMENSIONS (\d+) (\d+) (\d+)", head).groups()]
    spacing = np.array([float(x) for x in re.search(r"SPACING (\S+) (\S+) (\S+)", head).groups()])
    origin = [float(x) for x in re.search(r"ORIGIN (\S+) (\S+) (\S+)", head).groups()]
    if any(origin):
        raise ValueError(f"{path}: origin {origin}; the landmark frame assumes origin 0")
    n = dims[0] * dims[1] * dims[2]; start = i + len(VTK_KEY)
    if len(b) - start < n:
        raise ValueError(f"{path}: {len(b) - start} data bytes, {n} expected")
    return np.frombuffer(b[start:start + n], np.uint8).reshape(dims[2], dims[1], dims[0]), spacing


def read_case(us_dir: Path) -> Case:
    """All VTK frames, cross-checked voxel by voxel against the folder's one DICOM, with the DICOM frame time."""
    import pydicom
    us_dir = Path(us_dir); paths = sorted((us_dir / "VTK").glob("VTK*.vtk"))
    if not paths:
        raise FileNotFoundError(f"no VTK frames in {us_dir / 'VTK'}")
    frames = [read_vtk_volume(p) for p in paths]
    spacing = frames[0][1]
    if any(not np.allclose(s, spacing) for _, s in frames):
        raise ValueError(f"{us_dir}: spacing differs between frames")
    volume = np.stack([f for f, _ in frames])
    dicoms = sorted(us_dir.glob("IM_*"))                          # one per volunteer; the number differs (IM_0001, IM_0083, ...)
    if len(dicoms) != 1:
        raise FileNotFoundError(f"{us_dir}: expected one IM_* DICOM, found {[p.name for p in dicoms]}")
    dcm = pydicom.dcmread(dicoms[0])
    t, z, y, x = volume.shape
    raw = np.frombuffer(dcm.PixelData, np.uint8)
    if raw.size != t * z * y * x or int(dcm.NumberOfFrames) != t:
        raise ValueError(f"{us_dir}: DICOM holds {raw.size} voxels in {dcm.NumberOfFrames} frames, VTK {volume.shape}")
    # DICOM frames are (slices, rows, columns); the VTK files reverse the slices and swap rows and columns
    as_vtk = raw.reshape(t, z, x, y)[:, ::-1].transpose(0, 1, 3, 2)
    return Case(volume, spacing, float(dcm.FrameTime), float((as_vtk == volume).mean()))


def read_landmarks(path: Path) -> np.ndarray:
    """POLYDATA points of a landmark file, (N, 3) in mm."""
    text = Path(path).read_text()
    n = int(re.search(r"POINTS (\d+)", text).group(1))
    values = np.array(text.split("POINTS", 1)[1].split("\n", 1)[1].split()[:3 * n], float)
    return values.reshape(n, 3)


def _extents(mask: np.ndarray, spacing: np.ndarray) -> tuple[np.ndarray, ...]:
    """Per z-slice: index, widths and centres along x and y in mm, for non-empty slices."""
    zs, wx, wy, cx, cy = [], [], [], [], []
    for z in range(mask.shape[0]):
        s = mask[z]
        if not s.any():
            continue
        xs = np.flatnonzero(s.any(0)); ys = np.flatnonzero(s.any(1))
        zs.append(z); wx.append((xs[-1] - xs[0]) * spacing[0]); wy.append((ys[-1] - ys[0]) * spacing[1])
        cx.append((xs[-1] + xs[0]) / 2 * spacing[0]); cy.append((ys[-1] + ys[0]) / 2 * spacing[1])
    return tuple(np.array(v, float) for v in (zs, wx, wy, cx, cy))


MIN_TAPER_R2 = 0.98         # the widths must fall on straight lines along the taper
MAX_APEX_GAP_MM = 15.0      # the apex lies at most this far beyond the end of the visible region on the probe side


def sector_geometry(volume: np.ndarray, spacing: np.ndarray) -> Sector:
    """Probe apex and opening of the pyramidal sector, fitted to the widths of the visible region along z.

    The sector widens linearly with depth from the apex until the box or the far cap cuts it. On each side of the widest
    slice, the widths below 95% of the maximum (without the last two slices, a few voxels wide) are fitted with
    straight lines. The probe side is the side whose widths are straightest: a taper to the apex is linear, while a far
    cap is curved and a box-clipped plateau barely narrows, whichever run is longer. The lines' zero gives the apex
    depth. The function raises, rather than return a wrong apex, unless both fits are straight (R^2 >= 0.98) and the
    apex lies within 15 mm beyond the end of the visible region.
    """
    mask = (volume > 0).any(axis=0) if volume.ndim == 4 else volume > 0
    zs, wx, wy, cx, cy = _extents(mask, spacing)
    w = wx + wy; k = int(np.argmax(w)); sides = {True: np.arange(k, len(zs)), False: np.arange(0, k + 1)[::-1]}
    cands = {}
    for high, r in sides.items():
        run = r[(w[r] < 0.95 * w[k])][:-2]
        if len(run) < 5:
            continue
        z_mm = zs[run] * spacing[2]; fits = [np.polyfit(z_mm, v[run], 1) for v in (wx, wy)]
        r2 = min(1 - np.sum((np.polyval(f, z_mm) - v[run]) ** 2) / max(np.sum((v[run] - v[run].mean()) ** 2), 1e-9)
                 for f, v in zip(fits, (wx, wy)))
        cands[high] = (float(r2), run, fits, z_mm)
    if not cands:
        raise ValueError("too few tapering slices to fit the sector")
    toward_high = max(cands, key=lambda h: cands[h][0]); r2, run, fits, z_mm = cands[toward_high]
    if r2 < MIN_TAPER_R2:
        raise ValueError(f"sector widths are not straight along the taper (R^2 {r2:.3f})")
    z0 = float(np.mean([-b / a for a, b in fits]))
    z_end = zs[sides[toward_high][-1]] * spacing[2]
    gap = (z0 - z_end) if toward_high else (z_end - z0)
    if not -2.0 <= gap <= MAX_APEX_GAP_MM:
        raise ValueError(f"fitted apex lies {gap:.1f} mm beyond the visible region")
    centre = [np.polyfit(z_mm, v[run], 1) for v in (cx, cy)]
    apex = np.array([np.polyval(centre[0], z0), np.polyval(centre[1], z0), z0])
    tilt = float(np.degrees(np.arctan(np.hypot(centre[0][0], centre[1][0]))))
    rmse = float(np.sqrt(np.mean([(np.polyval(f, z_mm) - v[run]) ** 2 for f, v in zip(fits, (wx, wy))])))
    zz, yy, xx = np.nonzero(mask)
    pts = np.stack([xx * spacing[0], yy * spacing[1], zz * spacing[2]], 1)
    return Sector(apex, np.array([0.0, 0.0, -1.0 if toward_high else 1.0]),
                  float(np.degrees(np.arctan(abs(fits[0][0]) / 2))), float(np.degrees(np.arctan(abs(fits[1][0]) / 2))),
                  float(np.linalg.norm(pts - apex, axis=1).max()), tilt, rmse, r2, mask)


def _directions(n: int = 4000) -> np.ndarray:
    """Near-uniform unit vectors on a hemisphere (Fibonacci lattice)."""
    i = np.arange(n) + 0.5; z = i / n; r = np.sqrt(1 - z ** 2); a = np.pi * (1 + 5 ** 0.5) * i
    return np.stack([r * np.cos(a), r * np.sin(a), z], 1)


def landmark_levels(points: np.ndarray) -> list[np.ndarray]:
    """Indices of the three LV levels (basal, mid, apical walls; four landmarks each), apical first.

    The long axis is the direction along which the landmarks, sorted and cut into groups of four, have the smallest
    spread within groups relative to their total spread. The first principal axis is not used: the rings are about
    as wide as the LV is long, so it often lies across the ventricle. The end ring with the smaller radius is apical.
    """
    if len(points) != 12:
        raise ValueError(f"expected 12 landmarks (4 walls x 3 levels), got {len(points)}")
    proj = (points - points.mean(0)) @ _directions().T                      # (12, directions)
    order = np.argsort(proj, axis=0); s = np.take_along_axis(proj, order, axis=0)
    within = sum(s[k:k + 4].var(axis=0) for k in (0, 4, 8))
    best = int(np.argmin(within / proj.var(axis=0)))
    groups = [order[k:k + 4, best] for k in (0, 4, 8)]
    radius = [np.linalg.norm(points[g] - points[g].mean(0), axis=1).mean() for g in groups]
    return groups if radius[0] <= radius[2] else groups[::-1]


def grey_at(volume: np.ndarray, spacing: np.ndarray, p: np.ndarray) -> float:
    """Mean grey value of the 3x3x3 voxels around a point in mm (NaN outside the box)."""
    x, y, z = np.round(np.asarray(p) / spacing).astype(int)
    if not (0 <= z < volume.shape[0] and 0 <= y < volume.shape[1] and 0 <= x < volume.shape[2]):
        return float("nan")
    return float(volume[max(z - 1, 0):z + 2, max(y - 1, 0):y + 2, max(x - 1, 0):x + 2].mean())


def landmark_contrast(volume: np.ndarray, spacing: np.ndarray, points: np.ndarray, shift=(0.0, 0.0, 0.0),
                      levels: list[np.ndarray] | None = None) -> float:
    """Mean grey at the landmarks (on the wall) minus the mean grey at each level's centroid (in the cavity)."""
    p = points + np.asarray(shift); levels = landmark_levels(points) if levels is None else levels
    wall = np.nanmean([grey_at(volume, spacing, q) for q in p])
    cavity = np.nanmean([grey_at(volume, spacing, p[g].mean(0)) for g in levels])
    return float(wall - cavity)


def shift_rank(volume: np.ndarray, spacing: np.ndarray, points: np.ndarray, step: float = 4.0, reach: int = 3) -> tuple[int | None, int]:
    """Rank of the unshifted landmark set (1 = highest contrast) among copies shifted by up to reach*step mm per axis."""
    offsets = np.arange(-reach, reach + 1) * step; levels = landmark_levels(points)    # a shift keeps the levels
    scores = {s: landmark_contrast(volume, spacing, points, s, levels) for s in
              ((a, b, c) for a in offsets for b in offsets for c in offsets)}
    return _rank(scores, (0.0, 0.0, 0.0))


def _rank(scores: dict, key) -> tuple[int | None, int]:
    """Rank of `key` (1 = highest) among the finite scores. NaN candidates (landmarks outside the box) are dropped
    before sorting: NaN breaks the ordering of every other entry. None if `key` itself is NaN."""
    finite = {k: v for k, v in scores.items() if np.isfinite(v)}
    if key not in finite:
        return None, len(finite)
    ranked = sorted(finite, key=lambda k: -finite[k])
    return ranked.index(key) + 1, len(ranked)


def mapping_rank(volume: np.ndarray, spacing: np.ndarray, points: np.ndarray) -> tuple[int | None, int]:
    """Rank of the volume as read (1 = highest landmark contrast) among its 48 axis permutations and flips, each with
    its spacing permuted to match. Candidates that put more than two landmarks outside the box are skipped. This tests
    the axis order and directions; `shift_rank` tests millimetre registration."""
    import itertools
    levels = landmark_levels(points); scores = {}
    for perm in itertools.permutations(range(3)):
        for flips in itertools.product((False, True), repeat=3):
            v = volume.transpose(perm)
            for ax, f in enumerate(flips):
                if f:
                    v = v[(slice(None),) * ax + (slice(None, None, -1),)]
            sp = spacing[::-1][list(perm)][::-1]                  # spacing follows the array axes (z, y, x)
            if sum(np.isnan(grey_at(v, sp, q)) for q in points) > 2:
                continue
            scores[(perm, flips)] = landmark_contrast(v, sp, points, levels=levels)
    return _rank(scores, ((0, 1, 2), (False, False, False)))


def depth(points: np.ndarray, sector: Sector) -> np.ndarray:
    """Distance (mm) from the apex along the central beam."""
    return (np.atleast_2d(points) - sector.apex_mm) @ sector.axis


def margin_to_edge(points: np.ndarray, sector: Sector, spacing: np.ndarray) -> np.ndarray:
    """Distance (mm) from each point to the nearest voxel outside the visible region (0 if outside)."""
    edt = ndimage.distance_transform_edt(sector.mask, sampling=spacing[::-1])
    return np.array([edt[tuple(np.clip(np.round(q[::-1] / spacing[::-1]).astype(int), 0, np.array(edt.shape) - 1))]
                     for q in np.atleast_2d(points)])
