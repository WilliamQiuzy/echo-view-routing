"""CETUS 2014 (Bernard et al., IEEE TMI 35(4):967-977, 2016, doi:10.1109/TMI.2015.2503890): reader and checks for the
ED/ES volumes and LV annotations on the Human Heart Project database (collection 62eb991b73e9f0048c3a6c45). The data
are CC BY-NC-SA 4.0, for non-commercial research only, and may not be redistributed. Numbers only; no image of the
data leaves the server.

Each patient folder holds patientNN_{ED,ES}.nii.gz (Cartesian B-mode, integer grey values 0-255 stored as float32) and
patientNN_{ED,ES}_gt.nii.gz (LV cavity mask, 0/255) on one grid. Three quirks were found on the first case:
- the headers give the voxel size in metres while labelling it mm (0.000576 "mm"); the reader converts it to mm;
- outside the scan sector each volume is filled with one constant, which differs between volumes (13 at ED and 8 at ES
  in patient01); it is that volume's most frequent value;
- the last slice of each axis is 0.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage

from echo_routing.geometry.stacom import Sector, sector_geometry

PHASES = ("ED", "ES")
METRE_HEADER_LIMIT = 0.01      # a voxel size below 0.01 "mm" can only be metres


@dataclass(frozen=True)
class Case:
    patient: str
    volumes: dict                 # phase -> (I, J, K) uint8
    masks: dict                   # phase -> (I, J, K) bool, LV cavity
    spacing_mm: np.ndarray        # (I, J, K)
    affine: np.ndarray            # 4 x 4, as stored (header units)
    header_spacing: np.ndarray    # as stored
    mask_values: dict             # phase -> sorted distinct values of the annotation


def read_case(folder: Path) -> Case:
    """The four files of one patient, checked to share one grid, with the voxel size in mm."""
    import nibabel as nib
    folder = Path(folder); pid = folder.name
    imgs = {f"{ph}{s}": nib.load(folder / f"{pid}_{ph}{s}.nii.gz") for ph in PHASES for s in ("", "_gt")}
    ref = imgs["ED"]
    for k, im in imgs.items():
        if im.shape != ref.shape or not np.allclose(im.affine, ref.affine):
            raise ValueError(f"{pid}: {k} is not on the grid of ED ({im.shape} vs {ref.shape})")
    zooms = np.array(ref.header.get_zooms()[:3], float)
    spacing = zooms * 1000.0 if zooms.max() < METRE_HEADER_LIMIT else zooms
    vols, masks, values = {}, {}, {}
    for ph in PHASES:
        v = np.asanyarray(imgs[ph].dataobj)
        if v.min() < 0 or v.max() > 255 or not np.all(v == np.round(v)):
            raise ValueError(f"{pid} {ph}: grey values are not integers in 0-255")
        vols[ph] = v.astype(np.uint8)
        g = np.asanyarray(imgs[f"{ph}_gt"].dataobj)
        values[ph] = [float(x) for x in np.unique(g)]
        masks[ph] = g > 0
    return Case(pid, vols, masks, spacing, ref.affine, zooms, values)


def background(volume: np.ndarray) -> tuple[int, float]:
    """The constant fill outside the sector: the most frequent value, and the share of voxels holding it."""
    counts = np.bincount(volume.ravel(), minlength=256)
    v = int(np.argmax(counts))
    return v, float(counts[v] / volume.size)


def moved(a: np.ndarray, axis: int) -> np.ndarray:
    """(I, J, K) -> (Z, Y, X) with the given axis as Z and the other two in their order."""
    return np.moveaxis(a, axis, 0)


def spacing_xyz(spacing: np.ndarray, axis: int) -> np.ndarray:
    """Spacing in the (x, y, z) order sector_geometry expects for moved(a, axis)."""
    rest = [i for i in range(3) if i != axis]
    return np.array([spacing[rest[1]], spacing[rest[0]], spacing[axis]], float)


def fit_sector(volume: np.ndarray, spacing: np.ndarray, bg: int) -> tuple[int, Sector, dict]:
    """The beam axis is the array axis along which the visible region tapers straight to an apex (STACOM's fit).
    Returns that axis, its sector and the outcome for every axis; raises if no axis fits."""
    vis = np.where(volume > bg, volume, 0)
    outcome, fits = {}, {}
    for a in range(3):
        try:
            sec = sector_geometry(moved(vis, a), spacing_xyz(spacing, a))
            fits[a] = sec; outcome[a] = {"r2": sec.width_fit_r2}
        except ValueError as e:
            outcome[a] = {"error": str(e)}
    if not fits:
        raise ValueError(f"no axis gives a straight sector taper: {outcome}")
    best = max(fits, key=lambda a: fits[a].width_fit_r2)
    return best, fits[best], outcome


def points_mm(mask_zyx: np.ndarray, sp_xyz: np.ndarray) -> np.ndarray:
    zz, yy, xx = np.nonzero(mask_zyx)
    return np.stack([xx * sp_xyz[0], yy * sp_xyz[1], zz * sp_xyz[2]], 1)


def inside_sector(p: np.ndarray, sec: Sector) -> np.ndarray:
    """Points within the fitted pyramid (beam along the sector axis, opening by the two half-angles, depth rho_max)."""
    d = p - sec.apex_mm; depth = d @ sec.axis
    return ((depth > 0) & (np.abs(d[:, 0]) <= depth * np.tan(np.radians(sec.half_angle_x_deg)) + 1e-6)
            & (np.abs(d[:, 1]) <= depth * np.tan(np.radians(sec.half_angle_y_deg)) + 1e-6)
            & (np.linalg.norm(d, axis=1) <= sec.rho_max_mm + 1e-6))


def lv_geometry(mask: np.ndarray, spacing: np.ndarray, axis: int, sec: Sector) -> dict:
    """LV cavity: volume, components, long axis against the beam, depth range and share outside the sector."""
    lab, n = ndimage.label(mask)
    sizes = np.bincount(lab.ravel())[1:] if n else np.array([0])
    sp = spacing_xyz(spacing, axis); p = points_mm(moved(mask, axis), sp)
    c = p.mean(0); _, sv, vt = np.linalg.svd(p - c, full_matrices=False); long_axis = vt[0]
    depth = (p - sec.apex_mm) @ sec.axis
    return {"volume_ml": float(mask.sum() * np.prod(spacing) / 1000.0), "components": int(n),
            "largest_component_share": float(sizes.max() / max(sizes.sum(), 1)),
            "long_axis_vs_beam_deg": float(np.degrees(np.arccos(min(1.0, abs(float(long_axis @ sec.axis)))))),
            "depth_mm": [float(depth.min()), float(depth.max())],
            "length_along_long_axis_mm": float(np.ptp((p - c) @ long_axis)),
            "share_outside_sector": float(1.0 - inside_sector(p, sec).mean())}


def annotation_contrast(volume: np.ndarray, mask: np.ndarray, bg: int, shell: int = 3) -> dict:
    """Mean grey inside the cavity and in a shell just outside it (sector voxels only). Blood pool is darker than
    myocardium, so an aligned cavity mask has a darker inside."""
    ring = ndimage.binary_dilation(mask, iterations=shell) & ~mask & (volume > bg)
    inside = mask & (volume > bg)
    return {"mean_inside": float(volume[inside].mean()) if inside.any() else float("nan"),
            "mean_shell_outside": float(volume[ring].mean()) if ring.any() else float("nan")}
