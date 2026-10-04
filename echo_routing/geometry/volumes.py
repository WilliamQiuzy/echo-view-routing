"""3D+t echo volumes that can be sampled at probe-frame points (mm; x beam, y elevation, z azimuth; see planes.py).

- SphericalVolume holds EchoSlicer's acoustic grid (rho in mm, phi and theta in radians), whose frame is the probe's.
- CartesianVolume holds a scan-converted volume (STACOM) with its probe pose: the apex and the beam, elevation and
  azimuth axes in the volume's mm frame, fitted and checked by scripts/stacom_verify.py.
Both return grey values and an inside mask; nothing is rescaled per frame.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from scipy.interpolate import RegularGridInterpolator


@dataclass(frozen=True)
class SphericalVolume:
    data: np.ndarray                    # (rho, phi, theta, T)
    grid: tuple                         # (rho mm, phi rad, theta rad) linspaces
    times_s: np.ndarray
    subject: str
    source: str = "echoslicer"

    def inside(self, points: np.ndarray) -> np.ndarray:
        rpt = to_spherical(points)
        return np.all([(rpt[:, k] >= g[0]) & (rpt[:, k] <= g[-1]) for k, g in enumerate(self.grid)], axis=0)

    def sample(self, points: np.ndarray, t: int) -> tuple[np.ndarray, np.ndarray]:
        f = RegularGridInterpolator(self.grid, self.data[..., t].astype(np.float32), bounds_error=False, fill_value=0.0)
        return f(to_spherical(points)).astype(np.float32), self.inside(points)


@dataclass(frozen=True)
class CartesianVolume:
    volume: np.ndarray                  # (T, Z, Y, X)
    spacing: np.ndarray                 # (x, y, z) mm
    apex_mm: np.ndarray                 # probe position, (x, y, z) mm
    frame: np.ndarray                   # rows: beam, elevation, azimuth unit vectors in the volume's mm frame
    mask: np.ndarray                    # (Z, Y, X) visible region
    times_s: np.ndarray
    subject: str
    source: str = "stacom2011"

    def __post_init__(self):
        if not np.allclose(self.frame @ self.frame.T, np.eye(3), atol=1e-6):
            raise ValueError("probe frame rows must be orthonormal")
        # a point counts as inside only if all eight voxels of its trilinear support are visible
        object.__setattr__(self, "_support", ndimage.binary_erosion(self.mask, structure=np.ones((3, 3, 3), bool)))

    def _voxels(self, points: np.ndarray) -> np.ndarray:
        world = self.apex_mm + np.asarray(points, float) @ self.frame
        return world[:, ::-1] / self.spacing[::-1]                     # (z, y, x) voxel coordinates

    def inside(self, points: np.ndarray) -> np.ndarray:
        c = self._voxels(points); ok = np.all((c >= 0) & (c <= np.array(self.mask.shape) - 1), axis=1)
        out = np.zeros(len(c), bool); out[ok] = self._support[tuple(np.round(c[ok]).astype(int).T)]
        return out

    def sample(self, points: np.ndarray, t: int) -> tuple[np.ndarray, np.ndarray]:
        v = ndimage.map_coordinates(self.volume[t].astype(np.float32), self._voxels(points).T, order=1, cval=0.0)
        return v.astype(np.float32), self.inside(points)


def to_spherical(points: np.ndarray) -> np.ndarray:
    """Probe-frame points (N, 3) to EchoSlicer's (rho, phi, theta)."""
    p = np.asarray(points, float)
    return np.stack([np.linalg.norm(p, axis=1), np.arctan2(p[:, 2], p[:, 0]), np.arctan2(p[:, 1], np.hypot(p[:, 0], p[:, 2]))], 1)
