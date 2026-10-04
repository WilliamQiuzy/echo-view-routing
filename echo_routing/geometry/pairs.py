"""Slice pairs with a known relation, for pretraining. Diagnostic-safe: no view label is given to any slice.

A pair is two slices of one volume: same rotation psi about the beam, fan tilts alpha1 and alpha2, times t1 and t2.
Three kinds are drawn: same plane at two times (only the heart moved), two planes at one time (only the plane moved),
and both changed. The geometric target is the signed tilt d_alpha = alpha2 - alpha1, which is zero whenever only the
heart moved. Every arm that trains sees exactly the same pairs; only the loss on d_alpha differs.

Controls from the slice-artifact study (docs/results/2026-10-03_echoslicer_feasibility, section 5):
- both slices share one mask, the region visible at every tilt of the sampling range for that psi, so neither the
  visible shape nor its size reveals the tilt;
- each slice is standardised inside the mask to one fixed mean and SD, so global contrast cannot carry the tilt;
- `contrast_features` gives the per-slice statistics for the contrast-only baseline.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from .planes import ProbePlane

KINDS = ("same_plane", "same_time", "both")


@dataclass(frozen=True)
class SliceGrid:
    pixel_mm: float = 0.5
    width_mm: float = 160.0
    depth_mm: float = 150.0

    def axes(self) -> tuple[np.ndarray, np.ndarray]:
        w = int(round(self.width_mm / self.pixel_mm)); h = int(round(self.depth_mm / self.pixel_mm))
        return (np.arange(w) - (w - 1) / 2) * self.pixel_mm, np.arange(h) * self.pixel_mm


@dataclass(frozen=True)
class PairSpec:
    psi_deg: float
    alpha1_deg: float
    alpha2_deg: float
    t1: int
    t2: int
    kind: str

    @property
    def d_alpha(self) -> float:
        return self.alpha2_deg - self.alpha1_deg


PSI_RANGE = (-150.0, 30.0)


def sample_spec(rng: np.random.Generator, n_times: int, alpha_range: tuple[float, float] = (-15.0, 15.0),
                kind_p: tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3),
                psi_range: tuple[float, float] = PSI_RANGE) -> PairSpec:
    """psi uniform on psi_range; both tilts uniform on alpha_range; times uniform and distinct where the kind needs it.

    psi and psi + 180 give the same plane mirrored. The default range keeps the standard display orientation: the
    A4C start plane sits near psi = 0, and EchoSlicer's classifier finds A2C and A3C at negative psi (B_0: A2C 0.90 at
    -60, A3C 0.87-0.98 from -90 to -150), while positive psi gives their mirror images."""
    if n_times < 2:
        raise ValueError("pairs need at least two time points")
    kind = KINDS[int(rng.choice(3, p=kind_p))]; psi = float(rng.uniform(*psi_range))
    a1 = float(rng.uniform(*alpha_range)); a2 = a1 if kind == "same_plane" else float(rng.uniform(*alpha_range))
    while kind != "same_plane" and a2 == a1:
        a2 = float(rng.uniform(*alpha_range))
    t1 = int(rng.integers(n_times)); t2 = t1
    if kind != "same_time":
        t2 = int((t1 + rng.integers(1, n_times)) % n_times)
    return PairSpec(psi, a1, a2, t1, t2, kind)


def fixed_mask(volume, psi_deg: float, grid: SliceGrid, alpha_range: tuple[float, float], step_deg: float = 1.0,
               erode_px: int = 2) -> np.ndarray:
    """Pixels inside the volume for every tilt of alpha_range (inclusive, in steps) at this psi, shrunk by erode_px.

    Pairs draw alpha continuously, so a grid of tilts alone does not guarantee coverage in between. With 2.5-degree
    steps, 7 pixels of one STACOM volunteer fell outside over 1,000 random tilts; with 1-degree steps and the 2-pixel
    margin, none did. render_pair still raises if a pixel falls outside."""
    s, t = grid.axes(); m = None
    for a in np.arange(alpha_range[0], alpha_range[1] + 1e-9, step_deg):
        inside = volume.inside(ProbePlane(psi_deg, float(a)).points(s, t).reshape(-1, 3)).reshape(len(t), len(s))
        m = inside if m is None else m & inside
    if erode_px > 0 and m.any():
        m = ndimage.binary_erosion(m, structure=np.ones((3, 3), bool), iterations=erode_px)
    return m


def render(volume, plane: ProbePlane, t: int, grid: SliceGrid) -> tuple[np.ndarray, np.ndarray]:
    s, tt = grid.axes()
    v, inside = volume.sample(plane.points(s, tt).reshape(-1, 3), t)
    return v.reshape(len(tt), len(s)), inside.reshape(len(tt), len(s))


def standardise(img: np.ndarray, mask: np.ndarray, mean: float, sd: float) -> np.ndarray:
    """Grey values inside the mask moved to the given mean and SD (0-255, clipped); 0 outside."""
    x = img[mask]; z = (img - x.mean()) / max(float(x.std()), 1e-6)
    return np.where(mask, np.clip(mean + sd * z, 0.0, 255.0), 0.0).astype(np.float32)


def contrast_features(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Per-slice statistics a shortcut could use: mean, SD, sharpness (variance of the Laplacian) and relative
    sharpness, inside the mask shrunk by one pixel so the Laplacian never reaches outside it."""
    inner = ndimage.binary_erosion(mask, structure=np.ones((3, 3), bool))
    x = img[inner]; sh = float(ndimage.laplace(img.astype(np.float32))[inner].var())
    return np.array([x.mean(), x.std(), sh, sh / max(float(x.var()), 1e-6)], np.float32)


def render_pair(volume, spec: PairSpec, grid: SliceGrid, mask: np.ndarray, mean: float, sd: float) -> dict:
    """Both slices of a pair, masked and standardised, with the raw-image contrast features and the target."""
    out = {"spec": spec, "d_alpha": spec.d_alpha, "mask": mask}
    for k, (a, t) in enumerate(((spec.alpha1_deg, spec.t1), (spec.alpha2_deg, spec.t2)), start=1):
        img, inside = render(volume, ProbePlane(spec.psi_deg, a), t, grid)
        if not inside[mask].all():
            raise ValueError(f"mask reaches outside the volume at alpha {a}")
        out[f"img{k}"] = standardise(img, mask, mean, sd); out[f"raw_features{k}"] = contrast_features(img, mask)
    return out
