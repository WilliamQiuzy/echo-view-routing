"""Volume registry, person-balanced pair sampling and the final model input (geometry pretraining, Task 3).

- Registry. Every included volume belongs to a person. EchoSlicer's person A has 15 volumes and counts as one person.
  The held-out people (configs/experiments/geometry_pretraining_v1.yaml) never enter pretraining.
- Sampling. A pair draws a person uniformly, then one of the person's volumes, then psi from whole degrees whose fixed
  mask covers at least 20% of the slice, then tilts and times (pairs.sample_spec).
- Final input. A slice is standardised inside its mask, clipped and rounded to uint8, then put through E1's geometry
  (short side 256, bilinear; centre crop 224), as joint_visual.crop_uint8 does to real frames. The grey image is kept;
  the encoder gets it copied to three channels. The mask goes through the same geometry (nearest), so the contrast
  baseline can measure the image the model actually sees.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from .pairs import PairSpec, contrast_features, sample_spec, standardise

PSI_GRID = np.arange(-150, 31)                # whole degrees, pairs.PSI_RANGE
MIN_COVERAGE = 0.20


@dataclass(frozen=True)
class VolumeInfo:
    volume_id: str              # e.g. "echoslicer:A_3", "stacom:v7"
    person: str                 # e.g. "echoslicer:A", "stacom:v7"
    n_times: int


def people_of(volumes: list[VolumeInfo], held_out: set[str], pretraining: bool) -> dict[str, list[VolumeInfo]]:
    """Volumes grouped by person: the pretraining people, or only the held-out ones."""
    out: dict[str, list[VolumeInfo]] = {}
    for v in volumes:
        if (v.person in held_out) != pretraining:
            out.setdefault(v.person, []).append(v)
    return {p: sorted(vs, key=lambda x: x.volume_id) for p, vs in sorted(out.items())}


def sample_specs(rng: np.random.Generator, people: dict[str, list[VolumeInfo]], coverage: dict[str, np.ndarray], n: int,
                 alpha_range: tuple[float, float] = (-15.0, 15.0)) -> list[tuple[VolumeInfo, PairSpec]]:
    """n pairs: person uniform, volume uniform within the person, psi uniform over the whole degrees with coverage at
    least MIN_COVERAGE (coverage[volume_id] is indexed like PSI_GRID). A volume without any such psi is an error."""
    persons = list(people); out = []
    for v in (v for vs in people.values() for v in vs):
        if not (coverage[v.volume_id] >= MIN_COVERAGE).any():
            raise ValueError(f"{v.volume_id}: no psi reaches {MIN_COVERAGE:.0%} mask coverage; exclude it first")
    while len(out) < n:
        vols = people[persons[int(rng.integers(len(persons)))]]; v = vols[int(rng.integers(len(vols)))]
        ok = PSI_GRID[coverage[v.volume_id] >= MIN_COVERAGE]; psi = float(ok[int(rng.integers(len(ok)))])
        out.append((v, sample_spec(rng, v.n_times, alpha_range, psi_range=(psi, psi))))
    return out


def crop_like_e1(im: Image.Image, resample=Image.BILINEAR) -> Image.Image:
    """joint_visual.crop_uint8's geometry: short side to 256, centre crop 224."""
    w, h = im.size; s = 256 / min(w, h)
    im = im.resize((max(224, round(w * s)), max(224, round(h * s))), resample)
    w, h = im.size; left, top = (w - 224) // 2, (h - 224) // 2
    return im.crop((left, top, left + 224, top + 224))


def final_input(img: np.ndarray, mask: np.ndarray, mean: float, sd: float) -> tuple[np.ndarray, np.ndarray]:
    """(224, 224) uint8 image the encoder sees and its (224, 224) bool mask."""
    u8 = np.round(standardise(img, mask, mean, sd)).astype(np.uint8)
    x = np.asarray(crop_like_e1(Image.fromarray(u8)), dtype=np.uint8)
    m = np.asarray(crop_like_e1(Image.fromarray(mask.astype(np.uint8) * 255), Image.NEAREST)) > 127
    return x, m


def final_features(x: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Contrast features of the final input: mean, SD, sharpness, relative sharpness inside its mask."""
    return contrast_features(x.astype(np.float32), m)
