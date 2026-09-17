"""Sliding-window application of clip-level view classifiers to constructed streams.

A stream is a sequence of 10 Hz samples, each pointing at a source frame (video_id, 30-fps frame index, gamma edit).
Clip classifiers see a window of consecutive 30-fps frames centred on a sample; the window's class probabilities are
assigned to the samples nearest to its centre. Windows never receive join positions or labels.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from echo_routing.compose.recipe_builder import Recipe
from echo_routing.features.cache import VideoFeatures
from echo_routing.features.extract import VARIANT_GAMMA

# raw EV9V nine-code order used by STFM / EchoViewCLIP training (INDEXOFLABEL) -> family5 index (None = dropped)
RAW9_ORDER = ["PLHLA", "PMASA", "PMVLSA", "PASA", "A4C", "A5C", "PMPALA", "PPMLSA", "SC4C"]
FAMILY5 = ["PLAX", "PSAX", "A4C", "A5C", "SC4C"]
RAW9_TO_FAMILY5 = {"PLHLA": 0, "PMASA": 1, "PMVLSA": 1, "PASA": 1, "A4C": 2, "A5C": 3, "PMPALA": None, "PPMLSA": 1, "SC4C": 4}
ECHOPRIME_VIEWS = ["A2C", "A3C", "A4C", "A5C", "Apical_Doppler", "Doppler_Parasternal_Long", "Doppler_Parasternal_Short",
                   "Parasternal_Long", "Parasternal_Short", "SSN", "Subcostal"]
ECHOPRIME_TO_FAMILY5 = {"A4C": 2, "A5C": 3, "Parasternal_Long": 0, "Parasternal_Short": 1, "Subcostal": 4}


@dataclass(frozen=True)
class SampleRef:
    video_id: str
    frame_idx: int   # 30-fps frame index in the source cine
    gamma: float     # appearance edit applied to this fragment (1.0 = none)


def stream_plan(recipe: Recipe, loader: Callable[[str, str], VideoFeatures], frames_per_sample: int = 3) -> list[SampleRef]:
    """Per-sample source references for a recipe (same order as the assembled stream)."""
    refs: list[SampleRef] = []
    for fr in recipe.fragments:
        vf = loader(fr.video_id, "orig")
        gamma = float(VARIANT_GAMMA[fr.variant])
        for s in range(fr.start, fr.end):
            refs.append(SampleRef(fr.video_id, int(vf.frame_idx[s]), gamma))
    return refs


def window_centres(n_samples: int, stride: int) -> np.ndarray:
    """Centre sample indices: stride apart, always including the first and last sample neighbourhoods."""
    if n_samples <= 0:
        return np.array([], dtype=int)
    c = np.arange(stride // 2, n_samples, stride, dtype=int)
    if c.size == 0 or c[-1] < n_samples - 1 - stride // 2:
        c = np.append(c, n_samples - 1)
    return c


def window_samples(centre: int, n_samples: int, size: int) -> np.ndarray:
    """Sample indices covered by a window of `size` samples centred at `centre`, clamped to the stream."""
    half = size // 2
    start = max(0, min(centre - half, n_samples - size))
    return np.arange(start, min(n_samples, start + size), dtype=int)


def assign_to_samples(n_samples: int, centres: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """Piecewise-constant assignment: every sample takes the probabilities of the nearest window centre."""
    if centres.size == 0:
        raise ValueError("no windows")
    idx = np.abs(np.arange(n_samples)[:, None] - centres[None, :]).argmin(1)
    return np.asarray(probs)[idx]


def map_probs(raw: np.ndarray, names: Sequence[str], mapping: dict[str, int | None]) -> np.ndarray:
    """Sum class probabilities into family5 columns; mass of unmapped classes is dropped (rows may sum < 1)."""
    raw = np.asarray(raw, dtype=float)
    out = np.zeros((raw.shape[0], len(FAMILY5)))
    for j, name in enumerate(names):
        k = mapping.get(name)
        if k is not None:
            out[:, k] += raw[:, j]
    return out


def frame_indices_30fps(refs: Sequence[SampleRef], samples: np.ndarray, frames_per_sample: int = 3) -> list[tuple[str, int, float]]:
    """Expand a set of 10 Hz samples to the underlying consecutive 30-fps frames (3 per sample)."""
    out = []
    for s in samples:
        r = refs[int(s)]
        for k in range(frames_per_sample):
            out.append((r.video_id, r.frame_idx + k, r.gamma))
    return out
