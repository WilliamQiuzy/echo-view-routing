"""Stream-level inference helpers for the ViT family (pure NumPy parts are unit-tested).

Plans (data/window_plans/<hash>/plans.json) list, per stream, one reference per 10 Hz sample:
[video_id, 30-fps frame index, gamma]. Frame-wise encoders read those samples from their own feature cache (the
cache grid is identical: stride 3 from frame 0); clip models run the same sliding windows as STFM/EchoViewCLIP
(16 samples = 1.6 s, stride 5 samples = 0.5 s) and each sample takes the probabilities of the nearest window centre.
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from echo_routing.errors import CacheMissError
from echo_routing.features.cache import VideoFeatures
from echo_routing.temporal.windowed import assign_to_samples, window_centres, window_samples
from echo_routing.vit.clips import pad_window

WINDOW, STRIDE = 16, 5
GAMMA_TO_VARIANT = {1.0: "orig", 0.9: "gamma090", 1.1: "gamma110"}


def variant_of(gamma: float) -> str:
    for g, v in GAMMA_TO_VARIANT.items():
        if abs(float(gamma) - g) < 1e-6:
            return v
    raise CacheMissError(f"no cache variant for gamma {gamma}")


def sample_rows(vf: VideoFeatures, frame_idx: Sequence[int]) -> np.ndarray:
    """Row index of each 30-fps frame index in a cached cine (exact match required)."""
    frame_idx = np.asarray(frame_idx, dtype=int)
    pos = np.searchsorted(vf.frame_idx, frame_idx)
    if np.any(pos >= vf.frame_idx.size) or np.any(vf.frame_idx[np.minimum(pos, vf.frame_idx.size - 1)] != frame_idx):
        raise CacheMissError(f"frames {frame_idx.tolist()[:5]}... not on the cached grid of {vf.video_id}")
    return pos


def assemble(refs: Sequence[Sequence], loader: Callable[[str, str], VideoFeatures], field: str) -> np.ndarray:
    """Stack a cached per-sample array (`feat` or `prob`) along a plan's references, run by run."""
    out, i = [], 0
    while i < len(refs):
        vid, gamma = refs[i][0], float(refs[i][2]); j = i
        while j < len(refs) and refs[j][0] == vid and float(refs[j][2]) == gamma:
            j += 1
        vf = loader(vid, variant_of(gamma))
        rows = sample_rows(vf, [r[1] for r in refs[i:j]])
        out.append(np.asarray(getattr(vf, field))[rows]); i = j
    return np.concatenate(out)


def windows(n: int, size: int = WINDOW, stride: int = STRIDE) -> tuple[np.ndarray, list[np.ndarray]]:
    """Window centres and their (right-padded) sample indices for a stream of n samples."""
    centres = window_centres(n, stride)
    return centres, [pad_window(window_samples(int(c), n, size), size) for c in centres]


def windowed_probs(n: int, window_fn: Callable[[list[np.ndarray]], np.ndarray], size: int = WINDOW,
                   stride: int = STRIDE) -> tuple[np.ndarray, np.ndarray]:
    """(per-sample probabilities, per-window probabilities); window_fn maps a list of index windows to (W, C)."""
    centres, wins = windows(n, size, stride)
    wp = np.asarray(window_fn(wins))
    return assign_to_samples(n, centres, wp), wp
