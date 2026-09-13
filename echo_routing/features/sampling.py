"""Frame sampling policies: timestamp-spaced training frames, fixed-rate extraction, cine-balanced weights."""
from __future__ import annotations

from typing import Sequence

import numpy as np

from echo_routing.errors import FrameError


def spaced_indices(n_frames: int, k: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """k timestamp-spaced indices in [0, n_frames). Random phase if rng given, else centred."""
    if n_frames <= 0:
        raise FrameError("n_frames must be positive")
    k = min(k, n_frames)
    edges = np.linspace(0, n_frames, k + 1)
    if rng is None:
        offsets = (edges[1:] - edges[:-1]) / 2.0
    else:
        offsets = rng.uniform(0.0, 1.0, size=k) * (edges[1:] - edges[:-1])
    return np.minimum(np.floor(edges[:-1] + offsets).astype(int), n_frames - 1)


def stride_indices(n_frames: int, fps: float, target_hz: float) -> np.ndarray:
    """Indices sampled at `target_hz` from a `fps` playback stream (start at 0)."""
    if fps <= 0 or target_hz <= 0:
        raise FrameError("fps and target_hz must be positive")
    step = max(1, int(round(fps / target_hz)))
    return np.arange(0, n_frames, step, dtype=int)


def balanced_video_weights(label_indices: Sequence[int]) -> np.ndarray:
    """Per-video sampling weights so each class is drawn equally often (cine-level balance)."""
    labels = np.asarray(label_indices)
    counts = np.bincount(labels)
    weights = 1.0 / counts[labels]
    return weights / weights.sum()
