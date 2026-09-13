"""B0: raw per-frame argmax (no temporal processing) and clip-level majority."""
from __future__ import annotations

import numpy as np


def frame_argmax(prob: np.ndarray) -> np.ndarray:
    return np.asarray(prob).argmax(axis=1)


def clip_majority(prob: np.ndarray) -> int:
    """Majority vote of frame argmaxes; ties broken by mean probability."""
    labels = frame_argmax(prob)
    counts = np.bincount(labels, minlength=prob.shape[1])
    top = np.flatnonzero(counts == counts.max())
    if top.size == 1:
        return int(top[0])
    return int(top[np.argmax(np.asarray(prob).mean(0)[top])])
