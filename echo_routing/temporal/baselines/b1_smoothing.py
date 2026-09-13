"""B1: probability moving average / categorical majority filter + hysteresis."""
from __future__ import annotations

import numpy as np


def moving_average(prob: np.ndarray, window: int) -> np.ndarray:
    """Centred moving average over time with edge shrinkage (window must be odd, >=1)."""
    prob = np.asarray(prob, dtype=float)
    if window < 1 or window % 2 == 0:
        raise ValueError("window must be an odd positive integer")
    if window == 1:
        return prob.copy()
    half = window // 2
    cs = np.cumsum(np.pad(prob, ((1, 0), (0, 0))), axis=0)
    n = prob.shape[0]; idx = np.arange(n)
    lo = np.maximum(idx - half, 0); hi = np.minimum(idx + half + 1, n)
    return (cs[hi] - cs[lo]) / (hi - lo)[:, None]


def majority_filter(labels: np.ndarray, window: int) -> np.ndarray:
    """Centred categorical mode filter; never a numerical median of class ids."""
    labels = np.asarray(labels)
    if window < 1 or window % 2 == 0:
        raise ValueError("window must be an odd positive integer")
    half = window // 2; n = labels.size; out = labels.copy()
    for i in range(n):
        w = labels[max(0, i - half): min(n, i + half + 1)]
        counts = np.bincount(w); top = np.flatnonzero(counts == counts.max())
        out[i] = labels[i] if labels[i] in top else int(top[0])
    return out


def hysteresis(labels: np.ndarray, min_run: int) -> np.ndarray:
    """Accept a label change only once the new label has persisted for `min_run` samples."""
    labels = np.asarray(labels)
    if min_run <= 1 or labels.size == 0:
        return labels.copy()
    out = labels.copy(); current = labels[0]; i = 1; n = labels.size
    while i < n:
        if labels[i] == current:
            out[i] = current; i += 1; continue
        j = i
        while j < n and labels[j] == labels[i]:
            j += 1
        if j - i >= min_run:
            current = labels[i]; out[i:j] = current
        else:
            out[i:j] = current
        i = j
    return out


def smooth_labels(prob: np.ndarray, window: int, min_run: int, mode: str = "prob") -> np.ndarray:
    if mode == "prob":
        return hysteresis(moving_average(prob, window).argmax(1), min_run)
    if mode == "majority":
        return hysteresis(majority_filter(np.asarray(prob).argmax(1), window), min_run)
    raise ValueError(f"unknown mode {mode!r}")
