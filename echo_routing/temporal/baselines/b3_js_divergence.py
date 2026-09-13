"""B3: Jensen-Shannon divergence between left/right averaged class posteriors (semantic-change baseline)."""
from __future__ import annotations

import numpy as np

from echo_routing.temporal.segments import non_max_suppression

EPS = 1e-12


def kl(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float); q = np.asarray(q, float)
    return np.sum(p * (np.log(p + EPS) - np.log(q + EPS)), axis=-1)


def js_divergence(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """JSD in nats, symmetric, bounded by ln 2. Works row-wise on (..., C)."""
    m = 0.5 * (np.asarray(p, float) + np.asarray(q, float))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def boundary_scores(prob: np.ndarray, half_window: int) -> np.ndarray:
    """score[i] = JSD(mean prob over [i-h, i), mean prob over [i, i+h)); 0 where a side is empty."""
    prob = np.asarray(prob, float); n = prob.shape[0]; h = max(1, half_window)
    cs = np.cumsum(np.pad(prob, ((1, 0), (0, 0))), axis=0)
    scores = np.zeros(n)
    for i in range(1, n):
        lo, hi = max(0, i - h), min(n, i + h)
        left = (cs[i] - cs[lo]) / (i - lo); right = (cs[hi] - cs[i]) / (hi - i)
        scores[i] = js_divergence(left, right)
    return scores


def detect_boundaries(prob: np.ndarray, half_window: int, threshold: float, nms_radius: int,
                      require_label_change: bool = True) -> np.ndarray:
    """Peaks of JSD above threshold; optionally require the left/right argmax to differ (persistent change)."""
    prob = np.asarray(prob, float); scores = boundary_scores(prob, half_window)
    peaks = non_max_suppression(scores, threshold, nms_radius)
    if not require_label_change:
        return peaks
    keep = []
    for i in peaks:
        lo, hi = max(0, i - half_window), min(prob.shape[0], i + half_window)
        if prob[lo:i].mean(0).argmax() != prob[i:hi].mean(0).argmax():
            keep.append(int(i))
    return np.array(keep, dtype=int)


def labels_from_boundaries(prob: np.ndarray, boundaries: np.ndarray) -> np.ndarray:
    """Assign each segment between boundaries its mean-probability argmax."""
    prob = np.asarray(prob, float); n = prob.shape[0]
    cuts = np.concatenate([[0], np.asarray(boundaries, int), [n]])
    labels = np.empty(n, dtype=int)
    for s, e in zip(cuts[:-1], cuts[1:]):
        if e > s:
            labels[s:e] = int(prob[s:e].mean(0).argmax())
    return labels
