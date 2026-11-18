"""The change-point decoder ("B3"): cuts where the class probabilities on the two sides of a sample differ.

score[i] is the Jensen-Shannon divergence between the mean probability over the `half_window` samples before i and
the mean over the `half_window` samples from i on. A cut is a local peak of the score at or above the threshold
(weaker peaks within `nms_radius` samples are dropped) at which the two sides also name different views. Every
piece between cuts takes the view with the largest mean probability.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-12


def _kl(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    return np.sum(p * (np.log(p + EPS) - np.log(q + EPS)), axis=-1)


def js_divergence(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    p, q = np.asarray(p, float), np.asarray(q, float)
    m = 0.5 * (p + q)
    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def boundary_scores(prob: np.ndarray, half_window: int) -> np.ndarray:
    prob = np.asarray(prob, float)
    n, h = prob.shape[0], max(1, half_window)
    cumulative = np.cumsum(np.pad(prob, ((1, 0), (0, 0))), axis=0)
    scores = np.zeros(n)
    for i in range(1, n):
        lo, hi = max(0, i - h), min(n, i + h)
        left = (cumulative[i] - cumulative[lo]) / (i - lo)
        right = (cumulative[hi] - cumulative[i]) / (hi - i)
        scores[i] = js_divergence(left, right)
    return scores


def non_max_suppression(scores: np.ndarray, threshold: float, radius: int) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    candidates = np.flatnonzero(scores >= threshold)
    order = candidates[np.argsort(-scores[candidates], kind="stable")]
    kept: list[int] = []
    for i in order:
        if all(abs(i - k) > radius for k in kept):
            kept.append(int(i))
    return np.array(sorted(kept), dtype=int)


def detect_boundaries(prob: np.ndarray, half_window: int, threshold: float, nms_radius: int) -> np.ndarray:
    prob = np.asarray(prob, float)
    peaks = non_max_suppression(boundary_scores(prob, half_window), threshold, nms_radius)
    kept = []
    for i in peaks:
        lo, hi = max(0, i - half_window), min(prob.shape[0], i + half_window)
        if prob[lo:i].mean(0).argmax() != prob[i:hi].mean(0).argmax():
            kept.append(int(i))
    return np.array(kept, dtype=int)


def decode(prob: np.ndarray, half_window: int, threshold: float, nms_radius: int) -> np.ndarray:
    """Per-sample probabilities (N, classes) -> per-sample view indices (N,)."""
    prob = np.asarray(prob, float)
    n = prob.shape[0]
    cuts = np.concatenate([[0], detect_boundaries(prob, half_window, threshold, nms_radius), [n]]).astype(int)
    labels = np.empty(n, dtype=int)
    for start, end in zip(cuts[:-1], cuts[1:]):
        if end > start:
            labels[start:end] = int(prob[start:end].mean(0).argmax())
    return labels


def segments(labels: np.ndarray) -> list[tuple[int, int, int]]:
    """Runs of equal labels as (first sample, one past the last sample, view index)."""
    labels = np.asarray(labels, int)
    if not labels.size:
        return []
    edges = np.flatnonzero(np.diff(labels) != 0) + 1
    starts = np.concatenate([[0], edges])
    ends = np.concatenate([edges, [labels.size]])
    return [(int(a), int(b), int(labels[a])) for a, b in zip(starts, ends)]
