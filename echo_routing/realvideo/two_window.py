"""Component audit of the two-window change detector (docs/research_plan.md, audit round).

Every configuration runs B3's own pipeline and changes one component at a time:
score -> non-maximum suppression above a threshold (radius 5 samples) -> keep a peak only if the mean probabilities of
the window before and after it have different argmax -> label each segment by its mean probability.

Components:
- score: "js" (B3's Jensen-Shannon divergence, nats) or "tv" (total variation) between the mean probabilities of the
  window before and the window after sample t;
- window: a fixed number of samples per side, or K x the estimated period, with a fixed fallback where no period
  is available;
- thresholds: B3's raw grid, a grid at fixed quantiles of the validation native-clip scores (one monotone map, so it
  only re-parameterises the threshold), or quantiles taken separately for samples with and without a period (this
  also changes the ranking across the two groups).
"""
from __future__ import annotations

import numpy as np

from echo_routing.temporal.baselines.b3_js_divergence import js_divergence, labels_from_boundaries
from echo_routing.temporal.segments import non_max_suppression

FALLBACK_SAMPLES = 9
NMS_RADIUS = 5


def window_lengths(n: int, fixed: int | None = None, period_samples: np.ndarray | None = None, k: float | None = None,
                   fallback: int = FALLBACK_SAMPLES) -> tuple[np.ndarray, np.ndarray]:
    """Per-sample window length (samples per side) and a mask of samples where a period set it."""
    if fixed is not None:
        return np.full(n, int(fixed)), np.zeros(n, dtype=bool)
    per = np.asarray(period_samples, dtype=float)
    avail = np.isfinite(per) & (per >= 2)
    lengths = np.where(avail, np.maximum(1, np.rint(k * np.where(avail, per, 0))), fallback).astype(int)
    return lengths, avail


def _side_means(prob: np.ndarray, t: int, length: int) -> tuple[np.ndarray, np.ndarray] | None:
    n = len(prob); lo, hi = max(0, t - length), min(n, t + length)
    if t - lo == 0 or hi - t == 0:
        return None
    return prob[lo:t].mean(0), prob[t:hi].mean(0)


def two_window_score(prob: np.ndarray, lengths: np.ndarray, kind: str) -> np.ndarray:
    """score[t] = divergence(mean prob over [t-L, t), mean prob over [t, t+L)), L = lengths[t]; 0 if a side is empty."""
    if kind not in ("js", "tv"):
        raise ValueError("kind must be 'js' or 'tv'")
    prob = np.asarray(prob, dtype=float); n = len(prob)
    cs = np.cumsum(np.pad(prob, ((1, 0), (0, 0))), axis=0)
    t = np.arange(1, n); L = np.asarray(lengths)[1:]
    lo, hi = np.maximum(0, t - L), np.minimum(n, t + L)
    left = (cs[t] - cs[lo]) / (t - lo)[:, None]; right = (cs[hi] - cs[t]) / np.maximum(hi - t, 1)[:, None]
    s = js_divergence(left, right) if kind == "js" else 0.5 * np.abs(left - right).sum(-1)
    out = np.zeros(n); out[1:] = np.where(hi - t > 0, s, 0.0)
    return out


def detect_labels(prob: np.ndarray, score: np.ndarray, threshold: float, lengths: np.ndarray,
                  radius: int = NMS_RADIUS) -> np.ndarray:
    """B3's decoding with per-sample windows: NMS peaks, label-change check over the peak's window, segment labels."""
    prob = np.asarray(prob, dtype=float)
    keep = []
    for i in non_max_suppression(score, threshold, radius):
        sides = _side_means(prob, int(i), int(lengths[i]))
        if sides is not None and sides[0].argmax() != sides[1].argmax():
            keep.append(int(i))
    return labels_from_boundaries(prob, np.array(keep, dtype=int))


def quantile_thresholds(scores: np.ndarray, levels: tuple[float, ...]) -> list[float]:
    """Thresholds at fixed quantiles of reference scores (a monotone re-parameterisation of the raw threshold)."""
    s = np.asarray(scores, dtype=float)
    return [float(np.quantile(s, q)) for q in levels]
