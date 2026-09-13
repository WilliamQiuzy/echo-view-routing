"""Segment scoring and selective routing (accept / defer) with validation-selected thresholds."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from echo_routing.temporal.segments import Segment, labels_to_segments


@dataclass(frozen=True)
class RoutedInterval:
    start: int
    end: int
    view: int
    score: float
    accepted: bool


def segment_score(prob: np.ndarray, seg: Segment, method: str = "p10_agree") -> float:
    p = np.asarray(prob)[seg.start:seg.end]
    if p.shape[0] == 0:
        return 0.0
    if method == "mean":
        return float(p[:, seg.label].mean())
    if method == "p10_agree":
        return float(np.percentile(p[:, seg.label], 10) * (p.argmax(1) == seg.label).mean())
    if method == "max_softmax":
        return float(p.max(1).mean())
    raise ValueError(f"unknown method {method!r}")


def route(prob: np.ndarray, labels: np.ndarray, tau: float, method: str = "p10_agree",
          min_len: int = 1) -> list[RoutedInterval]:
    """Turn a decoded label sequence into scored intervals; accept those with score >= tau and length >= min_len."""
    out = []
    for seg in labels_to_segments(labels):
        score = segment_score(prob, seg, method)
        out.append(RoutedInterval(seg.start, seg.end, seg.label, score, score >= tau and seg.length >= min_len))
    return out


def select_threshold(scores: np.ndarray, wrong: np.ndarray, weights: np.ndarray, target_risk: float) -> float | None:
    """Smallest threshold whose accepted-duration contamination <= target_risk (max coverage).
    Returns None when no threshold meets the target with non-zero coverage."""
    scores = np.asarray(scores, float); wrong = np.asarray(wrong, bool); w = np.asarray(weights, float)
    best = None
    for tau in np.unique(scores)[::-1]:
        acc = scores >= tau
        if w[acc].sum() <= 0:
            continue
        risk = w[acc & wrong].sum() / w[acc].sum()
        if risk <= target_risk:
            best = float(tau)  # lower tau -> more coverage; keep going while feasible
        else:
            break
    return best
