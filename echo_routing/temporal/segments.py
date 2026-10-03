"""Label sequences <-> segments, and boundary utilities shared by all baselines."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Segment:
    start: int  # inclusive sample index
    end: int    # exclusive sample index
    label: int

    @property
    def length(self) -> int:
        return self.end - self.start


def labels_to_segments(labels: np.ndarray) -> list[Segment]:
    labels = np.asarray(labels)
    if labels.size == 0:
        return []
    change = np.flatnonzero(np.diff(labels)) + 1
    starts = np.concatenate([[0], change]); ends = np.concatenate([change, [labels.size]])
    return [Segment(int(s), int(e), int(labels[s])) for s, e in zip(starts, ends)]


def segments_to_labels(segments: list[Segment], n: int, fill: int = -1) -> np.ndarray:
    out = np.full(n, fill, dtype=int)
    for s in segments:
        out[s.start:s.end] = s.label
    return out


def boundaries_from_labels(labels: np.ndarray) -> np.ndarray:
    """Positions i where labels[i] != labels[i-1] (semantic change events)."""
    labels = np.asarray(labels)
    return np.flatnonzero(np.diff(labels)) + 1 if labels.size > 1 else np.array([], dtype=int)


def count_fragments(labels: np.ndarray) -> int:
    """Number of label changes (0 for a perfectly stable sequence)."""
    return int(boundaries_from_labels(labels).size)


def non_max_suppression(scores: np.ndarray, threshold: float, radius: int) -> np.ndarray:
    """Local peaks >= threshold, suppressing weaker peaks within `radius` samples."""
    scores = np.asarray(scores, dtype=float)
    cand = np.flatnonzero(scores >= threshold)
    order = cand[np.argsort(-scores[cand], kind="stable")]
    kept: list[int] = []
    for i in order:
        if all(abs(i - k) > radius for k in kept):
            kept.append(int(i))
    return np.array(sorted(kept), dtype=int)


def boundary_decode(prob: np.ndarray, boundary: np.ndarray, threshold: float, nms_radius: int) -> np.ndarray:
    """Intervals from boundary probabilities (peaks >= threshold after non-maximum suppression, position 0 excluded),
    each labelled with the argmax of its mean class probability; adjacent intervals with the same label merge because
    the labels are written per sample. Used by the round-2 boundary segments and both round-3 steps."""
    prob = np.asarray(prob, float); b = np.where(np.arange(len(boundary)) > 0, np.asarray(boundary, float), 0.0)
    cuts = np.concatenate([[0], non_max_suppression(b, threshold, nms_radius), [prob.shape[0]]])
    labels = np.empty(prob.shape[0], dtype=int)
    for a, e in zip(cuts[:-1], cuts[1:]):
        if e > a:
            labels[a:e] = int(prob[a:e].mean(0).argmax())
    return labels
