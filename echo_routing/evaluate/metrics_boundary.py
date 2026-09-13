"""Semantic boundary F1 with one-to-one tolerance matching and timing error."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BoundaryResult:
    n_true: int
    n_pred: int
    matched: int
    precision: float
    recall: float
    f1: float
    median_abs_error: float | None


def match_boundaries(pred: np.ndarray, true: np.ndarray, tol: float) -> list[tuple[int, int]]:
    """One-to-one greedy matching by absolute distance within tolerance (units = samples or seconds)."""
    pred = np.asarray(pred, float); true = np.asarray(true, float)
    if pred.size == 0 or true.size == 0:
        return []
    d = np.abs(pred[:, None] - true[None, :])
    pairs = sorted(((d[i, j], i, j) for i in range(pred.size) for j in range(true.size) if d[i, j] <= tol))
    used_p, used_t, out = set(), set(), []
    for _, i, j in pairs:
        if i not in used_p and j not in used_t:
            used_p.add(i); used_t.add(j); out.append((i, j))
    return out


def boundary_f1(pred: np.ndarray, true: np.ndarray, tol: float) -> BoundaryResult:
    pred = np.asarray(pred, float); true = np.asarray(true, float)
    pairs = match_boundaries(pred, true, tol)
    m = len(pairs)
    precision = m / pred.size if pred.size else (1.0 if true.size == 0 else 0.0)
    recall = m / true.size if true.size else (1.0 if pred.size == 0 else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    errs = [abs(pred[i] - true[j]) for i, j in pairs]
    return BoundaryResult(int(true.size), int(pred.size), m, precision, recall, f1, float(np.median(errs)) if errs else None)
