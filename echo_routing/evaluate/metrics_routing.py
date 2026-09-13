"""Selective routing endpoints: coverage C(tau) and contamination risk R(tau) (proposal §7.2)."""
from __future__ import annotations

import numpy as np


def coverage_and_risk(accepted: np.ndarray, wrong: np.ndarray, weights: np.ndarray | None = None) -> dict:
    """Coverage = accepted duration / all duration. Risk = wrong accepted duration / accepted duration (None if 0)."""
    accepted = np.asarray(accepted, bool); wrong = np.asarray(wrong, bool)
    w = np.ones(accepted.size) if weights is None else np.asarray(weights, float)
    total = w.sum(); acc = w[accepted].sum()
    return {
        "coverage": float(acc / total) if total > 0 else 0.0,
        "risk": float(w[accepted & wrong].sum() / acc) if acc > 0 else None,
        "accepted_duration": float(acc),
        "total_duration": float(total),
    }


def risk_coverage_curve(scores: np.ndarray, wrong: np.ndarray, weights: np.ndarray | None = None) -> list[dict]:
    """Curve over unique thresholds (descending): each point is coverage/risk when accepting score >= tau."""
    scores = np.asarray(scores, float)
    return [{"tau": float(t), **coverage_and_risk(scores >= t, wrong, weights)} for t in np.unique(scores)[::-1]]
