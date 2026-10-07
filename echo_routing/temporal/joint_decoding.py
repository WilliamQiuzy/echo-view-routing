"""Joint label/boundary decoding (configs/experiments/joint_boundary_decoding_v1.yaml).

One Viterbi pass over a whole sequence returns the label path y that maximises

    sum_t log p_t(y_t) + lam * sum_{t>=1} [c_t log b_t + (1 - c_t) log(1 - b_t)] - kappa * sum_{t>=1} c_t,

where c_t = 1 when y_t != y_{t-1}, p_t is a frozen classifier's class-probability vector at sample t and b_t is the
probability that a new segment starts at t. That is ASRF's convention: its boundary target marks t when
labels[t] != labels[t-1], so b_t pairs with a change between t-1 and t. With lam = 0 the boundary term vanishes and the
decoder is the fixed-switch-cost J0.

Inputs are clipped (p to [eps, 1], b to [eps, 1 - eps]) and handled in float64. Ties: at every step staying beats
switching when the two scores are equal; among equal predecessors, and among equal final states, the lowest class index
wins. The decoder reads p and b only, never reference labels, joins or source-recording boundaries.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

EPS = 1e-6


def log_terms(prob: np.ndarray, bprob: np.ndarray, eps: float = EPS) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """log p (T, C), log b (T,) and log(1 - b) (T,) after clipping."""
    p = np.asarray(prob, dtype=np.float64); b = np.asarray(bprob, dtype=np.float64)
    if p.ndim != 2 or p.shape[0] == 0 or not 2 <= p.shape[1] <= 127 or b.shape != (p.shape[0],):   # int8 back-pointers
        raise ValueError(f"expected prob (T, C) with T >= 1, 2 <= C <= 127 and bprob (T,); got {p.shape} and {b.shape}")
    if not (np.isfinite(p).all() and np.isfinite(b).all()) or (p < 0).any() or (p > 1 + 1e-5).any() or (b < 0).any() or (b > 1).any():
        raise ValueError("probabilities must be finite, with p in [0, 1] (rounding up to 1e-5 allowed) and b in [0, 1]")
    if not 0.0 < eps < 0.5:
        raise ValueError(f"eps must lie in (0, 0.5), got {eps}")
    bc = np.clip(b, eps, 1.0 - eps)
    return np.log(np.clip(p, eps, 1.0)), np.log(bc), np.log1p(-bc)


def check_params(params: Sequence[tuple[float, float]]) -> tuple[np.ndarray, np.ndarray]:
    lam = np.array([float(x) for x, _ in params]); kap = np.array([float(k) for _, k in params])
    if lam.size == 0 or not (np.isfinite(lam).all() and np.isfinite(kap).all()) or (lam < 0).any() or (kap < 0).any():
        raise ValueError(f"lambda and kappa must be finite and non-negative: {list(params)}")
    return lam, kap


def joint_viterbi_grid(prob: np.ndarray, bprob: np.ndarray, params: Sequence[tuple[float, float]], eps: float = EPS) -> np.ndarray:
    """The optimal label path for every (lam, kappa) in params, shape (G, T): one pass, vectorised over the settings."""
    logp, lb, l1b = log_terms(prob, bprob, eps); lam, kap = check_params(params)
    t_len, c = logp.shape; g = lam.size; cls = np.arange(c); diag = np.eye(c, dtype=bool)[None]
    delta = np.repeat(logp[:1], g, axis=0)
    back = np.zeros((t_len, g, c), dtype=np.int8)
    for t in range(1, t_len):
        other = np.where(diag, -np.inf, delta[:, :, None])                  # (G, from, to); a switch never comes from `to`
        src = other.argmax(axis=1)                                           # lowest index among equal predecessors
        best = np.take_along_axis(other, src[:, None, :], axis=1)[:, 0]
        stay = delta + (lam * l1b[t])[:, None]
        switch = best + (lam * lb[t] - kap)[:, None]
        keep = stay >= switch                                                # staying wins ties
        back[t] = np.where(keep, cls[None], src)
        delta = np.where(keep, stay, switch) + logp[t][None]
    path = np.empty((g, t_len), dtype=np.int64); path[:, -1] = delta.argmax(axis=1)
    rows = np.arange(g)
    for t in range(t_len - 1, 0, -1):
        path[:, t - 1] = back[t, rows, path[:, t]]
    return path


def joint_viterbi(prob: np.ndarray, bprob: np.ndarray, lam: float, kappa: float, eps: float = EPS) -> np.ndarray:
    return joint_viterbi_grid(prob, bprob, [(lam, kappa)], eps)[0]


def path_score(labels: np.ndarray, prob: np.ndarray, bprob: np.ndarray, lam: float, kappa: float, eps: float = EPS) -> float:
    """The objective of one label path (checks and tests)."""
    logp, lb, l1b = log_terms(prob, bprob, eps); y = np.asarray(labels, dtype=int)
    if y.shape != (logp.shape[0],):
        raise ValueError("one label per sample")
    change = y[1:] != y[:-1]
    return float(logp[np.arange(y.size), y].sum() + lam * np.where(change, lb[1:], l1b[1:]).sum() - kappa * change.sum())
