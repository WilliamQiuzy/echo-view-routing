"""Selective-routing threshold rules that respect cine reuse, and score diagnostics (proposal §6.4, §7.3).

Validation and test streams reuse cines across streams, so the upper bound on a threshold's risk comes from a
bootstrap that resamples cines, not samples or streams. All rules pick the smallest feasible threshold (the largest
coverage) on the validation data and are then frozen.
"""
from __future__ import annotations

import numpy as np

from echo_routing.evaluate.bootstrap import resample_weights
from echo_routing.temporal.decode_select import select_threshold


def temperature_scale(prob: np.ndarray, t: float) -> np.ndarray:
    """Divide log-probabilities by `t` and renormalise (t > 1 softens, t < 1 sharpens)."""
    logp = np.log(np.clip(prob, 1e-8, 1.0)) / t
    logp -= logp.max(1, keepdims=True)
    e = np.exp(logp)
    return e / e.sum(1, keepdims=True)


def fit_temperature(prob: np.ndarray, truth: np.ndarray, grid: np.ndarray | None = None) -> float:
    """Temperature minimising the per-sample negative log-likelihood of the true class (grid search)."""
    grid = np.geomspace(0.25, 8.0, 41) if grid is None else grid
    rows = np.arange(truth.size)
    nll = [-np.log(np.clip(temperature_scale(prob, t)[rows, truth], 1e-12, 1.0)).mean() for t in grid]
    return float(grid[int(np.argmin(nll))])


def threshold_grid(scores: np.ndarray, n: int = 400) -> np.ndarray:
    """Candidate thresholds: every finite score if there are at most `n`, else `n` quantiles of the per-sample scores
    (evenly spaced in coverage)."""
    finite = scores[np.isfinite(scores)]
    uniq = np.unique(finite)
    return uniq if uniq.size <= n else np.unique(np.quantile(finite, np.linspace(0.0, 1.0, n)))


def cine_counts(scores: np.ndarray, wrong: np.ndarray, cine: np.ndarray, taus: np.ndarray,
                ids: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per cine and threshold: accepted samples and accepted-wrong samples (score >= tau).
    `ids` fixes the cine order (cines absent from `cine` get zero rows)."""
    ids = np.unique(cine) if ids is None else ids
    pos = {c: i for i, c in enumerate(ids.tolist())}
    inv = np.array([pos[c] for c in cine.tolist()], dtype=int)
    acc = np.zeros((ids.size, taus.size)); wr = np.zeros((ids.size, taus.size))
    for k, t in enumerate(taus):
        a = scores >= t
        acc[:, k] = np.bincount(inv, weights=a, minlength=ids.size)
        wr[:, k] = np.bincount(inv, weights=a & wrong, minlength=ids.size)
    return ids, acc, wr


def select_threshold_ucb(acc: np.ndarray, wr: np.ndarray, taus: np.ndarray, target: float, weights: np.ndarray,
                         level: float = 0.95) -> float | None:
    """Smallest threshold whose bootstrap `level` quantile of risk is <= target.
    `weights` is (n_boot, n_cines) cine multiplicities; a single row of ones reproduces the point-estimate rule."""
    a, w = weights @ acc, weights @ wr
    with np.errstate(invalid="ignore", divide="ignore"):
        risk = np.where(a > 0, w / np.where(a > 0, a, 1.0), np.nan)
    finite_cols = ~np.all(np.isnan(risk), axis=0)
    ucb = np.full(taus.size, np.inf)
    ucb[finite_cols] = np.nanquantile(risk[:, finite_cols], level, axis=0)
    ok = np.flatnonzero((acc.sum(0) > 0) & (ucb <= target))
    return float(taus[ok[0]]) if ok.size else None


def select_classwise(scores: np.ndarray, wrong: np.ndarray, pred: np.ndarray, cine: np.ndarray, target: float,
                     rule: str = "point", n_boot: int = 1000, rng: np.random.Generator | None = None,
                     grid_n: int = 400) -> dict[int, float | None]:
    """One threshold per predicted class, each meeting the target within that class (so the total does too)."""
    ids = np.unique(cine)
    weights = None if rule == "point" else resample_weights(ids.size, n_boot, rng or np.random.default_rng(0))
    out = {}
    for c in np.unique(pred).tolist():
        m = pred == c
        if rule == "point":
            out[int(c)] = select_threshold(scores[m], wrong[m], np.ones(int(m.sum())), target)
        else:
            taus = threshold_grid(scores[m], grid_n)
            _, acc, wr = cine_counts(scores[m], wrong[m], cine[m], taus, ids)
            out[int(c)] = select_threshold_ucb(acc, wr, taus, target, weights) if taus.size else None
    return out


def accept_mask(scores: np.ndarray, pred: np.ndarray, tau: float | None | dict[int, float | None]) -> np.ndarray:
    """Accepted samples under a global threshold or a per-class dict of thresholds (None accepts nothing)."""
    if isinstance(tau, dict):
        t = np.array([np.inf if tau.get(int(c)) is None else tau[int(c)] for c in pred.tolist()])
        return scores >= t
    return np.zeros(scores.size, bool) if tau is None else scores >= tau


def aurc(scores: np.ndarray, wrong: np.ndarray) -> float:
    """Area under the risk-coverage curve over the reachable coverage range, ties grouped (lower is better)."""
    finite = np.isfinite(scores)
    if not finite.any():
        return float("nan")
    s, w = scores[finite], wrong[finite].astype(float)
    uniq, inv = np.unique(-s, return_inverse=True)                    # descending score order
    n_at = np.bincount(inv); w_at = np.bincount(inv, weights=w)
    cov = np.cumsum(n_at); risk = np.cumsum(w_at) / cov
    return float((risk * n_at).sum() / cov[-1])


def oracle_coverage(scores: np.ndarray, wrong: np.ndarray, target: float) -> float:
    """Largest coverage whose risk on this same data is <= target: a hindsight bound on what the score allows."""
    tau = select_threshold(scores, wrong, np.ones(scores.size), target)
    return 0.0 if tau is None else float((scores >= tau).mean())
