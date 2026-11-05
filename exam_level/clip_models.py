"""Exam-context decoding at clip level. Pure NumPy/SciPy on already-computed clip evidence."""
import itertools
from dataclasses import dataclass, replace

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

A4C, A5C = 2, 3
TAU_GRID = (.25, .5, 1., 2., 4., 8., 16., 32.)
ENUMERATION_CAP = 14
LOG_2PI = np.log(2 * np.pi)
MIN_SD = 1e-3


def clip_evidence(prob, floor=1e-8):
    """Mean log probability per class over a clip's own positions."""
    return np.log(np.maximum(np.asarray(prob, float), floor)).mean(0)


# --- first-order chain over clips ---

def _chain_from_counts(start, trans):
    return np.log(start / start.sum()), np.log(trans / trans.sum(1, keepdims=True))


def fit_chain(label_sequences, n=5, alpha=1.0):
    start, trans = np.full(n, alpha, float), np.full((n, n), alpha, float)
    for y in label_sequences:
        start[y[0]] += 1
        np.add.at(trans, (y[:-1], y[1:]), 1)
    return _chain_from_counts(start, trans)


def fit_bag_chain(label_sequences, n=5, alpha=1.0, seed=2026110399, n_perm=50):
    """The same estimator on randomly permuted sequences: it keeps session composition and removes order."""
    rng = np.random.default_rng(seed)
    start, trans = np.zeros(n), np.zeros((n, n))
    for y in label_sequences:
        for _ in range(n_perm):
            p = rng.permutation(y)
            start[p[0]] += 1
            np.add.at(trans, (p[:-1], p[1:]), 1)
    return _chain_from_counts(start / n_perm + alpha, trans / n_perm + alpha)


def prior_chain(pi):
    """A chain that carries only the global class prior, so decoding is independent across clips."""
    log_pi = np.log(np.asarray(pi, float))
    return log_pi, np.tile(log_pi, (len(log_pi), 1))


def decode_permuted(evidence, permutation, tau, log_start, log_trans):
    """Decode clips in a given order and return labels in the original clip order."""
    p = np.asarray(permutation)
    return viterbi(np.asarray(evidence, float)[p] / tau, log_start, log_trans)[np.argsort(p)]


def viterbi(evidence, log_start, log_trans):
    E = np.asarray(evidence, float)
    score = E[0] + log_start
    back = []
    for row in E[1:]:
        candidates = score[:, None] + log_trans
        back.append(candidates.argmax(0))
        score = candidates.max(0) + row
    path = [int(score.argmax())]
    for pointers in reversed(back):
        path.append(int(pointers[path[-1]]))
    return np.array(path[::-1], np.int64)


def _log_partition(E, log_start, log_trans):
    alpha = E[0] + log_start
    for row in E[1:]:
        alpha = logsumexp(alpha[:, None] + log_trans, axis=0) + row
    return logsumexp(alpha)


def conditional_ll(evidence, labels, log_start, log_trans):
    E, y = np.asarray(evidence, float), np.asarray(labels)
    path = E[np.arange(len(y)), y].sum() + log_start[y[0]] + log_trans[y[:-1], y[1:]].sum()
    return path - _log_partition(E, log_start, log_trans)


def fit_temperature(evidences, labels, log_start, log_trans, grid=TAU_GRID):
    """The grid value with the highest conditional log-likelihood; ties go to the earlier value."""
    scores = [sum(conditional_ll(E / tau, y, log_start, log_trans) for E, y in zip(evidences, labels)) for tau in grid]
    return float(grid[int(np.argmax(scores))])


# --- exam-level offset on the A5C minus A4C evidence ---

@dataclass(frozen=True)
class OffsetParams:
    mu4: float
    mu5: float
    s4: float
    s5: float
    tau: float
    converged: bool = True


def offset_loglik(d, bits, params):
    """log N(d; mu(bits), diag(s(bits)^2) + tau^2 11'), in closed form. bits: 0 = A4C, 1 = A5C."""
    d, bits = np.asarray(d, float), np.asarray(bits)
    mean = np.where(bits == 1, params.mu5, params.mu4)
    var = np.where(bits == 1, params.s5, params.s4) ** 2
    r = d - mean
    t2 = params.tau ** 2
    c = (1 / var).sum()
    b = (r / var).sum()
    quad = (r * r / var).sum() - t2 * b * b / (1 + t2 * c)
    return -.5 * (d.size * LOG_2PI + np.log(var).sum() + np.log1p(t2 * c) + quad)


def fit_offset(ds, ys, with_offset=True):
    """Maximum marginal likelihood given known labels. Each element is one session's apical clips."""
    d_all, y_all = np.concatenate(ds), np.concatenate(ys)
    mu = [d_all[y_all == k].mean() for k in (0, 1)]
    sd = [max(d_all[y_all == k].std(), MIN_SD) for k in (0, 1)]
    start = np.array([mu[0], mu[1], np.log(sd[0]), np.log(sd[1]), np.log(max(min(sd), MIN_SD))])

    def params(x):
        tau = float(np.exp(x[4])) if with_offset else 0.0
        return OffsetParams(float(x[0]), float(x[1]), float(np.exp(x[2])) + MIN_SD, float(np.exp(x[3])) + MIN_SD, tau)

    def negative(x):
        p = params(x)
        return -sum(offset_loglik(d, y, p) for d, y in zip(ds, ys))

    result = minimize(negative, start, method='L-BFGS-B', options={'maxiter': 500})
    return replace(params(result.x), converged=bool(result.success))


def _prior_score(labels, bits, log_pi, chain):
    if chain is None:
        return float(np.asarray(log_pi)[bits].sum())
    start, trans = chain
    return float(start[labels[0]] + trans[labels[:-1], labels[1:]].sum())


def decode_apical(evidence, base, params, log_pi=None, chain=None, cap=ENUMERATION_CAP):
    """MAP labels of clips that base calls A4C/A5C; other clips stay fixed. Returns (labels, used_fallback)."""
    if (log_pi is None) == (chain is None):
        raise ValueError('exactly one of log_pi (A4C, A5C) and chain is required')
    L, base = np.asarray(evidence, float), np.asarray(base)
    idx = np.flatnonzero((base == A4C) | (base == A5C))
    if not idx.size:
        return base.copy(), False
    d = L[idx, A5C] - L[idx, A4C]

    def labelled(bits):
        out = base.copy()
        out[idx] = A4C + bits
        return out

    def score(bits):
        return offset_loglik(d, bits, params) + _prior_score(labelled(bits), bits, log_pi, chain)

    if idx.size <= cap:
        candidates = (np.array(b, np.int64) for b in itertools.product((0, 1), repeat=idx.size))
        return labelled(max(candidates, key=score)), False
    bits = (base[idx] == A5C).astype(np.int64)
    best = score(bits)
    for _ in range(idx.size * 4):
        flips = [np.where(np.arange(idx.size) == i, 1 - bits, bits) for i in range(idx.size)]
        scores = [score(f) for f in flips]
        winner = int(np.argmax(scores))
        if scores[winner] <= best:
            break
        bits, best = flips[winner], scores[winner]
    return labelled(bits), True


# --- diagnostic: is the same-exam contrast stronger than the between-exam contrast? ---

def session_auc(d, y, sessions):
    """AUC of d for A5C (1) over A4C (0), separately on same-session and different-session pairs."""
    d, y, sessions = np.asarray(d, float), np.asarray(y), np.asarray(sessions)
    fives, fours = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    wins = (d[fives][:, None] > d[fours][None, :]) + .5 * (d[fives][:, None] == d[fours][None, :])
    same = sessions[fives][:, None] == sessions[fours][None, :]
    mean = lambda mask: float(wins[mask].mean()) if mask.any() else None
    return {'within_auc': mean(same), 'cross_auc': mean(~same), 'within_pairs': int(same.sum()), 'cross_pairs': int((~same).sum())}
