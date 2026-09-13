"""B2: HMM with training-estimated (sticky) transitions and Viterbi decoding over posteriors."""
from __future__ import annotations

import numpy as np

EPS = 1e-12


def estimate_transitions(label_sequences: list[np.ndarray], n_classes: int, pseudo: float = 1.0) -> np.ndarray:
    """Row-stochastic transition matrix from labelled sequences with additive smoothing."""
    counts = np.full((n_classes, n_classes), pseudo, dtype=float)
    for seq in label_sequences:
        seq = np.asarray(seq)
        for a, b in zip(seq[:-1], seq[1:]):
            counts[a, b] += 1.0
    return counts / counts.sum(1, keepdims=True)


def sticky_transitions(n_classes: int, stay: float) -> np.ndarray:
    """Analytic transition matrix: P(stay) = stay, remainder spread uniformly."""
    if not 0.0 < stay < 1.0:
        raise ValueError("stay must be in (0, 1)")
    off = (1.0 - stay) / max(n_classes - 1, 1)
    return np.full((n_classes, n_classes), off) + np.eye(n_classes) * (stay - off)


def viterbi(log_emission: np.ndarray, log_trans: np.ndarray, log_init: np.ndarray) -> np.ndarray:
    n, c = log_emission.shape
    score = np.empty((n, c)); back = np.zeros((n, c), dtype=int)
    score[0] = log_init + log_emission[0]
    for t in range(1, n):
        cand = score[t - 1][:, None] + log_trans  # (from, to)
        back[t] = cand.argmax(0); score[t] = cand.max(0) + log_emission[t]
    path = np.empty(n, dtype=int); path[-1] = int(score[-1].argmax())
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def hmm_decode(prob: np.ndarray, trans: np.ndarray, init: np.ndarray | None = None) -> np.ndarray:
    """Use classifier posteriors as (scaled) emissions; decode the MAP label path."""
    prob = np.asarray(prob, dtype=float); c = prob.shape[1]
    init = np.full(c, 1.0 / c) if init is None else np.asarray(init, dtype=float)
    return viterbi(np.log(prob + EPS), np.log(np.asarray(trans) + EPS), np.log(init + EPS))
