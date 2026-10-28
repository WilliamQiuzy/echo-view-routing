"""A44 (configs/experiments/video_risk_constrained_a5c_v1.yaml): the fit fold's Base-A4C repair opportunities as training
units, the fixed cyclic 600-update plan, the scalar head on A08's unchanged 512-d embed output, the C (class-balanced BCE)
and N (hinge false-promotion constraint with an adaptive dual variable) objectives with float64 references, and the
one-way weighted-mean deployment of a frozen model on original Base-A4C runs.

Units. A P run is a maximal equal-label run of Base's per-sample labels; its membership is the unique nearest-centre
request rows j with weights v_j = multiplicity / total (sum 1), exactly as A31 stored them. Run score r = sum_j v_j s_j.
Lists (fit fold only, original P order). eligible = Base label A4C; positives = eligible with reference A5C (S1);
protected negatives = eligible with reference A4C; other-class eligible runs are excluded from training only.
Plan. Update u: positive default_rng([20261028, 44, s, f, u]).choice(positives); negatives from one permutation
default_rng([20261028, 44, s, f, 1000]).permutation(negatives) at indices (3 u + arange(3)) % n_neg.
Objectives. A = softplus(-r_p); B = mean_k softplus(r_nk); H = mean_k relu(1 + r_nk).
C = A + B. N = A + lambda (H - alpha) with lambda constant in the primal step; afterwards
lambda <- clip(lambda + 0.1 (H_pre - alpha), 0, 100). alpha = 0.01, lambda_0 = 1.
Deployment. A Base run labelled A4C with r = weighted mean over its unique rows > 0 becomes A5C; every other label is copied.
"""
from __future__ import annotations

import hashlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from echo_routing.temporal.witness_mil import edit_sequences, one_way  # noqa: F401  (re-exported, unchanged A43 rule / relabel)

BASE, ROUND, NEG_KEY = 20261028, 44, 1000
A4C, A5C = 2, 3
EMBED, INIT_BIAS, N_NEG = 512, -2.0, 3
ALPHA, LAMBDA0, DUAL_STEP, LAMBDA_MIN, LAMBDA_MAX = 0.01, 1.0, 0.1, 0.0, 100.0
ARMS = ("C", "N")
HEAD_PARAMS = EMBED + 1
CONTRACT = ("units = original P runs (unique nearest-centre rows, multiplicity weights summing to 1), r = sum v s; lists = Base-A4C runs split by reference "
            "A5C (positives) / A4C (protected negatives) / other (excluded from training only); plan rng [20261028, 44, s, f, u].choice(pos) and "
            "[20261028, 44, s, f, 1000].permutation(neg)[(3u + arange(3)) % n]; head Linear(512, 1) init 0 / -2 on the unchanged embed; "
            "A = softplus(-r_p), B = mean softplus(r_n), H = mean relu(1 + r_n); C = A + B; N = A + lambda (H - 0.01), "
            "lambda <- clip(lambda + 0.1 (H_pre - 0.01), 0, 100); deploy: Base A4C run with weighted-mean score > 0 -> A5C")


# ----------------------------------------------------------------------------------------------- lists and plan
def training_lists(pk: dict) -> dict:
    """Fold-local P-run indices in original P order: eligible (Base A4C), positives (reference A5C), protected negatives
    (reference A4C) and the other-class eligible runs (excluded from binary training only)."""
    base, y = np.asarray(pk["base"]), np.asarray(pk["labels"]); el = base == A4C
    return {"eligible": np.flatnonzero(el), "positives": np.flatnonzero(el & (y == A5C)), "negatives": np.flatnonzero(el & (y == A4C)),
            "other": np.flatnonzero(el & (y != A5C) & (y != A4C))}


def draw_positive(pos: np.ndarray, s: int, f: int, u: int) -> int:
    return int(np.random.default_rng([BASE, ROUND, int(s), int(f), int(u)]).choice(np.asarray(pos)))


def negative_permutation(neg: np.ndarray, s: int, f: int) -> np.ndarray:
    return np.random.default_rng([BASE, ROUND, int(s), int(f), NEG_KEY]).permutation(np.asarray(neg))


def negatives_of(perm: np.ndarray, u: int) -> np.ndarray:
    return perm[(N_NEG * int(u) + np.arange(N_NEG)) % len(perm)]


def plan(pos: np.ndarray, neg: np.ndarray, s: int, f: int, updates: int) -> np.ndarray:
    """(updates, 4) int64 fold-local P-run indices [p, n1, n2, n3] per update (the same for both arms)."""
    if len(pos) == 0 or len(neg) < N_NEG:
        raise ValueError("the fit fold needs at least one positive and three protected negatives")
    perm = negative_permutation(neg, s, f)
    return np.asarray([[draw_positive(pos, s, f, u), *negatives_of(perm, u).tolist()] for u in range(int(updates))], np.int64)


def plan_sha(pl: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(pl, np.int64).tobytes()).hexdigest()


def members(pk: dict, i: int, off: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Fold-local unique rows and multiplicity weights of P run i (A31's stored membership)."""
    o = pk["offsets"]; a, b = int(o[i]), int(o[i + 1]); return np.asarray(pk["rows"][a:b]) - int(off), np.asarray(pk["weights"][a:b], np.float64)


def explicit_membership(nearest: np.ndarray, a: int, b: int) -> tuple[np.ndarray, np.ndarray]:
    """Independent loop: the rows of samples [a, b) under the nearest-centre map, counted one sample at a time."""
    counts = {}
    for t in range(int(a), int(b)):
        r = int(nearest[t]); counts[r] = counts.get(r, 0) + 1
    rows = np.asarray(sorted(counts), np.int64); n = float(b - a)
    return rows, np.asarray([counts[int(r)] / n for r in rows], np.float64)


def occurrences(pk: dict, runs, off: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Every window occurrence of the given runs in order: fold-local rows, weights and the run slot of each occurrence."""
    rows, ws, seg = [], [], []
    for k, i in enumerate(runs):
        r, w = members(pk, int(i), off); rows.append(r); ws.append(w); seg.append(np.full(len(r), k, np.int64))
    return np.concatenate(rows), np.concatenate(ws), np.concatenate(seg)


def exposure(pk: dict, pl: np.ndarray) -> dict:
    """Occurrence totals of a plan: per-update window occurrences (unique rows per run, summed over the four runs)."""
    o = pk["offsets"]; n = np.asarray([[int(o[i + 1] - o[i]) for i in row] for row in pl]); per = n.sum(1)
    return {"occurrences_total": int(per.sum()), "occurrences_max_update": int(per.max()), "occurrences_mean_update": float(per.mean()), "occurrences_min_update": int(per.min()),
            "positive_occurrences": int(n[:, 0].sum()), "negative_occurrences": int(n[:, 1:].sum()), "distinct_positives": int(len(np.unique(pl[:, 0]))), "distinct_negatives": int(len(np.unique(pl[:, 1:]))),
            "same_update_duplicate_negatives": int(sum(len(set(row[1:].tolist())) < N_NEG for row in pl))}


# ----------------------------------------------------------------------------------------------- head and scores
class ScalarHead(nn.Module):
    """Linear(512, 1): 513 parameters, weight exactly 0 and bias -2 at Init (every initial score is -2)."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(EMBED, 1)
        with torch.no_grad():
            self.fc.weight.zero_(); self.fc.bias.fill_(INIT_BIAS)

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        if e.ndim != 2 or e.shape[1] != EMBED:
            raise ValueError(f"embedding {tuple(e.shape)} is not (B, {EMBED})")
        return self.fc(e).squeeze(-1)


def score(m, head: ScalarHead, rgb: torch.Tensor, marker: torch.Tensor) -> torch.Tensor:
    """(B,) scores of one full original marker forward through the unchanged embed (graph kept when enabled)."""
    return head(m.embed(rgb, marker))


def run_means(s: torch.Tensor, v: torch.Tensor, seg: torch.Tensor, n_runs: int) -> torch.Tensor:
    """Weighted run means r_k = sum_{j in k} v_j s_j for every occurrence slot k (differentiable in s)."""
    if s.ndim != 1 or s.shape != v.shape or seg.shape != s.shape:
        raise ValueError("scores, weights and run slots must be equal-length vectors")
    return torch.zeros(n_runs, dtype=s.dtype, device=s.device).index_add(0, seg.to(s.device), v.to(s.dtype) * s)


# ----------------------------------------------------------------------------------------------- objectives
def objective(r: torch.Tensor, arm: str, lam: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """(loss, A, B, H) of one positive run mean r[0] and the protected negatives r[1:]. C = A + B; N = A + lam (H - alpha)."""
    if r.ndim != 1 or len(r) != 1 + N_NEG:
        raise ValueError(f"expected 1 positive and {N_NEG} negative run means")
    A = F.softplus(-r[0]); B = F.softplus(r[1:]).mean(); H = F.relu(1.0 + r[1:]).mean()
    if arm == "C":
        return A + B, A, B, H
    if arm == "N":
        return A + float(lam) * (H - ALPHA), A, B, H
    raise ValueError(f"unknown arm {arm!r}")


def dual_update(lam: float, H_pre: float) -> float:
    """lambda_{u+1} = clip(lambda_u + 0.1 (H_pre - alpha), 0, 100) with the detached pre-step minibatch H."""
    return float(min(LAMBDA_MAX, max(LAMBDA_MIN, float(lam) + DUAL_STEP * (float(H_pre) - ALPHA))))


def reference(s, v, seg, arm: str, lam: float) -> dict:
    """Independent float64 loop: run means, A / B / H, the loss and dL/ds of every occurrence (ReLU derivative 0 at 1 + r = 0)."""
    s, v, seg = (np.asarray(x) for x in (s, v, seg)); K = 1 + N_NEG; r = np.zeros(K)
    for j in range(len(s)):
        r[int(seg[j])] += float(v[j]) * float(s[j])
    sig = lambda x: 0.5 * (1.0 + np.tanh(0.5 * x))  # noqa: E731
    A = np.logaddexp(0.0, -r[0]); B = float(np.mean([np.logaddexp(0.0, r[k]) for k in range(1, K)])); H = float(np.mean([max(0.0, 1.0 + r[k]) for k in range(1, K)]))
    dr = np.zeros(K); dr[0] = -sig(-r[0])
    for k in range(1, K):
        dr[k] = sig(r[k]) / N_NEG if arm == "C" else float(lam) * (1.0 if 1.0 + r[k] > 0 else 0.0) / N_NEG
    g = np.asarray([dr[int(seg[j])] * float(v[j]) for j in range(len(s))])
    L = A + B if arm == "C" else A + float(lam) * (H - ALPHA)
    return {"r": r, "A": float(A), "B": B, "H": H, "loss": float(L), "g": g, "lambda_next": dual_update(lam, H) if arm == "N" else float(lam)}


# ----------------------------------------------------------------------------------------------- deployment
def run_weighted_mean(pk: dict, scores: np.ndarray, off: int, eligible: np.ndarray) -> np.ndarray:
    """r per P run: the multiplicity-weighted mean score over its unique rows (fold-local = global - off); -inf if not eligible."""
    r = np.full(len(pk["labels"]), -np.inf)
    for i in np.flatnonzero(eligible):
        rows, w = members(pk, int(i), off); r[i] = float(np.dot(w, np.asarray(scores[rows], np.float64)))
    return r


def run_weighted_mean_explicit(pk: dict, scores: np.ndarray, off: int, i: int, nearest: np.ndarray) -> float:
    """Independent check: the plain mean of the samples' window scores over the run's original samples [a, b)."""
    a, b = int(pk["a"][i]), int(pk["b"][i]); return float(np.mean([float(scores[int(nearest[t]) - int(off)]) for t in range(a, b)]))


def feasibility(r: np.ndarray, negatives: np.ndarray, positives: np.ndarray) -> dict:
    """Full-fit constraint report from per-run r: mean hinge H over the protected negatives, actual false promotions and
    positive recovery (r > 0)."""
    rn, rp = np.asarray(r[negatives], np.float64), np.asarray(r[positives], np.float64)
    return {"negatives": int(len(rn)), "H_full_fit": float(np.mean(np.maximum(0.0, 1.0 + rn))) if len(rn) else 0.0, "false_promotions": int((rn > 0).sum()),
            "false_promotion_fraction": float((rn > 0).mean()) if len(rn) else 0.0, "feasible_le_alpha": bool((np.mean(np.maximum(0.0, 1.0 + rn)) if len(rn) else 0.0) <= ALPHA),
            "positives": int(len(rp)), "positives_recovered": int((rp > 0).sum()), "positive_recovery_fraction": float((rp > 0).mean()) if len(rp) else 0.0,
            "r_negative": {"median": float(np.median(rn)), "q90": float(np.quantile(rn, 0.9)), "max": float(rn.max())} if len(rn) else {},
            "r_positive": {"median": float(np.median(rp)), "q10": float(np.quantile(rp, 0.1)), "min": float(rp.min()), "max": float(rp.max())} if len(rp) else {}}
