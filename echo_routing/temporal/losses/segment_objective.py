"""Training objectives of the segment objective round (configs/experiments/segment_objective_v1.yaml).

Three objectives for the same Linear(512, 5) head, from reference runs only:

- F (frame): sum_n sum_t alpha[r_t] e_t / sum_n sum_t alpha[r_t], A01's masked class-weighted cross-entropy.
- B (segment): every reference run contributes alpha[class] times its mean cross-entropy; the sum is divided by the
  summed run weights D_n = sum_r alpha[class(r)].
- O (order): (B + 0.5 L_order) / 1.5 with L_order = sum_n D_n q_n / sum_n D_n and q_n = -log P(y_n | x_n) / T_n, where
  P sums over every positive-duration path through the run-length-collapsed transcript y_n that starts in its first
  state, ends in its last and stays or advances one state per sample (no blank, no zero-length state, no maximum).

Runs merge adjacent equal classes, also across a same-view join; X-Y-X keeps three runs. Joins, ids and the native
flag never enter a loss. The objective is accumulated in float64 from the logits' own log-softmax values; padding
positions never enter a value or a gradient. Labels are a training input only: inference uses the logits alone.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

NUM_CLASSES = 5
ARMS = ("frame", "segment", "order")
ORDER_WEIGHT = 0.5
PAD = -100

__all__ = ["ARMS", "ORDER_WEIGHT", "Runs", "reference_runs", "transcript_nll", "edit_distance", "check_class_weights", "objective", "all_terms"]


# ----------------------------------------------------------------------------------------------- reference runs
@dataclass(frozen=True)
class Runs:
    """Maximal runs of equal reference labels of one sequence: [starts, ends) and their classes."""
    starts: np.ndarray
    ends: np.ndarray
    classes: np.ndarray

    @property
    def length(self) -> int:
        return int(self.ends[-1])

    @property
    def transcript(self) -> tuple[int, ...]:
        return tuple(int(c) for c in self.classes)


def reference_runs(labels) -> Runs:
    lab = np.asarray(labels)
    if lab.ndim != 1 or lab.size == 0:
        raise ValueError("a sequence needs a one-dimensional, non-empty label array")
    if not np.issubdtype(lab.dtype, np.integer) or (lab < 0).any() or (lab >= NUM_CLASSES).any():
        raise ValueError("a reference label lies outside the five classes")
    lab = lab.astype(np.int64); cut = np.flatnonzero(lab[1:] != lab[:-1]) + 1
    starts = np.concatenate([[0], cut]).astype(np.int64); ends = np.concatenate([cut, [lab.size]]).astype(np.int64)
    return Runs(starts, ends, lab[starts])


def edit_distance(a, b) -> int:
    """Levenshtein distance between two transcripts (unit insertion, deletion and substitution costs)."""
    a, b = list(a), list(b); prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def check_class_weights(alpha) -> np.ndarray:
    a = np.asarray(alpha, dtype=np.float64)
    if a.shape != (NUM_CLASSES,) or not np.isfinite(a).all() or (a <= 0).any():
        raise ValueError("the class weights must be five finite positive values")
    return a


# ----------------------------------------------------------------------------------------------- the forward recursion
def transcript_nll(logp: torch.Tensor, y: Sequence[int]) -> torch.Tensor:
    """q = -log P(y | x) / T for (T, 5) log-probabilities. prev[t] is the log-sum over the paths of samples 0..t that are
    in state k at t; a state entered at s >= k follows state k - 1 at s - 1. Every logcumsumexp input starts at a
    reachable state, so no all-negative-infinity prefix is ever differentiated."""
    if logp.ndim != 2 or logp.shape[1] != NUM_CLASSES:
        raise ValueError("the log-probabilities must have shape (T, 5)")
    y = [int(c) for c in y]; T, K = int(logp.shape[0]), len(y)
    if not 1 <= K <= T:
        raise ValueError("the transcript is empty or longer than its sequence")
    if any(not 0 <= c < NUM_CLASSES for c in y):
        raise ValueError("a transcript state lies outside the five classes")
    if any(y[k] == y[k - 1] for k in range(1, K)):
        raise ValueError("adjacent transcript states repeat a class")
    prev = logp[:, y[0]].cumsum(0)
    for k in range(1, K):
        pref = logp[:, y[k]].cumsum(0)
        entries = prev[k - 1:T - 1] - pref[k - 1:T - 1]
        valid_suffix = pref[k:] + torch.logcumsumexp(entries, dim=0)
        prev = torch.cat((logp.new_full((k,), -torch.inf), valid_suffix))
    return -prev[-1] / T


# ----------------------------------------------------------------------------------------------- batch terms
def _check(logits: torch.Tensor, runs: Sequence[Runs]) -> None:
    if logits.ndim != 3 or logits.shape[2] != NUM_CLASSES:
        raise ValueError("the logits must have shape (B, T, 5)")
    if len(runs) != logits.shape[0]:
        raise ValueError("one run record per sequence of the batch")
    if any(r.length > logits.shape[1] for r in runs):
        raise ValueError("a sequence is longer than the batch")


def _targets(runs: Sequence[Runs], T: int) -> np.ndarray:
    out = np.full((len(runs), T), PAD, np.int64)
    for i, r in enumerate(runs):
        out[i, :r.length] = np.repeat(r.classes, r.ends - r.starts)
    return out


def _log_probs(logits: torch.Tensor) -> torch.Tensor:
    """The logits' own log-softmax values (fp32 for the formal head), accumulated in float64 from here on."""
    return F.log_softmax(logits, dim=-1).to(torch.float64)


def _sample_errors(logp: torch.Tensor, runs: Sequence[Runs]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """e[n, t] = -log p[n, t, reference] at valid positions and exactly 0 at padding; the targets and the valid mask."""
    tgt = torch.as_tensor(_targets(runs, logp.shape[1]), device=logp.device); valid = tgt >= 0
    e = -logp.gather(-1, tgt.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    return torch.where(valid, e, torch.zeros((), dtype=e.dtype, device=e.device)), tgt, valid


def _run_index(runs: Sequence[Runs], device) -> tuple[torch.Tensor, ...]:
    n = np.concatenate([np.full(r.classes.size, i, np.int64) for i, r in enumerate(runs)])
    s = np.concatenate([r.starts for r in runs]); t = np.concatenate([r.ends for r in runs]); c = np.concatenate([r.classes for r in runs])
    return tuple(torch.as_tensor(v, device=device) for v in (n, s, t, c))


def frame_term(e: torch.Tensor, tgt: torch.Tensor, valid: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    w = torch.where(valid, alpha[tgt.clamp(min=0)], torch.zeros((), dtype=alpha.dtype, device=alpha.device))
    return (w * e).sum() / w.sum()


def segment_term(e: torch.Tensor, runs: Sequence[Runs], alpha: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(L_B, every run's mean cross-entropy, every run's weight alpha[class])."""
    n, s, t, c = _run_index(runs, e.device)
    cs = torch.cat((e.new_zeros(e.shape[0], 1), e.cumsum(1)), dim=1)
    means = (cs[n, t] - cs[n, s]) / (t - s).to(e.dtype); a = alpha[c]
    return (a * means).sum() / a.sum(), means, a


def order_term(logp: torch.Tensor, runs: Sequence[Runs], alpha: np.ndarray) -> torch.Tensor:
    q = torch.stack([transcript_nll(logp[i, :r.length], r.transcript) for i, r in enumerate(runs)])
    D = torch.as_tensor(np.array([alpha[r.classes].sum() for r in runs]), dtype=torch.float64, device=logp.device)
    return (D * q).sum() / D.sum()


def objective(arm: str, logits: torch.Tensor, runs: Sequence[Runs], alpha, order_weight: float = ORDER_WEIGHT) -> tuple[torch.Tensor, dict]:
    """The arm's training loss (float64; gradients reach the logits in their own dtype) and its detached parts."""
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    _check(logits, runs); a_np = check_class_weights(alpha); a = torch.as_tensor(a_np, device=logits.device)
    logp = _log_probs(logits); e, tgt, valid = _sample_errors(logp, runs)
    if arm == "frame":
        loss = frame_term(e, tgt, valid, a)
        return loss, {"frame": float(loss.detach())}
    seg, _, _ = segment_term(e, runs, a)
    if arm == "segment":
        return seg, {"segment": float(seg.detach())}
    order = order_term(logp, runs, a_np); loss = (seg + float(order_weight) * order) / (1.0 + float(order_weight))
    return loss, {"segment": float(seg.detach()), "order": float(order.detach())}


@torch.no_grad()
def all_terms(logits: torch.Tensor, runs: Sequence[Runs], alpha, order_weight: float = ORDER_WEIGHT) -> dict[str, float]:
    """Every arm's terms on the same logits (training monitors): L_F, L_B, L_order, L_O and the unweighted mean of the
    runs' mean cross-entropy."""
    _check(logits, runs); a_np = check_class_weights(alpha); a = torch.as_tensor(a_np, device=logits.device)
    logp = _log_probs(logits); e, tgt, valid = _sample_errors(logp, runs)
    seg, means, _ = segment_term(e, runs, a); order = order_term(logp, runs, a_np)
    return {"frame": float(frame_term(e, tgt, valid, a)), "segment": float(seg), "order": float(order),
            "objective_order": float((seg + float(order_weight) * order) / (1.0 + float(order_weight))), "segment_mean_ce": float(means.mean())}
