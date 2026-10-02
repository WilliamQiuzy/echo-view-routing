"""Losses of the boundary-controlled context model (plan Task 3, "损失及监督来源").

L = L_stream + 0.25 L_native_frame + 0.2 L_local + 0.2 L_boundary + 0.2 L_native_clip

- L_stream: per-sample cross-entropy on constructed streams. Targets are inherited from the source clips, so they
  are proxy labels.
- L_native_frame: per-sample cross-entropy on native clip windows. Each sample carries its clip's label, so it is
  weak supervision, not an expert frame label.
- L_local: cross-entropy of the local branch on every sequence, with the same valid mask.
- L_boundary: per sequence, sum(BCE on positives) / max(n_pos, 1) + sum(w * BCE on negatives) / max(n_neg, 1). A join
  whose source label changes is positive; a same-label join is a strong negative (w = 1); every other transition
  inside a fragment or a native clip is a weak negative (w = 0.1). The denominators count positions, not weights.
  Position 0 and padding are excluded.
- L_native_clip: negative log of the clip-mean class probability at the clip label (native windows).
Every term is normalised per sequence over its valid positions and then averaged over the sequences it applies to.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

WEIGHTS = {"stream": 1.0, "native_frame": 0.25, "local": 0.2, "boundary": 0.2, "native_clip": 0.2}
STRONG_NEGATIVE = 1.0
WEAK_NEGATIVE = 0.1


def boundary_targets(labels: np.ndarray, joins: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(target, weight, counted) per transition index t (between samples t-1 and t) of one sequence.

    `joins` are the recipe's physical joins (index of the first sample after a join); they define the training targets
    only and are never model inputs. Native windows have no joins, so all of their transitions are weak negatives."""
    n = int(np.asarray(labels).size)
    target = np.zeros(n, np.float32); weight = np.full(n, WEAK_NEGATIVE, np.float32)
    counted = np.ones(n, bool)
    if n:
        counted[0] = False; weight[0] = 0.0
    for j in np.asarray(joins, int):
        if not 1 <= j < n:
            raise ValueError(f"join {j} outside 1..{n - 1}")
        if labels[j] != labels[j - 1]:
            target[j] = 1.0; weight[j] = 1.0
        else:
            weight[j] = STRONG_NEGATIVE
    return target, weight, counted


def boundary_loss(logits: torch.Tensor, target: torch.Tensor, weight: torch.Tensor, counted: torch.Tensor) -> torch.Tensor:
    """logits/target/weight (B, T) and counted (B, T) bool -> mean over sequences of the normalised boundary loss."""
    bce = F.binary_cross_entropy_with_logits(logits.float(), target, reduction="none")
    pos = counted & (target > 0.5); neg = counted & (target <= 0.5)
    lp = (bce * pos).sum(1) / pos.sum(1).clamp(min=1)
    ln = (bce * weight * neg).sum(1) / neg.sum(1).clamp(min=1)
    return (lp + ln).mean()


def sequence_ce(logits: torch.Tensor, labels: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """logits (B, C, T), labels (B, T), valid (B, T) -> per-sequence mean cross-entropy (B,)."""
    ce = F.cross_entropy(logits.float(), labels.clamp(min=0), reduction="none")
    return (ce * valid).sum(1) / valid.sum(1).clamp(min=1)


def clip_nll(logits: torch.Tensor, labels: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """-log of the valid-mean class probability at each sequence's clip label (label of its first sample) -> (B,)."""
    prob = torch.softmax(logits.float(), dim=1)
    mean = (prob * valid[:, None]).sum(2) / valid.sum(1, keepdim=True).clamp(min=1)
    y = labels[:, 0].clamp(min=0)
    return -torch.log(mean.gather(1, y[:, None]).squeeze(1).clamp(min=1e-12))


def _mean_over(values: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
    return (values * rows).sum() / rows.sum() if bool(rows.any()) else values.sum() * 0.0


def total_loss(out: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], boundary_weight: float) -> tuple[torch.Tensor, dict[str, float]]:
    """Combined loss for one batch. `batch` carries labels (B, T) with -100 at padding, valid (B, T) float,
    is_native (B,) bool, and the boundary target, weight and counted arrays (B, T)."""
    valid, labels = batch["valid"], batch["labels"]
    native = batch["is_native"].float(); stream = 1.0 - native
    ce = sequence_ce(out["class_logits"], labels, valid)
    parts = {"stream": _mean_over(ce, stream), "native_frame": _mean_over(ce, native),
             "local": sequence_ce(out["local_logits"], labels, valid).mean(),
             "native_clip": _mean_over(clip_nll(out["class_logits"], labels, valid), native)}
    if boundary_weight > 0:
        parts["boundary"] = boundary_loss(out["boundary_logits"][:, 0], batch["b_target"], batch["b_weight"], batch["b_counted"])
    weights = {**WEIGHTS, "boundary": boundary_weight}
    total = sum(weights[k] * v for k, v in parts.items())
    return total, {k: float(v.detach()) for k, v in parts.items()}
