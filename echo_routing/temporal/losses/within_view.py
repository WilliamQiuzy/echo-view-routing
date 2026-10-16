"""Within-view output regularisation of the A08 round (configs/experiments/video_within_view_v1.yaml, loss;
docs/superpowers/plans/2026-10-16-video-within-view-regularization.md, section 3).

For the logits z of G groups of T neighbouring request centres and C classes (A08: G = 2, T = 4, C = 5):

    ell = log_softmax(z, dim=-1).to(float64)
    d[n, t, c] = clamp((ell[n, t, c] - stop_gradient(ell[n, t-1, c]))**2, min=0, max=16)      t = 1 .. T-1
    S_A = mean_n sum_(t, c) d[n, t, c] / ((T-1) C)                                               all neighbours (A)
    m[n, t] = 1{label[n, t] == label[n, t-1]}
    S_M = mean_n [sum_(t, c) m[n, t] d[n, t, c] / (C max(1, sum_t m[n, t]))]                    equal-label neighbours (M)

A group without a selected edge contributes exactly zero (its numerator is zero and its denominator is clamped to 1
before the division, so no zero denominator is ever evaluated). The objective is L_CE + 0.15 S with L_CE A07 F's plain
cross-entropy through the pinned A05 module (segment_objective.objective('frame', ...)). The penalty is MS-TCN's
truncated log-probability smoothing (coefficient 0.15, truncation 16, detached previous output); the label mask and
the per-group normalisation are this task's adaptation. Labels select edges during training only; inference uses the
model's logits alone.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from echo_routing.temporal.losses import segment_objective as so

ARMS = ("all", "masked")
COEFFICIENT = 0.15
TRUNCATION = 16.0

__all__ = ["ARMS", "COEFFICIENT", "TRUNCATION", "log_probabilities", "edge_penalties", "equal_label_mask", "all_neighbours", "equal_label",
           "regulariser", "objective", "monitor"]


def log_probabilities(z: torch.Tensor) -> torch.Tensor:
    """The logits' own log-softmax, then float64."""
    return F.log_softmax(z, dim=-1).to(torch.float64)


def edge_penalties(ell: torch.Tensor) -> torch.Tensor:
    """d of shape (G, T-1, C): truncated squared differences with the earlier endpoint of every edge detached."""
    if ell.ndim != 3 or ell.shape[1] < 2:
        raise ValueError("log-probabilities need the shape (groups, positions >= 2, classes)")
    return torch.clamp((ell[:, 1:] - ell[:, :-1].detach()) ** 2, min=0.0, max=TRUNCATION)


def equal_label_mask(labels) -> torch.Tensor:
    """m of shape (G, T-1): 1 where an edge joins two equal training labels (float64)."""
    lab = torch.as_tensor(np.asarray(labels), dtype=torch.int64)
    if lab.ndim != 2 or lab.shape[1] < 2:
        raise ValueError("labels need the shape (groups, positions >= 2)")
    return (lab[:, 1:] == lab[:, :-1]).to(torch.float64)


def all_neighbours(d: torch.Tensor) -> torch.Tensor:
    """S_A: every group's mean over its edges and classes, averaged over the groups."""
    return (d.sum(dim=(1, 2)) / float(d.shape[1] * d.shape[2])).mean()


def equal_label(d: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """S_M: every group's mean over its selected edges and classes (exactly zero without a selected edge), averaged."""
    mask = mask.to(device=d.device, dtype=d.dtype)
    if mask.shape != d.shape[:2]:
        raise ValueError("the mask needs one entry per edge of every group")
    count = mask.sum(dim=1)
    per = (d * mask[..., None]).sum(dim=(1, 2)) / (float(d.shape[2]) * count.clamp_min(1.0))
    return torch.where(count > 0, per, torch.zeros_like(per)).mean()


def regulariser(arm: str, d: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if arm == "all":
        return all_neighbours(d)
    if arm == "masked":
        return equal_label(d, mask)
    raise ValueError(f"unknown arm {arm!r}")


def objective(arm: str, z: torch.Tensor, labels, coefficient: float = COEFFICIENT) -> tuple[torch.Tensor, dict]:
    """L = L_CE + coefficient * S_arm (float64) and its detached parts; L_CE is A07 F's path through the A05 module."""
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    lab = np.asarray(labels).astype(np.int64)
    if lab.shape != tuple(z.shape[:2]):
        raise ValueError("one label per logit row is needed")
    ce, _ = so.objective("frame", z, [so.reference_runs(x) for x in lab], np.ones(z.shape[-1]))
    reg = regulariser(arm, edge_penalties(log_probabilities(z)), equal_label_mask(lab))
    loss = ce + float(coefficient) * reg
    return loss, {"ce": float(ce.detach()), "regulariser": float(reg.detach()), "coefficient": float(coefficient)}


@torch.no_grad()
def monitor(z: torch.Tensor, labels) -> dict:
    """Both regularisers and, per edge (its class mean of d), the summed penalties and counts of the equal-label and the
    different-label edges, from the detached logits (no gradient, no random draw)."""
    d = edge_penalties(log_probabilities(z.detach())); mask = equal_label_mask(labels).to(d.device); edge = d.mean(dim=-1)
    return {"S_A": float(all_neighbours(d)), "S_M": float(equal_label(d, mask)),
            "same_sum": float((edge * mask).sum()), "same_edges": int(mask.sum()),
            "diff_sum": float((edge * (1.0 - mask)).sum()), "diff_edges": int((1.0 - mask).sum()),
            "zero_edge_groups": int((mask.sum(dim=1) == 0).sum())}
