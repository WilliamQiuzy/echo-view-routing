"""Supervised contrastive term of the A09 round (configs/experiments/video_discriminative_v1.yaml, loss.contrast).

SupCon L_out (Khosla et al. 2020, arXiv:2004.11362) on L2-normalised embeddings, computed in float64. For anchor i the
positives are the other windows of the same class from a different session. The denominator holds every other window
except same-class windows of the same session, so a same-session repeat is neither a positive nor a negative. Anchors
without a positive are skipped. The known method is used unchanged; nothing here is a new loss.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

TAU = 0.1
FILL = -1.0e9                     # a finite fill keeps masked rows free of NaN in the backward pass


def masks(labels: torch.Tensor, sessions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(positives, candidates) as (n, n) booleans: positives = same label, different session; candidates = every other
    window except same-label same-session ones."""
    y, s = labels.view(-1), sessions.view(-1); n = y.numel()
    eye = torch.eye(n, dtype=torch.bool, device=y.device)
    same_y, same_s = y[:, None] == y[None, :], s[:, None] == s[None, :]
    return same_y & ~same_s & ~eye, ~eye & ~(same_y & same_s)


def supcon(emb: torch.Tensor, labels: torch.Tensor, sessions: torch.Tensor, tau: float = TAU) -> tuple[torch.Tensor, int]:
    """(S, anchors): the mean SupCon L_out over anchors with at least one positive, and their number. S is exactly zero
    (with a zero gradient) when no anchor has a positive."""
    u = F.normalize(emb.to(torch.float64), dim=1)
    pos, cand = masks(labels.to(u.device), sessions.to(u.device))
    logits = (u @ u.T / float(tau)).masked_fill(~cand, FILL)
    logp = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    has = pos.any(1); k = int(has.sum())
    if k == 0:
        return (u * 0.0).sum(), 0
    per = -(logp * pos).sum(1)[has] / pos.sum(1)[has].to(torch.float64)
    return per.mean(), k


@torch.no_grad()
def geometry(emb: torch.Tensor, labels: torch.Tensor, sessions: torch.Tensor, pair: tuple[int, int] = (2, 3)) -> dict:
    """Monitor from detached embeddings: mean cosine of cross-session same-class pairs, of different-class pairs and of
    the A4C-A5C pairs (class indices 2 and 3)."""
    u = F.normalize(emb.detach().to(torch.float64), dim=1); c = u @ u.T
    pos, _ = masks(labels, sessions); y = labels.view(-1); n = y.numel()
    off = ~torch.eye(n, dtype=torch.bool, device=y.device); diff = (y[:, None] != y[None, :]) & off
    ab = ((y[:, None] == pair[0]) & (y[None, :] == pair[1])) | ((y[:, None] == pair[1]) & (y[None, :] == pair[0]))
    mean = lambda m: float(c[m].mean()) if bool(m.any()) else float("nan")  # noqa: E731
    return {"cos_pos": mean(pos), "cos_neg": mean(diff), "cos_a4c_a5c": mean(ab)}
