"""Losses for geometry pretraining (configs/experiments/geometry_pretraining_v1.yaml, model).

A batch holds B pairs as 2B rows: rows 2i and 2i+1 are the two slices of pair i.
- InfoNCE (G1 and G2). The anchors are the slices of same_plane pairs (the same plane at two times), and an anchor's
  positive is its partner. The denominator holds every other row of the batch, the partner once, never the anchor
  itself. Slices of other pairs only serve as negatives.
- Tilt regression (G2 only). A head on [z1, z2, z1 - z2, z1 * z2] predicts d_alpha / 15 degrees for every pair; the
  loss is smooth L1. For same_plane pairs the target is 0: only the heart moved.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

D_ALPHA_SCALE = 15.0


def partners(same_plane: torch.Tensor) -> torch.Tensor:
    """For B pairs (bool, B), the partner row of each of the 2B rows, or -1 for rows that are not anchors."""
    b = same_plane.numel(); idx = torch.arange(2 * b, device=same_plane.device)
    partner = idx ^ 1                                          # 2i <-> 2i + 1
    return torch.where(same_plane.repeat_interleave(2), partner, torch.full_like(partner, -1))


def info_nce(z: torch.Tensor, partner: torch.Tensor, tau: float) -> torch.Tensor:
    """Mean over anchors of -log p(partner) with a softmax over every other row; 0 if the batch has no anchor."""
    anchors = partner >= 0
    if not bool(anchors.any()):
        return z.sum() * 0.0
    z = F.normalize(z, dim=1); sim = z @ z.T / tau
    sim = sim.masked_fill(torch.eye(len(z), dtype=torch.bool, device=z.device), float("-inf"))
    logp = sim[anchors].log_softmax(dim=1)
    return -logp.gather(1, partner[anchors].unsqueeze(1)).mean()


class Projector(nn.Module):
    def __init__(self, dim: int = 512, hidden: int = 512, out: int = 128) -> None:
        super().__init__(); self.net = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, out))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class TiltHead(nn.Module):
    """d_alpha / 15 from the two slices' embeddings."""

    def __init__(self, dim: int = 512, hidden: int = 256) -> None:
        super().__init__(); self.net = nn.Sequential(nn.Linear(4 * dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z1, z2, z1 - z2, z1 * z2], dim=1)).squeeze(1)


def tilt_loss(head: TiltHead, z: torch.Tensor, d_alpha_deg: torch.Tensor) -> torch.Tensor:
    return F.smooth_l1_loss(head(z[0::2], z[1::2]), d_alpha_deg / D_ALPHA_SCALE)
