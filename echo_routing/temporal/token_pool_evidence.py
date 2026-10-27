"""A39 (configs/experiments/video_patch_evidence_v1.yaml): pooled descriptors of the frozen A08's projected final tokens.

One original TargetMViT.embed forward gives e224 (CLS -> encoder.head) and, through a read-only forward hook on
encoder.norm that returns nothing, the normalised final tokens U (B, 1 + 8 x 7 x 7, 768) of the same forward. The 392
non-CLS tokens are projected one by one with the frozen eval encoder.head, V = head(U[:, 1:]) (B, 392, 512), and then
pooled over the joint time-space token axis:
  G  eG = V.mean(1)   channel mean
  P  eP = V.amax(1)   channel maximum; different channels can peak at different tokens, so it is not a location
Projection first, pooling second; no other normalisation. Both arms' residual head is A36's Residual with A39's seed.
Nothing here reads a label.
"""
from __future__ import annotations

import torch
import torch.nn as nn

import echo_routing.temporal.spatial_detail as sd
import echo_routing.vit.target_mvit as tm

GRID, SEED, ARMS = (8, 7, 7), 20261026 + 39, ("G", "P")
N_TOKENS = 1 + GRID[0] * GRID[1] * GRID[2]
CONTRACT = "encoder.norm output U [B, 393, 768]; CLS = U[:, 0] excluded; V = encoder.head(U[:, 1:]) token by token (eval); eG = V.mean(1), eP = V.amax(1)"


def embed_with_tokens(m: tm.TargetMViT, rgb: torch.Tensor, marker: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(e224 (B, 512) exactly as TargetMViT.embed, U (B, 393, 768) = a clone of encoder.norm's output in that forward)."""
    norm = m.encoder.norm
    if norm._forward_hooks:
        raise RuntimeError("encoder.norm already carries a forward hook")
    seen: list[torch.Tensor] = []

    def capture(_module, _inputs, out):
        seen.append(out.detach().clone())

    handle = norm.register_forward_hook(capture)
    try:
        e = m.embed(rgb, marker)
    finally:
        handle.remove()
    if len(seen) != 1:
        raise RuntimeError(f"the norm output was captured {len(seen)} times in one forward")
    u = seen[0]
    if u.ndim != 3 or tuple(u.shape[:2]) != (e.shape[0], N_TOKENS):
        raise ValueError(f"norm output {tuple(u.shape)} is not (B, {N_TOKENS}, C)")
    return e, u


def project_tokens(head: nn.Module, u: torch.Tensor) -> torch.Tensor:
    """V = head(U[:, 1:]): the frozen projection applied to every non-CLS token; the CLS token is excluded."""
    if head.training:
        raise RuntimeError("the projection must be in eval mode")
    return head(u[:, 1:])


def pool(v: torch.Tensor) -> dict[str, torch.Tensor]:
    """Channel mean (G) and channel maximum (P) over the token axis of projected tokens (B, n, 512)."""
    return {"G": v.mean(1), "P": v.amax(1)}


def descriptors(m: tm.TargetMViT, rgb: torch.Tensor, marker: torch.Tensor) -> dict[str, torch.Tensor]:
    """q, e224, eG and eP of one forward (no_grad)."""
    with torch.no_grad():
        e, u = embed_with_tokens(m, rgb, marker); d = pool(project_tokens(m.encoder.head, u))
        return {"e224": e.float(), "q": m.classifier(e).float(), "G": d["G"].float(), "P": d["P"].float()}


def head(fold: int) -> sd.Residual:
    """The G / P residual head of a fit fold (bit-identical for both arms)."""
    return sd.Residual(SEED + int(fold))
