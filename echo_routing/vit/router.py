"""ViT-Router: stream-level temporal Transformer on frozen ViT frame embeddings.

Design (each part borrowed from a baseline that is strong at it):
- ASFormer: self-attention restricted to a local window that doubles with depth (5, 9, 17, ... samples), and a
  dilated depthwise convolution in the feed-forward path that supplies relative position (no absolute encoding,
  so any stream length works).
- MS-TCN: multi-stage refinement. Later stages see only the previous stage's class probabilities, and every stage
  is trained with cross-entropy plus the truncated MSE smoothing loss on log-probabilities (tau = 4, weight 0.15).
- Input channel dropout (ASFormer's channel masking, p = 0.3) against over-reliance on a few feature channels.
Streams are processed one at a time (batch 1, as in the official MS-TCN/ASFormer code): no padding masks.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def band_mask(t: int, window: int, device=None) -> torch.Tensor:
    """(t, t) bool, True = attention NOT allowed (|i - j| > window // 2)."""
    if window < 1:
        raise ValueError("window must be >= 1")
    i = torch.arange(t, device=device)
    return (i[:, None] - i[None, :]).abs() > window // 2


class LocalBlock(nn.Module):
    def __init__(self, dim: int, heads: int, window: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.window = window
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.dw = nn.Conv1d(dim, dim, 3, padding=dilation, dilation=dilation, groups=dim)
        self.pw = nn.Sequential(nn.Conv1d(dim, 2 * dim, 1), nn.GELU(), nn.Dropout(dropout), nn.Conv1d(2 * dim, dim, 1))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, attn_mask=band_mask(x.shape[1], self.window, x.device), need_weights=False)
        x = x + self.drop(a)
        h = self.pw(F.gelu(self.dw(self.norm2(x).transpose(1, 2))))
        return x + self.drop(h.transpose(1, 2))


class Stage(nn.Module):
    def __init__(self, in_dim: int, dim: int, heads: int, n_layers: int, n_classes: int, dropout: float) -> None:
        super().__init__()
        self.inp = nn.Linear(in_dim, dim)
        self.blocks = nn.ModuleList(LocalBlock(dim, heads, 2 ** (i + 2) + 1, 2 ** i, dropout) for i in range(n_layers))
        self.norm = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.inp(x)
        for b in self.blocks:
            h = b(h)
        return self.out(self.norm(h))


class ViTRouter(nn.Module):
    def __init__(self, in_dim: int, n_classes: int, dim: int = 128, heads: int = 4, n_layers: int = 6, n_refine: int = 2,
                 refine_dim: int = 64, refine_layers: int = 4, dropout: float = 0.1, channel_dropout: float = 0.3) -> None:
        super().__init__()
        self.channel_drop = nn.Dropout1d(channel_dropout)
        self.stage1 = Stage(in_dim, dim, heads, n_layers, n_classes, dropout)
        self.refiners = nn.ModuleList(Stage(n_classes, refine_dim, heads, refine_layers, n_classes, dropout) for _ in range(n_refine))

    def forward(self, feat: torch.Tensor) -> list[torch.Tensor]:
        """(B, T, D) -> list of (B, T, C) logits, one per stage (last = final prediction)."""
        x = self.channel_drop(feat.transpose(1, 2)).transpose(1, 2)
        outs = [self.stage1(x)]
        for r in self.refiners:
            outs.append(r(torch.softmax(outs[-1], dim=-1)))
        return outs


def router_loss(outs: list[torch.Tensor], y: torch.Tensor, tmse_weight: float = 0.15, tau: float = 4.0) -> torch.Tensor:
    """Sum over stages of CE + tmse_weight * truncated MSE between consecutive log-probabilities (MS-TCN)."""
    total = torch.zeros((), device=y.device)
    for o in outs:
        total = total + F.cross_entropy(o.flatten(0, 1).float(), y.flatten())
        if o.shape[1] > 1:
            lp = F.log_softmax(o.float(), dim=-1)
            total = total + tmse_weight * torch.clamp((lp[:, 1:] - lp[:, :-1].detach()) ** 2, max=tau ** 2).mean()
    return total
