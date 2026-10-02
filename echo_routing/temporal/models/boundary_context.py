"""Boundary-controlled context aggregation (model development round 1, plan Task 3).

docs/superpowers/plans/2026-10-02-model-development.md, "明确的第一版网络". Input features (B, D, T) and mask
(B, 1, T); outputs class_logits (B, C, T), boundary_logits (B, 1, T) and local_logits (B, C, T).

1. A per-sample linear projection to `dim`.
2. A local branch of residual dilated convolutions (kernel 3, dilations 1/2/4) gives a local representation and
   local class logits.
3. A boundary head reads adjacent local representations and their difference and predicts b[t], the probability
   of a view change between samples t-1 and t. It runs before the context module. Position 0 has no transition.
4. A local multi-head attention (keys within +-radius samples, learnable relative position bias) aggregates context.
   With `use_path`, the score from t to j gets sum(log(1 - b[k])) over the transitions crossed between them (0 for
   j = t), so a predicted change limits mixing across it. b is clipped to [1e-4, 1 - 1e-4]; only padding is masked
   and no ground-truth boundary is read.
5. A scalar gate g_t mixes the two branches: logits = (1 - g) * local + g * context. `fixed_gate` replaces the learned
   gate with a constant (ablation). The gate is an internal weight; it is not a quality or cardiac-phase signal.
6. Optional `segment_pool` (round-1 follow-up): the output at t is the average of the mixed log-probabilities over
   all samples j of the sequence, weighted by prod(1 - b[k]) along the path from t to j, i.e. softmax_j of the same
   path term. With b near 0 inside a predicted segment and near 1 at its ends, this is the mean evidence of the
   predicted segment; it has no parameters and no window.

Ablations (Task 4): M2 = no path term and no boundary loss; M3 = boundary loss without the path term; M4 = full.
All three are built with the same modules in the same order, so one seed gives the same initial weights.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

B_CLIP = 1e-4            # boundary probabilities are clipped to [B_CLIP, 1 - B_CLIP] inside the path term
NO_TRANSITION_LOGIT = -20.0  # position 0 (no previous sample); never enters a path or a loss


@dataclass(frozen=True)
class BoundaryContextConfig:
    in_dim: int = 512
    dim: int = 128
    num_classes: int = 5
    dilations: tuple[int, ...] = (1, 2, 4)
    dropout: float = 0.2
    heads: int = 4
    radius: int = 20
    use_path: bool = True
    path_coef: float = 1.0
    fixed_gate: float | None = None
    segment_pool: bool = False

    @classmethod
    def from_dict(cls, d: dict) -> "BoundaryContextConfig":
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        if "dilations" in known:
            known["dilations"] = tuple(int(x) for x in known["dilations"])
        return cls(**known)


class LocalLayer(nn.Module):
    """Residual dilated convolution (kernel 3), as in MS-TCN's dilated residual layer."""

    def __init__(self, dim: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, padding=dilation, dilation=dilation)
        self.mix = nn.Conv1d(dim, dim, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return (x + self.drop(self.mix(F.relu(self.conv(x))))) * mask


class BoundaryHead(nn.Module):
    """MLP on [h(t-1), h(t), h(t) - h(t-1)] -> logit of a change between t-1 and t."""

    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3 * dim, dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(dim, 1))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h (B, T, dim) -> logits (B, T); position 0 gets NO_TRANSITION_LOGIT."""
        prev, cur = h[:, :-1], h[:, 1:]
        z = self.net(torch.cat([prev, cur, cur - prev], dim=-1)).squeeze(-1)
        return F.pad(z, (1, 0), value=NO_TRANSITION_LOGIT)


def segment_pool(values: torch.Tensor, path: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """values (B, T, C) averaged over j with weights softmax_j(path[t, j]) among valid j -> (B, T, C)."""
    w = torch.softmax(path.masked_fill(~valid[:, None, :], torch.finfo(path.dtype).min), dim=-1)
    return torch.einsum("btj,bjc->btc", w, values.float()).to(values.dtype)


def path_bias(boundary_logits: torch.Tensor) -> torch.Tensor:
    """(B, T) transition logits -> (B, T, T) with entry [t, j] = sum over k in (min(t,j), max(t,j)] of log(1 - b[k]).

    Uses the cumulative sum C[t] = sum_{k=1..t} log(1 - b[k]); C is non-increasing, so the path sum is -|C[t] - C[j]|.
    Entries that involve padded positions are masked by the caller."""
    b = torch.sigmoid(boundary_logits.float()).clamp(B_CLIP, 1.0 - B_CLIP)
    logs = torch.log1p(-b)
    logs = torch.cat([torch.zeros_like(logs[:, :1]), logs[:, 1:]], dim=1)
    c = torch.cumsum(logs, dim=1)
    return -(c[:, :, None] - c[:, None, :]).abs()


class BoundaryAttention(nn.Module):
    """Multi-head attention restricted to |j - t| <= radius with a learnable per-head relative position bias and an
    optional additive path term; residual connection and layer norm."""

    def __init__(self, dim: int, heads: int, radius: int, dropout: float) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError(f"dim {dim} not divisible by heads {heads}")
        self.heads, self.radius, self.dh = heads, radius, dim // heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)
        self.rel = nn.Parameter(torch.zeros(heads, 2 * radius + 1))
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def logits(self, q: torch.Tensor, k: torch.Tensor, valid: torch.Tensor, path: torch.Tensor | None) -> torch.Tensor:
        """Attention logits (B, H, T, T) after window, relative bias, path term and padding mask."""
        t = q.shape[2]
        s = (q @ k.transpose(-1, -2)).float() / math.sqrt(self.dh)
        pos = torch.arange(t, device=q.device)
        off = pos[None, :] - pos[:, None]                                  # j - t
        s = s + self.rel[:, off.clamp(-self.radius, self.radius) + self.radius].float()[None]
        if path is not None:
            s = s + path[:, None]
        allowed = (off.abs() <= self.radius)[None, None] & valid[:, None, None, :]
        return s.masked_fill(~allowed, torch.finfo(s.dtype).min)  # finite: an all-masked padded row stays finite

    def forward(self, h: torch.Tensor, valid: torch.Tensor, path: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        """h (B, T, dim), valid (B, T), path (B, T, T) or None -> (new representation, attention weights (B, H, T, T))."""
        bsz, t, d = h.shape
        q, k, v = self.qkv(h).view(bsz, t, 3, self.heads, self.dh).permute(2, 0, 3, 1, 4)
        attn = torch.softmax(self.logits(q, k, valid, path), dim=-1)
        ctx = (attn.to(v.dtype) @ v).transpose(1, 2).reshape(bsz, t, d)
        return self.norm(h + self.drop(self.out(ctx))), attn


class BoundaryContextNet(nn.Module):
    def __init__(self, cfg: BoundaryContextConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.proj = nn.Conv1d(cfg.in_dim, cfg.dim, 1)
        self.local = nn.ModuleList([LocalLayer(cfg.dim, d, cfg.dropout) for d in cfg.dilations])
        self.local_cls = nn.Linear(cfg.dim, cfg.num_classes)
        self.boundary = BoundaryHead(cfg.dim, cfg.dropout)
        self.attn = BoundaryAttention(cfg.dim, cfg.heads, cfg.radius, cfg.dropout)
        self.ctx_cls = nn.Linear(cfg.dim, cfg.num_classes)
        self.gate = nn.Linear(2 * cfg.dim, 1)

    def local_features(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.proj(x) * mask
        for layer in self.local:
            h = layer(h, mask)
        return h.transpose(1, 2)                                            # (B, T, dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        valid = mask[:, 0] > 0                                              # (B, T)
        h = self.local_features(x, mask)
        local_logits = self.local_cls(h)
        b_logits = self.boundary(h)
        full_path = self.cfg.path_coef * path_bias(b_logits) if (self.cfg.use_path or self.cfg.segment_pool) else None
        c, _ = self.attn(h, valid, full_path if self.cfg.use_path else None)
        ctx_logits = self.ctx_cls(c)
        if self.cfg.fixed_gate is None:
            g = torch.sigmoid(self.gate(torch.cat([h, c], dim=-1)))
        else:
            g = torch.full_like(local_logits[..., :1], float(self.cfg.fixed_gate))
        logits = (1.0 - g) * local_logits + g * ctx_logits
        if self.cfg.segment_pool:
            logits = segment_pool(torch.log_softmax(logits, dim=-1), full_path, valid)
        vm = valid[..., None].to(logits.dtype)
        return {"class_logits": (logits * vm).transpose(1, 2), "local_logits": (local_logits * vm).transpose(1, 2),
                "context_logits": (ctx_logits * vm).transpose(1, 2), "boundary_logits": b_logits[:, None, :],
                "gate": (g * vm).transpose(1, 2)}
