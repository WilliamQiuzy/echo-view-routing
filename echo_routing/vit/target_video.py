"""E1 frame encoder plus a two-layer temporal Transformer with positions relative to the target sample
(configs/experiments/ev9v_center_supervision_v1.yaml, 2026-10-05).

Token flow: frames (B*16, 224, 224, 3) -> E1 features (B, 16, 512) -> CLS + 16 frame tokens (B, 17, 512) -> window
logits (B, 5) for the target time. An auxiliary linear head reads the frame features before the temporal layers and
gives 16 frame logits. Each frame token carries the learned embedding of its offset s[i] - c from the target sample;
the CLS token has its own learned position. Self-attention over (token, offset) pairs does not depend on the order of
the pairs, so only the offsets locate the target. The same module serves the three arms; only the main target differs.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from echo_routing.temporal.joint_visual import VisualEncoder

NUM_CLASSES = 5


class TargetTimeHead(nn.Module):
    def __init__(self, dim: int = 512, layers: int = 2, heads: int = 8, ffn: int = 2048, dropout: float = 0.1,
                 max_offset: int = 15, num_classes: int = NUM_CLASSES, pos_std: float = 0.02) -> None:
        super().__init__()
        self.max_offset = int(max_offset)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.cls_pos = nn.Parameter(torch.zeros(1, 1, dim))
        self.rel_pos = nn.Parameter(torch.zeros(2 * self.max_offset + 1, dim))
        nn.init.trunc_normal_(self.cls_pos, std=pos_std); nn.init.trunc_normal_(self.rel_pos, std=pos_std)
        layer = nn.TransformerEncoderLayer(dim, heads, dim_feedforward=ffn, dropout=dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.temporal = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        self.frame_head = nn.Linear(dim, num_classes)

    def positions(self, rel: torch.Tensor) -> torch.Tensor:
        if rel.dtype.is_floating_point:
            raise TypeError("relative positions must be integers")
        if int(rel.abs().max()) > self.max_offset:
            raise ValueError(f"relative position {int(rel.abs().max())} outside +-{self.max_offset}")
        return self.rel_pos[rel.long() + self.max_offset]

    def forward(self, emb: torch.Tensor, rel: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """emb (B, T, D), rel (B, T) -> window logits (B, C), frame logits (B, T, C)."""
        if emb.shape[:2] != rel.shape:
            raise ValueError(f"features {tuple(emb.shape)} and positions {tuple(rel.shape)} disagree")
        b = emb.shape[0]
        tokens = torch.cat([self.cls.expand(b, -1, -1) + self.cls_pos, emb + self.positions(rel)], dim=1)
        return self.head(self.norm(self.temporal(tokens)[:, 0])), self.frame_head(emb)

    def init_from_probe(self, probe_state: dict) -> None:
        """Both heads start from the frozen E1 probe (Linear 512 -> 5) and are trained separately afterwards."""
        with torch.no_grad():
            for lin in (self.head, self.frame_head):
                lin.weight.copy_(probe_state["weight"]); lin.bias.copy_(probe_state["bias"])


class TargetVideoModel(nn.Module):
    def __init__(self, encoder: VisualEncoder, head: TargetTimeHead) -> None:
        super().__init__()
        self.encoder, self.head = encoder, head

    def forward(self, u8: torch.Tensor, rel: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """u8 (B*T, H, W, 3) frames in window order, rel (B, T) offsets -> window and frame logits."""
        b, t = rel.shape
        if u8.shape[0] != b * t:
            raise ValueError(f"{u8.shape[0]} frames for {b} windows of {t}")
        return self.head(self.encoder(u8).view(b, t, -1), rel)

    def head_parameters(self) -> list[nn.Parameter]:
        return list(self.head.parameters())


def window_target(arm: str, labels: torch.Tensor, target_labels: torch.Tensor, num_classes: int = NUM_CLASSES) -> torch.Tensor:
    """S: the share of each class among the window's frame labels. N and C: the one-hot of the target sample's label
    (for N's single-source windows this equals the recording label)."""
    if arm == "S":
        return F.one_hot(labels.long(), num_classes).float().mean(dim=1)
    if arm in ("N", "C"):
        return F.one_hot(target_labels.long(), num_classes).float()
    raise ValueError(f"unknown arm {arm!r}")


def losses(arm: str, logits: torch.Tensor, frame_logits: torch.Tensor, labels: torch.Tensor, target_labels: torch.Tensor,
           aux_weight: float) -> tuple[torch.Tensor, dict[str, float]]:
    """Main soft cross-entropy on the window target plus aux_weight times the frame cross-entropy. One forward pass
    feeds both terms."""
    target = window_target(arm, labels, target_labels, logits.shape[-1])
    main = -(target * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
    aux = F.cross_entropy(frame_logits.reshape(-1, frame_logits.shape[-1]), labels.reshape(-1).long())
    return main + float(aux_weight) * aux, {"main": float(main.detach()), "aux": float(aux.detach())}
