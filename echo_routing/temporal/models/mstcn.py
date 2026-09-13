"""MS-TCN (Abu Farha & Gall, CVPR 2019) on frozen frame features — B4 learned temporal baseline.

Compact re-implementation (multi-stage dilated residual TCN with per-stage refinement). Not the authors' code.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation: int, channels: int) -> None:
        super().__init__()
        self.conv_dilated = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(channels, channels, 1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.conv_dilated(x))
        out = self.dropout(self.conv_1x1(out))
        return (x + out) * mask


class SingleStageTCN(nn.Module):
    def __init__(self, num_layers: int, channels: int, in_dim: int, num_classes: int) -> None:
        super().__init__()
        self.conv_in = nn.Conv1d(in_dim, channels, 1)
        self.layers = nn.ModuleList([DilatedResidualLayer(2 ** i, channels) for i in range(num_layers)])
        self.conv_out = nn.Conv1d(channels, num_classes, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        out = self.conv_in(x)
        for layer in self.layers:
            out = layer(out, mask)
        return self.conv_out(out) * mask


class MultiStageTCN(nn.Module):
    """Input (B, D, T) features + (B, 1, T) mask -> list of per-stage logits (B, C, T)."""

    def __init__(self, num_stages: int, num_layers: int, channels: int, in_dim: int, num_classes: int) -> None:
        super().__init__()
        self.stage1 = SingleStageTCN(num_layers, channels, in_dim, num_classes)
        self.stages = nn.ModuleList([SingleStageTCN(num_layers, channels, num_classes, num_classes) for _ in range(num_stages - 1)])
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> list[torch.Tensor]:
        out = self.stage1(x, mask); outputs = [out]
        for s in self.stages:
            out = s(F.softmax(out, dim=1) * mask, mask); outputs.append(out)
        return outputs


def mstcn_loss(outputs: list[torch.Tensor], target: torch.Tensor, mask: torch.Tensor, smooth_weight: float = 0.15,
               class_weight: torch.Tensor | None = None) -> torch.Tensor:
    """Cross-entropy on every stage + truncated MSE smoothing of log-probs (as in the paper)."""
    loss = torch.zeros((), device=target.device)
    for out in outputs:
        ce = F.cross_entropy(out.transpose(1, 2).reshape(-1, out.shape[1]), target.reshape(-1), weight=class_weight,
                             ignore_index=-100)
        lp = F.log_softmax(out, dim=1)
        sm = torch.clamp((lp[:, :, 1:] - lp[:, :, :-1].detach()) ** 2, max=16.0)
        loss = loss + ce + smooth_weight * (sm * mask[:, :, 1:]).sum() / mask[:, :, 1:].sum().clamp(min=1)
    return loss
