"""Target-logit residual readout (configs/experiments/target_logit_residual_v1.yaml, 2026-10-06).

The model is the C model of the center-supervision round: E1 encoder plus TargetTimeHead, whose forward gives the CLS
readout delta = W h + b and the auxiliary frame logits. Only the final logits differ between the two arms:
  Z: final = delta                     (the CLS readout; W and b start at zero)
  R: final = a_c + delta               (a_c: the frame logits at the target sample; the same zero-initialised delta)
a_c is read at the positions whose offset from the target is 0; when a short clip repeats the target frame, those
positions are averaged, and a window without the target is refused. Logits are added before the softmax; there is no
probability averaging, gate, fixed mixing weight, stop-gradient or extra parameter. Both arms share one parameter tree
and one forward pass, so the same dropout calls happen in both.
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn

from echo_routing.temporal.joint_visual import VisualEncoder
from echo_routing.vit.target_video import TargetTimeHead

MODES = ("Z", "R")


def target_logits(frame_logits: torch.Tensor, rel: torch.Tensor) -> torch.Tensor:
    """frame_logits (B, T, C), rel (B, T) -> the logits at the target sample (B, C); repeats are averaged."""
    at_target = rel.eq(0)
    if not bool(at_target.any(dim=1).all()):
        raise ValueError("target sample is absent from the window")
    weights = at_target.to(frame_logits.dtype)
    weights = weights / weights.sum(dim=1, keepdim=True)
    return (frame_logits * weights.unsqueeze(-1)).sum(dim=1)


def readout(delta: torch.Tensor, frame_logits: torch.Tensor, rel: torch.Tensor, mode: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Final logits of the arm and the frame logits (unchanged)."""
    target = target_logits(frame_logits, rel)            # computed in both arms, so the target check runs in both
    if mode == "Z":
        return delta, frame_logits
    if mode == "R":
        return target + delta, frame_logits
    raise ValueError(f"unknown mode {mode!r}")


class ResidualTargetModel(nn.Module):
    def __init__(self, encoder: VisualEncoder, head: TargetTimeHead, mode: str) -> None:
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}")
        self.encoder, self.head, self.mode = encoder, head, mode

    def forward(self, u8: torch.Tensor, rel: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, t = rel.shape
        if u8.shape[0] != b * t:
            raise ValueError(f"{u8.shape[0]} frames for {b} windows of {t}")
        delta, frame_logits = self.head(self.encoder(u8).view(b, t, -1), rel)
        return readout(delta, frame_logits, rel, self.mode)

    def head_parameters(self) -> list[nn.Parameter]:
        return list(self.head.parameters())


def zero_output_state(head_state: dict) -> dict:
    """The source head state with the window output layer (head.weight, head.bias) set to zero; all else equal."""
    out = copy.deepcopy(head_state)
    for key in ("head.weight", "head.bias"):
        if key not in out:
            raise KeyError(f"{key} missing from the head state")
        out[key] = torch.zeros_like(out[key])
    return out


TEMPORAL_PREFIXES = ("cls", "cls_pos", "rel_pos", "temporal.", "norm.")


def temporal_named(head: TargetTimeHead) -> list[tuple[str, nn.Parameter]]:
    """Parameters of the temporal path before the window output layer."""
    return [(n, p) for n, p in head.named_parameters() if n.startswith(TEMPORAL_PREFIXES)]
