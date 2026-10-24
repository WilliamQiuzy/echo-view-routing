"""Interval clips and the recognition branch of A28 (configs/experiments/video_interval_visual_adaptation_v1.yaml).

One 16-frame clip per interval [a, b): slot k shows sample a + floor((2k + 1) L / 32), so a long interval is covered evenly
and a short one repeats samples in time order; every request lies inside the interval. The marker sits where A08 puts it
for the target sample s[8]. The recognition branch is an independent copy of A08's TargetMViT; its forward is split at
the last MultiscaleBlock so that the frozen prefix can run without gradients while the suffix (last block, final norm,
512-d projection, classifier) may train.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn

import echo_routing.vit.target_mvit as tm

N_SLOTS = 16
ADAPTED_MODULES = ("encoder.blocks.15", "encoder.norm", "encoder.head.1")


def clip_positions(a: int, b: int) -> np.ndarray:
    """Sample positions of the 16 slots of interval [a, b)."""
    L = int(b) - int(a)
    if L < 1:
        raise ValueError(f"empty interval [{a}, {b})")
    return int(a) + (np.arange(N_SLOTS) * 2 + 1) * L // (2 * N_SLOTS)


def clip_marker(s: np.ndarray) -> int:
    return tm.marker_slot(np.asarray(s), int(np.asarray(s)[N_SLOTS // 2]))


def branch(model: tm.TargetMViT, adapted: bool) -> tm.TargetMViT:
    """An independent eval-mode copy; only the classifier (and, if adapted, the last block, norm and projection) trains."""
    m = copy.deepcopy(model).eval()
    for n, p in m.named_parameters():
        p.requires_grad_(n.startswith("classifier.") or (adapted and any(n.startswith(k + ".") for k in ADAPTED_MODULES)))
    return m


def trainable(m: nn.Module) -> list[tuple[str, nn.Parameter]]:
    return [(n, p) for n, p in m.named_parameters() if p.requires_grad]


def prefix(m: tm.TargetMViT, rgb: torch.Tensor, marker: torch.Tensor) -> tuple[torch.Tensor, tuple]:
    """conv_proj (+ marker stem), positional encoding and blocks 0..14: TargetMViT.embed's computation up to the last block."""
    enc = m.encoder; x = enc.conv_proj(rgb); x = x + m.marker_stem(marker.to(x.dtype)); x = enc.pos_encoding(x.flatten(2).transpose(1, 2))
    thw = (enc.pos_encoding.temporal_size,) + enc.pos_encoding.spatial_size
    for blk in enc.blocks[:-1]:
        x, thw = blk(x, thw)
    return x, thw


def suffix(m: tm.TargetMViT, x: torch.Tensor, thw: tuple) -> torch.Tensor:
    """The last block, the final norm, the class token and the 512-d projection."""
    enc = m.encoder; x, _ = enc.blocks[-1](x, thw); return enc.head(enc.norm(x)[:, 0])
