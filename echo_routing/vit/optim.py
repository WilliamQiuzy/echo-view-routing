"""Optimiser helpers: ViT layer-wise learning-rate decay, no weight decay on norms/biases/position tables,
and a per-step linear-warmup + cosine schedule."""
from __future__ import annotations

import math
import re

import torch.nn as nn

_BLOCK = re.compile(r"(?:^|\.)blocks\.(\d+)\.")
_EMBED = ("cls_token", "pos_embed", "patch_embed", "reg_token")
_NO_DECAY = ("pos_embed", "cls_token", "reg_token", "rel_pos", "pos", "cls")


def vit_layer_id(name: str, vit_prefix: str, num_layers: int) -> int:
    """0 = patch/position embeddings, i+1 = transformer block i, num_layers+1 = everything above the ViT trunk."""
    if not name.startswith(vit_prefix + "."):
        return num_layers + 1
    rest = name[len(vit_prefix) + 1:]
    if rest.split(".")[0] in _EMBED:
        return 0
    m = _BLOCK.search("." + rest)
    return int(m.group(1)) + 1 if m else num_layers + 1


def _no_decay(name: str, p) -> bool:
    leaf = name.split(".")[-1]
    return p.ndim <= 1 or leaf in _NO_DECAY or any(leaf.startswith(k) for k in ("rel_pos",))


def param_groups(model: nn.Module, lr: float, weight_decay: float, layer_decay: float = 1.0,
                 vit_prefix: str | None = None, num_layers: int = 0) -> list[dict]:
    """AdamW parameter groups; with `vit_prefix`, lr is scaled by layer_decay ** (num_layers + 1 - layer_id)."""
    groups: dict[tuple[int, bool], dict] = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        lid = vit_layer_id(name, vit_prefix, num_layers) if vit_prefix else num_layers + 1
        nd = _no_decay(name, p)
        g = groups.setdefault((lid, nd), {"params": [], "weight_decay": 0.0 if nd else weight_decay,
                                          "lr_scale": layer_decay ** (num_layers + 1 - lid), "layer_id": lid})
        g["params"].append(p)
    out = []
    for g in groups.values():
        g["lr"] = lr * g["lr_scale"]; out.append(g)
    return out


def warmup_cosine(step: int, total_steps: int, warmup_steps: int, min_ratio: float = 0.0) -> float:
    """LR multiplier: linear warm-up from ~0 to 1, then cosine to `min_ratio` at `total_steps`."""
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if warmup_steps > 0 and step < warmup_steps:
        return (step + 1) / warmup_steps
    t = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
    return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * t))


def set_lr(optimizer, base_lr: float, factor: float) -> None:
    for g in optimizer.param_groups:
        g["lr"] = base_lr * g.get("lr_scale", 1.0) * factor
