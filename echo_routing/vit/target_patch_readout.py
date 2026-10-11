"""Target-anchored readout of the raw-video context round's TargetMViT (configs/experiments/target_patch_readout_v1.yaml;
docs/superpowers/plans/2026-10-11-target-patch-readout.md).

The torchvision MViT runs conv_proj -> pos_encoding -> blocks -> norm -> x[:, 0] -> head. ReadoutMViT keeps every
parameter of TargetMViT and changes only which 768-d vector reaches the head. A forward hook on encoder.norm, present
only while ReadoutMViT runs and removed in finally, replaces token 0 of a clone of the norm output.
  cls             no replacement; the model equals the source TargetMViT
  global_patches  the patch tokens (token 0 excluded) averaged over every time slot and spatial position
  target_patches  the patch tokens averaged over space within each time slot, then combined with fixed weights w
w comes from the input marker's single 1 position among the 16 frames, convolved with an all-ones kernel of the RGB stem's
temporal kernel, stride, padding and dilation and normalised to sum 1. The token grid (T', H', W') comes from the
positional-encoding grid and every block's query-pool stride; tokens 1.. are in (T', H', W') order, as conv_proj's
flatten(2) and every Pool's reshape leave them. Nothing here reads a label, a boundary or a recording name.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

import echo_routing.vit.target_mvit as tm

READOUTS = ("cls", "global_patches", "target_patches")
NAMES = {"cls": "Continued CLS", "global_patches": "Global patches", "target_patches": "Target patches"}
EXPECTED_GRID = (8, 7, 7)


def token_grid(encoder) -> tuple[tuple[int, int, int], list[tuple[int, int, int]]]:
    """(T', H', W') of the final tokens and every block's query stride, read from the network."""
    thw = (int(encoder.pos_encoding.temporal_size), *(int(x) for x in encoder.pos_encoding.spatial_size))
    strides = []
    for blk in encoder.blocks:
        pool = blk.attn.pool_q
        s = tuple(int(x) for x in pool.pool.stride) if pool is not None else (1, 1, 1)
        strides.append(s)
        thw = tuple(n // k for n, k in zip(thw, s))
    return thw, strides


def marker_vector(marker: torch.Tensor) -> torch.Tensor:
    """(B, 1, 16, H, W) 0/1 marker -> (B, 16) holding each window's single marked slot."""
    if marker.ndim != 5 or marker.shape[1] != 1:
        raise ValueError(f"expected a (B, 1, T, H, W) marker, got {tuple(marker.shape)}")
    hi, lo = marker[:, 0].amax(dim=(-2, -1)), marker[:, 0].amin(dim=(-2, -1))
    if not torch.equal(hi, lo) or not bool(((hi == 0) | (hi == 1)).all()) or not bool((hi.sum(1) == 1).all()):
        raise ValueError("every window needs exactly one whole marker plane")
    return hi


def time_weights(m16: torch.Tensor, kernel: int = 3, stride: int = 2, padding: int = 1, dilation: int = 1) -> torch.Tensor:
    """(B, 16) one-hot marker -> (B, T') weights: a fixed all-ones kernel with the stem's temporal geometry, sum 1."""
    ones = torch.ones(1, 1, int(kernel), dtype=m16.dtype, device=m16.device)
    w = F.conv1d(m16.unsqueeze(1), ones, stride=int(stride), padding=int(padding), dilation=int(dilation)).squeeze(1)
    total = w.sum(1, keepdim=True)
    if bool((total <= 0).any()):
        raise ValueError("a marker falls outside every time slot")
    return w / total


def readout_vector(tokens: torch.Tensor, mode: str, weights: torch.Tensor, grid: tuple[int, int, int]) -> torch.Tensor:
    """(B, 1 + T'H'W', C) final tokens -> (B, C): the mean of all patch tokens or their target-weighted time slots."""
    b, n, c = tokens.shape
    t, h, w = grid
    if n != 1 + t * h * w:
        raise ValueError(f"{n} tokens do not fit the grid {grid} plus the class token")
    p = tokens[:, 1:].reshape(b, t, h, w, c).mean(dim=(2, 3))
    if mode == "global_patches":
        return p.mean(1)
    if mode == "target_patches":
        if tuple(weights.shape) != (b, t):
            raise ValueError(f"weights {tuple(weights.shape)} do not fit ({b}, {t})")
        return (weights.to(p.dtype).unsqueeze(-1) * p).sum(1)
    raise ValueError(f"no patch readout {mode!r}")


def replace_token0(tokens: torch.Tensor, mode: str, weights: torch.Tensor, grid: tuple[int, int, int]) -> torch.Tensor:
    """A copy of the final tokens whose token 0 is the readout; Continued CLS returns the tokens unchanged."""
    if mode == "cls":
        return tokens
    out = tokens.clone()
    out[:, 0] = readout_vector(tokens, mode, weights, grid)
    return out


class ReadoutMViT(tm.TargetMViT):
    """TargetMViT whose head reads Continued CLS, Global patches or Target patches; no parameter is added."""

    def __init__(self, encoder, readout: str, num_classes: int = tm.NUM_CLASSES) -> None:
        super().__init__(encoder, num_classes)
        if readout not in READOUTS:
            raise ValueError(f"unknown readout {readout!r}")
        self.readout = readout
        self.grid, self.query_strides = token_grid(encoder)
        conv = encoder.conv_proj
        self.stem_time = (int(conv.kernel_size[0]), int(conv.stride[0]), int(conv.padding[0]), int(conv.dilation[0]))
        if any(s[0] != 1 for s in self.query_strides) or self.grid != EXPECTED_GRID:
            raise SystemExit(f"the encoder's token grid {self.grid} or query time strides {self.query_strides} differ from the plan")

    def embed(self, rgb: torch.Tensor, marker: torch.Tensor) -> torch.Tensor:
        if self.readout == "cls":
            return super().embed(rgb, marker)
        w = time_weights(marker_vector(marker), *self.stem_time)
        if w.shape[1] != self.grid[0]:
            raise ValueError(f"{w.shape[1]} marker time slots for a grid of {self.grid[0]}")
        norm = self.encoder.norm
        if norm._forward_hooks:
            raise RuntimeError("encoder.norm already carries a forward hook")
        mode, grid = self.readout, self.grid

        def replace(_module, _inputs, out):
            return replace_token0(out, mode, w, grid)

        handle = norm.register_forward_hook(replace)
        try:
            return super().embed(rgb, marker)
        finally:
            handle.remove()


def build_readout_model(readout: str, state: dict) -> ReadoutMViT:
    """A ReadoutMViT strictly loaded with a full TargetMViT state (encoder, projection, marker stem and head)."""
    model = ReadoutMViT(tm.build_source_encoder(None), readout)
    model.load_state_dict(state, strict=True)
    return model
