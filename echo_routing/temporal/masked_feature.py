"""A40 (configs/experiments/video_masked_feature_recognition_v1.yaml): masked-region HOG prediction for TargetMViT.

Target. From the clean uint8 crop / gamma frames (B, 16, 224, 224, 3), before masking and detached: gray = mean of R, G, B
as float32 / 255; centred differences with replicated edges, gx = I[y, x+1] - I[y, x-1], gy = I[y+1, x] - I[y-1, x];
magnitude sqrt(gx^2 + gy^2), unsigned orientation atan2(gy, gx) mod pi. Nine bins of width pi / 9 centred at
(k + 0.5) pi / 9; each pixel votes its magnitude linearly into the two neighbouring circular bins. Every 8 x 8 cell sums its
votes and is L2 normalised alone, h / sqrt(sum(h^2) + 1e-6). A 32 x 32 block (4 x 4 cells) gives a 144-d target, ordered
cell row, cell column, bin. Token time t takes input slot 2t.
Mask. 25 of the 7 x 7 blocks per window (stateless numpy RNG, label-free), the same on all 16 slots; the normalised RGB is 0
there on all three channels; the marker planes are not touched.
Tokens. The masked forward's encoder.norm output (B, 1 + 8 x 7 x 7, 768), captured with autograd by a hook that returns
None and is removed in finally; non-CLS tokens reshaped time-major, y, x. A fixed alignment, not a receptive-field claim.
Nothing here reads a label.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

import echo_routing.vit.target_mvit as tm

GRID, BLOCK, CELL, BINS = (8, 7, 7), 32, 8, 9
N_BLOCKS, N_MASK, HOG_DIM, EPS = GRID[1] * GRID[2], 25, (BLOCK // CELL) ** 2 * BINS, 1e-6
N_TOKENS, DIM, BASE, ROUND = 1 + GRID[0] * GRID[1] * GRID[2], 768, 20261026, 40
CONTRACT = ("HOG144 of slot 2t (gray mean RGB / 255, centred replicate differences, 9 unsigned bins centred (k + 0.5) pi / 9, "
            "linear circular votes, 8 x 8 cells L2 / sqrt(sum + 1e-6), 4 x 4 cells row-major then bins); 25 of 49 32 x 32 tube "
            "blocks of normalised RGB set to 0; encoder.norm output with autograd, U[:, 1:] -> (8, 7 x 7) time-major, y, x; "
            "Linear(768, 144) N(0, 0.02^2) seed 20261026 + 40 + 10 s + f; MSE on masked blocks only")


# ----------------------------------------------------------------------------------------------- HOG target
def gray_of(u8: torch.Tensor) -> torch.Tensor:
    """(..., H, W, 3) uint8 -> (..., H, W) float32 in [0, 1]: the channel mean / 255."""
    if u8.dtype != torch.uint8 or u8.shape[-1] != 3:
        raise ValueError(f"expected uint8 (..., H, W, 3), got {tuple(u8.shape)} {u8.dtype}")
    return u8.float().mean(-1) / 255.0


def cell_histograms(gray: torch.Tensor) -> torch.Tensor:
    """(N, H, W) gray -> (N, H / 8, W / 8, 9) L2-normalised cell histograms."""
    n, h, w = gray.shape
    p = torch.nn.functional.pad(gray[:, None], (1, 1, 1, 1), mode="replicate")[:, 0]
    gx, gy = p[:, 1:-1, 2:] - p[:, 1:-1, :-2], p[:, 2:, 1:-1] - p[:, :-2, 1:-1]
    mag, theta = torch.sqrt(gx * gx + gy * gy), torch.remainder(torch.atan2(gy, gx), math.pi)
    b = theta / (math.pi / BINS) - 0.5; fl = torch.floor(b); w_hi = b - fl; lo = torch.remainder(fl.long(), BINS); hi = torch.remainder(lo + 1, BINS)
    oh = lambda k: torch.nn.functional.one_hot(k, BINS).to(gray.dtype)  # noqa: E731 - lo != hi, so no index collides
    votes = oh(lo) * (mag * (1 - w_hi))[..., None] + oh(hi) * (mag * w_hi)[..., None]
    cells = votes.reshape(n, h // CELL, CELL, w // CELL, CELL, BINS).sum((2, 4))
    return cells / torch.sqrt((cells * cells).sum(-1, keepdim=True) + EPS)


def block_features(cells: torch.Tensor) -> torch.Tensor:
    """(N, 28, 28, 9) cells -> (N, 49, 144): block by * 7 + bx, then cell row, cell column, bin."""
    n, k = cells.shape[0], BLOCK // CELL
    return cells.reshape(n, GRID[1], k, GRID[2], k, BINS).permute(0, 1, 3, 2, 4, 5).reshape(n, N_BLOCKS, HOG_DIM)


@torch.no_grad()
def hog_targets(u8: torch.Tensor) -> torch.Tensor:
    """(B, 16, 224, 224, 3) clean uint8 frames -> (B, 8, 49, 144) detached targets of slots 0, 2, ..., 14."""
    if u8.ndim != 5 or u8.shape[1] != tm.N_SLOTS or u8.shape[2:4] != (224, 224):
        raise ValueError(f"expected (B, 16, 224, 224, 3), got {tuple(u8.shape)}")
    b = u8.shape[0]; g = gray_of(u8[:, 0::2]).reshape(b * GRID[0], 224, 224)
    return block_features(cell_histograms(g)).reshape(b, GRID[0], N_BLOCKS, HOG_DIM).detach()


# ----------------------------------------------------------------------------------------------- mask
def mask_rng(seed: int, fold: int, update: int, window: int) -> np.random.Generator:
    return np.random.default_rng([BASE, ROUND, int(seed), int(fold), int(update), int(window)])


def mask_blocks(seed: int, fold: int, update: int, n_windows: int) -> np.ndarray:
    """(n_windows, 49) bool: the 25 masked blocks of each window of an update (stateless, label-free)."""
    out = np.zeros((n_windows, N_BLOCKS), bool)
    for w in range(n_windows):
        out[w, mask_rng(seed, fold, update, w).choice(N_BLOCKS, N_MASK, replace=False)] = True
    return out


def pixel_mask(sel: torch.Tensor) -> torch.Tensor:
    """(B, 49) bool -> (B, 1, 1, 224, 224) bool: True on every pixel of a selected 32 x 32 block."""
    b = sel.shape[0]; m = sel.reshape(b, GRID[1], 1, GRID[2], 1).expand(b, GRID[1], BLOCK, GRID[2], BLOCK)
    return m.reshape(b, 1, 1, GRID[1] * BLOCK, GRID[2] * BLOCK)


def masked_rgb(rgb: torch.Tensor, sel: torch.Tensor) -> torch.Tensor:
    """A new (B, 3, 16, 224, 224) tensor: the normalised RGB with every selected block 0 on all slots and channels."""
    return rgb.masked_fill(pixel_mask(sel.to(rgb.device)), 0.0)


# ----------------------------------------------------------------------------------------------- tokens, decoder, loss
def embed_with_tokens_grad(m: tm.TargetMViT, rgb: torch.Tensor, marker: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(e (B, 512), U (B, 393, 768)) of one TargetMViT.embed forward; U keeps its autograd graph (no detach, no clone)."""
    norm = m.encoder.norm
    if norm._forward_hooks:
        raise RuntimeError("encoder.norm already carries a forward hook")
    seen: list[torch.Tensor] = []

    def capture(_module, _inputs, out) -> None:
        seen.append(out)

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


def token_grid(u: torch.Tensor) -> torch.Tensor:
    """(B, 393, C) -> (B, 8, 49, C): CLS dropped, tokens time-major then y, x."""
    return u[:, 1:].reshape(u.shape[0], GRID[0], N_BLOCKS, u.shape[-1])


def decoder(seed: int, fold: int) -> nn.Linear:
    """Linear(768, 144): weight N(0, 0.02^2) from its own generator, bias 0; the global RNG is not touched."""
    g = torch.Generator().manual_seed(BASE + ROUND + 10 * int(seed) + int(fold)); lin = nn.utils.skip_init(nn.Linear, DIM, HOG_DIM)   # no default init draw
    with torch.no_grad():
        lin.weight.copy_(torch.randn(HOG_DIM, DIM, generator=g) * 0.02); lin.bias.zero_()
    return lin


def hog_loss(pred: torch.Tensor, target: torch.Tensor, sel: torch.Tensor) -> torch.Tensor:
    """Mean squared error over the selected blocks x 8 times x 144 dims; unselected positions do not enter."""
    w = sel.to(pred.device)[:, None, :, None].to(pred.dtype)
    return ((pred - target) ** 2 * w).sum() / (w.sum() * GRID[0] * HOG_DIM)


def masked_loss(m: tm.TargetMViT, dec: nn.Linear, rgb: torch.Tensor, marker: torch.Tensor, target: torch.Tensor, sel: torch.Tensor) -> torch.Tensor:
    """L_HOG of one masked forward: the masked RGB through the encoder, the decoder on the non-CLS final tokens."""
    _, u = embed_with_tokens_grad(m, masked_rgb(rgb, sel), marker)
    return hog_loss(dec(token_grid(u)), target, sel)


def shared_trunk(m: tm.TargetMViT) -> list[tuple[str, nn.Parameter]]:
    """Parameters both losses reach: the encoder without encoder.head, plus marker_stem."""
    return [(n, p) for n, p in m.named_parameters() if (n.startswith("encoder.") and not n.startswith("encoder.head.")) or n.startswith("marker_stem.")]


def grad_norm(params) -> float:
    """L2 norm of the gradients (a None gradient counts as 0), in float64."""
    return float(math.sqrt(sum(float(p.grad.double().pow(2).sum()) for p in params if p.grad is not None)))


# ----------------------------------------------------------------------------------------------- independent reference (checks only)
def reference_block_hog(gray: np.ndarray) -> np.ndarray:
    """Pixel loop in float64 over one (224, 224) gray image -> (49, 144); shares no code with the vectorised target."""
    h, w = gray.shape; cells = np.zeros((h // CELL, w // CELL, BINS))
    for y in range(h):
        for x in range(w):
            gx = float(gray[y, min(x + 1, w - 1)]) - float(gray[y, max(x - 1, 0)]); gy = float(gray[min(y + 1, h - 1), x]) - float(gray[max(y - 1, 0), x])
            mag = math.hypot(gx, gy); theta = math.atan2(gy, gx) % math.pi; b = theta * BINS / math.pi - 0.5; k = math.floor(b); f = b - k
            cells[y // CELL, x // CELL, k % BINS] += mag * (1 - f); cells[y // CELL, x // CELL, (k + 1) % BINS] += mag * f
    cells = cells / np.sqrt((cells ** 2).sum(-1, keepdims=True) + EPS); out = np.zeros((N_BLOCKS, HOG_DIM)); k4 = BLOCK // CELL
    for by in range(GRID[1]):
        for bx in range(GRID[2]):
            out[by * GRID[2] + bx] = [cells[by * k4 + i, bx * k4 + j, q] for i in range(k4) for j in range(k4) for q in range(BINS)]
    return out
