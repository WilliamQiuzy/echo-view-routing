"""GPU preprocessing for the ViT family: JPEG bytes -> nvJPEG decode -> gamma edit -> letterbox -> normalise.

DataLoader workers only read file bytes; decoding and resizing run batched on the GPU (the server has 8 vCPUs,
which cap PIL decoding at ~0.5k frames/s per core). Geometry follows ingest.frames.letterbox (PIL ImageOps.pad,
centred, black padding); the gamma edit uses the same lookup table as ingest.frames.apply_gamma.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class Norm:
    """Per-channel normalisation applied after dividing uint8 pixels by `scale`."""

    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    scale: float = 255.0


IMAGENET = Norm((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
KINETICS = Norm((0.45, 0.45, 0.45), (0.225, 0.225, 0.225))
ECHOPRIME = Norm((29.110628, 28.076836, 29.096405), (47.989223, 46.456997, 47.20083), scale=1.0)  # EchoPrime model.py
NORMS = {"imagenet": IMAGENET, "kinetics": KINETICS, "echoprime": ECHOPRIME}


def letterbox_geometry(height: int, width: int, size: int) -> tuple[int, int, int, int]:
    """(new_h, new_w, top, left) exactly as PIL ImageOps.pad with centring 0.5."""
    if height <= 0 or width <= 0 or size <= 0:
        raise ValueError("height, width and size must be positive")
    if width / height > 1.0:
        new_w, new_h = size, round(height / width * size)
    elif width / height < 1.0:
        new_w, new_h = round(width / height * size), size
    else:
        new_w = new_h = size
    return new_h, new_w, round((size - new_h) * 0.5), round((size - new_w) * 0.5)


def gamma_lut(gamma: float) -> torch.Tensor:
    """uint8 lookup table identical to ingest.frames.apply_gamma."""
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    return torch.tensor([int(255 * ((i / 255.0) ** gamma) + 0.5) for i in range(256)], dtype=torch.uint8)


def read_bytes(path: Path) -> bytes:
    """Raw file bytes. Plain `bytes` cross DataLoader worker queues by pickling; tensors from
    torchvision.io.read_file fail there (shared-storage fd duplication -> EBADF) and stall the loader."""
    return Path(path).read_bytes()


def _as_tensor(data: bytes | torch.Tensor) -> torch.Tensor:
    return data if isinstance(data, torch.Tensor) else torch.frombuffer(bytearray(data), dtype=torch.uint8)


def _apply_gammas(imgs: torch.Tensor, gammas: Sequence[float], luts: dict[float, torch.Tensor]) -> torch.Tensor:
    out = imgs
    for g in sorted(set(float(x) for x in gammas)):
        if abs(g - 1.0) < 1e-9:
            continue
        lut = luts.setdefault(g, gamma_lut(g).to(imgs.device))
        sel = torch.tensor([abs(float(x) - g) < 1e-9 for x in gammas], device=imgs.device)
        out = torch.where(sel.view(-1, 1, 1, 1), lut[out.long()], out)
    return out


def letterbox_batch(imgs: torch.Tensor, size: int, norm: Norm) -> torch.Tensor:
    """(B,3,H,W) uint8 -> (B,3,size,size) float32 normalised, antialiased bilinear resize, black padding."""
    b, _, h, w = imgs.shape
    new_h, new_w, top, left = letterbox_geometry(h, w, size)
    x = F.interpolate(imgs.float(), size=(new_h, new_w), mode="bilinear", antialias=True, align_corners=False)
    x = x.round_().clamp_(0, 255)
    out = torch.zeros(b, 3, size, size, device=imgs.device)
    out[:, :, top:top + new_h, left:left + new_w] = x
    mean = torch.tensor(norm.mean, device=imgs.device).view(1, 3, 1, 1)
    std = torch.tensor(norm.std, device=imgs.device).view(1, 3, 1, 1)
    return (out / norm.scale - mean) / std


class GpuDecoder:
    """Batched decode of JPEG byte tensors on one device; frames of different sizes are handled per size group."""

    def __init__(self, size: int, norm: Norm, device: torch.device) -> None:
        self.size, self.norm, self.device = size, norm, device
        self._luts: dict[float, torch.Tensor] = {}

    def __call__(self, datas: Sequence[bytes | torch.Tensor], gammas: Sequence[float] | None = None) -> torch.Tensor:
        from torchvision.io import ImageReadMode, decode_jpeg

        if len(datas) == 0:
            raise ValueError("no frames to decode")
        gammas = [1.0] * len(datas) if gammas is None else list(gammas)
        if len(gammas) != len(datas):
            raise ValueError("one gamma per frame required")
        dev = self.device if self.device.type == "cuda" else "cpu"
        decoded = decode_jpeg([_as_tensor(d) for d in datas], mode=ImageReadMode.RGB, device=dev)
        groups: dict[tuple[int, int], list[int]] = {}
        for i, im in enumerate(decoded):
            groups.setdefault((int(im.shape[1]), int(im.shape[2])), []).append(i)
        out = torch.empty(len(datas), 3, self.size, self.size, device=self.device)
        for idx in groups.values():
            batch = torch.stack([decoded[i] for i in idx]).to(self.device)
            batch = _apply_gammas(batch, [gammas[i] for i in idx], self._luts)
            out[idx] = letterbox_batch(batch, self.size, self.norm)
        return out
