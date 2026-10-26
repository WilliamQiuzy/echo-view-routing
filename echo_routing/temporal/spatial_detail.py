"""A36 (configs/experiments/video_native_spatial_detail_v1.yaml): matched native / reduced spatial-detail inputs on a 320
canvas, the frozen A08 encoder on the 320 grid, and the zero-initialised residual recognition head.

Inputs. A08's decode and gamma rule, then an aspect-preserving resize to long side 320 (320 x 240 frames unchanged),
centred on a 320 x 320 canvas of RGB(29, 28, 29) (odd spare pixels: left / top floor, right / bottom one more).
  H  the long-side-320 image (native detail).
  L  the same image reduced with PIL BILINEAR to long side 224 and enlarged back to H's content size.
Only the spatial detail differs; field of view, canvas, marker, times and gamma are shared.

Encoder. The source A08 TargetMViT is reused frozen; only encoder.pos_encoding.spatial_size changes to (80, 80), so the
torchvision MViTv2-S forward runs on the 320 grid with its own runtime interpolation of rel_pos_h / rel_pos_w.

Head. z = q + g(concat(e224, e320)), g = Linear(1024, 32) -> GELU -> Linear(32, 5) with a zero last layer, so z = q
before any update.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

import echo_routing.vit.target_mvit as tm
from echo_routing.temporal.joint_visual import gamma_lut

CANVAS, LOW, FILL, GRID = 320, 224, (29, 28, 29), (80, 80)


def long_side_320(im: Image.Image) -> Image.Image:
    w, h = im.size
    if max(w, h) == CANVAS:
        return im
    s = CANVAS / max(w, h); return im.resize((round(w * s), round(h * s)), Image.BILINEAR)


def reduce_detail(im: Image.Image) -> Image.Image:
    w, h = im.size; s = LOW / max(w, h)
    return im.resize((round(w * s), round(h * s)), Image.BILINEAR).resize((w, h), Image.BILINEAR)


def content_box(w: int, h: int) -> tuple[int, int, int, int]:
    left, top = (CANVAS - w) // 2, (CANVAS - h) // 2; return left, top, left + w, top + h


def on_canvas(im: Image.Image) -> np.ndarray:
    c = Image.new("RGB", (CANVAS, CANVAS), FILL); c.paste(im, content_box(*im.size)[:2]); return np.asarray(c, dtype=np.uint8)


def gamma_rgb(im: Image.Image, gamma: float) -> Image.Image:
    im = im.convert("RGB"); return im.point(gamma_lut(gamma)) if abs(gamma - 1.0) > 1e-9 else im


def pair(im: Image.Image, gamma: float) -> tuple[np.ndarray, np.ndarray]:
    """(H, L) uint8 320 x 320 x 3 canvases from one decoded frame."""
    base = long_side_320(gamma_rgb(im, gamma)); return on_canvas(base), on_canvas(reduce_detail(base))


def load_pair(path, gamma: float) -> tuple[np.ndarray, np.ndarray]:
    with Image.open(path) as im:
        return pair(im, round(float(gamma), 6))


# ----------------------------------------------------------------------------------------------- encoder on the 320 grid
def encoder320(model: tm.TargetMViT) -> tm.TargetMViT:
    m = copy.deepcopy(model).eval(); m.encoder.pos_encoding.spatial_size = GRID
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def materialised320(model: tm.TargetMViT) -> tuple[tm.TargetMViT, list]:
    """torchvision MViTv2-S built at spatial_size 320 with model's weights; rel_pos_h / rel_pos_w resized once with the
    runtime rule (linear, align_corners False). Used only to check that encoder320's runtime path is the same computation."""
    from torchvision.models.video import mvit as M
    orig = M._mvit
    M._mvit = lambda **kw: orig(**{**kw, "spatial_size": (CANVAS, CANVAS)})
    try:
        net = M.mvit_v2_s()
    finally:
        M._mvit = orig
    net.head[-1] = nn.Linear(net.head[-1].in_features, tm.EMBED_DIM); m = tm.TargetMViT(net); src = model.state_dict(); dst = m.state_dict(); out, changed = {}, []
    for k, v in src.items():
        if v.shape == dst[k].shape:
            out[k] = v.clone()
        elif k.endswith(("rel_pos_h", "rel_pos_w")):
            out[k] = F.interpolate(v.permute(1, 0).unsqueeze(0), size=dst[k].shape[0], mode="linear", align_corners=False).squeeze(0).permute(1, 0).contiguous(); changed.append((k, list(v.shape), list(dst[k].shape)))
        else:
            raise ValueError(f"unexpected shape change at {k}")
    m.load_state_dict(out, strict=True); return m.to(next(model.parameters()).device).eval(), changed


def block_grids(m: tm.TargetMViT, rgb: torch.Tensor, marker: torch.Tensor) -> list:
    """The token grid entering every block (and after the last), read from a real forward."""
    enc = m.encoder
    with torch.no_grad():
        x = enc.conv_proj(rgb); x = x + m.marker_stem(marker.to(x.dtype)); stem = tuple(x.shape[2:]); x = enc.pos_encoding(x.flatten(2).transpose(1, 2))
        thw = (enc.pos_encoding.temporal_size,) + tuple(enc.pos_encoding.spatial_size); grids = [thw]
        for blk in enc.blocks:
            x, thw = blk(x, thw); grids.append(tuple(int(t) for t in thw))
    return [stem] + grids


# ----------------------------------------------------------------------------------------------- residual head
class Residual(nn.Module):
    def __init__(self, seed: int) -> None:
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(1024, 32), nn.Linear(32, 5)
        g = torch.Generator().manual_seed(int(seed))
        with torch.no_grad():
            self.fc1.weight.copy_(torch.randn(32, 1024, generator=g) * 0.02); self.fc1.bias.zero_(); self.fc2.weight.zero_(); self.fc2.bias.zero_()

    def forward(self, q: torch.Tensor, e224: torch.Tensor, e320: torch.Tensor) -> torch.Tensor:
        return q + self.fc2(F.gelu(self.fc1(torch.cat([e224, e320], -1))))
