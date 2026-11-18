"""Input augmentation on the GPU for the five-view window model (training stage 3).

A window is 16 grey frames. The model's input (short side 256, centre crop 224) is expressed as an affine map from the
224 x 224 output to the source frame, so a random zoom, aspect change, rotation and shift cost nothing extra: one
bilinear resampling produces the augmented input directly from the source frame. Grey-level changes follow: a random
monotone grey map (another scanner's grey scale), gain, gamma, a gain ramp over depth, blur or sharpening,
speckle-like and additive noise, dark borders and a slight colour tint.

Parameters are drawn per fragment: a window cut from two recordings can carry two independent sets (as two recordings
would) or one shared set. With `Strength(0)` the output is the plain input of the grey frame (a test checks this).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ..model import MEAN, STD

OUT = 224
MAX_FRAGMENTS = 3
TONE_KNOTS = 8


@dataclass(frozen=True)
class Strength:
    """Half-ranges of the random draws; `scale` multiplies all of them (0 switches augmentation off). Three further
    multipliers switch whole groups: `geometry` (zoom, aspect, rotation, shift), `levels` (grey map, gain, gamma, depth
    ramp, tint) and `degrade` (blur, sharpening, noise, dark borders)."""
    scale: float = 1.0
    geometry: float = 1.0
    levels: float = 1.0
    degrade: float = 1.0
    zoom: float = 0.30            # log zoom in [-zoom, zoom]
    aspect: float = 0.10          # log aspect
    rotate: float = 12.0          # degrees
    shift: float = 0.08           # share of the output side
    identity: float = 0.15        # probability of no geometric change
    gain_low: float = 0.60        # log gain in [-gain_low, gain_high]
    gain_high: float = 0.45
    gamma: float = 0.50           # log gamma
    ramp: float = 0.60            # log gain difference between the top and the bottom row
    tone: float = 0.80            # largest blend of a random monotone grey-level curve (another scanner's grey map)
    tone_p: float = 0.70
    blur: float = 0.35            # probability of a blur blend
    sharpen: float = 0.15         # probability of sharpening
    speckle: float = 0.15         # largest sigma of the multiplicative noise
    speckle_p: float = 0.30
    additive: float = 0.02        # largest sigma of the additive noise (share of full scale)
    additive_p: float = 0.30
    border: float = 0.12          # largest dark strip per side (share of the output side)
    border_p: float = 0.30
    tint: float = 0.06            # largest per-channel gain difference
    tint_p: float = 0.30
    shared: float = 0.50          # probability that all fragments of a window share one parameter set


def base_map(height: int, width: int) -> tuple[float, float, float, float]:
    """(ax, bx, ay, by): the plain input as x_norm = ax * u_norm + bx, y_norm = ay * v_norm + by (pixel centres)."""
    s = 256.0 / min(height, width)
    wide, high = max(OUT, round(width * s)), max(OUT, round(height * s))
    left, top = (wide - OUT) // 2, (high - OUT) // 2
    return OUT / wide, (OUT + 2 * left) / wide - 1.0, OUT / high, (OUT + 2 * top) / high - 1.0


def _uniform(shape, low, high, generator, device):
    return torch.rand(shape, generator=generator, device=device) * (high - low) + low


def draw_groups(batch: int, group: int, strength: Strength, generator: torch.Generator, device) -> dict:
    """One parameter set per group of `group` neighbouring windows, repeated for each of its windows, so that the
    windows of a group differ only in content (the neighbour penalty compares their outputs)."""
    if batch % group:
        raise ValueError("the batch must hold whole groups")
    return {name: value.repeat_interleave(group, 0) for name, value in draw(batch // group, strength, generator, device).items()}


def draw(batch: int, strength: Strength, generator: torch.Generator, device) -> dict:
    """Random parameters, one set per (window, fragment); fragments share a set with probability `shared`."""
    shape = (batch, MAX_FRAGMENTS)
    k, g, d = strength.scale * strength.levels, strength.scale * strength.geometry, strength.scale * strength.degrade
    u = lambda low, high, size=shape: _uniform(size, low, high, generator, device)
    still = (u(0, 1) < strength.identity) | (g == 0)
    params = {
        "zoom": torch.where(still, torch.zeros(shape, device=device), u(-strength.zoom * g, strength.zoom * g)).exp(),
        "aspect": torch.where(still, torch.zeros(shape, device=device), u(-strength.aspect * g, strength.aspect * g)).exp(),
        "angle": torch.where(still, torch.zeros(shape, device=device), u(-strength.rotate * g, strength.rotate * g)) * (math.pi / 180),
        "shift_x": torch.where(still, torch.zeros(shape, device=device), u(-strength.shift * g, strength.shift * g)) * 2,
        "shift_y": torch.where(still, torch.zeros(shape, device=device), u(-strength.shift * g, strength.shift * g)) * 2,
        "gain": u(-strength.gain_low * k, strength.gain_high * k).exp(),
        "gamma": u(-strength.gamma * k, strength.gamma * k).exp(),
        "ramp": u(-strength.ramp * k, strength.ramp * k),
    }
    steps = -torch.log(u(1e-6, 1.0, (batch, MAX_FRAGMENTS, TONE_KNOTS)))            # Dirichlet(1, ..., 1) increments
    params["tone_w"] = steps / steps.sum(-1, keepdim=True)
    params["tone"] = (u(0, 1) < strength.tone_p * min(1.0, k)).float() * u(0, strength.tone * min(1.0, k))
    shared = u(0, 1, (batch, 1)) < strength.shared
    params = {name: torch.where(shared if value.ndim == 2 else shared.unsqueeze(-1), value[:, :1].expand_as(value), value)
              for name, value in params.items()}
    on = lambda p: (u(0, 1, (batch,)) < p * min(1.0, d)).float()
    tinted = (u(0, 1, (batch,)) < strength.tint_p * min(1.0, k)).float()
    params.update({
        "blur": on(strength.blur) * u(0.3, 1.0, (batch,)), "blur_wide": (u(0, 1, (batch,)) < 0.35).float(),
        "sharpen": on(strength.sharpen) * u(0.3, 1.0, (batch,)),
        "speckle": on(strength.speckle_p) * u(0, strength.speckle, (batch,)),
        "additive": on(strength.additive_p) * u(0, strength.additive, (batch,)),
        "border": on(strength.border_p)[:, None] * u(0, strength.border, (batch, 4)),
        "tint": 1.0 + tinted[:, None] * u(-strength.tint, strength.tint, (batch, 3)),
    })
    return params


def _per_slot(value: torch.Tensor, fragment: torch.Tensor) -> torch.Tensor:
    """(B, MAX_FRAGMENTS) parameter -> (B, 16) by each slot's fragment number."""
    return torch.gather(value, 1, fragment.clamp(0, MAX_FRAGMENTS - 1))


def _gaussian(x: torch.Tensor, sigma: float) -> torch.Tensor:
    radius = max(1, math.ceil(2.5 * sigma))
    grid = torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype)
    kernel = torch.exp(-grid ** 2 / (2 * sigma ** 2))
    kernel = (kernel / kernel.sum()).view(1, 1, -1)
    n, t, h, w = x.shape
    flat = x.reshape(n * t, 1, h, w)
    flat = F.conv2d(F.pad(flat, (radius, radius, 0, 0), mode="replicate"), kernel.unsqueeze(2))
    flat = F.conv2d(F.pad(flat, (0, 0, radius, radius), mode="replicate"), kernel.unsqueeze(3))
    return flat.view(n, t, h, w)


def warp(frames: torch.Tensor, fragment: torch.Tensor, params: dict) -> torch.Tensor:
    """(B, 16, H, W) uint8 source frames -> (B, 16, 224, 224) float in [0, 1], geometry applied."""
    b, t, h, w = frames.shape
    ax, bx, ay, by = base_map(h, w)
    zoom, aspect, angle = (_per_slot(params[k], fragment) for k in ("zoom", "aspect", "angle"))
    sx, sy = 1.0 / (zoom * aspect.sqrt()), aspect.sqrt() / zoom
    cos, sin = angle.cos(), angle.sin()
    tx, ty = _per_slot(params["shift_x"], fragment), _per_slot(params["shift_y"], fragment)
    theta = torch.stack([torch.stack([ax * cos * sx, -ax * sin * sy, ax * tx + bx], -1),
                         torch.stack([ay * sin * sx, ay * cos * sy, ay * ty + by], -1)], -2).view(b * t, 2, 3)
    grid = F.affine_grid(theta, (b * t, 1, OUT, OUT), align_corners=False)
    out = F.grid_sample(frames.reshape(b * t, 1, h, w).float() / 255.0, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    return out.view(b, t, OUT, OUT)


def grey_levels(x: torch.Tensor, fragment: torch.Tensor, params: dict, generator: torch.Generator) -> torch.Tensor:
    """Gain, gamma, depth ramp, blur or sharpening, noise and dark borders on (B, 16, 224, 224) in [0, 1]."""
    b, t = x.shape[:2]
    slot = lambda name: _per_slot(params[name], fragment).view(b, t, 1, 1)
    rows = torch.linspace(-0.5, 0.5, OUT, device=x.device).view(1, 1, OUT, 1)
    if bool((params["tone"] > 0).any()):
        weights = torch.gather(params["tone_w"], 1, fragment.clamp(0, MAX_FRAGMENTS - 1).unsqueeze(-1).expand(b, t, TONE_KNOTS))
        curve = torch.zeros_like(x)
        for knot in range(TONE_KNOTS):
            curve = curve + weights[:, :, knot].view(b, t, 1, 1) * (x * TONE_KNOTS - knot).clamp(0.0, 1.0)
        x = x + slot("tone") * (curve - x)
    x = x.clamp_min(1e-6).pow(slot("gamma")) * slot("gain") * torch.exp(slot("ramp") * rows)
    window = lambda name: params[name].view(b, 1, 1, 1)
    if bool((params["blur"] > 0).any()) or bool((params["sharpen"] > 0).any()):
        soft = _gaussian(x, 1.0)
        wide = _gaussian(x, 2.2) if bool((params["blur"] * params["blur_wide"] > 0).any()) else soft
        target = torch.where(window("blur_wide") > 0, wide, soft)
        x = x + window("blur") * (target - x) + window("sharpen") * (x - soft)
    if bool((params["speckle"] > 0).any()):
        grain = torch.randn((b * t, 1, OUT // 2, OUT // 2), generator=generator, device=x.device)
        grain = F.interpolate(grain, size=(OUT, OUT), mode="bilinear", align_corners=False).view(b, t, OUT, OUT)
        x = x * torch.exp(window("speckle") * grain)
    if bool((params["additive"] > 0).any()):
        x = x + window("additive") * torch.randn(x.shape, generator=generator, device=x.device)
    x = x.clamp(0.0, 1.0)
    if bool((params["border"] > 0).any()):
        ramp = torch.linspace(0.0, 1.0, OUT, device=x.device)
        edge = params["border"]
        keep_x = (ramp.view(1, OUT) >= edge[:, 0:1]) & (ramp.view(1, OUT) <= 1.0 - edge[:, 1:2])
        keep_y = (ramp.view(1, OUT) >= edge[:, 2:3]) & (ramp.view(1, OUT) <= 1.0 - edge[:, 3:4])
        x = x * (keep_y.view(b, 1, OUT, 1) & keep_x.view(b, 1, 1, OUT)).to(x.dtype)
    return x


def to_model_input(x: torch.Tensor, tint: torch.Tensor | None = None) -> torch.Tensor:
    """(B, 16, 224, 224) in [0, 1] -> (B, 3, 16, 224, 224) normalised with the encoder's mean and std."""
    mean = torch.tensor(MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1, 1)
    std = torch.tensor(STD, device=x.device, dtype=x.dtype).view(1, 3, 1, 1, 1)
    rgb = x.unsqueeze(1) * 255.0
    if tint is not None:
        rgb = (rgb * tint.view(-1, 3, 1, 1, 1)).clamp(0.0, 255.0)
    else:
        rgb = rgb.expand(-1, 3, -1, -1, -1)
    return (rgb - mean) / std


def augment(frames: torch.Tensor, fragment: torch.Tensor, strength: Strength, generator: torch.Generator, group: int = 1) -> torch.Tensor:
    """(B, 16, H, W) uint8 grey source frames and (B, 16) fragment numbers -> the model's (B, 3, 16, 224, 224) input.
    With group > 1 the windows come in groups of that size and every group shares one set of changes."""
    batch, device = frames.shape[0], frames.device
    params = draw(batch, strength, generator, device) if group == 1 else draw_groups(batch, group, strength, generator, device)
    x = warp(frames, fragment, params)
    if strength.scale > 0 and (strength.levels > 0 or strength.degrade > 0):
        x = grey_levels(x, fragment, params, generator)
        return to_model_input(x, params["tint"])
    return to_model_input(x)
