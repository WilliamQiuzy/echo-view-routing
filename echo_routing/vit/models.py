"""ViT-family models. Every model is built from a `ModelSpec`, which is stored in its checkpoint.

frame  ViTFrame             plain ViT-S/16 classifier on single frames (B0's role, transformer encoder)
clip   FactorizedVideoViT   ViViT factorised encoder: shared ViT-S frame encoder -> temporal Transformer over the
                            clip's frame tokens -> clip head; plus an auxiliary per-frame head (STFM-style
                            spatial/temporal heads). The frame encoder is frame-wise, so its embeddings can be cached
                            and the temporal Transformer applied to any window afterwards.
clip   MViTClip             MViTv2-S (joint space-time pooling attention), Kinetics-400 or EchoPrime echo init
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

VIT_S = "vit_small_patch16_224.augreg_in21k_ft_in1k"
KINDS = {"vit_frame": "frame", "vivit_fe": "clip", "mvit_k400": "clip", "mvit_echoprime": "clip"}
DEFAULT_NORM = {"vit_frame": "imagenet", "vivit_fe": "imagenet", "mvit_k400": "kinetics", "mvit_echoprime": "echoprime"}


@dataclass(frozen=True)
class ModelSpec:
    name: str                      # key of KINDS
    num_classes: int
    task: str                      # family5 | raw9
    classes: tuple[str, ...]
    image_size: int = 224
    n_frames: int = 16             # clip models: frames per clip
    step: int = 3                  # clip models: 30-fps frames between clip frames (3 -> 10 Hz)
    arch: str = VIT_S
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def kind(self) -> str:
        return KINDS[self.name]

    @property
    def norm(self) -> str:
        return str(self.extra.get("norm", DEFAULT_NORM[self.name]))


def _timm_vit(arch: str, pretrained: bool, drop_path: float) -> nn.Module:
    import timm  # lazy: only needed when a ViT is built

    return timm.create_model(arch, pretrained=pretrained, num_classes=0, drop_path_rate=drop_path)


class ViTFrame(nn.Module):
    def __init__(self, num_classes: int, arch: str = VIT_S, pretrained: bool = True, drop_path: float = 0.1) -> None:
        super().__init__()
        self.vit = _timm_vit(arch, pretrained, drop_path)
        self.feature_dim = int(self.vit.num_features)
        self.head = nn.Linear(self.feature_dim, num_classes)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.vit(x)

    def forward_with_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        f = self.features(x)
        return self.head(f), f

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


class FactorizedVideoViT(nn.Module):
    def __init__(self, num_classes: int, arch: str = VIT_S, pretrained: bool = True, n_frames: int = 16, depth: int = 2,
                 heads: int = 6, drop_path: float = 0.1, dropout: float = 0.1) -> None:
        super().__init__()
        self.spatial = _timm_vit(arch, pretrained, drop_path)
        d = int(self.spatial.num_features)
        self.feature_dim, self.n_frames = d, n_frames
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.pos = nn.Parameter(torch.zeros(1, n_frames + 1, d))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(d, heads, dim_feedforward=4 * d, dropout=dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.temporal = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, num_classes)
        self.frame_head = nn.Linear(d, num_classes)

    def frame_embed(self, frames: torch.Tensor) -> torch.Tensor:
        """(N,3,H,W) -> (N,d) frame embeddings (cacheable: depends on one frame only)."""
        return self.spatial(frames)

    def temporal_logits(self, emb: torch.Tensor) -> torch.Tensor:
        """(B,T,d) frame embeddings -> (B,C) clip logits; T <= n_frames."""
        b, t, _ = emb.shape
        if t > self.n_frames:
            raise ValueError(f"clip of {t} frames exceeds the {self.n_frames} positional slots")
        tokens = torch.cat([self.cls.expand(b, -1, -1), emb], dim=1) + self.pos[:, : t + 1]
        return self.head(self.norm(self.temporal(tokens)[:, 0]))

    def forward(self, clips: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, t = clips.shape[:2]
        emb = self.frame_embed(clips.flatten(0, 1)).view(b, t, -1)
        return self.temporal_logits(emb), self.frame_head(emb)


class MViTClip(nn.Module):
    def __init__(self, num_classes: int, init: str = "k400", echoprime_ckpt: str | Path | None = None) -> None:
        super().__init__()
        from torchvision.models.video import MViT_V2_S_Weights, mvit_v2_s

        if init == "k400":
            net = mvit_v2_s(weights=MViT_V2_S_Weights.KINETICS400_V1)
        elif init == "echoprime":
            if echoprime_ckpt is None or not Path(echoprime_ckpt).is_file():
                raise FileNotFoundError(f"EchoPrime encoder weights not found: {echoprime_ckpt}")
            net = mvit_v2_s()
            net.head[-1] = nn.Linear(net.head[-1].in_features, 512)       # EchoPrime's projection (load_for_finetuning.py)
            net.load_state_dict(torch.load(echoprime_ckpt, map_location="cpu", weights_only=True))
        elif init == "none":
            net = mvit_v2_s()
        else:
            raise ValueError(f"unknown MViT init {init!r}")
        net.head[-1] = nn.Linear(net.head[-1].in_features, num_classes)
        self.net = net

    def forward(self, clips: torch.Tensor) -> tuple[torch.Tensor, None]:
        return self.net(clips.permute(0, 2, 1, 3, 4)), None   # (B,T,3,H,W) -> (B,3,T,H,W)


def build_model(spec: ModelSpec, pretrained: bool = True) -> nn.Module:
    x = spec.extra
    if spec.name == "vit_frame":
        return ViTFrame(spec.num_classes, spec.arch, pretrained, float(x.get("drop_path", 0.1)))
    if spec.name == "vivit_fe":
        return FactorizedVideoViT(spec.num_classes, spec.arch, pretrained, spec.n_frames, int(x.get("temporal_depth", 2)),
                                  int(x.get("temporal_heads", 6)), float(x.get("drop_path", 0.1)), float(x.get("dropout", 0.1)))
    if spec.name == "mvit_k400":
        return MViTClip(spec.num_classes, "k400" if pretrained else "none")
    if spec.name == "mvit_echoprime":
        return MViTClip(spec.num_classes, "echoprime" if pretrained else "none", x.get("echoprime_ckpt"))
    raise ValueError(f"unknown model {spec.name!r}")


def save_vit_checkpoint(path: Path, model: nn.Module, spec: ModelSpec, extra: dict[str, Any]) -> Path:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save({"state_dict": model.state_dict(), "spec": dataclasses.asdict(spec), **extra}, tmp)
    tmp.replace(path)   # atomic: a preempted write never leaves a truncated checkpoint
    return path


def spec_from_dict(d: dict[str, Any]) -> ModelSpec:
    return ModelSpec(**{**d, "classes": tuple(d["classes"])})


def load_vit_checkpoint(path: Path, map_location: str = "cpu") -> tuple[nn.Module, ModelSpec, dict[str, Any]]:
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    spec = spec_from_dict(payload["spec"])
    model = build_model(spec, pretrained=False)
    model.load_state_dict(payload["state_dict"])
    return model, spec, {k: v for k, v in payload.items() if k not in ("state_dict", "spec")}
