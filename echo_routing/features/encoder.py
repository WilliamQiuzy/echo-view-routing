"""ResNet-18 frame encoder with a view head; exposes pooled 512-d features for caching."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torchvision

FEATURE_DIM = {"resnet18": 512, "resnet34": 512, "resnet50": 2048}


class FrameEncoder(nn.Module):
    def __init__(self, num_classes: int, backbone: str = "resnet18", pretrained: bool = True) -> None:
        super().__init__()
        if backbone not in FEATURE_DIM:
            raise ValueError(f"unsupported backbone {backbone!r}")
        weights = "IMAGENET1K_V1" if pretrained else None
        net = getattr(torchvision.models, backbone)(weights=weights)
        self.backbone_name = backbone
        self.feature_dim = FEATURE_DIM[backbone]
        self.trunk = nn.Sequential(*list(net.children())[:-1])  # up to global avg pool
        self.head = nn.Linear(self.feature_dim, num_classes)
        self.num_classes = num_classes

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return torch.flatten(self.trunk(x), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))

    def forward_with_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        f = self.features(x)
        return self.head(f), f


def save_checkpoint(path: Path, model: FrameEncoder, extra: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "backbone": model.backbone_name,
        "num_classes": model.num_classes,
        **extra,
    }
    torch.save(payload, path)
    return path


def load_checkpoint(path: Path, map_location: str = "cpu") -> tuple[FrameEncoder, dict[str, Any]]:
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    model = FrameEncoder(payload["num_classes"], payload["backbone"], pretrained=False)
    model.load_state_dict(payload["state_dict"])
    meta = {k: v for k, v in payload.items() if k != "state_dict"}
    return model, meta


def checkpoint_hash(path: Path, n_chars: int = 12) -> str:
    """Short content hash of a checkpoint file; used as the feature-cache key."""
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n_chars]
