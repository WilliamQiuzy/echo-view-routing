"""The window model.

A window is 16 frames of a 10 Hz sequence around one requested time. The network is torchvision's MViTv2-S video
encoder with a 512-d projection, a second input stem that reads a 0/1 marker video (one whole plane set to 1 at the
requested frame) and a linear five-class head. The output is the view at the marked frame.
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

CLASSES = ("PLAX", "PSAX", "A4C", "A5C", "SC4C")
N_SLOTS = 16
SIZE = 224
EMBED_DIM = 512
# The encoder's own input statistics, on pixel values 0-255.
MEAN = (29.110628, 28.076836, 29.096405)
STD = (47.989223, 46.456997, 47.20083)


class WindowModel(nn.Module):
    """forward(rgb, marker) -> five-class logits for the marked frame."""

    def __init__(self) -> None:
        super().__init__()
        from torchvision.models.video import mvit_v2_s

        encoder = mvit_v2_s()
        encoder.head[-1] = nn.Linear(encoder.head[-1].in_features, EMBED_DIM)
        stem = encoder.conv_proj
        self.encoder = encoder
        self.marker_stem = nn.Conv3d(1, stem.out_channels, stem.kernel_size, stem.stride, stem.padding, stem.dilation, bias=False)
        self.classifier = nn.Linear(EMBED_DIM, len(CLASSES))

    def forward(self, rgb: torch.Tensor, marker: torch.Tensor) -> torch.Tensor:
        """rgb (B, 3, 16, 224, 224) normalised, marker (B, 1, 16, 224, 224) of 0/1 -> logits (B, 5)."""
        if rgb.ndim != 5 or rgb.shape[1] != 3 or marker.shape != (rgb.shape[0], 1, *rgb.shape[2:]):
            raise ValueError(f"rgb {tuple(rgb.shape)} and marker {tuple(marker.shape)} do not fit")
        stem, marker_stem = self.encoder.conv_proj, self.marker_stem

        def add_marker(_module: nn.Module, _inputs: tuple, out: torch.Tensor) -> torch.Tensor:
            return out + marker_stem(marker.to(out.dtype))

        handle = stem.register_forward_hook(add_marker)
        try:
            embedding = self.encoder(rgb)
        finally:
            handle.remove()
        return self.classifier(embedding)

    def new_parameters(self) -> list[nn.Parameter]:
        """The parameters that are not part of the pretrained encoder: the marker stem and the five-class head."""
        return [*self.marker_stem.parameters(), *self.classifier.parameters()]

    def set_encoder_trainable(self, flag: bool) -> None:
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(bool(flag))


def read_state(weights: str | Path) -> dict[str, torch.Tensor]:
    """The tensors of a weight file: safetensors, or a plain PyTorch state dict read without running any pickled code."""
    path = Path(weights)
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        return load_file(str(path), device="cpu")
    return torch.load(path, map_location="cpu", weights_only=True)


def load_model(weights: str | Path, device: torch.device) -> WindowModel:
    """One trained seed, loaded strictly: every tensor of the file must fit the network and none may be missing."""
    model = WindowModel()
    model.load_state_dict(read_state(weights), strict=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model.eval().to(device)


def normalise(frames_u8: torch.Tensor) -> torch.Tensor:
    """(B, 16, H, W, 3) uint8 -> (B, 3, 16, H, W) float32."""
    if frames_u8.dtype != torch.uint8 or frames_u8.ndim != 5 or frames_u8.shape[-1] != 3:
        raise ValueError(f"expected uint8 (B, T, H, W, 3), got {tuple(frames_u8.shape)} {frames_u8.dtype}")
    mean = torch.tensor(MEAN, device=frames_u8.device).view(1, 3, 1, 1, 1)
    std = torch.tensor(STD, device=frames_u8.device).view(1, 3, 1, 1, 1)
    return (frames_u8.permute(0, 4, 1, 2, 3).float() - mean) / std


def marker_planes(slots: torch.Tensor) -> torch.Tensor:
    """(B,) marker positions -> (B, 1, 16, 224, 224) float video with the marked plane set to 1."""
    out = torch.zeros(slots.numel(), 1, N_SLOTS, SIZE, SIZE, device=slots.device)
    out[torch.arange(slots.numel(), device=slots.device), 0, slots] = 1.0
    return out
