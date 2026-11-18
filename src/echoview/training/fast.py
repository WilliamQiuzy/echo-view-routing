"""A hook-free forward of the window model and a lean weight average, for training speed.

`WindowModel.forward` adds the marker stem's output to the patch embedding through a forward hook that it registers
and removes on every call. That is exact but opaque to torch.compile. `embed_direct` writes the same computation out
(torchvision's MViT.forward with the marker added after the patch embedding); a test checks that both give the same
output.
"""
from __future__ import annotations

import torch
from torch import nn


def embed_direct(model: nn.Module, rgb: torch.Tensor, marker: torch.Tensor) -> torch.Tensor:
    """(B, 3, 16, 224, 224) input and (B, 1, 16, 224, 224) marker -> the (B, 512) embedding."""
    encoder = model.encoder
    x = encoder.conv_proj(rgb)
    x = x + model.marker_stem(marker.to(x.dtype))
    x = x.flatten(2).transpose(1, 2)
    x = encoder.pos_encoding(x)
    thw = (encoder.pos_encoding.temporal_size, *encoder.pos_encoding.spatial_size)
    for block in encoder.blocks:
        x, thw = block(x, thw)
    x = encoder.norm(x)
    return encoder.head(x[:, 0])


class Direct(nn.Module):
    """forward(rgb, marker) -> logits through the hook-free path; shares the wrapped model's parameters."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, rgb: torch.Tensor, marker: torch.Tensor) -> torch.Tensor:
        return self.model.classifier(embed_direct(self.model, rgb, marker))


class Average:
    """Exponential moving average of a model's parameters, updated with one multi-tensor call per step."""

    def __init__(self, model: nn.Module) -> None:
        self.names = [name for name, _ in model.named_parameters()]
        self.live = [p for _, p in model.named_parameters()]
        self.mean = [p.detach().clone() for p in self.live]

    @torch.no_grad()
    def update(self, decay: float) -> None:
        if decay <= 0.0:
            for mean, live in zip(self.mean, self.live):
                mean.copy_(live)
        else:
            torch._foreach_lerp_(self.mean, [p.detach() for p in self.live], 1.0 - decay)

    def state(self) -> list[torch.Tensor]:
        """A copy of the averages, in the order of the model's parameters (for a checkpoint)."""
        return [mean.detach().clone() for mean in self.mean]

    @torch.no_grad()
    def load(self, means: list[torch.Tensor]) -> None:
        """Takes the averages a checkpoint stored."""
        if len(means) != len(self.mean):
            raise ValueError(f"the stored average holds {len(means)} tensors, the model {len(self.mean)}")
        for mean, stored in zip(self.mean, means):
            mean.copy_(stored)

    def state_dict(self, model: nn.Module) -> dict:
        state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        for name, mean in zip(self.names, self.mean):
            state[name] = mean.clone()
        return state
