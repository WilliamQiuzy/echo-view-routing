"""Time-token supervision with the preserved CLS output (configs/experiments/video_dense_supervision_v1.yaml;
docs/superpowers/plans/2026-10-13-video-dense-supervision.md, Task 2).

DenseMViT is the raw-video context round's TargetMViT with no parameter added. Every arm runs forward_with_time:
  1. the full original TargetMViT.forward(rgb, marker), whose CLS logits are the model's output; a forward hook on
     encoder.norm, registered for this call only and removed in finally, stores a clone of the norm output X
     (B, 1 + 8 x 7 x 7, 768) with autograd and returns nothing, so token 0 and the CLS path are untouched;
  2. P = X[:, 1:] reshaped to (B, 8, 7, 7, 768) and averaged over space, read by the shared encoder.head (dropout and the
     768->512 projection) and the five-class classifier: z_time (B, 8, 5), one logit vector per final time position.
The time positions correspond to the input-slot anchors j * stride - padding + (kernel - 1) * dilation / 2 of the RGB
stem's time geometry (slots 0, 2, ..., 14); the final tokens have interacted with every time and the CLS token, so they
are not local per-frame features. The arms differ only in the loss outside the model: lambda 0 (CLS continuation), the
aligned anchor labels, or the same labels permuted within the window. Nothing here reads a label, a boundary or a name.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

import echo_routing.vit.target_mvit as tm
import echo_routing.vit.target_patch_readout as tp

ARMS = ("center_only", "aligned_dense", "shuffled_dense")
NAMES = {"center_only": "CLS continuation", "aligned_dense": "Aligned token supervision", "shuffled_dense": "Shuffled token supervision"}
LAMBDA = {"center_only": 0.0, "aligned_dense": 0.5, "shuffled_dense": 0.5}
EXPECTED_GRID = (8, 7, 7)
STEM_TIME = (3, 2, 1, 1)                      # conv_proj time kernel, stride, padding, dilation
ANCHORS = (0, 2, 4, 6, 8, 10, 12, 14)


def anchor_slots(kernel: int, stride: int, padding: int, dilation: int, n_time: int) -> list[int]:
    """Input-slot centre of every stem time position; integers inside the 16 slots or ValueError."""
    out = [j * int(stride) - int(padding) + (int(kernel) - 1) * int(dilation) / 2 for j in range(int(n_time))]
    if any(x != int(x) or not 0 <= x < tm.N_SLOTS for x in out):
        raise ValueError(f"time geometry {(kernel, stride, padding, dilation)} gives anchors {out}")
    return [int(x) for x in out]


def time_vectors(tokens: torch.Tensor, grid: tuple[int, int, int]) -> torch.Tensor:
    """(B, 1 + T'H'W', C) final tokens -> (B, T', C): the patch tokens averaged over space at every time position."""
    b, n, c = tokens.shape; t, h, w = grid
    if n != 1 + t * h * w:
        raise ValueError(f"{n} tokens do not fit the grid {grid} plus the class token")
    return tokens[:, 1:].reshape(b, t, h, w, c).mean(dim=(2, 3))


class DenseMViT(tm.TargetMViT):
    """TargetMViT whose training forward also returns 8 time logits from the shared head; no parameter is added."""

    def __init__(self, encoder, num_classes: int = tm.NUM_CLASSES) -> None:
        super().__init__(encoder, num_classes)
        self.grid, self.query_strides = tp.token_grid(encoder)
        conv = encoder.conv_proj
        self.stem_time = (int(conv.kernel_size[0]), int(conv.stride[0]), int(conv.padding[0]), int(conv.dilation[0]))
        if self.grid != EXPECTED_GRID or any(s[0] != 1 for s in self.query_strides) or self.stem_time != STEM_TIME:
            raise SystemExit(f"grid {self.grid}, query time strides {sorted({s[0] for s in self.query_strides})} or stem {self.stem_time} differ from the plan")
        self.anchors = anchor_slots(*self.stem_time, self.grid[0])
        if tuple(self.anchors) != ANCHORS:
            raise SystemExit(f"anchors {self.anchors} differ from the plan {ANCHORS}")

    def forward_with_time(self, rgb: torch.Tensor, marker: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(CLS logits (B, 5), time logits (B, 8, 5)); the CLS logits are exactly TargetMViT.forward's."""
        norm = self.encoder.norm
        if norm._forward_hooks:
            raise RuntimeError("encoder.norm already carries a forward hook")
        seen: list[torch.Tensor] = []

        def capture(_module, _inputs, out):
            seen.append(out.clone())

        handle = norm.register_forward_hook(capture)
        try:
            z_cls = self.forward(rgb, marker)
        finally:
            handle.remove()
        if len(seen) != 1:
            raise RuntimeError(f"the norm output was captured {len(seen)} times in one forward")
        z_time = self.classifier(self.encoder.head(time_vectors(seen[0], self.grid)))
        return z_cls, z_time


def dense_loss(z_cls: torch.Tensor, z_time: torch.Tensor, target: torch.Tensor, aux: torch.Tensor, lam: float) -> tuple[torch.Tensor, dict]:
    """(L_center + lam * L_time) / (1 + lam): CLS cross-entropy (batch mean) and the per-window mean over the 8 anchors."""
    b, t, k = z_time.shape
    if tuple(aux.shape) != (b, t):
        raise ValueError(f"aux labels {tuple(aux.shape)} do not fit ({b}, {t})")
    l_center = F.cross_entropy(z_cls, target)
    l_time = F.cross_entropy(z_time.reshape(-1, k), aux.reshape(-1), reduction="none").reshape(b, t).mean(1).mean()
    loss = (l_center + float(lam) * l_time) / (1.0 + float(lam))
    return loss, {"center": float(l_center.detach()), "time": float(l_time.detach())}


def build_dense_model(state: dict) -> DenseMViT:
    """A DenseMViT strictly loaded with a full TargetMViT state (encoder, projection, marker stem and head)."""
    model = DenseMViT(tm.build_source_encoder(None))
    model.load_state_dict(state, strict=True)
    return model
