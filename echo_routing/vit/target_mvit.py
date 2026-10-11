"""Target-conditioned MViTv2-S for target-time view recognition from raw video
(configs/experiments/raw_video_context_v1.yaml; docs/superpowers/plans/2026-10-10-raw-video-recognition.md).

Model. The video encoder is torchvision's mvit_v2_s built as EchoPrime's load_for_finetuning.py builds it: the default
constructor, the last head layer replaced by Linear(768, 512), and the EchoPrime encoder weights loaded strictly. That
512-d projection is kept, and a new Linear(512, 5) reads it. A zero-initialised Conv3d(1, 96) with the geometry of the
RGB patch embedding (conv_proj) reads a 0/1 target-marker video. Its output is added to conv_proj's output by a
forward hook that exists only while TargetMViT.forward runs and is removed in `finally`. With zero marker weights the
512-d output equals the source encoder's. Freezing the encoder only switches its parameters off; the forward always
keeps autograd, so the marker can learn through the frozen trunk.

Inputs. A window holds 16 sample positions s[0:16] around a requested sample c
(echo_routing.evaluate.window_target.window_frames, clamped at sequence ends). The target slots are the positions with
s == c, and the marker plane sits at the middle one (sorted index len // 2). Three input conditions use the same windows.
  Ordered context   the 16 frames in time order.
  Shuffled context  the same 16 frames; the target slots stay, and the other slots are permuted among themselves.
  Target-only       the target frame, with its gamma, in all 16 slots.
The marker slot is the same in all three. Nothing in this module reads a class label.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

N_SLOTS = 16
NUM_CLASSES = 5
EMBED_DIM = 512
GROUPS = ("Target-only", "Shuffled context", "Ordered context")
KEYS = {"Target-only": "target_only", "Shuffled context": "shuffled", "Ordered context": "ordered"}
TRAIN_PERM_BASE = 20261010
MEAN = (29.110628, 28.076836, 29.096405)   # EchoPrime, pixel values 0-255 (no division by 255)
STD = (47.989223, 46.456997, 47.20083)


# ----------------------------------------------------------------------------------------------- windows
def target_slots(s: np.ndarray, c: int) -> np.ndarray:
    """Positions of the window whose sample index is the requested sample c (sorted)."""
    slots = np.flatnonzero(np.asarray(s) == int(c))
    if slots.size == 0:
        raise ValueError(f"sample {c} is not in the window {np.asarray(s).tolist()}")
    return slots


def marker_slot(s: np.ndarray, c: int) -> int:
    """The marker position: the middle of the target slots (sorted index len // 2)."""
    slots = target_slots(s, c)
    return int(slots[slots.size // 2])


def train_rng(seed: int, update: int, window: int) -> np.random.Generator:
    """The data RNG of one training window; independent of the model RNG."""
    return np.random.default_rng([TRAIN_PERM_BASE, int(seed), int(update), int(window)])


def inference_rng(seed: int, sequence_id: str, centre: int) -> np.random.Generator:
    """The fixed RNG of one inference window: the first 8 bytes (big endian) of SHA256('seed|sequence id|centre')."""
    digest = hashlib.sha256(f"{int(seed)}|{sequence_id}|{int(centre)}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


def shuffle_order(rng: np.random.Generator, s: np.ndarray, c: int) -> np.ndarray:
    """order[i] = the time-ordered slot shown at slot i: target slots map to themselves, the others are permuted."""
    s = np.asarray(s)
    target_slots(s, c)
    order = np.arange(s.size)
    other = np.flatnonzero(s != int(c))
    if other.size:
        order[other] = other[rng.permutation(other.size)]
    return order


def group_order(group: str, s: np.ndarray, c: int, shuffled: np.ndarray) -> np.ndarray:
    """Index into the time-ordered slots for each input slot of a group."""
    if group == "Ordered context":
        return np.arange(len(s))
    if group == "Shuffled context":
        order = np.asarray(shuffled, dtype=int)
        keep = target_slots(s, c)
        if sorted(order.tolist()) != list(range(len(s))) or not np.array_equal(order[keep], keep):
            raise ValueError("a shuffled order must be a permutation that keeps every target slot")
        return order
    if group == "Target-only":
        return np.full(len(s), marker_slot(s, c), dtype=int)
    raise ValueError(f"unknown group {group!r}")


def marker_planes(slots: torch.Tensor, n_slots: int = N_SLOTS, size: int = 224) -> torch.Tensor:
    """(B,) marker slots -> (B, 1, n_slots, size, size) float 0/1 video with one whole plane set to 1."""
    slots = torch.as_tensor(slots, dtype=torch.long)
    if slots.ndim != 1 or bool(((slots < 0) | (slots >= n_slots)).any()):
        raise ValueError(f"marker slots must be a vector in [0, {n_slots})")
    out = torch.zeros(slots.numel(), 1, n_slots, size, size, device=slots.device)
    out[torch.arange(slots.numel(), device=slots.device), 0, slots] = 1.0
    return out


# ----------------------------------------------------------------------------------------------- pixels
def frame_hash(u8: np.ndarray) -> bytes:
    """16-byte digest of one decoded uint8 frame (shape and bytes)."""
    a = np.ascontiguousarray(u8, dtype=np.uint8)
    return hashlib.blake2b(str(a.shape).encode() + a.tobytes(), digest_size=16).digest()


def window_digest(slot_hashes) -> bytes:
    """SHA256 over a window's 16 slot digests in input order."""
    h = hashlib.sha256()
    for x in slot_hashes:
        h.update(bytes(x))
    return h.digest()


def to_rgb(u8: torch.Tensor) -> torch.Tensor:
    """(B, T, H, W, 3) uint8 frames -> (B, 3, T, H, W) float normalised with EchoPrime's 0-255 mean and std."""
    if u8.dtype != torch.uint8 or u8.ndim != 5 or u8.shape[-1] != 3:
        raise ValueError(f"expected uint8 (B, T, H, W, 3), got {tuple(u8.shape)} {u8.dtype}")
    mean = torch.tensor(MEAN, device=u8.device).view(1, 3, 1, 1, 1)
    std = torch.tensor(STD, device=u8.device).view(1, 3, 1, 1, 1)
    return (u8.permute(0, 4, 1, 2, 3).float() - mean) / std


# ----------------------------------------------------------------------------------------------- model
def build_source_encoder(checkpoint: str | Path | None) -> nn.Module:
    """torchvision mvit_v2_s with head[-1] = Linear(768, 512), strictly loaded with the EchoPrime encoder weights
    (third_party/EchoPrime/load_for_finetuning.py at the pinned commit). None leaves the projection randomly
    initialised (tests only)."""
    from torchvision.models.video import mvit_v2_s

    net = mvit_v2_s()
    net.head[-1] = nn.Linear(net.head[-1].in_features, EMBED_DIM)
    if checkpoint is not None:
        state = torch.load(Path(checkpoint), map_location="cpu", weights_only=True)
        net.load_state_dict(state, strict=True)
    return net


class TargetMViT(nn.Module):
    """forward(rgb, marker) -> five-class logits for the marked target time."""

    def __init__(self, encoder: nn.Module, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        conv = encoder.conv_proj
        if conv.in_channels != 3 or encoder.head[-1].out_features != EMBED_DIM:
            raise ValueError("the encoder must be the three-channel MViT with its 512-d projection")
        self.encoder = encoder
        self.marker_stem = nn.Conv3d(1, conv.out_channels, conv.kernel_size, conv.stride, conv.padding, conv.dilation, bias=False)
        nn.init.zeros_(self.marker_stem.weight)
        self.classifier = nn.Linear(EMBED_DIM, num_classes)

    def embed(self, rgb: torch.Tensor, marker: torch.Tensor) -> torch.Tensor:
        """(B, 3, 16, 224, 224) normalised RGB and (B, 1, 16, 224, 224) marker -> (B, 512) encoder output."""
        if rgb.ndim != 5 or rgb.shape[1] != 3 or marker.shape != (rgb.shape[0], 1, *rgb.shape[2:]):
            raise ValueError(f"rgb {tuple(rgb.shape)} and marker {tuple(marker.shape)} do not fit")
        conv = self.encoder.conv_proj
        if conv._forward_hooks:
            raise RuntimeError("conv_proj already carries a forward hook")
        stem = self.marker_stem

        def add_marker(_module, _inputs, out):
            return out + stem(marker.to(out.dtype))

        handle = conv.register_forward_hook(add_marker)
        try:
            return self.encoder(rgb)
        finally:
            handle.remove()

    def forward(self, rgb: torch.Tensor, marker: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.embed(rgb, marker))

    def new_parameters(self) -> list[nn.Parameter]:
        return [*self.marker_stem.parameters(), *self.classifier.parameters()]

    def encoder_parameters(self) -> list[nn.Parameter]:
        return list(self.encoder.parameters())

    def set_encoder_trainable(self, flag: bool) -> None:
        for p in self.encoder.parameters():
            p.requires_grad_(bool(flag))

    def new_state(self) -> dict:
        return {"marker_stem": self.marker_stem.state_dict(), "classifier": self.classifier.state_dict()}

    def load_new_state(self, state: dict) -> None:
        self.marker_stem.load_state_dict(state["marker_stem"], strict=True)
        self.classifier.load_state_dict(state["classifier"], strict=True)


def initial_new_state(seed: int, num_classes: int = NUM_CLASSES, stem_channels: int = 96,
                      kernel: tuple[int, int, int] = (3, 7, 7)) -> dict:
    """The seed's common initial state of the new parameters: classifier weight N(0, 0.02^2) from a generator seeded
    with the seed, bias 0, marker stem all zero."""
    g = torch.Generator().manual_seed(int(seed))
    weight = torch.randn(num_classes, EMBED_DIM, generator=g) * 0.02
    return {"marker_stem": {"weight": torch.zeros(stem_channels, 1, *kernel)},
            "classifier": {"weight": weight, "bias": torch.zeros(num_classes)}}


def build_model(checkpoint: str | Path | None, init: dict) -> TargetMViT:
    model = TargetMViT(build_source_encoder(checkpoint))
    model.load_new_state(init)
    return model
