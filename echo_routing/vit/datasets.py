"""Datasets that return JPEG *bytes* (decoded later on the GPU) for frame- and clip-level ViT training.

Sampling mirrors the B0 recipe: every epoch draws one cine per training cine, class-balanced with replacement
(features.sampling.balanced_video_weights); frame models take `frames_per_video` timestamp-spaced frames per draw,
clip models one random clip per draw. The appearance augmentation is the same gamma edit (U[0.9, 1.1]).

Transition-aware clips (mix_prob > 0): a clip may join the first k frames of one cine's clip to the first 16-k
frames of a second, independently drawn training cine (k uniform in 1..15, one gamma edit per fragment), mirroring
the view changes a 1.6 s evaluation window straddles in a routed stream. Each frame keeps its own label.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from echo_routing.features.sampling import balanced_video_weights, spaced_indices
from echo_routing.ingest.frames import VideoRecord, list_frame_paths
from echo_routing.vit.clips import random_clip, spaced_clips
from echo_routing.vit.gpu_frames import read_bytes

GAMMA_RANGE = (0.9, 1.1)


@dataclass(frozen=True)
class Segment:
    video_idx: int
    frames: tuple[int, ...]
    gamma: float


@dataclass(frozen=True)
class Item:
    segments: tuple[Segment, ...]


@dataclass(frozen=True)
class Batch:
    datas: list            # flat list of JPEG bytes, item-major
    labels: torch.Tensor   # (B,) label of the item's first segment (the only one unless mixed)
    frame_labels: torch.Tensor  # (B, T) label of every frame
    vids: torch.Tensor     # (B,) video index of the first segment
    gammas: list           # per frame
    t: int                 # frames per item


def _item(video_idx: int, frames, gamma: float) -> Item:
    return Item((Segment(int(video_idx), tuple(int(f) for f in frames), float(gamma)),))


class _BytesBase(Dataset):
    def __init__(self, records: Sequence[VideoRecord]) -> None:
        self.records = list(records)
        self.items: list[Item] = []
        self._paths: dict[int, list] = {}

    def _path(self, video_idx: int, frame_idx: int):
        if video_idx not in self._paths:
            self._paths[video_idx] = list_frame_paths(self.records[video_idx].frames_dir)
        paths = self._paths[video_idx]
        return paths[min(frame_idx, len(paths) - 1)]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        datas, labels, gammas = [], [], []
        for s in self.items[i].segments:
            datas.extend(read_bytes(self._path(s.video_idx, f)) for f in s.frames)
            labels.extend([self.records[s.video_idx].label_index] * len(s.frames))
            gammas.extend([s.gamma] * len(s.frames))
        return datas, labels, self.items[i].segments[0].video_idx, gammas


def _balanced_draws(records: Sequence[VideoRecord], n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.choice(len(records), size=n, replace=True, p=balanced_video_weights([r.label_index for r in records]))


class EpochFrames(_BytesBase):
    """Frame model training set (one frame per item)."""

    def __init__(self, records, frames_per_video: int, videos_per_epoch: int | None, seed: int, augment: bool = True) -> None:
        super().__init__(records)
        self.frames_per_video, self.seed, self.augment = frames_per_video, seed, augment
        self.videos_per_epoch = videos_per_epoch or len(self.records)
        self.resample(0)

    def resample(self, epoch: int) -> None:
        rng = np.random.default_rng(self.seed + epoch)
        items = []
        for vi in _balanced_draws(self.records, self.videos_per_epoch, rng):
            for fi in spaced_indices(self.records[vi].n_frames, self.frames_per_video, rng):
                items.append(_item(vi, (fi,), float(rng.uniform(*GAMMA_RANGE)) if self.augment else 1.0))
        self.items = items


class EpochClips(_BytesBase):
    """Clip model training set: one random clip per class-balanced draw (one gamma per fragment), optionally
    transition-aware (see module docstring)."""

    def __init__(self, records, n_frames: int, step: int, videos_per_epoch: int | None, seed: int, augment: bool = True,
                 mix_prob: float = 0.0) -> None:
        super().__init__(records)
        if not 0.0 <= mix_prob <= 1.0:
            raise ValueError("mix_prob must be in [0, 1]")
        self.n_frames, self.step, self.seed, self.augment, self.mix_prob = n_frames, step, seed, augment, mix_prob
        self.videos_per_epoch = videos_per_epoch or len(self.records)
        self.resample(0)

    def _gamma(self, rng) -> float:
        return float(rng.uniform(*GAMMA_RANGE)) if self.augment else 1.0

    def resample(self, epoch: int) -> None:
        rng = np.random.default_rng(self.seed + epoch)
        draws = _balanced_draws(self.records, self.videos_per_epoch, rng)
        # partners are drawn only when mixing, so mix_prob = 0 reproduces the plain sampler's random stream exactly
        partners = _balanced_draws(self.records, self.videos_per_epoch, rng) if self.mix_prob > 0 else draws
        items = []
        for vi, vj in zip(draws, partners):
            a = random_clip(self.records[vi].n_frames, self.n_frames, self.step, rng)
            if self.mix_prob > 0 and rng.random() < self.mix_prob:
                k = int(rng.integers(1, self.n_frames))
                b = random_clip(self.records[vj].n_frames, self.n_frames, self.step, rng)
                items.append(Item((Segment(int(vi), tuple(int(x) for x in a[:k]), self._gamma(rng)),
                                   Segment(int(vj), tuple(int(x) for x in b[: self.n_frames - k]), self._gamma(rng)))))
            else:
                items.append(_item(vi, a, self._gamma(rng)))
        self.items = items


class FixedFrames(_BytesBase):
    """Deterministic evaluation: `frames_per_video` centred spaced frames of every cine (as FixedFrameDataset)."""

    def __init__(self, records, frames_per_video: int) -> None:
        super().__init__(records)
        self.items = [_item(vi, (fi,), 1.0) for vi, r in enumerate(self.records) for fi in spaced_indices(r.n_frames, frames_per_video)]


class FixedClips(_BytesBase):
    """Deterministic evaluation: k evenly spaced clips of every cine."""

    def __init__(self, records, n_frames: int, step: int, clips_per_video: int) -> None:
        super().__init__(records)
        self.items = [_item(vi, c, 1.0) for vi, r in enumerate(self.records) for c in spaced_clips(r.n_frames, n_frames, step, clips_per_video)]


def collate_bytes(batch) -> Batch:
    datas, gammas = [], []
    for d, _, _, g in batch:
        datas.extend(d); gammas.extend(g)
    frame_labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    return Batch(datas, frame_labels[:, 0].clone(), frame_labels, torch.tensor([b[2] for b in batch], dtype=torch.long),
                 gammas, frame_labels.shape[1])


def soft_targets(frame_labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """(B, T) frame labels -> (B, C) share of the clip's frames per class (one-hot for single-view clips)."""
    return torch.nn.functional.one_hot(frame_labels, num_classes).float().mean(1)
