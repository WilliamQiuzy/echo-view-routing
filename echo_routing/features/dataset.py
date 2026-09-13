"""PyTorch datasets built on echo_routing.data.frames (server-side use)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from echo_routing.features.sampling import balanced_video_weights, spaced_indices
from echo_routing.ingest.frames import VideoRecord, list_frame_paths, load_frame


@dataclass(frozen=True)
class Sample:
    video_idx: int
    frame_idx: int


class EpochFrameDataset(Dataset):
    """Cine-balanced frame sampler: each epoch draws `videos_per_epoch` cines (class-balanced,
    with replacement) and `frames_per_video` timestamp-spaced frames from each."""

    def __init__(self, records: Sequence[VideoRecord], frames_per_video: int, image_size: int,
                 videos_per_epoch: int | None, seed: int, augment: bool = False) -> None:
        self.records = list(records)
        self.frames_per_video = frames_per_video
        self.image_size = image_size
        self.videos_per_epoch = videos_per_epoch or len(self.records)
        self.seed = seed
        self.augment = augment
        self._paths: dict[int, list] = {}
        self.samples: list[Sample] = []
        self.resample(epoch=0)

    def resample(self, epoch: int) -> None:
        rng = np.random.default_rng(self.seed + epoch)
        weights = balanced_video_weights([r.label_index for r in self.records])
        chosen = rng.choice(len(self.records), size=self.videos_per_epoch, replace=True, p=weights)
        samples: list[Sample] = []
        for vi in chosen:
            for fi in spaced_indices(self.records[vi].n_frames, self.frames_per_video, rng):
                samples.append(Sample(int(vi), int(fi)))
        self.samples = samples

    def _frame_path(self, video_idx: int, frame_idx: int):
        if video_idx not in self._paths:
            self._paths[video_idx] = list_frame_paths(self.records[video_idx].frames_dir)
        paths = self._paths[video_idx]
        return paths[min(frame_idx, len(paths) - 1)]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        s = self.samples[i]
        rec = self.records[s.video_idx]
        gamma = 1.0
        if self.augment:
            gamma = float(np.random.default_rng(self.seed * 7919 + i).uniform(0.9, 1.1))
        x = load_frame(self._frame_path(s.video_idx, s.frame_idx), self.image_size, gamma=gamma)
        return torch.from_numpy(x), rec.label_index, s.video_idx


class FixedFrameDataset(Dataset):
    """Deterministic evaluation set: `frames_per_video` centred spaced frames from every cine."""

    def __init__(self, records: Sequence[VideoRecord], frames_per_video: int, image_size: int) -> None:
        self.records = list(records)
        self.image_size = image_size
        self.samples = [
            Sample(vi, int(fi))
            for vi, r in enumerate(self.records)
            for fi in spaced_indices(r.n_frames, frames_per_video)
        ]
        self._paths: dict[int, list] = {}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        s = self.samples[i]
        if s.video_idx not in self._paths:
            self._paths[s.video_idx] = list_frame_paths(self.records[s.video_idx].frames_dir)
        paths = self._paths[s.video_idx]
        x = load_frame(paths[min(s.frame_idx, len(paths) - 1)], self.image_size)
        return torch.from_numpy(x), self.records[s.video_idx].label_index, s.video_idx
