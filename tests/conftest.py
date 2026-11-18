"""Shared test helpers: the parity file of the research code, synthetic pictures and a stand-in window model."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from echoview.model import MEAN, STD

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def parity() -> dict:
    """Inputs and outputs of the research code's window and decoder functions (see fixtures/make_fixtures_from_research_code.py)."""
    with np.load(FIXTURES / "research_parity.npz") as stored:
        return {key: stored[key] for key in stored.files}


class BrightnessModel(nn.Module):
    """Stands in for a trained window model: the view is read off the grey level of the marked frame.

    Grey level 0 gives view 0, 50 view 1, 100 view 2, 150 view 3, 200 view 4. `shift` moves the level a seed reads.
    """

    def __init__(self, shift: float = 0.0, sharpness: float = 8.0) -> None:
        super().__init__()
        self.shift, self.sharpness = shift, sharpness
        self.calls = 0

    def forward(self, rgb: torch.Tensor, marker: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        level = (rgb[:, 0] * STD[0] + MEAN[0]) * marker[:, 0]
        level = level.sum((1, 2, 3)) / marker[:, 0].sum((1, 2, 3)) + self.shift
        centres = torch.arange(5, device=rgb.device) * 50.0
        return -self.sharpness * ((level[:, None] - centres[None, :]) / 50.0) ** 2


def flat_frames(levels: list[int]) -> np.ndarray:
    """One 224 x 224 grey frame per level, as the model reads frames: (n, 224, 224, 3) uint8."""
    return np.stack([np.full((224, 224, 3), level, np.uint8) for level in levels])


def write_video(path: Path, levels: list[int], fps: float = 25.0, size: tuple[int, int] = (320, 240)) -> Path:
    """An mp4 file with one flat grey frame per level (a stand-in for a recording; no real data in the tests)."""
    import cv2

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        raise RuntimeError("OpenCV cannot write mp4 files on this machine")
    for level in levels:
        writer.write(np.full((size[1], size[0], 3), level, np.uint8))
    writer.release()
    return path
