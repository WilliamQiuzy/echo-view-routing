"""Shared test helpers: the parity file of the research code, synthetic pictures and a stand-in window model."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from echoview.evaluation.ev9v import recording_key
from echoview.model import MEAN, STD
from echoview.training import cache, train

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def no_download_outside_weights_tests(request, monkeypatch):
    """A test of the default suite must pass on a machine without access to the weights repository. Any download it
    tries fails loudly here, instead of passing on the one machine whose cache happens to hold the files."""
    if "weights" in request.keywords:
        return
    import huggingface_hub

    def refuse(**_kwargs):
        raise AssertionError("this test tried to download from the Hub: give it a stand-in model, or mark it `weights`")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", refuse)


@pytest.fixture(scope="session")
def parity() -> dict:
    """Inputs and outputs of the research code's window and decoder functions (see fixtures/make_fixtures_from_research_code.py)."""
    with np.load(FIXTURES / "research_parity.npz") as stored:
        return {key: stored[key] for key in stored.files}


TRAINING_NAMES = ("rec_a", "rec_b", "rec_c")


@pytest.fixture()
def tiny_training(tmp_path) -> tuple[Path, Path, Path]:
    """A frame cache of three recordings (two for training) and a plan of groups over the training ones."""
    images = tmp_path / "Images"
    items = []
    for r, (name, split) in enumerate(zip(TRAINING_NAMES, ("train", "train", "selection"))):
        (images / name).mkdir(parents=True)
        for k in range(20):
            Image.fromarray(np.full((240, 320, 3), 60 * r + k, np.uint8)).save(images / name / f"{3 * k}.jpg", quality=95)
        items.append({"key": recording_key(name), "split": split, "label": r % 2, "n_frames": 60})
    cache.build(images, tmp_path / "cache", workers=1, items=items)
    vid = np.zeros((4, 8, 16), np.int32)
    vid[:, 4:, 8:] = 1
    target = np.zeros((4, 8), np.int8)
    target[:, 4:] = 1
    plan = tmp_path / "plan.npz"
    np.savez_compressed(plan, recordings=np.array([recording_key(n) for n in TRAINING_NAMES[:2]]), vid=vid,
                        frame=np.tile(np.arange(16, dtype=np.int32) * 3, (4, 8, 1)),
                        gamma=np.ones((4, 8, 16), np.float32), marker=np.full((4, 8), 8, np.int8), target=target, kind=(vid.max(-1) > 0).astype(np.int8))
    start = tmp_path / "start.pt"
    torch.manual_seed(3)
    torch.save(train.initial_model(None, 0).state_dict(), start)
    return tmp_path / "cache", plan, start


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
