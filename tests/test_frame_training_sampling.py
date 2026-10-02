"""Per-epoch resampling must reach the DataLoader workers (2026-10-02 plan, Task 1).

EpochFrameDataset.resample() changes the sample list in the main process. These checks read the (video, frame) pairs
that actual worker batches return, not the main-process list.
"""
from __future__ import annotations

import pytest
from PIL import Image
from torch.utils.data import DataLoader

from echo_routing.features.dataset import EpochFrameDataset, RateFrameDataset
from echo_routing.features.train_encoder import train_loader_kwargs
from echo_routing.ingest.frames import VideoRecord


@pytest.fixture()
def recs(tmp_path):
    out = []
    for k, (n, lab) in enumerate([(40, 0), (50, 1), (60, 0), (45, 1)]):
        d = tmp_path / f"v{k}"; d.mkdir()
        for f in range(n):
            Image.new("RGB", (8, 8), (f * 5 % 255, k * 40, 0)).save(d / f"{f}.jpg")
        out.append(VideoRecord(f"v{k}", d, n, lab, "train"))
    return out


def _epochs(ds: EpochFrameDataset, loader_kw: dict, epochs: int = 2):
    """(keys returned by workers, keys in the main process) for each epoch."""
    dl = DataLoader(ds, batch_size=4, shuffle=False, **loader_kw)
    got, expected = [], []
    for e in range(epochs):
        ds.resample(e)
        expected.append(sorted((s.video_idx, s.frame_idx) for s in ds.samples))
        got.append(sorted((int(v), int(f)) for _, _, vv, ff in dl for v, f in zip(vv, ff)))
    return got, expected


def test_persistent_workers_keep_serving_the_first_epoch(recs):
    ds = EpochFrameDataset(recs, frames_per_video=4, image_size=8, videos_per_epoch=8, seed=0)
    got, expected = _epochs(ds, dict(num_workers=2, persistent_workers=True))
    assert expected[0] != expected[1]          # the main process did resample
    assert got[0] == got[1] == expected[0]     # but the workers never saw it: the defect


def test_training_loader_restarts_workers_each_epoch(recs):
    ds = EpochFrameDataset(recs, frames_per_video=4, image_size=8, videos_per_epoch=8, seed=0)
    kw = train_loader_kwargs(num_workers=2, cuda=False)
    assert kw["persistent_workers"] is False
    got, expected = _epochs(ds, kw)
    assert got == expected and got[0] != got[1]


def test_rate_dataset_uses_the_10hz_grid_in_order(recs):
    ds = RateFrameDataset(recs[:1], image_size=8, hz=10.0, fps=30.0)
    assert [s.frame_idx for s in ds.samples] == list(range(0, 40, 3))
    x, y, v = ds[0]
    assert x.shape == (3, 8, 8) and y == 0 and v == 0
