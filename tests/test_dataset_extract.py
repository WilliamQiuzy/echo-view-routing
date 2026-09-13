from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

torch = pytest.importorskip("torch")

from echo_routing.features.cache import load_video_features, read_index, variant_dir  # noqa: E402
from echo_routing.features.dataset import EpochFrameDataset, FixedFrameDataset  # noqa: E402
from echo_routing.features.encoder import FrameEncoder, checkpoint_hash, load_checkpoint, save_checkpoint  # noqa: E402
from echo_routing.features.extract import extract_split, extract_video  # noqa: E402
from echo_routing.ingest.frames import VideoRecord  # noqa: E402


@pytest.fixture(scope="module")
def tiny_images(tmp_path_factory) -> tuple[Path, list[VideoRecord]]:
    root = tmp_path_factory.mktemp("Images")
    recs = []
    for i, (vid, n, label) in enumerate([("va", 12, 0), ("vb", 9, 1), ("vc", 30, 0)]):
        d = root / vid; d.mkdir()
        for f in range(1, n + 1):
            Image.new("RGB", (32, 24), (f * 7 % 255, 50, 100)).save(d / f"frame_{f:06d}.jpg")
        recs.append(VideoRecord(vid, d, n, label, "train"))
    return root, recs


def test_epoch_dataset_balanced_and_resampled(tiny_images):
    _, recs = tiny_images
    ds = EpochFrameDataset(recs, frames_per_video=4, image_size=32, videos_per_epoch=6, seed=0, augment=True)
    assert len(ds) == 6 * 4
    x, y, v = ds[0]
    assert x.shape == (3, 32, 32) and y in (0, 1) and 0 <= v < 3
    before = [(s.video_idx, s.frame_idx) for s in ds.samples]
    ds.resample(1)
    assert [(s.video_idx, s.frame_idx) for s in ds.samples] != before


def test_fixed_dataset_deterministic(tiny_images):
    _, recs = tiny_images
    ds = FixedFrameDataset(recs, frames_per_video=3, image_size=32)
    assert len(ds) == 9  # 3 videos x 3 frames (vb has 9 frames >= 3)
    x, y, v = ds[4]
    assert x.shape == (3, 32, 32) and v == 1


def test_encoder_checkpoint_roundtrip(tmp_path: Path):
    m = FrameEncoder(num_classes=2, backbone="resnet18", pretrained=False).eval()
    out = m(torch.zeros(2, 3, 32, 32)); assert out.shape == (2, 2)
    logits, feats = m.forward_with_features(torch.zeros(1, 3, 32, 32)); assert feats.shape == (1, 512)
    p = save_checkpoint(tmp_path / "c.pt", m, {"epoch": 3, "classes": ["a", "b"]})
    m2, meta = load_checkpoint(p)
    assert meta["epoch"] == 3 and m2.num_classes == 2 and len(checkpoint_hash(p)) == 12
    with pytest.raises(ValueError):
        FrameEncoder(2, backbone="vgg")


def test_extract_video_and_split(tiny_images, tmp_path: Path):
    root, recs = tiny_images
    model = FrameEncoder(2, pretrained=False).eval(); dev = torch.device("cpu")
    vf = extract_video(model, dev, recs[2].frames_dir, "vc", 30, fps=30.0, target_hz=10.0, image_size=32, gamma=1.0,
                       batch_size=8, num_workers=0)
    assert vf.n == 10 and vf.prob.shape == (10, 2) and np.allclose(vf.prob.sum(1), 1) and vf.feat.shape == (10, 512)
    rows = pd.DataFrame([{"video_id": r.video_id, "split": r.split, "raw_label": "X", "family5_index": r.label_index,
                          "frames_dir": str(r.frames_dir), "n_frames": r.n_frames, "fps_playback": 30.0} for r in recs])
    out = extract_split(model, dev, rows, "family5_index", tmp_path, "abc", "gamma090", 10.0, 32, batch_size=8, num_workers=0)
    assert out == variant_dir(tmp_path, "abc", "gamma090")
    idx = read_index(out)
    assert idx["n_samples"].tolist() == [4, 3, 10] and (out / "meta.json").is_file()
    loaded = load_video_features(out / "va.npz")
    assert loaded.video_id == "va" and loaded.frame_idx.tolist() == [0, 3, 6, 9]
    # second call is a cache hit (no overwrite) and keeps the index consistent
    extract_split(model, dev, rows, "family5_index", tmp_path, "abc", "gamma090", 10.0, 32, batch_size=8, num_workers=0)
    assert read_index(out)["n_samples"].tolist() == [4, 3, 10]
