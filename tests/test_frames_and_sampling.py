from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from echo_routing.errors import FrameError
from echo_routing.features.sampling import balanced_video_weights, spaced_indices, stride_indices
from echo_routing.ingest.frames import apply_gamma, letterbox, list_frame_paths, load_frame, to_tensor_array


def test_spaced_indices_deterministic_and_in_range():
    idx = spaced_indices(100, 4)
    assert idx.tolist() == [12, 37, 62, 87]
    assert spaced_indices(3, 10).tolist() == [0, 1, 2]  # k clipped to n
    rnd = spaced_indices(100, 4, np.random.default_rng(0))
    assert rnd.size == 4 and np.all(np.diff(rnd) > 0) and rnd.max() < 100
    with pytest.raises(FrameError):
        spaced_indices(0, 4)


def test_stride_indices_10hz_from_30fps():
    assert stride_indices(10, 30.0, 10.0).tolist() == [0, 3, 6, 9]
    with pytest.raises(FrameError):
        stride_indices(10, 0, 10)


def test_balanced_weights_equalise_classes():
    w = balanced_video_weights([0, 0, 0, 1])
    assert w.sum() == pytest.approx(1.0) and w[3] == pytest.approx(0.5)


def test_letterbox_and_gamma_and_tensor(tmp_path: Path):
    img = Image.new("RGB", (320, 240), (128, 128, 128))
    lb = letterbox(img, 224)
    assert lb.size == (224, 224) and lb.getpixel((0, 0)) == (0, 0, 0)  # padded rows are black
    assert lb.getpixel((112, 112)) == (128, 128, 128)
    dark = apply_gamma(img, 1.5).getpixel((0, 0))[0]
    bright = apply_gamma(img, 0.7).getpixel((0, 0))[0]
    assert dark < 128 < bright and apply_gamma(img, 1.0) is img
    with pytest.raises(ValueError):
        apply_gamma(img, 0)
    arr = to_tensor_array(lb)
    assert arr.shape == (3, 224, 224) and arr.dtype == np.float32
    p = tmp_path / "f.jpg"; img.save(p)
    assert load_frame(p, 64).shape == (3, 64, 64)


def test_list_frame_paths_sorted_and_errors(tmp_path: Path):
    d = tmp_path / "vid"; d.mkdir()
    for i in (3, 1, 2):
        (d / f"frame_{i:06d}.jpg").write_bytes(b"")
    (d / "notes.txt").write_text("x")
    assert [p.name for p in list_frame_paths(d)] == ["frame_000001.jpg", "frame_000002.jpg", "frame_000003.jpg"]
    empty = tmp_path / "empty"; empty.mkdir()
    with pytest.raises(FrameError):
        list_frame_paths(empty)
