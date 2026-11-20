"""The demo renderer: a stream with the timeline of its views under it."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

SPEC = importlib.util.spec_from_file_location("render_demo", Path(__file__).resolve().parents[1] / "tools" / "render_demo.py")
render_demo = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(render_demo)


@pytest.mark.parametrize("suffix", [".gif", ".webp", ".mp4"])
def test_every_frame_gets_a_picture_with_the_timeline_under_it(tmp_path, suffix):
    frames = np.random.default_rng(0).integers(0, 255, (12, 64, 64, 3), dtype=np.uint8)
    labels = np.array([0] * 5 + [-1] * 2 + [3] * 5)  # two views with an uncertain stretch between them
    out = render_demo.render(frames, labels, tmp_path / f"demo{suffix}", reference=np.array([0] * 6 + [3] * 6))
    assert out.is_file() and out.stat().st_size > 0
    if suffix != ".mp4":
        with Image.open(out) as picture:
            assert picture.n_frames == 12 and picture.size[0] == 128 and picture.size[1] > 128  # the frames enlarged, the lanes below


def test_one_label_per_frame_is_needed(tmp_path):
    frames = np.zeros((4, 32, 32, 3), np.uint8)
    with pytest.raises(ValueError):
        render_demo.render(frames, np.zeros(3, int), tmp_path / "x.gif")
