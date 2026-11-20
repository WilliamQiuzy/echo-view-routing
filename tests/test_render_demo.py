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


def test_a_run_narrower_than_a_pixel_is_still_drawn(tmp_path):
    """60 s of stream on a lane 104 pixels wide: a run of one sample has less than a pixel."""
    labels = np.zeros(600, int)
    labels[[1, 77, 300, 598]] = 1
    labels[450] = -1
    out = render_demo.render(np.zeros((600, 64, 64, 3), np.uint8), labels, tmp_path / "long.gif")
    with Image.open(out) as picture:  # the writer joins pictures that are equal, so there are fewer than 600
        assert picture.n_frames > 1 and picture.size[0] == 128


def test_the_output_format_and_the_scale_are_checked_before_anything_is_drawn(tmp_path):
    frames, labels = np.zeros((4, 32, 32, 3), np.uint8), np.zeros(4, int)
    for name, scale in (("x.jpg", 2), ("x", 2), ("x.gif", 0)):
        with pytest.raises(ValueError, match="must end in one of"):
            render_demo.render(frames, labels, tmp_path / name, scale=scale)
    assert not list(tmp_path.iterdir())


def test_the_command_refuses_bad_arguments_before_it_reads_or_runs_anything(tmp_path, monkeypatch):
    import echoview

    def never(*args, **kwargs):
        raise AssertionError("nothing may be read and no model may be loaded for arguments that are refused")

    monkeypatch.setattr(echoview, "read_recording", never)
    monkeypatch.setattr(echoview, "load", never)
    clip = tmp_path / "clip.gif"
    clip.write_bytes(b"the recording")
    cases = ((["--out", str(clip)], "one of the inputs"), (["--out", str(tmp_path / "demo.png")], "must end in one of"),
             (["--out", str(tmp_path / "d.gif"), "--reference", "A4C,PLAX"], "for each of the 1 inputs"),
             (["--out", str(tmp_path / "d.gif"), "--reference", "A3C"], "--reference needs one of"),
             (["--out", str(tmp_path / "d.gif"), "--scale", "0"], "--scale must be"), (["--out", str(tmp_path / "d.gif"), "--seconds", "inf"], "--seconds"))
    for arguments, message in cases:
        with pytest.raises(SystemExit, match=message):
            render_demo.main([str(clip), *arguments])
    assert clip.read_bytes() == b"the recording"


def test_the_uncertain_option_segments_with_the_released_rule(tmp_path, monkeypatch):
    import echoview
    from echoview import uncertain

    asked = []

    class System:
        def segment(self, frames, rule=None):
            asked.append(rule)
            labels = np.array([0] * 10 + [-1] * 6 + [2] * 14) if rule is not None else np.array([0] * 13 + [2] * 17)
            return echoview.Segmentation(labels=labels, probabilities=np.full((30, 5), 0.2), segments=())

    monkeypatch.setattr(echoview, "read_recording", lambda path: echoview.Recording("clip", np.zeros((30, 224, 224, 3), np.uint8), 10.0, 30, "test"))
    monkeypatch.setattr(echoview, "load", lambda weights, device="auto": System())
    for arguments in ([], ["--uncertain"]):
        assert render_demo.main(["clip.mp4", "--out", str(tmp_path / "d.gif"), *arguments]) == 0
    assert asked == [None, uncertain.released()]


def test_a_recording_that_cannot_be_read_ends_with_its_message(tmp_path):
    with pytest.raises(SystemExit) as stop:
        render_demo.main([str(tmp_path / "missing.mp4"), "--out", str(tmp_path / "d.gif")])
    assert isinstance(stop.value.code, str) and not (tmp_path / "d.gif").exists()
