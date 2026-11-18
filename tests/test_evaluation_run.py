"""The evaluation runner end to end on a tiny made-up benchmark, with the stand-in model."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from echoview import Ensemble, cli
from echoview.evaluation import benchmark, ev9v, run
from echoview.reference import B3

from .conftest import BrightnessModel

LEVELS = (0, 50, 100, 150, 200)  # the grey level the stand-in model reads as view 0 to 4


def fragment(video: str, label: int, first: int, count: int, gamma: float = 1.0) -> dict:
    """A fragment of the recording folder `video`, named by its key as in the published benchmark."""
    return {"video": ev9v.recording_key(video), "label": label, "gamma": gamma, "frames": list(range(first, first + count))}


def tiny_half(tag: str) -> list[dict]:
    natives = [{"id": f"{tag}_v{k}", "kind": "native", "fragments": [fragment(f"{tag}_v{k}", k, 0, 30)]} for k in range(5)]
    multi = [{"id": f"{tag}_multi{m}", "kind": "multi", "cell": "x",
              "fragments": [fragment(f"{tag}_v{(m + j) % 5}", (m + j) % 5, 0, 25) for j in range(4)]} for m in range(3)]
    same = {"id": f"{tag}_same", "kind": "multi", "cell": "x",
            "fragments": [fragment(f"{tag}_v1", 1, 0, 30), fragment(f"{tag}_v1", 1, 5, 25), fragment(f"{tag}_v3", 3, 0, 30)]}
    pair = {"id": f"{tag}_pair", "kind": "pairs", "cell": "x", "fragments": [fragment(f"{tag}_v0", 0, 0, 30), fragment(f"{tag}_v4", 4, 0, 30)]}
    exam = {"id": f"{tag}_exam", "kind": "exam", "fragments": [fragment(f"{tag}_v2", 2, 0, 30), fragment(f"{tag}_v0", 0, 0, 30)]}
    return [*natives, *multi, same, pair, exam]


@pytest.fixture()
def tiny(tmp_path) -> tuple[Path, Path]:
    """A benchmark file of two small halves and an Images folder whose frames carry their view as a grey level."""
    document = {"format": benchmark.FORMAT, "classes": ["PLAX", "PSAX", "A4C", "A5C", "SC4C"], "hz": 10.0,
                "halves": {half: {"sequences": tiny_half(half[:3])} for half in benchmark.HALVES}}
    path = tmp_path / "tiny.json"
    path.write_text(json.dumps(document))
    images = tmp_path / "Images"
    for half in benchmark.HALVES:
        for k, level in enumerate(LEVELS):
            folder = images / f"{half[:3]}_v{k}"
            folder.mkdir(parents=True)
            for frame in range(30):
                Image.fromarray(np.full((240, 320, 3), level, np.uint8)).save(folder / f"{frame}.jpg", quality=95)
    return path, images


def test_a_sequence_knows_its_labels_joins_and_view_changes(tiny):
    halves = benchmark.load(tiny[0])
    same = next(s for s in halves["calibration"] if s.id == "cal_same")
    assert len(same) == 85 and same.joins.tolist() == [30, 55] and same.view_changes.tolist() == [55]
    assert same.labels[:55].tolist() == [1] * 55 and same.pieces.max() == 1  # two fragments of one recording are one piece
    assert same.refs[30] == (ev9v.recording_key("cal_v1"), 5, 1.0)


def test_the_frame_table_reads_each_frame_once_and_can_be_cached(tiny, tmp_path):
    halves = benchmark.load(tiny[0])
    table = ev9v.FrameTable.build(halves["calibration"], tiny[1], workers=2, cache=tmp_path / "cache")
    assert len(table.refs) == 5 * 30 and table.grey.shape == (150, 224, 224)
    frames = table.frames(next(s for s in halves["calibration"] if s.id == "cal_pair"))
    assert frames.shape == (60, 224, 224, 3) and abs(int(frames[0, 0, 0, 0]) - 0) <= 2 and abs(int(frames[59, 0, 0, 0]) - 200) <= 2
    again = ev9v.FrameTable.build(halves["calibration"], tmp_path / "no_such_folder", cache=tmp_path / "cache")
    assert np.array_equal(again.grey, table.grey)


def test_a_gamma_changes_the_grey_levels_of_a_frame(tiny):
    path = ev9v.frame_path(tiny[1], "cal_v2", 0)
    plain, darker = ev9v.load_frame(path, 1.0), ev9v.load_frame(path, 2.0)
    assert abs(int(plain[100, 100, 0]) - 100) <= 2 and abs(int(darker[100, 100, 0]) - round(255 * (100 / 255) ** 2)) <= 2


def test_a_missing_dataset_is_named(tiny, tmp_path):
    halves = benchmark.load(tiny[0])
    with pytest.raises(ev9v.DataError, match="not a folder"):
        ev9v.FrameTable.build(halves["calibration"], tmp_path / "nowhere")
    (tiny[1] / "cal_v0" / "7.jpg").unlink()
    with pytest.raises(ev9v.DataError, match=r"7\.jpg"):
        ev9v.FrameTable.build(halves["calibration"], tiny[1])
    (tiny[1] / "cal_v3").rename(tiny[1] / "renamed")
    with pytest.raises(ev9v.DataError, match="1 of the 5 recordings"):
        ev9v.FrameTable.build(halves["calibration"], tiny[1])


def test_a_perfect_model_scores_perfectly_on_every_system(tiny):
    pytest.importorskip("scipy")
    halves = benchmark.load(tiny[0])
    tables = {half: ev9v.FrameTable.build(halves[half], tiny[1], workers=2) for half in benchmark.HALVES}
    system = Ensemble.from_models([BrightnessModel(), BrightnessModel(3.0)], B3, device="cpu", batch=16)
    report = run.evaluate_reference(system, halves, tables)
    assert set(report["systems"]) == {"model_0", "model_1", "ensemble", "ensemble_published_setting"}
    for name, entry in report["systems"].items():
        for half in benchmark.HALVES:
            scores = entry[half]
            assert scores["system"]["segment_f1_50_multi"] == 1.0, name
            assert scores["system"]["boundary_recall_multi"] == 1.0 and scores["system"]["same_class_cut_rate_multi"] == 0.0
            assert scores["native"]["errors"] == 0 and scores["native"]["recordings"] == 5
            assert scores["streams_exam"]["A6_f1"]["50"] == 1.0 and scores["streams_multi"]["frame_error"] == 0.0
    assert report["systems"]["ensemble"]["b3"]["status"] == "feasible"
    assert report["systems"]["ensemble_published_setting"]["b3"]["params"] == B3
    assert len(run.headline(report)) == 4


def test_the_evaluate_command_needs_the_dataset(tmp_path):
    with pytest.raises(SystemExit, match="not a folder"):
        cli.main(["evaluate", "--images", str(tmp_path / "nowhere")])
