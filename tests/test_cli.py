"""The command line, run on synthetic inputs with a stand-in model (no weights needed)."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from echoview import Ensemble, cli, mimic_table, weights
from echoview.reference import B3

from .conftest import BrightnessModel
from .dicom_samples import write_compressed_cine


@pytest.fixture(autouse=True)
def stand_in_model(monkeypatch):
    """Every command of these tests gets the stand-in model instead of the published weights."""
    monkeypatch.setattr(cli, "build", lambda args, verify=True: Ensemble.from_models([BrightnessModel()], B3, device="cpu"))


def frame_folder(path: Path, level: int, count: int = 30) -> Path:
    path.mkdir(parents=True)
    for k in range(count):
        Image.fromarray(np.full((240, 320), level, np.uint8)).save(path / f"{k}.png")
    return path


def rows_of(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


# ------------------------------------------------------------------------------------------------ classify
def test_classify_writes_one_row_per_recording(tmp_path, capsys):
    a, b = frame_folder(tmp_path / "in" / "a", 100), frame_folder(tmp_path / "in" / "b", 0)
    out = tmp_path / "out" / "views.csv"
    assert cli.main(["classify", str(a), str(b), "--fps", "10", "--out", str(out)]) == 0
    rows = rows_of(out)
    assert tuple(rows[0]) == cli.CLASSIFY_HEADER
    assert [(Path(r["file"]).name, r["predicted_view"], r["samples_10hz"], r["windows"]) for r in rows] == [("a", "A4C", "30", "6"), ("b", "PLAX", "30", "6")]
    assert rows[0]["reader"] == "image folder" and rows[0]["view_classifier_version"].startswith("custom ensemble")
    assert "A4C" in capsys.readouterr().out


def test_classify_skips_what_it_cannot_read_and_says_so(tmp_path, capsys):
    good = frame_folder(tmp_path / "good", 150)
    (tmp_path / "broken.dcm").write_text("not a dicom file")
    assert cli.main(["classify", str(good), str(tmp_path / "broken.dcm"), "--fps", "10"]) == 0
    captured = capsys.readouterr()
    assert "SKIPPED" in captured.err and "1 of 2 inputs could not be read" in captured.err
    assert cli.main(["classify", str(tmp_path / "broken.dcm")]) == 1


def test_classify_without_any_recording_stops(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(SystemExit, match="no recording found"):
        cli.main(["classify", str(tmp_path / "empty")])


# ------------------------------------------------------------------------------------------------- segment
def test_segment_joins_the_inputs_in_the_given_order(tmp_path):
    a, b = frame_folder(tmp_path / "a", 200, 40), frame_folder(tmp_path / "b", 50, 40)
    pieces, samples = tmp_path / "pieces.csv", tmp_path / "samples.csv"
    assert cli.main(["segment", str(a), str(b), "--fps", "10", "--out", str(pieces), "--samples", str(samples)]) == 0
    assert [(r["start_s"], r["end_s"], r["view"]) for r in rows_of(pieces)] == [("0.0", "4.0", "SC4C"), ("4.0", "8.0", "PSAX")]
    sample_rows = rows_of(samples)
    assert len(sample_rows) == 80
    assert sample_rows[39]["file"] == "a" and sample_rows[40]["file"] == "b" and sample_rows[40]["time_s"] == "4.0"


def test_segment_builds_no_stream_when_one_input_cannot_be_read(tmp_path):
    good = frame_folder(tmp_path / "good", 100)
    with pytest.raises(SystemExit, match="no stream was built"):
        cli.main(["segment", str(good), str(tmp_path / "missing.mp4"), "--fps", "10"])


# --------------------------------------------------------------------------------------------- mimic-table
@pytest.fixture()
def mimic(tmp_path) -> tuple[Path, Path]:
    """A label table of three rows: one readable cine, one missing file and one file that is no image."""
    root = tmp_path / "mimic"
    root.mkdir()
    write_compressed_cine(root / "cine.dcm")
    (root / "report.dcm").write_text("not a dicom file")
    labels = tmp_path / "labels.csv"
    with labels.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(mimic_table.V4_COLUMNS)
        writer.writerow(["r1", "1", "2", "cine.dcm", "30", "m", "mm", "A4C_general", "0.9", "v4"])
        writer.writerow(["r2", "1", "2", "missing.dcm", "30", "m", "mm", "PLAX_LV", "0.9", "v4"])
        writer.writerow(["r3", "1", "2", "report.dcm", "30", "m", "mm", "PLAX_LV", "0.9", "v4"])
    return labels, root


def table(labels: Path, root: Path, out: Path, *more: str) -> int:
    return cli.main(["mimic-table", "--labels", str(labels), "--root", str(root), "--out", str(out), *more])


def test_mimic_table_writes_readable_rows_and_lists_the_others(mimic, tmp_path):
    labels, root = mimic
    out = tmp_path / "ours.csv"
    out.write_text("")  # an empty file left by an interrupted start is taken as new
    assert table(labels, root, out) == 0
    rows = rows_of(out)
    assert tuple(rows[0]) == mimic_table.V4_COLUMNS and [r["video_name"] for r in rows] == ["r1"]
    assert rows[0]["subject_id"] == "1" and rows[0]["predicted_view"] in ("PLAX", "PSAX", "A4C", "A5C", "SC4C")
    assert sorted(r["video_name"] for r in rows_of(tmp_path / "ours.errors.csv")) == ["r2", "r3"]


def test_mimic_table_continues_and_tries_unreadable_rows_again(mimic, tmp_path, capsys):
    labels, root = mimic
    out = tmp_path / "ours.csv"
    table(labels, root, out)
    capsys.readouterr()
    assert table(labels, root, out) == 1  # nothing new could be written
    assert "already written 1" in capsys.readouterr().err
    assert len(rows_of(out)) == 1


def test_mimic_table_never_writes_into_another_table(mimic, tmp_path):
    labels, root = mimic
    with pytest.raises(SystemExit, match="must not be the label table"):
        table(labels, root, labels)
    other = tmp_path / "other.csv"
    other.write_text("a,b\n1,2\n")
    with pytest.raises(SystemExit, match="not a table written by this command"):
        table(labels, root, other)
    assert other.read_text() == "a,b\n1,2\n"


def test_mimic_table_limit_and_bad_options(mimic, tmp_path):
    labels, root = mimic
    assert table(labels, root, tmp_path / "limited.csv", "--limit", "1") == 0
    with pytest.raises(SystemExit, match="--workers"):
        table(labels, root, tmp_path / "x.csv", "--workers", "0")


# ------------------------------------------------------------------------------------------------- options
def test_seed_lists_are_checked():
    assert cli.parse_seeds("0,2") == (0, 2)
    for text in ("a,b", "0,7", ""):
        with pytest.raises(SystemExit, match="--seeds"):
            cli.parse_seeds(text)


def test_the_weights_command_reports_every_file(tmp_path, monkeypatch, capsys):
    for seed in (0, 1, 2):
        (tmp_path / f"C2_s{seed}").mkdir()
        (tmp_path / f"C2_s{seed}" / "final.pt").write_bytes(b"x")
    monkeypatch.setattr(weights, "is_published", lambda path, seed: seed != 1)
    assert cli.main(["weights", "--weights", str(tmp_path)]) == 1
    printed = capsys.readouterr().out
    assert "ok   checksum C2_s0/final.pt" in printed and "FAIL checksum C2_s1/final.pt" in printed and str(tmp_path) in printed


def test_your_own_weights_need_the_unverified_switch(tmp_path, monkeypatch):
    monkeypatch.undo()  # the real build, not the stand-in model
    for seed in (0, 1, 2):
        (tmp_path / f"C2_s{seed}").mkdir()
        (tmp_path / f"C2_s{seed}" / "final.pt").write_bytes(b"weights of my own")
    args = cli.parser().parse_args(["classify", "x", "--weights", str(tmp_path)])
    with pytest.raises(SystemExit, match="not a published file"):
        cli.build(args)
    assert cli.parser().parse_args(["classify", "x", "--unverified"]).unverified is True
