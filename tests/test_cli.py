"""The command line, run on synthetic inputs with a stand-in model (no weights needed)."""
from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import echoview
from echoview import Ensemble, cli, mimic_table, reference, weights
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
    assert sample_rows[39]["file"] == str(a) and sample_rows[40]["file"] == str(b) and sample_rows[40]["time_s"] == "4.0"


def test_segment_declines_around_the_cut_when_asked(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(reference, "UNCERTAIN", {"band": 2, "f_low": 1.0, "f_high": 1.0, "margin": 0.0, "seeds": False})
    a, b = frame_folder(tmp_path / "a", 200, 40), frame_folder(tmp_path / "b", 50, 40)
    pieces, samples = tmp_path / "pieces.csv", tmp_path / "samples.csv"
    assert cli.main(["segment", str(a), str(b), "--fps", "10", "--uncertain", "--out", str(pieces), "--samples", str(samples)]) == 0
    assert [(r["start_s"], r["end_s"], r["view"]) for r in rows_of(pieces)] == [
        ("0.0", "3.8", "SC4C"), ("3.8", "4.2", "uncertain"), ("4.2", "8.0", "PSAX")]
    assert [r["view"] for r in rows_of(samples)[37:43]] == ["SC4C", "uncertain", "uncertain", "uncertain", "uncertain", "PSAX"]
    printed = capsys.readouterr()
    assert "uncertain: 0.4 s of 8.0 s (5.0%)" in printed.out
    assert re.search(r"^     0\.0 -    3\.8 s  SC4C      \d\.\d{4}$", printed.out, re.M)
    assert re.search(r"^     3\.8 -    4\.2 s  uncertain \d\.\d{4}$", printed.out, re.M)
    assert "chosen and checked for the reference ensemble" in printed.err  # the stand-in model is not the reference


def test_segment_names_every_sample_without_the_option(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(reference, "UNCERTAIN", {"band": 2, "f_low": 1.0, "f_high": 1.0, "margin": 0.0, "seeds": False})
    a, b = frame_folder(tmp_path / "a", 200, 40), frame_folder(tmp_path / "b", 50, 40)
    samples = tmp_path / "samples.csv"
    assert cli.main(["segment", str(a), str(b), "--fps", "10", "--samples", str(samples)]) == 0
    assert {r["view"] for r in rows_of(samples)} == {"SC4C", "PSAX"}
    printed = capsys.readouterr()
    assert "uncertain" not in printed.out and "chosen and checked" not in printed.err
    # the lines of the default output are as they were before the option existed: the view in a column of five
    assert re.search(r"^     0\.0 -    4\.0 s  SC4C  \d\.\d{4}$", printed.out, re.M)
    assert re.search(r"^     4\.0 -    8\.0 s  PSAX  \d\.\d{4}$", printed.out, re.M)


def test_segment_reads_more_often_when_asked_and_says_that_this_is_not_the_published_system(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(Ensemble, "is_reference", property(lambda self: True))  # the stand-in passes for the published system
    a, b = frame_folder(tmp_path / "a", 200, 43), frame_folder(tmp_path / "b", 50, 37)
    pieces = tmp_path / "pieces.csv"
    assert cli.main(["segment", str(a), str(b), "--fps", "10", "--out", str(pieces)]) == 0
    assert [(r["start_s"], r["end_s"]) for r in rows_of(pieces)] == [("0.0", "4.5"), ("4.5", "8.0")]  # the join at 4.3 s is off the 0.5 s grid
    assert "not the published system" not in capsys.readouterr().err
    dense = tmp_path / "dense.csv"
    assert cli.main(["segment", str(a), str(b), "--fps", "10", "--read-every", "0.1", "--out", str(dense)]) == 0
    rows = rows_of(dense)
    assert [(r["start_s"], r["end_s"], r["view"]) for r in rows] == [("0.0", "4.3", "SC4C"), ("4.3", "8.0", "PSAX")]
    assert {r["view_classifier_version"] for r in rows} == {f"{Ensemble.from_models([BrightnessModel()], B3).version}, read every 0.1 s"}
    assert {r["view_classifier_version"] for r in rows_of(pieces)} == {Ensemble.from_models([BrightnessModel()], B3).version}
    said = capsys.readouterr().err
    assert "not the published system" in said and "nearer cuts at a cost" in said and "docs/FOLLOWUP.md" in said
    assert "784 of 800 single recordings right where the published system reads 785" in said and "not confirmed" not in said
    fifth = tmp_path / "fifth.csv"
    assert cli.main(["segment", str(a), str(b), "--fps", "10", "--read-every", "0.2", "--out", str(fifth)]) == 0
    rows = rows_of(fifth)
    assert [(r["start_s"], r["end_s"]) for r in rows] == [("0.0", "4.3"), ("4.3", "8.0")]  # sample 43 has an output of its own every 0.2 s
    assert {r["view_classifier_version"] for r in rows} == {f"{Ensemble.from_models([BrightnessModel()], B3).version}, read every 0.2 s"}
    said = capsys.readouterr().err
    assert "a read-out every 0.2 s is not the published system" in said and "2.5 times as many windows" in said
    assert "nearer cuts at a cost" in said
    with pytest.raises(SystemExit):
        cli.main(["segment", str(a), str(b), "--fps", "10", "--read-every", "0.3"])
    capsys.readouterr()
    with pytest.raises(SystemExit):
        cli.main(["segment", "--help"])
    said = " ".join(capsys.readouterr().out.split())  # the help says what the read of the test benchmark found
    assert "both have nearer cuts at a cost: one single recording more of 800 is misread" in said and "not confirmed" not in said


@pytest.mark.parametrize("every", ["0.2", "0.1"])
def test_a_denser_read_out_that_cannot_be_given_stops_before_any_work(tmp_path, monkeypatch, every):
    a, b = frame_folder(tmp_path / "a", 200, 40), frame_folder(tmp_path / "b", 50, 40)
    command = ["segment", str(a), str(b), "--fps", "10", "--read-every", every]
    with pytest.raises(SystemExit, match="published system"):  # the stand-in model of these tests is not it
        cli.main(command)

    def never(*args, **kwargs):
        raise AssertionError("an input was read or a model was loaded for a command that cannot run")

    monkeypatch.setattr(cli, "build", never)
    monkeypatch.setattr(cli.fr, "read_recording", never)
    with pytest.raises(SystemExit, match="--uncertain was studied with the published read-out"):
        cli.main([*command, "--uncertain"])
    for seeds in ("0", "0,1", "2,1,0"):
        with pytest.raises(SystemExit, match="published system: leave --seeds at 0,1,2"):
            cli.main([*command, "--seeds", seeds])
    with pytest.raises(AssertionError):  # with the published read-out the same command goes on to its work
        cli.main(["segment", str(a), str(b), "--fps", "10", "--seeds", "0"])


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


def test_version_names_the_package_and_the_model(capsys):
    with pytest.raises(SystemExit) as stopped:
        cli.main(["--version"])
    assert stopped.value.code == 0
    assert capsys.readouterr().out.strip() == f"echoview {echoview.__version__} ({reference.NAME} {reference.VERSION})"


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
