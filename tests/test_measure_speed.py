"""The speed measurement: what it times, how it sums up the runs and what it refuses."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from echoview.windows import STRIDE, window_plan

SPEC = importlib.util.spec_from_file_location("measure_speed", Path(__file__).resolve().parents[1] / "tools" / "measure_speed.py")
measure_speed = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(measure_speed)


class Clock:
    """A clock that a fake system moves forward by the cost it is given for every window it reads."""

    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


class FakeSystem:
    """Stands for the ensemble: `segment` costs a fixed time per window, more on the runs named in `slow`."""

    def __init__(self, clock: Clock, per_window: float = 0.01, slow: dict | None = None) -> None:
        self.clock, self.per_window, self.slow, self.calls = clock, per_window, slow or {}, []

    def segment(self, frames: np.ndarray, stride: int = STRIDE):
        nth = sum(1 for _, s in self.calls if s == stride)  # 0 is the run before the timed ones
        self.calls.append((len(frames), stride))
        self.clock.now += self.per_window * len(window_plan(len(frames), stride)[0]) * self.slow.get((stride, nth), 1.0)


def test_a_stream_is_the_clips_joined_and_repeated_to_the_asked_length():
    parts = [np.full((3, 2, 2, 3), 1, np.uint8), np.full((2, 2, 2, 3), 2, np.uint8)]
    stream = measure_speed.stream_of(parts, 12)
    assert stream.shape == (12, 2, 2, 3) and stream.dtype == np.uint8
    assert stream[:, 0, 0, 0].tolist() == [1, 1, 1, 2, 2, 1, 1, 1, 2, 2, 1, 1]  # joined, repeated, cut to the length
    assert measure_speed.stream_of(parts, 4)[:, 0, 0, 0].tolist() == [1, 1, 1, 2]


def test_a_stream_needs_clips_and_a_length():
    with pytest.raises(ValueError, match="no frames"):
        measure_speed.stream_of([], 10)
    with pytest.raises(ValueError, match="no frames"):
        measure_speed.stream_of([np.zeros((0, 2, 2, 3), np.uint8)], 10)
    with pytest.raises(ValueError, match="at least one sample"):
        measure_speed.stream_of([np.zeros((3, 2, 2, 3), np.uint8)], 0)


def test_every_step_is_run_once_untimed_and_then_timed_as_often_as_asked():
    clock = Clock()
    system = FakeSystem(clock)
    frames = np.zeros((600, 2, 2, 3), np.uint8)  # one minute of stream
    rows = measure_speed.time_steps(system, frames, repeats=3, clock=clock)
    assert [stride for _, stride in system.calls] == [5] * 4 + [2] * 4 + [1] * 4  # one run before the three timed ones
    assert {n for n, _ in system.calls} == {600}
    assert [row["read_every_s"] for row in rows] == [0.5, 0.2, 0.1]
    assert [row["samples_between_outputs"] for row in rows] == [5, 2, 1]
    for row in rows:
        windows = len(window_plan(600, row["samples_between_outputs"])[0])
        assert row["windows"] == windows and row["runs"] == 3
        assert row["seconds"] == pytest.approx([0.01 * windows] * 3)
        assert row["seconds_median"] == pytest.approx(0.01 * windows)
        assert row["seconds_per_minute_of_stream"] == pytest.approx(0.01 * windows)  # the stream is one minute long
        assert row["times_real_time"] == pytest.approx(60.0 / (0.01 * windows))
    assert rows[0]["times_the_published_step"] == pytest.approx(1.0)
    assert rows[2]["times_the_published_step"] == pytest.approx(rows[2]["windows"] / rows[0]["windows"])
    assert rows[2]["windows_times_the_published_step"] == pytest.approx(600 / 120)


def test_the_summary_is_the_median_and_the_range_and_the_first_run_does_not_count():
    clock = Clock()
    # the untimed run of the published step is 50 times slower (a model that warms up), and one timed run is twice as slow
    system = FakeSystem(clock, slow={(5, 0): 50.0, (5, 2): 2.0})
    frames = np.zeros((300, 2, 2, 3), np.uint8)  # half a minute
    row = measure_speed.time_steps(system, frames, repeats=5, clock=clock)[0]
    one = 0.01 * row["windows"]
    assert sorted(row["seconds"]) == pytest.approx([one, one, one, one, 2 * one])
    assert row["seconds_median"] == pytest.approx(one)
    assert row["seconds_min"] == pytest.approx(one) and row["seconds_max"] == pytest.approx(2 * one)
    assert row["seconds_per_minute_of_stream"] == pytest.approx(2 * one)  # half a minute took `one`


def test_the_number_of_runs_must_be_odd_and_at_least_three():
    frames = np.zeros((100, 2, 2, 3), np.uint8)
    for repeats in (0, 1, 2, 4):
        with pytest.raises(ValueError, match="odd number of runs, three or more"):
            measure_speed.time_steps(FakeSystem(Clock()), frames, repeats=repeats, clock=Clock())


def test_the_command_refuses_bad_arguments_before_it_loads_anything(tmp_path, monkeypatch):
    import echoview

    def never(*args, **kwargs):
        raise AssertionError("no model may be loaded for arguments that are refused")

    monkeypatch.setattr(echoview, "Ensemble", never)
    for arguments in (["--minutes", "0"], ["--repeats", "2"], ["--repeats", "4"], ["--minutes", "-1"], ["--minutes", "inf"], ["--minutes", "nan"],
                      ["--batch", "-1"]):
        with pytest.raises(SystemExit) as stop:
            measure_speed.main([*arguments, "--out", str(tmp_path / "speed.json")])
        assert stop.value.code not in (0, None)
    assert not list(tmp_path.iterdir())
    kept = tmp_path / "kept.json"
    kept.write_text("an earlier measurement")
    with pytest.raises(SystemExit) as stop:  # a file that exists is refused before the models are loaded
        measure_speed.main(["--out", str(kept)])
    assert stop.value.code not in (0, None) and kept.read_text() == "an earlier measurement"


def test_a_report_is_never_written_in_the_place_of_a_file(tmp_path):
    out = tmp_path / "speed.json"
    out.write_text("kept")
    with pytest.raises(FileExistsError):
        measure_speed.write({"steps": []}, out)
    assert out.read_text() == "kept" and [p.name for p in tmp_path.iterdir()] == ["speed.json"]


def test_the_report_names_the_machine_the_system_and_what_was_timed(tmp_path):
    clock = Clock()
    system = FakeSystem(clock)
    system.device, system.batch, system.version, system.is_reference = "cpu", 4, "C2_ens_v1", True
    frames = np.zeros((600, 2, 2, 3), np.uint8)
    report = measure_speed.report(system, frames, repeats=3, clock=clock)
    assert report["stream_seconds"] == 60.0 and report["system"] == "C2_ens_v1" and report["reference_system"] is True
    assert report["timed"] == measure_speed.TIMED
    assert report["gpu_memory_peak_gib"] is None  # measured on an NVIDIA GPU only
    for key in ("device", "batch", "torch", "python", "platform", "echoview"):
        assert key in report["environment"], key
    assert [row["read_every_s"] for row in report["steps"]] == [0.5, 0.2, 0.1]
    out = tmp_path / "speed.json"
    measure_speed.write(report, out)
    assert json.loads(out.read_text()) == json.loads(json.dumps(report))
    assert [p.name for p in tmp_path.iterdir()] == ["speed.json"]  # written whole, nothing left beside it
