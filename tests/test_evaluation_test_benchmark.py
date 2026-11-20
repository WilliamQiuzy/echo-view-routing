"""The frozen EV9V test benchmark: its definition file, the evaluation at the fixed decoder setting and the intervals."""
from __future__ import annotations

import gzip
import hashlib
import json
import re
from importlib import resources
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from echoview import Ensemble, cli, windows
from echoview.evaluation import benchmark, bootstrap, ev9v, metrics, run
from echoview.reference import B3

from .conftest import BrightnessModel

FIXTURES = Path(__file__).parent / "fixtures"
LEVELS = (0, 50, 100, 150, 200)  # the grey level the stand-in model reads as view 0 to 4
TEST_BENCHMARK_SHA256 = "70e58cf7baec049776c1c01ca284083ec01aca7925d0b1348c2c391fb958119e"  # of the uncompressed definition
FIRST_DRAWS = [21, 80, 54, 99, 27, 5]  # the bootstrap generator's first integers below 125


def fragment(view: int, first: int = 0, count: int = 30) -> dict:
    return {"video": ev9v.recording_key(f"t_v{view}"), "label": view, "gamma": 1.0, "frames": list(range(first, first + count))}


@pytest.fixture()
def tiny(tmp_path) -> tuple[Path, Path]:
    """A test benchmark of five recordings in three examinations, three multi streams and two pair streams, with frames."""
    natives = [{"id": f"native_test_{k:04d}", "kind": "native", "group": (0, 0, 1, 1, 2)[k], "fragments": [fragment(k)]} for k in range(5)]
    multi = [{"id": f"multi_test_{m:04d}", "kind": "multi", "cell": "x", "fragments": [fragment((m + j) % 5, 0, 25) for j in range(4)]}
             for m in (2, 0, 1)]  # not in the order of their names
    pairs = [{"id": "pairs_test_x_0001", "kind": "pairs", "cell": "x", "fragments": [fragment(0), fragment(4)]},
             {"id": "pairs_test_x_0000", "kind": "pairs", "cell": "x", "fragments": [fragment(1), fragment(1, 5, 25)]}]
    document = {"format": benchmark.FORMAT, "classes": ["PLAX", "PSAX", "A4C", "A5C", "SC4C"], "hz": 10.0,
                "halves": {benchmark.TEST: {"sequences": [*natives, *pairs, *multi]}}}
    path = tmp_path / "tiny_test.json"
    path.write_text(json.dumps(document))
    images = tmp_path / "Images"
    for view, level in enumerate(LEVELS):
        folder = images / f"t_v{view}"
        folder.mkdir(parents=True)
        for frame in range(30):
            Image.fromarray(np.full((240, 320, 3), level, np.uint8)).save(folder / f"{frame}.jpg", quality=95)
    return path, images


def test_the_test_benchmark_is_one_part_and_names_the_examinations(tiny):
    sequences = benchmark.load_test(tiny[0])
    assert [s.kind for s in sequences] == ["native"] * 5 + ["pairs"] * 2 + ["multi"] * 3
    assert [s.group for s in sequences] == [0, 0, 1, 1, 2, -1, -1, -1, -1, -1]
    with pytest.raises(ValueError, match="calibration"):
        benchmark.load(tiny[0])  # a test file is not a development file
    with pytest.raises(ValueError, match="test"):
        benchmark.load_test(resources.files("echoview").joinpath("data", benchmark.EV9V_DEV))


def test_a_perfect_model_scores_perfectly_at_the_fixed_setting(tiny):
    sequences = benchmark.load_test(tiny[0])
    table = ev9v.FrameTable.build(sequences, tiny[1], workers=2)
    system = Ensemble.from_models([BrightnessModel(), BrightnessModel(3.0)], B3, device="cpu", batch=16)
    report = run.evaluate_test_benchmark(system, sequences, table)
    assert report["b3"] == {"status": "fixed", "params": B3}  # nothing is chosen on the test benchmark
    assert report["sequences"] == {"native": 5, "pairs": 2, "multi": 3} and report["frames"] == 150
    scores = report["system"]
    assert scores["segment_f1_50_multi"] == scores["segment_f1_50_pairs"] == scores["native_accuracy"] == scores["native_macro_f1"] == 1.0
    intervals = report["intervals"]
    assert intervals["segment_f1_50_multi"] == intervals["segment_f1_50_pairs"] == intervals["native_accuracy"] == [1.0, 1.0]
    assert intervals["native_macro_f1"][1] == 1.0 and len(intervals) == 4  # a resample without a view scores that view 0
    lines = run.headline_of_test(report)
    assert len(lines) == 3 and "1.0000" in lines[0] and "1.000 to 1.000" in lines[0]


def test_the_evaluate_command_runs_the_test_benchmark_on_request(tiny, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "build", lambda args, verify=True: Ensemble.from_models([BrightnessModel()], B3, device="cpu"))
    monkeypatch.setattr(benchmark, "load_test", lambda path=None, read=benchmark.load_test: read(tiny[0]))
    out = tmp_path / "test_report.json"
    assert cli.main(["evaluate", "--benchmark", "test", "--images", str(tiny[1]), "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "multi streams" in printed and "segment F1@50 1.0000" in printed
    assert json.loads(out.read_text())["benchmark"].startswith("EV9V test benchmark")
    kept = tmp_path / "kept" / "logits.npz"  # the model outputs behind the numbers, for a later look without a GPU
    assert cli.main(["evaluate", "--benchmark", "test", "--images", str(tiny[1]), "--save-logits", str(kept)]) == 0
    with np.load(kept) as stored:
        sequences = benchmark.load_test(tiny[0])
        assert set(stored.files) == {s.id for s in sequences}
        assert all(stored[s.id].shape == (len(windows.window_centres(len(s))), 1, 5) and stored[s.id].dtype == np.float32 for s in sequences)
    assert not list(kept.parent.glob("*partial*"))
    with pytest.raises(SystemExit):
        cli.main(["evaluate", "--benchmark", "holdout", "--images", str(tiny[1])])


def test_recordings_without_examinations_stop_the_run_before_the_models_do_any_work(tiny):
    sequences = benchmark.load_test(tiny[0])
    without = tuple(benchmark.Sequence(s.id, s.kind, s.fragments, s.cell) for s in sequences)
    model = BrightnessModel()
    system = Ensemble.from_models([model], B3, device="cpu")
    with pytest.raises(ValueError, match="examinations"):
        run.evaluate_test_benchmark(system, without, ev9v.FrameTable.build(without, tiny[1], workers=2))
    assert model.calls == 0


def test_the_intervals_resample_examinations_then_streams(tiny):
    sequences = benchmark.load_test(tiny[0])
    decoded = {s.id: s.labels.copy() for s in sequences}
    decoded["native_test_0004"][:] = 0  # the only recording of examination 2 is read as view 0
    decoded["multi_test_0001"][:] = 1  # one stream decoded as a single piece
    intervals = bootstrap.intervals(sequences, decoded, draws=200)
    natives, multi = (benchmark.of_kind(sequences, kind) for kind in ("native", "multi"))
    cells = bootstrap.native_sums(natives, decoded)
    assert cells.shape == (3, 25) and cells[2].tolist() == [0.0] * 20 + [1.0, 0.0, 0.0, 0.0, 0.0] and cells.sum() == 5
    sums = bootstrap.stream_sums(multi, decoded)
    assert sums.tolist() == [[4.0, 4.0, 4.0], [0.0, 1.0, 4.0], [4.0, 4.0, 4.0]]  # by name: 0000, 0001, 0002
    generator = np.random.default_rng(bootstrap.SEED)  # one generator: recordings first, then multi, then pairs
    first = bootstrap.resampling_weights(generator, 3, 200)
    second = bootstrap.resampling_weights(generator, 3, 200)
    assert first.shape == (200, 3) and (first.sum(1) == 3).all()
    assert intervals["native_accuracy"] == bootstrap.percentile_interval(first, cells, bootstrap.accuracy)
    assert intervals["segment_f1_50_multi"] == bootstrap.percentile_interval(second, sums, bootstrap.segment_f1)
    assert intervals["native_accuracy"][0] < 0.8 < intervals["native_accuracy"][1] == 1.0
    assert intervals["segment_f1_50_pairs"] == [1.0, 1.0]
    assert float(bootstrap.segment_f1(sums.sum(0))) == metrics.evaluate(decoded, sequences)["segment_f1_50_multi"]
    with pytest.raises(ValueError, match="examinations"):
        bootstrap.native_sums(benchmark.load()["selection"][:3], {})  # development recordings carry no examination


def test_the_intervals_are_those_of_the_research_bootstrap():
    """Made-up counts, and the intervals the research code's bootstrap gave them (see the fixture's generator)."""
    fixture = json.loads((FIXTURES / "bootstrap_parity.json").read_text())
    expected = fixture["expected"]
    first_draws = np.random.default_rng(bootstrap.SEED).integers(0, 125, size=6).tolist()
    assert first_draws == FIRST_DRAWS, "this NumPy draws other integers from the same seed: the stored intervals cannot be recomputed with it"
    cells = np.zeros((1 + max(row[0] for row in fixture["natives"]), 25))
    for examination, reference, decided in fixture["natives"]:
        cells[examination, 5 * reference + decided] += 1
    generator = np.random.default_rng(bootstrap.SEED)
    weights = bootstrap.resampling_weights(generator, len(cells))
    for name, statistic in (("clip macro-F1", bootstrap.macro_f1), ("clip accuracy", bootstrap.accuracy)):
        assert float(statistic(cells.sum(0))) == pytest.approx(expected[name]["value"], abs=1e-15)
        assert bootstrap.percentile_interval(weights, cells, statistic) == pytest.approx(expected[name]["ci95"], abs=1e-12)
    for part, name in (("multi", "segment F1@50, multi"), ("pairs", "segment F1@50, pairs")):
        sums = np.array([row[1:] for row in sorted(fixture[part])], float)  # streams in the order of their names
        weights = bootstrap.resampling_weights(generator, len(sums))
        assert float(bootstrap.segment_f1(sums.sum(0))) == pytest.approx(expected[name]["value"], abs=1e-15)
        assert bootstrap.percentile_interval(weights, sums, bootstrap.segment_f1) == pytest.approx(expected[name]["ci95"], abs=1e-12)


def test_the_packaged_test_benchmark_has_the_shape_of_the_project_s_frozen_one():
    payload = gzip.decompress(resources.files("echoview").joinpath("data", benchmark.EV9V_TEST).read_bytes())
    assert hashlib.sha256(payload).hexdigest() == TEST_BENCHMARK_SHA256  # the file as the export tool wrote it
    sequences = benchmark.load_test()
    natives, pairs, multi = (benchmark.of_kind(sequences, kind) for kind in ("native", "pairs", "multi"))
    assert (len(natives), len(pairs), len(multi)) == (800, 240, 120) and len(sequences) == 1160
    assert sum(len(s) for s in sequences) == 72185
    assert (sum(len(s) for s in natives), sum(len(s) for s in pairs), sum(len(s) for s in multi)) == (46507, 10391, 15287)
    assert len(ev9v.FrameTable.references(sequences)) == 55288
    assert sum(len(windows.window_centres(len(s))) for s in sequences) == 14871
    assert sorted({s.group for s in natives}) == list(range(125)) and all(s.group == -1 for s in (*pairs, *multi))
    assert len({f.video for s in multi for f in s.fragments}) == 405
    assert [sum(len(metrics.labels_to_segments(s.labels)) for s in part) for part in (multi, pairs)] == [499, 360]
    assert sum(len(s.view_changes) for s in multi) == 379 and sum(len(s.joins) - len(s.view_changes) for s in multi) == 202
    assert all(len(s.fragments) == 1 for s in natives) and all(len(s.fragments) == 2 for s in pairs)
    assert {len(s.fragments) for s in multi} <= set(range(4, 9))
    assert {f.gamma for s in sequences for f in s.fragments} <= {1.0, 0.9}


def test_the_packaged_test_benchmark_names_no_recording_and_shares_none_with_training_or_development():
    raw = gzip.decompress(resources.files("echoview").joinpath("data", benchmark.EV9V_TEST).read_bytes()).decode()
    assert not re.search(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}", raw)  # EV9V folder names carry a date and a time
    sequences = benchmark.load_test()
    keys = {f.video for s in sequences for f in s.fragments}
    assert all(re.fullmatch(r"[0-9a-f]{16}", key) for key in keys)
    assert [s.id for s in benchmark.of_kind(sequences, "native")] == [f"native_test_{k:04d}" for k in range(800)]
    listed = json.loads(gzip.decompress(resources.files("echoview").joinpath("data", "ev9v_recordings.json.gz").read_bytes()))
    assert not keys & {entry["key"] for entry in listed["recordings"]}  # the list holds the training and development recordings
    development = {f.video for half in benchmark.load().values() for s in half for f in s.fragments}
    assert not keys & development
