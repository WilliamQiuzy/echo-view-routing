"""The evaluation must say what the research code says, on made-up predictions for the whole development benchmark.

The stored file holds a random distortion of the reference labels for every sequence and the research code's
metrics, scorecards and decoder choice for them (fixtures/make_metric_fixtures_from_research_code.py).
"""
import json

import numpy as np
import pytest

from echoview.evaluation import benchmark, calibrate, metrics, scorecard

from .conftest import FIXTURES


@pytest.fixture(scope="module")
def stored() -> tuple[dict, dict]:
    with np.load(FIXTURES / "metrics_parity.npz") as arrays:
        data = {key: arrays[key] for key in arrays.files}
    return data, json.loads((FIXTURES / "metrics_parity.json").read_text())


@pytest.fixture(scope="module")
def halves() -> dict:
    return benchmark.load()


def same(got, want, where: str = "") -> None:
    """Equal structures; floats to 1e-12."""
    if isinstance(want, dict):
        assert set(got) == set(want), where
        for key in want:
            same(got[key], want[key], f"{where}/{key}")
    elif isinstance(want, list):
        assert len(got) == len(want), where
        for k, (a, b) in enumerate(zip(got, want)):
            same(a, b, f"{where}[{k}]")
    elif isinstance(want, float):
        assert got == pytest.approx(want, rel=1e-12, abs=1e-12), where
    else:
        assert got == want, where


def test_the_benchmark_holds_the_published_counts(halves):
    counts = {half: {kind: len(benchmark.of_kind(seqs, kind)) for kind in benchmark.KINDS} for half, seqs in halves.items()}
    assert counts == {"calibration": {"native": 237, "pairs": 115, "multi": 120, "exam": 36},
                      "selection": {"native": 284, "pairs": 137, "multi": 120, "exam": 36}}
    assert sum(len(s) for s in halves["selection"] if s.kind != "exam") == 37287


@pytest.mark.parametrize("half", benchmark.HALVES)
def test_metrics_equal_the_research_code(half, stored, halves):
    data, report = stored
    preds = {s.id: data[f"{half}|pred|{s.id}"].astype(int) for s in halves[half] if s.kind != "exam"}
    same(metrics.evaluate(preds, halves[half]), report[half]["metrics"])


@pytest.mark.parametrize("half", benchmark.HALVES)
@pytest.mark.parametrize("kind", ["multi", "exam"])
def test_scorecard_equals_the_research_code(half, kind, stored, halves):
    pytest.importorskip("scipy")
    data, report = stored
    key = "exam" if kind == "exam" else "pred"
    streams = [(s.labels, s.pieces, data[f"{half}|{key}|{s.id}"].astype(int)) for s in benchmark.of_kind(halves[half], kind)]
    same(scorecard.summarise(streams), report[half][f"scorecard_{kind}"])


@pytest.mark.parametrize("half", benchmark.HALVES)
def test_decoder_choice_equals_the_research_code(half, stored, halves):
    data, report = stored
    probabilities = {s.id: data[f"{half}|prob|{s.id}"] for s in halves[half] if s.kind != "exam"}
    chosen = calibrate.choose(probabilities, halves[half])
    want = report[half]["b3"]
    assert chosen["status"] == want["status"] and chosen["n_feasible"] == want["n_feasible"]
    same(chosen["params"], want["params"])
    same(chosen["best_unconstrained"], want["best_unconstrained"])
    decoded = calibrate.decode_all(chosen["params"], probabilities)
    scores = metrics.evaluate(decoded, halves[half])
    for name, value in report[half]["decoded_metrics_at_chosen_setting"].items():
        same(scores[name], value, name)


def record(order: int, f1: float, native: float = 0.99, recall: float = 0.97) -> dict:
    return {"order": order, "params": {}, "segment_f1_50_multi": f1, "native_macro_f1": native, "boundary_recall_multi": recall}


def test_the_rule_takes_the_best_feasible_setting_and_the_earlier_one_on_a_tie():
    records = [record(0, 0.99, native=0.97), record(1, 0.95), record(2, 0.96), record(3, 0.96), record(4, 0.98, recall=0.9)]
    chosen = calibrate.select_setting(records)
    assert chosen["status"] == "feasible" and chosen["chosen"]["order"] == 2 and chosen["n_feasible"] == 3


def test_the_rule_says_when_nothing_is_feasible():
    chosen = calibrate.select_setting([record(0, 0.99, native=0.9), record(1, 0.9, recall=None)])
    assert chosen == {"status": "infeasible", "chosen": None, "n_feasible": 0, "n_settings": 2}
