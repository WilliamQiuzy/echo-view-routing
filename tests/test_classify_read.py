"""The read of the test benchmark for what `echoview classify` returns (docs/EVALUATION.md)."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

from echoview import Ensemble, windows
from echoview.evaluation import benchmark, classify_read, comparison, run
from echoview.evaluation.classify_read import CLASSIFY, LESS, MAJORITY, MORE, NOT_SHOWN
from echoview.evaluation.comparison import ROUTER
from echoview.model import CLASSES
from echoview.pipeline import classification, softmax
from echoview.reference import B3

from .conftest import BrightnessModel, flat_frames
from .test_comparison import logits_from, sequences
from .test_followup import first_read_of, published_of, two_halves

MEMBERS = (0, 1, 2)
PLAX, PSAX, A4C, A5C, SC4C = range(5)


def leaning(view: int, other: int, by: float = 0.2) -> np.ndarray:
    """Logits of one model for one window that lean to `view` over `other`: 0.55 against 0.45 for `by` = 0.2."""
    row = np.full(5, -8.0)
    row[view], row[other] = by, 0.0
    return row


def sure(view: int) -> np.ndarray:
    row = np.full(5, -8.0)
    row[view] = 8.0
    return row


def weak(view: int, other: int) -> np.ndarray:
    return np.tile(leaning(view, other), (3, 1))  # the same for three seeds


def strong(view: int) -> np.ndarray:
    return np.tile(sure(view), (3, 1))


def two_rules_apart(seqs: tuple[benchmark.Sequence, ...], rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Stored logits on which the two rules differ twice. In the first recording most windows lean to the reference
    view and one window is sure of another: the majority is right and the mean probability is wrong. In the fifth,
    a recording of another examination, most windows lean to another view and three are sure of the reference view:
    the other way round."""
    stored = logits_from(seqs, rng, noise=0.05)
    n = len(windows.window_centres(len(seqs[0])))
    assert n == 8 and seqs[0].labels[0] == 0 and seqs[4].labels[0] == 4 and seqs[0].group != seqs[4].group
    stored[seqs[0].id] = np.stack([weak(0, 1)] * (n - 1) + [strong(1)])
    stored[seqs[4].id] = np.stack([weak(2, 4)] * (n - 3) + [strong(4)] * 3)
    return stored


def no_name_of_a_recording(text: str, seqs: tuple[benchmark.Sequence, ...]) -> bool:
    """Pooled numbers only: no id of a sequence, no key of a recording, nothing that looks like a key (16 hex digits)."""
    names = {s.id for s in seqs} | {f.video for s in seqs for f in s.fragments}
    key = r"(?<![0-9a-f.])(?=\d*[a-f])[0-9a-f]{16}(?![0-9a-f])"  # not the digits of a number: those follow a point or have no letter
    return not any(name in text for name in names) and not re.search(key, text)


# ------------------------------------------------------------------------------------------------- the rule
def test_the_rule_is_the_one_classify_returns():
    system = Ensemble.from_models([BrightnessModel(shift) for shift in (0.0, 20.0, -20.0)], B3, device="cpu")
    frames = flat_frames([100] * 12 + [150] * 8)
    logits = system.window_logits(frames)
    returned, direct = system.classify(frames), classification(logits)
    assert returned.view == direct.view and returned.probability == direct.probability and returned.seeds_agree == direct.seeds_agree
    np.testing.assert_array_equal(returned.probabilities, softmax(logits).mean(0).mean(0))
    np.testing.assert_array_equal(returned.by_seed, direct.by_seed)
    assert direct.n_windows == len(logits) and direct.view == CLASSES[int(direct.probabilities.argmax())]
    for wrong in (np.zeros((0, 3, 5)), np.zeros((4, 0, 5)), np.zeros((4, 3, 4)), np.zeros((4, 5))):
        with pytest.raises(ValueError, match="expected logits"):
            classification(wrong)


def test_the_view_is_the_largest_mean_probability_and_not_a_seed_or_a_vote():
    # two seeds lean to A4C (0.55 against 0.45 for A5C), the third is sure of A5C: the mean is 0.37 for A4C and 0.63 for A5C
    one_window = np.stack([leaning(A4C, A5C), leaning(A4C, A5C), sure(A5C)])
    result = classification(np.stack([one_window] * 4))
    assert result.view == "A5C" and not result.seeds_agree and result.n_windows == 4
    by_seed = softmax(one_window)  # every window is the same, so a seed's mean over its windows is its window
    assert 0.54 < by_seed[0, A4C] < 0.56 and 0.44 < by_seed[0, A5C] < 0.46 and by_seed[2, A5C] > 0.999
    assert result.probability == pytest.approx(by_seed[:, A5C].mean()) and 0.63 < result.probability < 0.64
    np.testing.assert_allclose(result.by_seed, by_seed)
    np.testing.assert_allclose(result.probabilities, by_seed.mean(0))
    assert result.probabilities.sum() == pytest.approx(1.0) and result.probabilities[A4C] == pytest.approx(2 * by_seed[0, A4C] / 3, abs=1e-6)
    # the windows count, not their majority: three windows lean to PLAX and one is sure of PSAX, in every seed
    by_window = classification(np.stack([np.tile(leaning(PLAX, PSAX), (3, 1))] * 3 + [np.tile(sure(PSAX), (3, 1))]))
    assert by_window.view == "PSAX" and by_window.seeds_agree
    # only the middle seed names another view: the seeds do not agree
    middle = classification(np.stack([np.stack([sure(A4C), sure(A5C), sure(A4C)])] * 2))
    assert middle.view == "A4C" and not middle.seeds_agree and middle.probability == pytest.approx(2 / 3, abs=1e-6)
    assert classification(np.stack([np.stack([sure(SC4C)] * 3)] * 2)).seeds_agree


def test_the_numbers_are_those_of_the_development_reports_table():
    seqs = sequences()
    natives = benchmark.of_kind(seqs, "native")
    stored = two_rules_apart(seqs, np.random.default_rng(1))
    table = run.recording_table(stored, natives, MEMBERS)
    got = classify_read.described(classify_read.decisions(stored, natives), natives, draws=20, seed=3)
    assert got["recordings"] == table["recordings"] == 12 and got["errors"] == table["errors"] == 1
    assert got["confusion"] == table["confusion"] and got["confusion"][0][1] == 1  # the first recording, read as PSAX
    assert got["probability"]["ece"] == pytest.approx(table["ece"], abs=1e-12)
    assert got["probability"]["median_probability"] == pytest.approx(table["median_confidence"], abs=1e-12)
    assert got["units_drawn"] == 4 and got["units_with_a_wrong_recording"] == 1
    stored[seqs[1].id] = np.stack([strong(3)] * 8)  # a second wrong recording in the first examination,
    stored[seqs[5].id] = np.stack([strong(3)] * 8)  # and one in the second
    more = classify_read.described(classify_read.decisions(stored, natives), natives, draws=20, seed=3)
    assert more["errors"] == 3 and more["units_with_a_wrong_recording"] == 2  # examinations are counted, not recordings


def test_recordings_are_drawn_by_examination_where_the_benchmark_names_one():
    natives = benchmark.of_kind(sequences(), "native")
    assert classify_read.units(natives).tolist() == [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]
    unnamed = benchmark.of_kind(two_halves()["selection"], "native")
    assert classify_read.units(unnamed).tolist() == list(range(12))  # the development benchmark names none: every recording is its own unit
    unit = np.array([0, 0, 1, 2, 2, 2])
    counts = classify_read.drawn(unit, draws=300, seed=4)
    assert counts.shape == (300, 6)
    assert (counts[:, 0] == counts[:, 1]).all() and (counts[:, 3] == counts[:, 4]).all() and (counts[:, 4] == counts[:, 5]).all()  # a unit comes whole
    assert (counts[:, [0, 2, 3]].sum(1) == 3).all() and not (counts[:, 0] == counts[:, 2]).all()  # three units are drawn, each draw another mix
    np.testing.assert_array_equal(counts, classify_read.drawn(unit, draws=300, seed=4))
    assert not np.array_equal(counts, classify_read.drawn(unit, draws=300, seed=5))


# ------------------------------------------------------------------------------------------- the probability
def test_the_bins_are_those_of_the_calibration_error():
    probability = np.array([0.05, 0.31, 0.35, 0.60, 0.90, 0.95, 0.99, 1.0])
    right = np.array([0, 0, 1, 1, 1, 0, 1, 1], bool)
    bins = classify_read.reliability(probability, right)
    assert len(bins) == 10 and sum(b["recordings"] for b in bins) == 8 and sum(b["right"] for b in bins) == 5
    assert [b["recordings"] for b in bins] == [1, 0, 0, 2, 0, 1, 0, 0, 1, 3]  # 0.90 lies in the bin up to 0.9
    assert bins[1]["mean_probability"] is None and bins[1]["share_right"] is None
    assert bins[9] == {"above": pytest.approx(0.9), "up_to": 1.0, "recordings": 3, "right": 2, "mean_probability": pytest.approx(0.98),
                       "share_right": pytest.approx(2 / 3)}
    assert classify_read._gap(bins) == pytest.approx(run.expected_calibration_error(probability, right))


def test_parts_of_equal_count_follow_the_order_of_the_probabilities():
    probability = np.array([0.99, 0.50, 0.97, 0.97, 0.80, 0.98, 0.60])
    right = np.array([1, 0, 1, 0, 1, 1, 0], bool)
    parts = classify_read.equal_count(probability, right, bins=3)
    assert [p["recordings"] for p in parts] == [3, 2, 2] and [(p["from"], p["to"]) for p in parts] == [(0.50, 0.80), (0.97, 0.97), (0.98, 0.99)]
    assert [p["right"] for p in parts] == [1, 1, 2] and parts[0]["mean_probability"] == pytest.approx((0.5 + 0.6 + 0.8) / 3)
    assert parts[1]["share_right"] == 0.5 and sum(p["recordings"] for p in parts) == 7
    by_hand = 3 / 7 * abs(1 / 3 - 1.9 / 3) + 2 / 7 * abs(0.5 - 0.97) + 2 / 7 * abs(1.0 - 0.985)
    assert classify_read._gap(parts) == pytest.approx(by_hand)
    assert len(classify_read.equal_count(probability[:2], right[:2], bins=10)) == 2  # fewer recordings than parts: no empty part
    eighty = classify_read.equal_count(np.linspace(0.5, 1.0, 800), np.ones(800, bool))
    assert [p["recordings"] for p in eighty] == [80] * 10
    tied = classify_read.equal_count(np.full(4, 0.9), np.array([0, 1, 1, 1], bool), bins=2)
    assert [p["right"] for p in tied] == [1, 2]  # recordings of one probability stay in the order in which they were given


def test_the_separation_is_the_chance_that_a_right_answer_has_the_higher_probability():
    right = np.array([1, 1, 1, 0, 0], bool)
    assert classify_read.separation(np.array([0.9, 0.8, 0.7, 0.6, 0.5]), right) == 1.0
    assert classify_read.separation(np.array([0.3, 0.2, 0.1, 0.6, 0.5]), right) == 0.0
    assert classify_read.separation(np.full(5, 0.7), right) == 0.5  # every pair is a tie
    assert classify_read.separation(np.array([0.9, 0.8, 0.4, 0.6, 0.5]), right) == pytest.approx(4 / 6)
    assert classify_read.separation(np.array([0.9, 0.8, 0.7]), np.ones(3, bool)) is None  # nothing is wrong
    assert classify_read.separation(np.array([0.9, 0.8, 0.7]), np.zeros(3, bool)) is None
    probability, counts = np.array([0.9, 0.8, 0.4, 0.6, 0.5]), np.array([2, 0, 3, 1, 2])
    repeated = classify_read.separation(np.repeat(probability, counts), np.repeat(right, counts))
    assert classify_read.separation(probability, right, counts) == pytest.approx(repeated)  # a weight is a recording counted that often
    assert classify_read.separation(probability, right, np.array([1, 1, 1, 0, 0])) is None  # a draw without a wrong recording


def test_the_interval_of_the_separation_draws_whole_examinations():
    rng = np.random.default_rng(12)
    unit = np.repeat(np.arange(40), 3)  # 40 examinations of three recordings
    right = rng.random(120) > 0.15
    probability = np.where(right, rng.uniform(0.6, 1.0, 120), rng.uniform(0.4, 0.9, 120))
    counts = classify_read.drawn(unit, draws=500, seed=5)
    got = classify_read.separation_interval(probability, right, counts)
    assert got["value"] == pytest.approx(classify_read.separation(probability, right)) and got["level"] == 95.0 and got["draws_without_a_number"] == 0
    by_hand = np.array([classify_read.separation(probability, right, row) for row in counts])
    assert got["interval"] == pytest.approx(np.percentile(by_hand, [2.5, 97.5]).tolist())
    narrower = classify_read.separation_interval(probability, right, counts, level=90.0)
    assert narrower["interval"] == pytest.approx(np.percentile(by_hand, [5.0, 95.0]).tolist()) and narrower["interval"][0] > got["interval"][0]
    few = np.array([0, 0, 1, 1, 2, 2])  # three examinations, the wrong recordings in two of them
    one_wrong = np.array([1, 1, 1, 0, 0, 1], bool)
    six = np.array([0.9, 0.8, 0.4, 0.6, 0.5, 0.95])
    small = classify_read.drawn(few, draws=400, seed=5)
    rare = classify_read.separation_interval(six, one_wrong, small)
    without = sum(classify_read.separation(six, one_wrong, row) is None for row in small)
    assert rare["draws_without_a_number"] == without > 0  # the examination without a wrong recording, drawn three times
    none_wrong = classify_read.separation_interval(np.linspace(0.5, 0.9, 6), np.ones(6, bool), small)
    assert none_wrong == {"value": None, "level": 95.0, "interval": None, "draws_without_a_number": 400}


def test_a_mean_has_the_interval_of_its_resamples():
    values = np.array([1.0, 3.0, 5.0, 7.0])
    counts = np.array([[1, 1, 1, 1], [2, 0, 2, 0], [0, 0, 0, 4], [4, 0, 0, 0], [0, 2, 2, 0]], float)
    got = classify_read.mean_interval(values, counts)
    means = np.array([4.0, 3.0, 7.0, 1.0, 4.0])
    assert got["value"] == 4.0 and got["level"] == 95.0 and got["interval"] == pytest.approx(np.percentile(means, [2.5, 97.5]).tolist())
    unit = np.array([0, 1, 1, 2, 2, 2, 2, 2])  # units of one, two and five recordings: a resample holds 3 to 15 recordings
    values = np.array([9.0, 1.0, 2.0, 5.0, 5.0, 6.0, 7.0, 4.0])
    counts = classify_read.drawn(unit, draws=300, seed=2)
    assert len(set(counts.sum(1).tolist())) > 3
    slow = [sum(c * v for c, v in zip(row, values)) / sum(row) for row in counts]
    uneven = classify_read.mean_interval(values, counts)
    assert uneven["interval"] == pytest.approx(np.percentile(slow, [2.5, 97.5]).tolist())
    assert uneven["interval"] != pytest.approx(np.percentile(counts @ values / len(values), [2.5, 97.5]).tolist())  # a mean of what was drawn


def test_the_probability_part_holds_what_the_plan_names():
    probabilities = np.array([[0.90, 0.10, 0.0, 0.0, 0.0],   # PLAX, right
                              [0.20, 0.80, 0.0, 0.0, 0.0],   # PSAX, right
                              [0.00, 0.00, 0.6, 0.4, 0.0],   # A5C read as A4C
                              [0.00, 0.00, 0.3, 0.7, 0.0],   # A5C, right
                              [0.05, 0.00, 0.0, 0.0, 0.95]])  # SC4C, right
    truth = np.array([PLAX, PSAX, A5C, A5C, SC4C])
    unit = np.array([0, 0, 1, 1, 2])
    got = classify_read.probability_part(probabilities, truth, unit, draws=200, seed=9)
    probability, right = np.array([0.9, 0.8, 0.6, 0.7, 0.95]), np.array([1, 1, 0, 1, 1], bool)
    assert got["mean_probability"] == pytest.approx(0.79) and got["share_right"] == 0.8
    assert got["median_probability"] == pytest.approx(0.8) and got["median_probability_right"] == pytest.approx(0.85)
    assert got["median_probability_wrong"] == pytest.approx(0.6)
    assert got["ece"] == pytest.approx(run.expected_calibration_error(probability, right)) and got["bins"] == classify_read.reliability(probability, right)
    assert got["bins_equal_count"] == classify_read.equal_count(probability, right)
    assert got["ece_equal_count"] == pytest.approx(np.mean([0.6, 0.3, 0.2, 0.1, 0.05]))  # five recordings, each a part of its own
    counts = classify_read.drawn(unit, 200, 9)  # the examinations hold two, two and one recording

    def slow(values) -> list[float]:
        return np.percentile([sum(c * v for c, v in zip(row, values)) / sum(row) for row in counts], [2.5, 97.5]).tolist()

    gap, brier = (probability - right).tolist(), [0.01 + 0.01, 0.04 + 0.04, 0.36 + 0.36, 0.09 + 0.09, 0.0025 + 0.0025]
    loss = (-np.log([0.9, 0.8, 0.4, 0.7, 0.95])).tolist()
    for name, values in (("probability_minus_right", gap), ("brier", brier), ("log_loss", loss)):
        assert got[name]["value"] == pytest.approx(np.mean(values)) and got[name]["level"] == 95.0
        assert got[name]["interval"] == pytest.approx(slow(values))
    assert got["probability_minus_right"]["value"] == pytest.approx(0.79 - 0.8)
    assert slow(gap) != pytest.approx(slow(brier)) and slow(brier) != pytest.approx(slow(loss))
    assert got["separation"] == classify_read.separation_interval(probability, right, counts) and got["separation"]["value"] == 1.0
    low, high = got["brier"]["interval"]
    assert low < got["brier"]["value"] < high
    never = classify_read.probability_part(np.array([[1.0, 0, 0, 0, 0], [1.0, 0, 0, 0, 0]]), np.array([PLAX, PSAX]), np.array([0, 1]), draws=20, seed=1)
    assert never["log_loss"]["value"] == pytest.approx(-np.log(1e-12) / 2) and never["brier"]["value"] == 1.0  # a probability of zero counts as 1e-12
    # seven recordings on which the two calibration errors differ: two of them share a bin of equal width, none a part of equal count
    seven, read_right = np.array([0.99, 0.50, 0.97, 0.97, 0.80, 0.98, 0.60]), np.array([1, 0, 1, 0, 1, 1, 0], bool)
    two_views = np.zeros((7, 5))
    two_views[:, PLAX], two_views[:, PSAX] = np.where(read_right, seven, 1 - seven), np.where(read_right, 1 - seven, seven)
    apart = classify_read.probability_part(two_views, np.full(7, PLAX), np.arange(7), draws=20, seed=1)
    assert apart["ece_equal_count"] == pytest.approx((0.01 + 0.5 + 0.03 + 0.97 + 0.2 + 0.02 + 0.6) / 7)
    assert apart["ece"] == pytest.approx((0.5 + 0.6 + 0.2 + 4 * abs(0.75 - 0.9775)) / 7) and apart["ece"] != pytest.approx(apart["ece_equal_count"])


def test_the_numbers_about_the_probability_share_one_set_of_draws():
    rng = np.random.default_rng(21)
    unit = np.repeat(np.arange(30), 3)  # 30 examinations of three recordings
    truth = rng.integers(0, 5, 90)
    logits = 3.0 * np.eye(5)[truth] + rng.normal(size=(90, 5)) * 1.5
    probabilities = softmax(logits)
    got = classify_read.probability_part(probabilities, truth, unit, draws=300, seed=11)
    counts = classify_read.drawn(unit, 300, 11)
    probability, right = probabilities.max(1), probabilities.argmax(1) == truth
    assert 5 < (~right).sum() < 45

    def slow(values) -> list[float]:
        return np.percentile([sum(c * v for c, v in zip(row, values)) / sum(row) for row in counts], [2.5, 97.5]).tolist()

    by_hand = {"probability_minus_right": probability - right, "brier": ((probabilities - np.eye(5)[truth]) ** 2).sum(1),
               "log_loss": -np.log(probabilities[np.arange(90), truth])}
    for name, values in by_hand.items():
        assert got[name]["value"] == pytest.approx(values.mean()) and got[name]["interval"] == pytest.approx(slow(values.tolist()))
    separations = [classify_read.separation(probability, right, row) for row in counts]
    assert got["separation"]["interval"] == pytest.approx(np.percentile(separations, [2.5, 97.5]).tolist())
    other = classify_read.drawn(unit, 300, 12)  # other draws give other intervals: the test can tell
    assert classify_read.mean_interval(by_hand["log_loss"], other)["interval"] != pytest.approx(got["log_loss"]["interval"])
    assert classify_read.separation_interval(probability, right, other)["interval"] != pytest.approx(got["separation"]["interval"])


def test_the_seeds_are_counted_against_the_answer():
    agree = np.array([1, 1, 0, 0, 0, 1], bool)
    right = np.array([1, 0, 1, 0, 0, 1], bool)
    assert classify_read.seeds_part(agree, right) == {"recordings_where_the_seeds_differ": 3, "of_them_wrong": 2, "wrong_where_the_seeds_agree": 1}


def test_the_word_needs_the_interval_and_the_exact_test():
    assert classify_read.word({"interval": [0.001, 0.02]}, 0.01) == MORE
    assert classify_read.word({"interval": [-0.02, -0.001]}, 0.01) == LESS
    assert classify_read.word({"interval": [0.001, 0.02]}, 0.125) == NOT_SHOWN  # four recordings against none: the interval alone is not enough
    assert classify_read.word({"interval": [0.001, 0.02]}, 0.05) == NOT_SHOWN  # below 0.05, not at it
    assert classify_read.word({"interval": [-0.01, 0.01]}, 0.01) == NOT_SHOWN
    assert classify_read.word({"interval": [0.0, 0.01]}, 0.01) == NOT_SHOWN  # an end at exactly zero is not on one side
    assert classify_read.word({"interval": [-0.01, 0.0]}, 0.01) == NOT_SHOWN
    assert classify_read.word({"interval": None}, 0.01) == NOT_SHOWN
    assert classify_read.word({"interval": [0.001, 0.02]}, float("nan")) == NOT_SHOWN


# ------------------------------------------------------------------------------------------------- the trial
def development(rng: np.random.Generator) -> tuple[dict, dict, dict]:
    halves = two_halves()
    both = (*halves["calibration"], *halves["selection"])
    stored = logits_from(both, rng, noise=0.05)
    stored[halves["selection"][0].id] = np.stack([weak(0, 1)] * 7 + [strong(1)])
    report = {"systems": {classify_read.TRIAL_SYSTEM: {half: {"native": run.recording_table(stored, benchmark.of_kind(halves[half], "native"), MEMBERS)}
                                                       for half in benchmark.HALVES}}}
    return halves, stored, json.loads(json.dumps(report))


def test_the_trial_holds_the_functions_to_the_development_report():
    halves, stored, report = development(np.random.default_rng(2))
    held = classify_read.trial(stored, halves, report)
    assert held["calibration"]["errors"] == 0 and held["selection"]["errors"] == 1 and held["selection"]["recordings"] == 12
    for change in ("errors", "ece", "median_confidence", "recordings"):
        other = json.loads(json.dumps(report))
        other["systems"][classify_read.TRIAL_SYSTEM]["selection"]["native"][change] += 1 if change in ("errors", "recordings") else 1e-4
        with pytest.raises(SystemExit, match="do not come out: selection"):
            classify_read.trial(stored, halves, other)
    other = json.loads(json.dumps(report))
    other["systems"][classify_read.TRIAL_SYSTEM]["calibration"]["native"]["confusion"][0][0] -= 1
    with pytest.raises(SystemExit, match="do not come out: calibration: confusion"):
        classify_read.trial(stored, halves, other)
    with pytest.raises(SystemExit, match="not the development report"):
        classify_read.trial(stored, halves, {"systems": {"ensemble": {}}})
    near = json.loads(json.dumps(report))  # logits of another run: the sixth decimal of a probability moves, no count does
    near["systems"][classify_read.TRIAL_SYSTEM]["selection"]["native"]["ece"] += 1e-6
    assert classify_read.trial(stored, halves, near) == held
    for key in ("ece", "median_confidence"):  # a number that is not finite does not pass for close
        not_a_number = json.loads(json.dumps(report))
        not_a_number["systems"][classify_read.TRIAL_SYSTEM]["calibration"]["native"][key] = float("nan")
        with pytest.raises(SystemExit, match="do not come out: calibration"):
            classify_read.trial(stored, halves, not_a_number)


def test_the_trial_command_reads_the_development_benchmark_only(tmp_path, monkeypatch, capsys):
    halves, stored, report = development(np.random.default_rng(3))
    np.savez(tmp_path / "dev.npz", **stored)
    (tmp_path / "report.json").write_text(json.dumps(report))
    monkeypatch.setattr(classify_read.benchmark, "load", lambda: halves)
    monkeypatch.setattr(classify_read.benchmark, "load_test", lambda: pytest.fail("the trial does not read the test benchmark"))
    out = tmp_path / "trial.json"
    assert classify_read.main(["trial", "--logits", str(tmp_path / "dev.npz"), "--report", str(tmp_path / "report.json"), "--out", str(out)]) == 0
    written = json.loads(out.read_text())
    assert written["benchmark"] == "development" and written["held_to"] == classify_read.TRIAL_SYSTEM and written["close"] == classify_read.CLOSE
    assert written["numbers_reproduced"]["selection"]["errors"] == 1 and "wrote" in capsys.readouterr().out
    assert written["sha256"] == {"logits": comparison.file_sha256(tmp_path / "dev.npz"), "report": comparison.file_sha256(tmp_path / "report.json")}
    both = (*halves["calibration"], *halves["selection"])
    assert classify_read.trial_held(written) and no_name_of_a_recording(out.read_text().replace(json.dumps(written["sha256"], indent=1), ""), both)
    with pytest.raises(SystemExit, match="does not replace"):
        classify_read.main(["trial", "--logits", str(tmp_path / "dev.npz"), "--report", str(tmp_path / "report.json"), "--out", str(out)])


# -------------------------------------------------------------------------------------------------- the read
HELD_TRIAL = {"benchmark": "development", "held_to": classify_read.TRIAL_SYSTEM, "close": classify_read.CLOSE,
              "sha256": {"logits": "a" * 64, "report": "b" * 64},
              "numbers_reproduced": {half: {"recordings": 12, "errors": 0, "confusion": [], "ece": 0.01, "median_probability": 0.98}
                                     for half in benchmark.HALVES}}


def test_a_trial_that_held_is_a_whole_output_of_the_trial_command():
    assert classify_read.trial_held(HELD_TRIAL)
    one_half = HELD_TRIAL["numbers_reproduced"]["selection"]
    for change in ({"benchmark": "test"}, {"held_to": "ensemble"}, {"close": 1e-3}, {"sha256": {"logits": "a" * 64}},
                   {"sha256": {"logits": "a" * 63, "report": "b" * 64}}, {"sha256": {"logits": "g" * 64, "report": "b" * 64}}, {"sha256": "a" * 64},
                   {"numbers_reproduced": {"selection": one_half}}, {"numbers_reproduced": {"calibration": one_half, "selection": {}}},
                   {"numbers_reproduced": {"calibration": one_half, "selection": {**one_half, "recordings": 0}}},
                   {"numbers_reproduced": {"calibration": one_half, "selection": {k: v for k, v in one_half.items() if k != "ece"}}},
                   {"numbers_reproduced": [one_half, one_half]}):
        assert not classify_read.trial_held({**HELD_TRIAL, **change}), change
    for key in HELD_TRIAL:
        assert not classify_read.trial_held({k: v for k, v in HELD_TRIAL.items() if k != key}), key
    for other in ([], "trial", None, 3):
        assert not classify_read.trial_held(other)


def test_the_read_scores_what_classify_returns_beside_the_majority_rule(monkeypatch):
    seqs = sequences()
    natives = benchmark.of_kind(seqs, "native")
    stored = two_rules_apart(seqs, np.random.default_rng(4))
    monkeypatch.setattr(comparison, "PUBLISHED", published_of(seqs, stored))
    first = first_read_of(seqs, stored, 80, 7)
    given, said = [], []
    part, word = classify_read.probability_part, classify_read.word
    monkeypatch.setattr(classify_read, "probability_part", lambda *args: given.append(args) or part(*args))
    monkeypatch.setattr(classify_read, "word", lambda difference, p: said.append((difference, p)) or word(difference, p))
    result = classify_read.read(stored, seqs, first, draws=80, seed=7)
    assert result["first_read_rows_reproduced"] == [ROUTER, "Perfect on the 0.5 s grid"]
    assert result["published_numbers_reproduced"] == comparison.PUBLISHED[ROUTER]
    majority, by_classify = result["rules"][MAJORITY], result["rules"][CLASSIFY]
    assert majority == first["rows"][ROUTER]["recordings"]  # the published rule, with the first read's intervals
    assert majority["accuracy"]["value"] == by_classify["accuracy"]["value"] == pytest.approx(11 / 12)
    assert "cuts_per_recording" in majority and "cuts_per_recording" not in by_classify
    assert result["confusion_majority"][4][2] == 1 and result["classify"]["confusion"][0][1] == 1  # each rule misreads one, not the same one
    assert result["classify"]["errors"] == 1 and result["classify"]["recordings"] == len(natives)
    assert result["classify"]["units_drawn"] == 4 and result["classify"]["units_with_a_wrong_recording"] == 1
    assert result["recordings_only_one_rule_reads_right"] == {"only_classify": 1, "only_the_majority": 1, "p_two_sided": 1.0}
    difference = result["classify_minus_majority"]
    assert tuple(difference) == ("accuracy", "macro_f1", "recall_A4C", "recall_A5C", "a4c_as_a5c", "a5c_as_a4c")
    assert difference["accuracy"]["value"] == 0.0 and difference["accuracy"]["level"] == 95.0
    low, high = difference["accuracy"]["interval"]
    assert low < 0 < high and result["word"] == NOT_SHOWN  # one recording each way round, in two examinations
    assert said == [(difference["accuracy"], 1.0)] and difference["accuracy"] != difference["macro_f1"]  # the word rests on the accuracy
    (_, truth, unit, draws, seed), = given
    assert unit.tolist() == [s.group for s in natives] and (draws, seed) == (80, 7)  # the probability's intervals draw examinations too
    assert truth.tolist() == [int(s.labels[0]) for s in natives]
    probability = result["classify"]["probability"]
    assert sum(b["recordings"] for b in probability["bins"]) == 12 and probability["share_right"] == pytest.approx(11 / 12)
    assert probability["median_probability_wrong"] == pytest.approx(classification(stored[seqs[0].id]).probability)
    assert probability["median_probability_right"] > 0.9 > probability["median_probability_wrong"]
    assert probability["separation"]["value"] == 1.0  # the one wrong recording has the lowest probability of all
    assert result["classify"]["seeds"] == {"recordings_where_the_seeds_differ": 0, "of_them_wrong": 0, "wrong_where_the_seeds_agree": 1}
    assert no_name_of_a_recording(json.dumps(comparison._clean(result), allow_nan=False), seqs)


def test_two_rules_that_agree_differ_by_nothing_in_every_draw(monkeypatch):
    seqs = sequences()
    stored = logits_from(seqs, np.random.default_rng(13), noise=0.05)
    stored[seqs[2].id] = np.stack([strong(4)] * 8)  # one recording that both rules misread in the same way
    monkeypatch.setattr(comparison, "PUBLISHED", published_of(seqs, stored))
    result = classify_read.read(stored, seqs, first_read_of(seqs, stored, 80, 7), draws=80, seed=7)
    assert result["rules"][CLASSIFY]["accuracy"] == result["rules"][MAJORITY]["accuracy"]
    assert result["rules"][MAJORITY]["accuracy"]["value"] == pytest.approx(11 / 12)
    assert result["rules"][MAJORITY]["accuracy"]["interval"][0] < result["rules"][MAJORITY]["accuracy"]["interval"][1]
    for name in ("accuracy", "macro_f1"):  # paired: the same draws for both rules, so not one draw tells them apart
        assert result["classify_minus_majority"][name]["value"] == 0.0 and result["classify_minus_majority"][name]["interval"] == [0.0, 0.0]
    assert result["recordings_only_one_rule_reads_right"] == {"only_classify": 0, "only_the_majority": 0, "p_two_sided": 1.0}
    assert result["word"] == NOT_SHOWN and result["confusion_majority"] == result["classify"]["confusion"]


def test_the_word_follows_the_interval_and_the_exact_test(monkeypatch):
    seqs = sequences()
    stored = logits_from(seqs, np.random.default_rng(5), noise=0.05)
    for k in (1, 4, 7, 10):  # one recording in every examination: the majority is wrong, the mean probability is right
        stored[seqs[k].id] = np.stack([weak((k + 1) % 5, k % 5)] * 5 + [strong(k % 5)] * 3)
    monkeypatch.setattr(comparison, "PUBLISHED", published_of(seqs, stored))
    result = classify_read.read(stored, seqs, first_read_of(seqs, stored, 80, 7), draws=80, seed=7)
    assert result["recordings_only_one_rule_reads_right"] == {"only_classify": 4, "only_the_majority": 0, "p_two_sided": 0.125}
    difference = result["classify_minus_majority"]["accuracy"]
    assert difference["value"] == pytest.approx(4 / 12) and difference["interval"][0] > 0
    assert result["word"] == NOT_SHOWN  # the interval lies above zero, and four recordings against none are not enough for the exact test
    for k in (0, 2, 3, 5):  # eight against none: p = 2 / 256
        stored[seqs[k].id] = np.stack([weak((k + 1) % 5, k % 5)] * 5 + [strong(k % 5)] * 3)
    monkeypatch.setattr(comparison, "PUBLISHED", published_of(seqs, stored))
    more = classify_read.read(stored, seqs, first_read_of(seqs, stored, 80, 7), draws=80, seed=7)
    assert more["recordings_only_one_rule_reads_right"] == {"only_classify": 8, "only_the_majority": 0, "p_two_sided": 2 / 256}
    assert more["classify_minus_majority"]["accuracy"]["value"] == pytest.approx(8 / 12) and more["word"] == MORE


def test_nothing_of_classify_is_computed_before_the_checks_hold(monkeypatch):
    seqs = sequences()
    stored = two_rules_apart(seqs, np.random.default_rng(6))
    numbers = published_of(seqs, stored)
    first = first_read_of(seqs, stored, 50, 3)
    monkeypatch.setattr(classify_read, "decisions", lambda *given: pytest.fail("classify's rule was applied before the checks held"))
    monkeypatch.setattr(classify_read, "classification", lambda *given: pytest.fail("classify's rule was applied before the checks held"))
    monkeypatch.setattr(comparison, "PUBLISHED", {ROUTER: {**numbers[ROUTER], "native_accuracy": numbers[ROUTER]["native_accuracy"] - 0.01}})
    with pytest.raises(SystemExit, match="do not give the published numbers"):
        classify_read.read(stored, seqs, first, draws=50, seed=3)
    monkeypatch.setattr(comparison, "PUBLISHED", numbers)
    changed = json.loads(json.dumps(first))
    changed["rows"][ROUTER]["recordings"]["accuracy"]["value"] += 1e-9
    with pytest.raises(SystemExit, match="first read does not come out again: row: EchoRouter"):
        classify_read.read(stored, seqs, changed, draws=50, seed=3)
    with pytest.raises(SystemExit, match="first read does not come out again: draws"):
        classify_read.read(stored, seqs, first, draws=51, seed=3)


def read_inputs(tmp_path, monkeypatch, rng):
    seqs = sequences()
    stored = two_rules_apart(seqs, rng)
    np.savez(tmp_path / "logits.npz", **stored)
    monkeypatch.setattr(classify_read, "DRAWS", 60)
    (tmp_path / "first.json").write_text(json.dumps(first_read_of(seqs, stored, 60, classify_read.SEED)))
    (tmp_path / "trial.json").write_text(json.dumps(HELD_TRIAL))
    monkeypatch.setattr(classify_read.benchmark, "load_test", lambda: seqs)
    monkeypatch.setattr(classify_read.benchmark, "load", lambda: pytest.fail("the read does not load the development benchmark"))
    monkeypatch.setattr(comparison, "PUBLISHED", published_of(seqs, stored))
    monkeypatch.setattr(comparison, "TEST_LOGITS_SHA256", comparison.file_sha256(tmp_path / "logits.npz"))
    monkeypatch.setattr(comparison, "FIRST_READ_SHA256", comparison.file_sha256(tmp_path / "first.json"))
    monkeypatch.setattr(classify_read, "READ_SHA256", None)  # this made-up benchmark has not been read for the question

    def command(out="classify_test.json", **other) -> list[str]:
        files = {"logits": "logits.npz", "earlier": "first.json", "trial": "trial.json", **other}
        given = [arg for key, name in files.items() if name is not None for arg in (f"--{key}", str(tmp_path / name))]
        return ["read", *given, "--draws", "60", "--out", str(tmp_path / out)]

    return seqs, stored, command


def test_the_read_command_reads_once(tmp_path, monkeypatch, capsys):
    seqs, stored, command = read_inputs(tmp_path, monkeypatch, np.random.default_rng(8))
    out = tmp_path / "classify_test.json"
    (tmp_path / "other.json").write_text(json.dumps({**json.loads((tmp_path / "first.json").read_text()), "seed": 1}))
    (tmp_path / "failed_trial.json").write_text(json.dumps({"benchmark": "development", "held_to": classify_read.TRIAL_SYSTEM}))
    (tmp_path / "broken_trial.json").write_text("{not json")
    with monkeypatch.context() as held_back:  # while a check of the command fails, the benchmark is not loaded and nothing is scored
        held_back.setattr(classify_read, "read", lambda *given, **more: pytest.fail("the read ran although a check failed"))
        held_back.setattr(classify_read.benchmark, "load_test", lambda: pytest.fail("the test benchmark was loaded although a check failed"))
        with pytest.raises(SystemExit, match="the test benchmark is not read: the plan fixes 60 draws"):
            classify_read.main([*command()[:-3], "50", "--out", str(out)])
        with pytest.raises(SystemExit, match="the read needs --out"):
            classify_read.main(command()[:-2])
        with pytest.raises(SystemExit, match="the test benchmark is not read: not a file"):
            classify_read.main(command(trial="no_trial.json"))
        held_back.setattr(comparison, "TEST_LOGITS_SHA256", "0" * 64)
        with pytest.raises(SystemExit, match="the test benchmark is not read: the logits file is not the one the plan names"):
            classify_read.main(command())
        held_back.setattr(comparison, "TEST_LOGITS_SHA256", comparison.file_sha256(tmp_path / "logits.npz"))
        with pytest.raises(SystemExit, match="the test benchmark is not read: --earlier must be the output of the comparison's first read"):
            classify_read.main(command(earlier="other.json"))
        for name in ("failed_trial.json", "broken_trial.json", "first.json"):
            with pytest.raises(SystemExit, match="the test benchmark is not read: --trial must be the output of a trial on development data that held"):
                classify_read.main(command(trial=name))
    np.savez(tmp_path / "few.npz", **{s.id: stored[s.id] for s in seqs[1:]})
    monkeypatch.setattr(comparison, "TEST_LOGITS_SHA256", comparison.file_sha256(tmp_path / "few.npz"))
    with pytest.raises(SystemExit, match="lacks the logits of 1 of the"):
        classify_read.main(command(logits="few.npz"))
    monkeypatch.setattr(comparison, "TEST_LOGITS_SHA256", comparison.file_sha256(tmp_path / "logits.npz"))
    assert not out.exists() and not list(tmp_path.glob("*.partial"))  # a stopped read leaves no output
    described = classify_read.described
    monkeypatch.setattr(classify_read, "described", lambda *given: (_ for _ in ()).throw(RuntimeError("stopped after the checks")))
    with pytest.raises(RuntimeError, match="stopped after the checks"):
        classify_read.main(command())
    assert not out.exists() and not list(tmp_path.glob("*.partial"))  # a run that stops after the checks shows no number
    monkeypatch.setattr(classify_read, "described", described)
    assert classify_read.main(command()) == 0 and "wrote" in capsys.readouterr().out
    result = json.loads(out.read_text())
    assert result["benchmark"] == "test" and result["sequences"] == len(seqs) and result["draws"] == 60 and result["seed"] == classify_read.SEED
    assert result["sha256"] == {key: comparison.file_sha256(tmp_path / name) for key, name in
                                (("logits", "logits.npz"), ("first_read", "first.json"), ("trial", "trial.json"))}
    direct = classify_read.read(stored, seqs, json.loads((tmp_path / "first.json").read_text()), draws=60, seed=classify_read.SEED)
    assert {key: result[key] for key in direct} == json.loads(json.dumps(comparison._clean(direct)))
    assert no_name_of_a_recording(json.dumps({k: v for k, v in result.items() if k != "sha256"}), seqs) and "NaN" not in out.read_text()
    before = out.read_bytes()
    with pytest.raises(SystemExit, match="does not replace"):
        classify_read.main(command())
    assert out.read_bytes() == before


def test_after_the_read_the_command_only_reproduces_it(tmp_path, monkeypatch):
    command = read_inputs(tmp_path, monkeypatch, np.random.default_rng(9))[-1]
    out = tmp_path / "classify_test.json"
    with pytest.raises(SystemExit, match="there is no read of it to reproduce"):  # before the read nothing can be reproduced
        classify_read.main(command(out="never.json", reproduce="first.json"))
    assert classify_read.main(command()) == 0
    monkeypatch.setattr(classify_read, "READ_SHA256", comparison.file_sha256(out))  # from now on the read has been made
    with pytest.raises(SystemExit, match="the test benchmark is not read: it has been read for this question"):
        classify_read.main(command(out="never.json"))
    changed = json.loads(out.read_text())
    changed["classify"]["probability"]["ece"] += 1e-9
    (tmp_path / "changed.json").write_text(json.dumps(changed))
    with pytest.raises(SystemExit, match="the test benchmark is not read: it has been read for this question"):  # another file than the read's output
        classify_read.main(command(out="never.json", reproduce="changed.json"))
    assert classify_read.main(command(out="again.json", reproduce="classify_test.json")) == 0
    assert json.loads((tmp_path / "again.json").read_text()) == json.loads(out.read_text())
    monkeypatch.setattr(classify_read, "READ_SHA256", comparison.file_sha256(tmp_path / "changed.json"))
    with pytest.raises(SystemExit, match="the read does not come out again"):
        classify_read.main(command(out="never.json", reproduce="changed.json"))
    assert not (tmp_path / "never.json").exists()


def test_the_code_names_the_read_once_it_exists():
    """Until the read is made the code names no output and the documents hold none; from then on the command is held to it."""
    folder = Path(__file__).parents[1] / "docs" / "evaluation"
    output, tried = folder / "classify_test.json", folder / "classify_trial.json"
    if output.is_file():
        assert classify_read.READ_SHA256 == comparison.file_sha256(output)
        assert json.loads(output.read_text())["sha256"] == {"logits": comparison.TEST_LOGITS_SHA256, "first_read": comparison.FIRST_READ_SHA256,
                                                            "trial": comparison.file_sha256(tried)}
    else:
        assert classify_read.READ_SHA256 is None
