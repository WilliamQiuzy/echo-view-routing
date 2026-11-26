"""What part 1 left open, on the development benchmark (docs/ABLATIONS.md, part 3)."""
from __future__ import annotations

import json
import re

import numpy as np
import pytest

from echoview import decoder
from echoview.evaluation import ablations, benchmark, calibrate, comparison, metrics, open_points
from echoview.evaluation.ablations import CANDIDATE, NOT_SUPPORTED
from echoview.evaluation.comparison import ROUTER
from echoview.evaluation.open_points import BY_THE_RULE, KEPT_SETTING, STAYS, WITH_HIGHER
from echoview.reference import B3

from .test_ablations import MEMBERS, development, rows_of
from .test_comparison import logits_from
from .test_followup import two_halves

NOISE = 5.0  # stand-in logits that are wrong often enough for the settings, the halves and the decoders to differ


def both_halves(stored: dict, halves: dict) -> dict:
    return {half: comparison.member_probabilities(stored, sequences, MEMBERS) for half, sequences in halves.items()}


def numbers_at(setting: dict, probabilities: dict, sequences: tuple) -> dict:
    result = metrics.evaluate(calibrate.decode_all(setting, probabilities), sequences)
    return {key: result[key] for key in calibrate.REPORTED}


def f1_of(scored_numbers: dict) -> float:
    return scored_numbers["multi"]["segment_f1_50"]["value"]


# ------------------------------------------------------------------------------------------------- the rule
def entry(order: int, f1: float, macro: float, recall: float) -> dict:
    return {"order": order, "here": {"segment_f1_50_multi": f1, "native_macro_f1": macro, "boundary_recall_multi": recall},
            "there": {"segment_f1_50_multi": 1.0 - f1, "native_macro_f1": 0.5, "boundary_recall_multi": 0.5}}


def test_the_rule_keeps_both_floors_takes_the_highest_and_of_equals_the_earliest():
    table = [entry(0, 0.90, 0.99, 0.94),   # the highest, and its recall is below the floor
             entry(1, 0.80, 0.99, 0.96),
             entry(2, 0.85, 0.97, 0.99),   # its macro-F1 is below the floor
             entry(3, 0.80, 0.98, 0.95)]   # on the floors is enough, and it ties with the second
    found = open_points.by_the_rule(table, "here")
    assert found == {"status": "feasible", "n_feasible": 2, "n_settings": 4, "floors_dropped": False, "chosen": 1}
    # on the other half no setting keeps the floors: they are dropped, and the highest segment F1 takes their place
    dropped = open_points.by_the_rule(table, "there")
    assert dropped == {"status": "infeasible", "n_feasible": 0, "n_settings": 4, "floors_dropped": True, "chosen": 1}
    table[3]["there"]["segment_f1_50_multi"] = table[1]["there"]["segment_f1_50_multi"]
    assert open_points.by_the_rule(table, "there")["chosen"] == 1  # of equals the earlier
    assert (calibrate.NATIVE_FLOOR, calibrate.RECALL_FLOOR) == (0.980, 0.95)


def test_the_published_setting_is_known_by_its_three_numbers_and_is_never_an_added_one():
    assert open_points.is_published(dict(B3)) and open_points.is_published({**B3, "threshold": B3["threshold"] + comparison.THRESHOLD_CLOSE / 2})
    assert not open_points.is_published({**B3, "threshold": B3["threshold"] + 2 * comparison.THRESHOLD_CLOSE})
    assert not open_points.is_published({**B3, "half_window": B3["half_window"] + 2})
    assert not open_points.is_published({**B3, "nms_radius": B3["nms_radius"] + 1})
    assert open_points.is_published({**B3, "added": False}) and not open_points.is_published({**B3, "added": True})


# ----------------------------------------------------------------------------------- thresholds above the grid
def test_the_extended_grid_is_the_rules_grid_with_higher_thresholds_after_it():
    halves, stored = development(np.random.default_rng(11), noise=NOISE)
    natives = [comparison.member_probabilities(stored, halves["calibration"], MEMBERS)[s.id] for s in benchmark.of_kind(halves["calibration"], "native")]
    grid = open_points.extended_grid(natives)
    per_window = len(calibrate.QUANTILES) + len(open_points.HIGHER)
    assert len(grid) == len(calibrate.HALF_WINDOWS) * per_window and open_points.HIGHER == (0.9995, 0.9999, 1.0)
    assert min(open_points.HIGHER) > max(calibrate.QUANTILES)
    kept = [{key: value for key, value in setting.items() if key != "added"} for setting in grid if not setting["added"]]
    assert kept == calibrate.grid(natives)  # the rule's own settings, with the thresholds the rule gives them, in its order
    for k, half_window in enumerate(calibrate.HALF_WINDOWS):
        block = grid[k * per_window:(k + 1) * per_window]
        assert [s["half_window"] for s in block] == [half_window] * per_window
        assert [s["threshold_quantile"] for s in block] == [*calibrate.QUANTILES, *open_points.HIGHER]
        assert [s["added"] for s in block] == [False] * len(calibrate.QUANTILES) + [True] * len(open_points.HIGHER)
        scores = np.concatenate([decoder.boundary_scores(p, half_window)[1:] for p in natives])
        for setting in block:  # every threshold is the quantile of the scores inside the single recordings that its name says
            assert setting["threshold"] == pytest.approx(float(np.quantile(scores, setting["threshold_quantile"])), abs=1e-15)
        assert block[-1]["threshold"] == pytest.approx(scores.max())  # at quantile 1 only the largest score inside a single recording is a peak
        assert len({s["threshold"] for s in block}) > 3


def test_the_rule_may_take_a_higher_threshold_and_the_selection_half_reports(monkeypatch):
    halves, stored = development(np.random.default_rng(12), noise=NOISE)
    found = open_points.higher_thresholds(stored, halves, draws=60, seed=3)
    probabilities = both_halves(stored, halves)
    table = found["settings"]
    grid = open_points.extended_grid([probabilities["calibration"][s.id] for s in benchmark.of_kind(halves["calibration"], "native")])
    assert [{key: row[key] for key in grid[0]} for row in table] == grid  # the thresholds come from the calibration half
    assert [row["order"] for row in table] == list(range(55))
    for row in table:  # every setting on both halves, each half under its own name
        for half in halves:
            assert row[half] == numbers_at(row, probabilities[half], halves[half])
    assert sum(row["calibration"] != row["selection"] for row in table) > 40  # the two halves tell themselves apart on these stand-ins
    rule = found["rule"]
    on_calibration = open_points.by_the_rule(table, "calibration")
    assert {key: rule[key] for key in on_calibration} == on_calibration  # the rule reads the calibration half
    assert on_calibration["chosen"] != open_points.by_the_rule(table, "selection")["chosen"]  # and the selection half would choose another setting
    chosen = table[rule["chosen"]]
    assert rule["is_the_published_setting"] == open_points.is_published(chosen) is False
    assert rule["is_at_the_highest_quantile"] == (chosen["threshold_quantile"] == 1.0)
    rows = [comparison.Row(ROUTER, ROUTER, "A", comparison.decoded(probabilities["selection"], B3)),
            comparison.Row("chosen", ROUTER, "D", comparison.decoded(probabilities["selection"], chosen))]
    scored = comparison.score_rows(rows, halves["selection"], 60, 3)
    selection = found["selection"]
    assert selection["published"] == ablations.numbers(scored[ROUTER]) and selection["chosen"] == ablations.numbers(scored["chosen"])
    assert f1_of(selection["published"]) != f1_of(selection["chosen"])  # the two rows differ here, so exchanging them would show
    apart = selection["chosen_minus_published"]
    assert apart == ablations.minus(scored["chosen"], scored[ROUTER])  # the chosen one minus the published one
    assert apart["multi"]["segment_f1_50"]["value"] == pytest.approx(f1_of(selection["chosen"]) - f1_of(selection["published"]))
    assert apart["multi"]["segment_f1_50"]["level"] == 95.0
    published = metrics.evaluate(comparison.decoded(probabilities["selection"], B3), halves["selection"])["segment_f1_50_multi"]
    added = [row["selection"]["segment_f1_50_multi"] for row in table if row["added"]]
    assert found["added_on_selection"] == {"settings": 15, "above_the_published": sum(v > published for v in added), "highest": max(added)}
    assert max(added) != max(row["calibration"]["segment_f1_50_multi"] for row in table if row["added"])

    # the word: the published setting stays, or the choice is a candidate, or it is not supported
    real = ablations.minus

    def with_interval(interval, accuracy):
        def changed(first, second, level=comparison.DESCRIBED):
            out = real(first, second, level)
            out["multi"]["segment_f1_50"] = {**out["multi"]["segment_f1_50"], "interval": interval}
            out["recordings"]["accuracy"] = {**out["recordings"]["accuracy"], "value": accuracy}
            return out
        return changed

    for interval, accuracy, word in (([0.001, 0.02], 0.0, CANDIDATE), ([0.001, 0.02], -0.004, NOT_SUPPORTED), ([0.0, 0.02], 0.0, NOT_SUPPORTED),
                                     ([-0.01, 0.02], 0.01, NOT_SUPPORTED), ([-0.03, -0.01], 0.01, NOT_SUPPORTED), (None, 0.0, NOT_SUPPORTED)):
        monkeypatch.setattr(ablations, "minus", with_interval(interval, accuracy))
        assert open_points.higher_thresholds(stored, halves, draws=20, seed=3)["word"] == word, (interval, accuracy)
    monkeypatch.setattr(open_points, "is_published", lambda setting: True)
    monkeypatch.setattr(ablations, "minus", with_interval([0.001, 0.02], 0.0))
    assert open_points.higher_thresholds(stored, halves, draws=20, seed=3)["word"] == STAYS == "the published setting stays"
    assert WITH_HIGHER != ROUTER
    # a choice at quantile 1 lies at the end of the thresholds tried, and the output says so
    monkeypatch.undo()
    by_the_rule = open_points.by_the_rule
    for order, quantile, at_the_end in ((10, 1.0, True), (8, 0.9995, False), (7, 0.999, False)):
        monkeypatch.setattr(open_points, "by_the_rule", lambda table, half, order=order: {**by_the_rule(table, half), "chosen": order})
        forced = open_points.higher_thresholds(stored, halves, draws=20, seed=3)
        assert forced["settings"][forced["rule"]["chosen"]]["threshold_quantile"] == quantile and forced["rule"]["is_at_the_highest_quantile"] is at_the_end


def test_when_no_setting_keeps_the_floors_they_are_dropped_and_the_output_says_so():
    halves, stored = development(np.random.default_rng(20), noise=0.8)
    wrong = {key: np.roll(value, 1, axis=2) for key, value in stored.items()}  # every sample sure of another view: no setting keeps the floors
    found = open_points.higher_thresholds(wrong, halves, draws=20, seed=3)
    table, rule = found["settings"], found["rule"]
    assert (rule["status"], rule["n_feasible"], rule["floors_dropped"]) == ("infeasible", 0, True)
    scores = [row["calibration"]["segment_f1_50_multi"] for row in table]
    assert rule["chosen"] == scores.index(max(scores))  # the highest segment F1 on the calibration half, the earlier of equals
    exchanged = open_points.halves_exchanged(wrong, halves)
    assert exchanged["rule"]["floors_dropped"] is True and exchanged["rule"]["chosen"] == exchanged["highest_on_selection_whatever_the_floors"]
    others = open_points.numbers_of_the_other_decoders(wrong, halves)
    assert all(found["rule"]["floors_dropped"] and found["rule"]["n_feasible"] == 0 for found in others.values())
    for name, found in others.items():
        part_one = ablations.choose_number(name, both_halves(wrong, halves)["calibration"], halves["calibration"])
        assert part_one["floors_dropped"] and found["chosen"] == part_one["value"]


# ---------------------------------------------------------------------------------------- the halves exchanged
def test_the_halves_exchanged_weigh_every_setting_on_the_selection_half():
    halves, stored = development(np.random.default_rng(13), noise=NOISE)
    found = open_points.halves_exchanged(stored, halves)
    probabilities = both_halves(stored, halves)
    # the thresholds come from the selection half
    grid = calibrate.grid([probabilities["selection"][s.id] for s in benchmark.of_kind(halves["selection"], "native")])
    assert grid != calibrate.grid([probabilities["calibration"][s.id] for s in benchmark.of_kind(halves["calibration"], "native")])
    table = found["settings"]
    assert len(table) == 40 and [{key: row[key] for key in grid[0]} for row in table] == grid
    for row in table:
        for half in halves:
            assert row[half] == numbers_at(row, probabilities[half], halves[half])
        sel = row["selection"]
        assert row["feasible_on_selection"] == (sel["native_macro_f1"] >= 0.980 and (sel["boundary_recall_multi"] or 0) >= 0.95)
    rule = found["rule"]
    assert rule == open_points.by_the_rule(table, "selection") and rule["n_feasible"] == sum(row["feasible_on_selection"] for row in table)
    other_half = open_points.by_the_rule(table, "calibration")
    assert (rule["chosen"], rule["n_feasible"]) != (other_half["chosen"], other_half["n_feasible"])  # the calibration half would say something else
    again = comparison.choose(probabilities["selection"], halves["selection"], drop_floors=True)  # what part 1 reported of the exchanged choice
    assert rule["status"] == again["status"] and {key: table[rule["chosen"]][key] for key in again["params"]} == again["params"]
    scores = [row["selection"]["segment_f1_50_multi"] for row in table]
    assert found["highest_on_selection_whatever_the_floors"] == scores.index(max(scores))  # of equal ones the earliest
    assert scores != [row["calibration"]["segment_f1_50_multi"] for row in table]
    # the floors are read on the selection half: stand-ins that read every view right on the calibration half and wrong on the selection half
    clean_halves, clean = development(np.random.default_rng(22), noise=0.05)
    selection_ids = {s.id for s in clean_halves["selection"]}
    one_sided = {key: (np.roll(value, 1, axis=2) if key in selection_ids else value) for key, value in clean.items()}
    lopsided = open_points.halves_exchanged(one_sided, clean_halves)
    assert any(row["calibration"]["native_macro_f1"] >= 0.980 and (row["calibration"]["boundary_recall_multi"] or 0) >= 0.95 for row in lopsided["settings"])
    assert not any(row["feasible_on_selection"] for row in lopsided["settings"]) and lopsided["rule"]["n_feasible"] == 0
    assert lopsided["rule"]["floors_dropped"] is True


# ------------------------------------------------------------------------------------------------ the view test
def test_the_view_test_is_counted_peak_by_peak(monkeypatch):
    halves, stored = development(np.random.default_rng(14), noise=NOISE)
    found = open_points.view_test(stored, halves)
    probabilities = both_halves(stored, halves)
    half_window, threshold, radius = B3["half_window"], B3["threshold"], B3["nms_radius"]
    for half, sequences in halves.items():
        assert set(found[half]) == {"native", "pairs", "multi"}
        for kind, count in found[half].items():
            streams = benchmark.of_kind(sequences, kind)
            given = [probabilities[half][s.id] for s in streams]
            assert count["streams"] == len(streams) and count["samples"] == sum(len(p) for p in given)
            peaks = [decoder.non_max_suppression(decoder.boundary_scores(p, half_window), threshold, radius) for p in given]
            kept = [decoder.detect_boundaries(p, half_window, threshold, radius) for p in given]
            assert count["peaks"] == sum(int((found_peaks > 0).sum()) for found_peaks in peaks) and count["kept"] == sum(k.size for k in kept)
            assert count["dropped"] == count["peaks"] - count["kept"] >= 0
            assert count["streams_with_a_dropped_peak"] == sum(int((a > 0).sum()) > b.size for a, b in zip(peaks, kept))
            differ = [int((decoder.decode(p, half_window, threshold, radius) != ablations.without_the_view_test(p, B3)).sum()) for p in given]
            assert count["samples_that_change"] == sum(differ) and count["streams_that_change"] == sum(d > 0 for d in differ)
    assert sum(count["peaks"] for half in found.values() for count in half.values()) > 0

    def every(rows: np.ndarray) -> dict:
        return {half: {s.id: rows for s in sequences} for half, sequences in halves.items()}

    natives = len(benchmark.of_kind(halves["selection"], "native"))
    # one view throughout, sure in the first half and unsure in the second: the score has a peak, both sides name the same view, the test drops it
    flat = np.concatenate([rows_of([2] * 20, sure=0.99), rows_of([2] * 20, sure=0.3)])
    monkeypatch.setattr(open_points, "ensemble_probabilities", lambda stored, halves, temperature=1.0: every(flat))
    dropped = open_points.view_test(stored, halves)["selection"]["native"]
    assert dropped["peaks"] == dropped["dropped"] == dropped["streams_with_a_dropped_peak"] == natives and dropped["kept"] == 0
    assert dropped["streams_that_change"] == dropped["samples_that_change"] == 0 and dropped["samples"] == 40 * natives
    # two views with a cut between them: the peak is kept, and without the test nothing else is cut
    two = np.concatenate([rows_of([1] * 20, sure=0.99), rows_of([3] * 20, sure=0.99)])
    monkeypatch.setattr(open_points, "ensemble_probabilities", lambda stored, halves, temperature=1.0: every(two))
    kept = open_points.view_test(stored, halves)["selection"]["native"]
    assert kept["peaks"] == kept["kept"] == natives and kept["dropped"] == kept["samples_that_change"] == 0
    # a dropped peak that shows: the windows on its two sides name the same view, and the piece after it as a whole names another
    def mixed(second: float, third: float, n: int) -> np.ndarray:
        rows = np.full((n, 5), (1.0 - second - third) / 3)
        rows[:, 2], rows[:, 3] = second, third
        return rows

    shows = np.concatenate([rows_of([2] * 20, sure=0.99), mixed(0.50, 0.44, 8), mixed(0.42, 0.52, 14)])
    assert decoder.detect_boundaries(shows, half_window, threshold, radius).size == 0  # the released decoder makes no cut
    without = ablations.without_the_view_test(shows, B3)
    assert set(without[:20].tolist()) == {2} and set(without[20:].tolist()) == {3}  # without the test the piece after the peak is read as another view
    monkeypatch.setattr(open_points, "ensemble_probabilities", lambda stored, halves, temperature=1.0: every(shows))
    seen = open_points.view_test(stored, halves)["selection"]["native"]
    assert (seen["peaks"], seen["kept"], seen["dropped"], seen["streams_with_a_dropped_peak"]) == (natives, 0, natives, natives)
    assert seen["streams_that_change"] == natives and seen["samples_that_change"] == 22 * natives


# ---------------------------------------------------------------------------------- the other decoders' numbers
def test_every_other_decoder_is_given_at_every_value_of_its_number():
    halves, stored = development(np.random.default_rng(15), noise=NOISE)
    found = open_points.numbers_of_the_other_decoders(stored, halves)
    probabilities = both_halves(stored, halves)
    assert list(found) == list(ablations.GRIDS)
    not_the_best, spread = 0, 0
    for name, values in ablations.GRIDS.items():
        given = found[name]
        table = given["values"]
        assert [row["value"] for row in table] == list(values) and [row["order"] for row in table] == list(range(len(values)))
        for row in table:
            for half in halves:
                direct = metrics.evaluate(ablations.decode_with(name, row["value"], probabilities[half]), halves[half])
                assert row[half] == {key: direct[key] for key in calibrate.REPORTED}
        assert any(row["calibration"] != row["selection"] for row in table)
        part_one = ablations.choose_number(name, probabilities["calibration"], halves["calibration"])  # the value part 1 chose for this decoder
        assert given["rule"] == open_points.by_the_rule(table, "calibration")
        assert given["chosen"] == part_one["value"] == table[given["rule"]["chosen"]]["value"]
        assert (given["rule"]["n_feasible"], given["rule"]["floors_dropped"]) == (part_one["n_feasible"], part_one["floors_dropped"])
        scores = [row["selection"]["segment_f1_50_multi"] for row in table]
        at = scores[given["rule"]["chosen"]]
        summary = given["selection_segment_f1_50_multi"]
        assert (summary["of_the_chosen"], summary["lowest"], summary["highest"], summary["values"]) == (at, min(scores), max(scores), len(values))
        assert summary["within_0.01_of_the_chosen"] == sum(abs(s - at) <= 0.01 for s in scores) >= 1  # the chosen one is among them
        assert summary["within_0.02_of_the_chosen"] == sum(abs(s - at) <= 0.02 for s in scores) >= summary["within_0.01_of_the_chosen"]
        assert summary["above_the_chosen"] == sum(s > at for s in scores)
        not_the_best += at < max(scores)
        spread += summary["within_0.01_of_the_chosen"] < summary["within_0.02_of_the_chosen"] < len(values)
    assert not_the_best >= 1 and spread >= 1  # the summary is taken around the chosen value, and its two counts are not the same number


# ------------------------------------------------------------------------------------------- lower temperatures
def test_lower_temperatures_are_tried_with_those_of_part_one(monkeypatch):
    assert open_points.LOWER == (0.05, 0.10, 0.15, 0.20, 0.25) and open_points.TEMPERATURES == open_points.LOWER + ablations.TEMPERATURES
    assert list(open_points.TEMPERATURES) == sorted(set(open_points.TEMPERATURES)) and len(open_points.TEMPERATURES) == 40
    halves = two_halves()
    both = (*halves["calibration"], *halves["selection"])
    unsure = {key: value / 8.0 for key, value in logits_from(both, np.random.default_rng(16), noise=0.05).items()}  # right, and far from sure
    found = open_points.lower_temperatures(unsure, halves, draws=60, seed=7)
    assert found["chosen"] == 0.05 and found["is_the_lowest_tried"] is True and len(found["calibration_curve"]) == 40 and found["word"] == CANDIDATE
    # the recordings' part is part 1's own function on the longer list
    part_one = ablations.one_temperature(unsure, halves, draws=60, seed=7, temperatures=open_points.TEMPERATURES)
    assert {key: found[key] for key in part_one} == part_one
    assert ablations.one_temperature(unsure, halves, draws=60, seed=7)["chosen"] == 0.3  # and without the list it tries part 1's values, as before
    # a fit inside the values tried is not at their end: recordings that are sure of a wrong view want a temperature above 1
    wrong = {key: np.roll(value, 1, axis=2) * 3 for key, value in unsure.items()}
    monkeypatch.setattr(open_points, "TEMPERATURES", (0.5, 1.0, 2.0, 4.0))
    high = open_points.lower_temperatures(wrong, halves, draws=40, seed=7)
    assert high["chosen"] in (2.0, 4.0) and high["is_the_lowest_tried"] is False
    assert high["word"] == CANDIDATE  # by the rule of part 1: the log loss is lower, and no more recordings are wrong than before
    monkeypatch.setattr(open_points, "TEMPERATURES", (1.0,))  # nothing but the published probabilities: not supported
    assert open_points.lower_temperatures(unsure, halves, draws=40, seed=7)["word"] == NOT_SUPPORTED


def test_the_streams_are_scored_under_the_temperature(monkeypatch):
    halves, stored = development(np.random.default_rng(17), noise=NOISE)
    monkeypatch.setattr(open_points, "TEMPERATURES", (0.5,))  # the fit has one value to take
    found = open_points.lower_temperatures(stored, halves, draws=60, seed=7)
    assert found["chosen"] == 0.5
    plain = both_halves(stored, halves)
    sharper = both_halves({key: np.asarray(value, np.float64) / 0.5 for key, value in stored.items()}, halves)
    setting = comparison.choose(sharper["calibration"], halves["calibration"], drop_floors=True)  # the rule on the changed probabilities, its grid of 40
    rows = [comparison.Row(ROUTER, ROUTER, "A", comparison.decoded(plain["selection"], B3)),
            comparison.Row(KEPT_SETTING, ROUTER, "D", comparison.decoded(sharper["selection"], B3)),
            comparison.Row(BY_THE_RULE, ROUTER, "D", comparison.decoded(sharper["selection"], setting["params"]))]
    scored = comparison.score_rows(rows, halves["selection"], 60, 7)
    streams = found["streams"]
    assert streams["published"] == ablations.numbers(scored[ROUTER])
    assert streams["setting_by_the_rule"] == {"status": setting["status"], "n_feasible": setting["n_feasible"], "floors_dropped": setting["floors_dropped"],
                                              **setting["params"]}
    for name in (KEPT_SETTING, BY_THE_RULE):
        assert streams[name]["selection"] == ablations.numbers(scored[name])
        apart = streams[name]["minus_published"]
        assert apart == ablations.minus(scored[name], scored[ROUTER])  # under the temperature minus as published
        assert apart["multi"]["wrong_samples"]["value"] == pytest.approx(
            streams[name]["selection"]["multi"]["wrong_samples"]["value"] - streams["published"]["multi"]["wrong_samples"]["value"])
    # the three rows are three different things on these stand-ins: the plain probabilities, the changed ones, and the changed ones at another setting
    assert streams[KEPT_SETTING]["selection"] != streams["published"] and streams[BY_THE_RULE]["selection"] != streams[KEPT_SETTING]["selection"]
    thresholds = [row["threshold"] for row in calibrate.grid([sharper["calibration"][s.id] for s in benchmark.of_kind(halves["calibration"], "native")])]
    assert streams["grid_of_the_rule"] == {"settings": 40, "distinct_thresholds": len({round(t, 12) for t in thresholds}),
                                           "thresholds_at_zero": sum(t <= 0.0 for t in thresholds)}
    # when no setting keeps the floors under the temperature they are dropped, and the output says so
    wrong = {key: np.roll(value, 1, axis=2) for key, value in stored.items()}
    dropped = open_points.lower_temperatures(wrong, halves, draws=20, seed=7)["streams"]["setting_by_the_rule"]
    sharper_wrong = both_halves({key: np.asarray(value, np.float64) / 0.5 for key, value in wrong.items()}, halves)
    by_hand = comparison.choose(sharper_wrong["calibration"], halves["calibration"], drop_floors=True)
    assert dropped["floors_dropped"] is True and {key: dropped[key] for key in by_hand["params"]} == by_hand["params"]
    # at a temperature of 1 the published setting gives the published numbers again
    monkeypatch.setattr(open_points, "TEMPERATURES", (1.0,))
    unchanged = open_points.lower_temperatures(stored, halves, draws=40, seed=7)["streams"]
    assert unchanged[KEPT_SETTING]["selection"] == unchanged["published"]
    assert all(entry["value"] == 0.0 for part in unchanged[KEPT_SETTING]["minus_published"].values() for entry in part.values())


def test_sharp_probabilities_can_make_the_thresholds_of_the_grid_fall_together(monkeypatch):
    halves, stored = development(np.random.default_rng(21), noise=0.05)
    sure = {key: value * 6.0 for key, value in stored.items()}  # every sample sure of its view: almost every score inside a recording is zero
    monkeypatch.setattr(open_points, "TEMPERATURES", (0.05,))
    grid = open_points.lower_temperatures(sure, halves, draws=20, seed=7)["streams"]["grid_of_the_rule"]
    assert grid["settings"] == 40 and grid["distinct_thresholds"] < 40 and grid["thresholds_at_zero"] > 0


def test_the_ensembles_probabilities_at_a_temperature_of_one_are_those_of_part_one():
    halves, stored = development(np.random.default_rng(18), noise=NOISE)
    plain = open_points.ensemble_probabilities(stored, halves)
    direct = both_halves(stored, halves)
    assert set(plain) == set(halves) and all(set(plain[half]) == {s.id for s in halves[half]} for half in halves)
    assert all(np.array_equal(plain[half][key], direct[half][key]) for half in halves for key in direct[half])
    colder = open_points.ensemble_probabilities(stored, halves, 0.5)
    by_hand = both_halves({key: np.asarray(value, np.float64) / 0.5 for key, value in stored.items()}, halves)
    assert all(np.allclose(colder[half][key], by_hand[half][key]) for half in halves for key in by_hand[half])
    key = next(iter(direct["selection"]))
    assert colder["selection"][key].max(1).mean() > direct["selection"][key].max(1).mean()  # a temperature below 1 makes the system surer
    assert np.allclose(colder["selection"][key].sum(1), 1.0)


# ------------------------------------------------------------------------------------------------- the command
def test_the_command_holds_the_logits_to_the_published_number_and_writes_once(tmp_path, monkeypatch, capsys):
    whole, stored = development(np.random.default_rng(19), noise=0.6)
    halves = {"calibration": whole["calibration"][:-1], "selection": whole["selection"]}  # halves of two sizes, so that a count tells its half
    np.savez(tmp_path / "dev.npz", **stored)
    monkeypatch.setattr(ablations.benchmark, "load", lambda: halves)
    monkeypatch.setattr(ablations.benchmark, "load_test", lambda: pytest.fail("this study does not read the test benchmark"))
    selection = halves["selection"]
    f1 = metrics.evaluate(comparison.decoded(comparison.member_probabilities(stored, selection, MEMBERS), B3), selection)["segment_f1_50_multi"]
    out = tmp_path / "open.json"
    monkeypatch.setattr(comparison, "PUBLISHED_SELECTION_F1", round(f1, 4) + 0.01)
    with pytest.raises(SystemExit, match="must give"):
        open_points.main(["--logits", str(tmp_path / "dev.npz"), "--draws", "40", "--out", str(out)])
    assert not out.exists()
    monkeypatch.setattr(comparison, "PUBLISHED_SELECTION_F1", round(f1, 4))
    assert open_points.main(["--logits", str(tmp_path / "dev.npz"), "--draws", "40", "--out", str(out)]) == 0 and "wrote" in capsys.readouterr().out
    result = json.loads(out.read_text())
    assert result["benchmark"] == "development" and result["draws"] == 40 and result["sha256"] == {"logits": comparison.file_sha256(tmp_path / "dev.npz")}
    assert set(result) == {"sha256", "published_selection_f1", "benchmark", "draws", "seed", "sequences", "higher_thresholds", "halves_exchanged", "view_test",
                           "other_decoders", "temperature"}
    assert result["sequences"] == {"calibration": len(whole["calibration"]) - 1, "selection": len(selection)}
    assert result["published_selection_f1"] == pytest.approx(f1) and result["seed"] == comparison.SEED
    names = {s.id for half in whole.values() for s in half} | {f.video for half in whole.values() for s in half for f in s.fragments}
    assert not any(name in out.read_text() for name in names) and not re.search(r"NaN|Infinity", out.read_text())  # pooled numbers only
    with pytest.raises(SystemExit, match="does not replace"):
        open_points.main(["--logits", str(tmp_path / "dev.npz"), "--draws", "40", "--out", str(out)])


def test_the_plan_of_part_three_names_what_the_code_holds():
    """Section 10 of docs/ABLATIONS.md was written before the results of part 3: what it fixes is what the code holds."""
    import math
    from pathlib import Path

    plan = (Path(__file__).parents[1] / "docs" / "ABLATIONS.md").read_text()
    section = plan[plan.index("## 10. Part 3: what part 1 left open"):plan.index("## 11. Results of part 3")]
    assert "`python -m echoview.evaluation.open_points` is the command" in section
    assert "0.9995, 0.9999 and 1, which is the largest score inside a single recording" in section and open_points.HIGHER == (0.9995, 0.9999, 1.0)
    assert "15 more settings, 55 in all" in section and len(calibrate.HALF_WINDOWS) * len(open_points.HIGHER) == 15
    assert len(calibrate.HALF_WINDOWS) * (len(calibrate.QUANTILES) + len(open_points.HIGHER)) == 55
    assert "40 of the 55 settings below and 35 of the 40 temperatures are those of part 1" in section
    assert len(calibrate.HALF_WINDOWS) * len(calibrate.QUANTILES) == 40 and (len(ablations.TEMPERATURES), len(open_points.TEMPERATURES)) == (35, 40)
    assert "The added settings of a half window come after its own eight" in section and len(calibrate.QUANTILES) == 8
    natives = benchmark.of_kind(ablations.development_halves()["calibration"], "native")
    scores = sum(len(s) - 1 for s in natives)  # one score between every two neighbouring samples of a single recording
    assert f"give {scores:,} scores for a half window" in section and scores == 14_086
    assert (math.floor((1 - 0.9995) * (scores - 1)), math.floor((1 - 0.9999) * (scores - 1))) == (7, 1)  # the eighth largest; between the second and the third
    assert "0.05, 0.10, 0.15, 0.20 and 0.25" in section and open_points.LOWER == (0.05, 0.10, 0.15, 0.20, 0.25)
    assert "Part 1 chose 0.30, the lowest value that was tried" in section and min(ablations.TEMPERATURES) == 0.3
    assert "A probability counts as at least 1e-12, so one recording adds at most 27.6" in section and ablations.EPS == 1e-12
    assert round(-math.log(ablations.EPS), 1) == 27.6
    for word in (STAYS, CANDIDATE, NOT_SUPPORTED):
        assert f'"{word}"' in section
    assert "the mode filter, the mean filter, the minimum duration and the Viterbi path are given at every value of section 3" in section
    assert list(ablations.GRIDS) == ["mode filter", "mean filter", "minimum duration", "Viterbi path"]
    assert "A grid here has 7 or 8 values where the released decoder's has 40" in section and {len(values) for values in ablations.GRIDS.values()} == {7, 8}
    assert "the 95% interval of the difference in segment F1 at 50% on the multi streams lies wholly above zero" in section and comparison.DESCRIBED == 95.0
    assert open_points.KINDS == ("native", "pairs", "multi") and "for single recordings, pair streams and multi streams apart" in section
    assert "the floors are dropped. The setting with the highest segment F1 takes their place, the earlier of equals, and the output says so" in section
    assert "from its grid of 40" in section and "how many of the 40 are distinct and how many are zero" in section
