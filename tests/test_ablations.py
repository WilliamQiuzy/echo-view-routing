"""What the parts of the published system add, on the development benchmark (docs/ABLATIONS.md, part 1)."""
from __future__ import annotations

import itertools
import json
import re

import numpy as np
import pytest

from echoview.evaluation import ablations, benchmark, calibrate, comparison, metrics
from echoview.evaluation.ablations import AHEAD, BEHIND, CANDIDATE, DURATION, MEAN, MODE, NONE, NOT_SHOWN, NOT_SUPPORTED, VITERBI
from echoview.evaluation.comparison import ROUTER
from echoview.reference import B3

from .test_comparison import logits_from
from .test_followup import two_halves

MEMBERS = (0, 1, 2)


def rows_of(labels: list[int], sure: float = 0.9) -> np.ndarray:
    """Probabilities (samples, 5) whose likeliest view is the given one."""
    prob = np.full((len(labels), 5), (1 - sure) / 4)
    prob[np.arange(len(labels)), labels] = sure
    return prob


def development(rng: np.random.Generator, noise: float = 0.05) -> tuple[dict, dict]:
    halves = two_halves()
    return halves, logits_from((*halves["calibration"], *halves["selection"]), rng, noise=noise)


# ---------------------------------------------------------------------------------------------- other decoders
def test_a_mode_filter_drops_a_view_that_stands_alone():
    assert ablations.mode_filter(rows_of([0, 0, 1, 0, 0]), 3).tolist() == [0, 0, 0, 0, 0]
    assert ablations.mode_filter(rows_of([0, 0, 1, 1, 0, 0]), 3).tolist() == [0, 0, 1, 1, 0, 0]  # two samples hold against a window of three
    assert ablations.mode_filter(rows_of([0, 0, 1, 1, 0, 0]), 5).tolist() == [0, 0, 0, 0, 0, 0]
    assert ablations.mode_filter(rows_of([0, 1]), 3).tolist() == [0, 1]  # a tie: every sample keeps its own view
    assert ablations.mode_filter(rows_of([2, 0, 1]), 3).tolist() == [2, 0, 1]
    # at the ends the window is cut off: the first sample sees 3, 1, 1 and the last sees 0, 0, 3
    assert ablations.mode_filter(rows_of([3, 1, 1, 0, 0, 3]), 5).tolist() == [1, 1, 1, 0, 0, 0]


def test_a_mean_filter_averages_the_probabilities():
    prob = rows_of([0, 1, 0], sure=0.6)
    prob[1] = [0.38, 0.62, 0.0, 0.0, 0.0]
    assert ablations.mean_filter(prob, 3).tolist() == [0, 0, 0]  # 0.6 + 0.38 + 0.6 against 0.1 + 0.62 + 0.1
    sure = rows_of([0, 1, 0], sure=0.7)
    sure[1] = [0.0, 1.0, 0.0, 0.0, 0.0]
    assert ablations.mean_filter(sure, 3).tolist() == [1, 0, 1]  # at the ends two samples are summed: 0.7 against 1.075; in the middle 1.4 against 1.15
    flat = rows_of([2] * 7)
    assert ablations.mean_filter(flat, 5).tolist() == [2] * 7 and ablations.mean_filter(flat, 31).tolist() == [2] * 7


def test_a_minimum_duration_gives_a_short_piece_to_its_longer_neighbour():
    assert ablations.minimum_duration(rows_of([0] * 5 + [1] * 2 + [0] * 5), 3).tolist() == [0] * 12
    assert ablations.minimum_duration(rows_of([0] * 2 + [1] * 5), 3).tolist() == [1] * 7  # a piece at the start has one neighbour
    assert ablations.minimum_duration(rows_of([0] * 3 + [1] + [2] * 3), 2).tolist() == [0, 0, 0, 0, 2, 2, 2]  # of equal neighbours the earlier
    assert ablations.minimum_duration(rows_of([0] * 3 + [1] + [2] * 4), 2).tolist() == [0, 0, 0, 2, 2, 2, 2, 2]  # the longer neighbour
    assert ablations.minimum_duration(rows_of([0] * 5 + [1] * 3 + [0] * 5), 3).tolist() == [0] * 5 + [1] * 3 + [0] * 5  # three samples last
    assert ablations.minimum_duration(rows_of([0, 1]), 5).tolist() in ([0, 0], [1, 1])  # two short pieces: one is left
    assert ablations.minimum_duration(rows_of([4] * 6), 30).tolist() == [4] * 6
    chain = ablations.minimum_duration(rows_of([0, 0, 0, 0, 1, 2, 2, 0, 0, 0, 0]), 3)  # the single sample goes first, to the longer side
    assert chain.tolist() == [0] * 11


def test_the_viterbi_path_is_the_best_of_all_paths():
    rng = np.random.default_rng(0)
    for penalty in (0.0, 0.3, 1.0, 5.0):
        prob = rng.dirichlet(np.ones(5), size=5)
        logp = np.log(prob)

        def value(path, logp=logp, penalty=penalty) -> float:
            return sum(logp[t, v] for t, v in enumerate(path)) - penalty * sum(a != b for a, b in zip(path, path[1:]))

        best = max(value(path) for path in itertools.product(range(5), repeat=5))
        found = ablations.viterbi(prob, penalty)
        assert value(tuple(found)) == pytest.approx(best)
    prob = rng.dirichlet(np.ones(5), size=20)
    assert ablations.viterbi(prob, 0.0).tolist() == prob.argmax(1).tolist()  # a change that costs nothing: the likeliest view of every sample
    assert len(set(ablations.viterbi(prob, 1e6).tolist())) == 1 and ablations.viterbi(prob, 1e6)[0] == np.log(prob).sum(0).argmax()
    assert ablations.viterbi(rows_of([1]), 2.0).tolist() == [1]


def test_the_one_number_is_chosen_by_the_projects_rule(monkeypatch):
    halves, stored = development(np.random.default_rng(1))
    calibration = halves["calibration"]
    probabilities = comparison.member_probabilities(stored, calibration, MEMBERS)
    for name in ablations.OTHERS:
        chosen = ablations.choose_number(name, probabilities, calibration)
        assert chosen["value"] in ablations.GRIDS[name] and chosen["grid"] == len(ablations.GRIDS[name])
        direct = metrics.evaluate(ablations.decode_with(name, chosen["value"], probabilities), calibration)
        assert chosen["calibration"] == {key: direct[key] for key in calibrate.REPORTED}
        assert chosen["status"] in ("feasible", "infeasible") and chosen["floors_dropped"] == (chosen["status"] != "feasible")
    seen = []

    def by_value(name, value, given):  # every number gives the same labels: the earliest of equals must win
        seen.append(value)
        return {key: p.argmax(1) for key, p in given.items()}

    monkeypatch.setattr(ablations, "decode_with", by_value)
    assert ablations.choose_number(MODE, probabilities, calibration)["value"] == ablations.GRIDS[MODE][0] and seen == list(ablations.GRIDS[MODE])
    wrong = {key: np.roll(p, 1, axis=1) for key, p in probabilities.items()}  # every recording misread: no number keeps the floors
    equal = ablations.choose_number(MODE, wrong, calibration)  # the floors are dropped, and of equal numbers the earliest still wins
    assert equal["floors_dropped"] and equal["value"] == ablations.GRIDS[MODE][0]
    monkeypatch.undo()
    dropped = ablations.choose_number(VITERBI, wrong, calibration)
    assert dropped["floors_dropped"] and dropped["status"] == "infeasible" and dropped["n_feasible"] == 0 and dropped["value"] in ablations.GRIDS[VITERBI]


def test_the_word_for_another_decoder_needs_an_interval_on_one_side_of_zero():
    assert ablations.word({"interval": [0.01, 0.05]}) == AHEAD and ablations.word({"interval": [-0.05, -0.01]}) == BEHIND
    assert ablations.word({"interval": [-0.01, 0.05]}) == NOT_SHOWN and ablations.word({"interval": [0.0, 0.05]}) == NOT_SHOWN
    assert ablations.word({"interval": None}) == NOT_SHOWN


# ----------------------------------------------------------------------------------------- the four questions
def test_every_subset_of_the_seeds_is_a_system_of_its_own():
    halves, stored = development(np.random.default_rng(2), noise=2.5)
    found = ablations.ensemble_size(stored, halves, draws=60, seed=5)
    names = ["seed 0", "seed 1", "seed 2", "seeds 0+1", "seeds 0+2", "seeds 1+2", "seeds 0+1+2"]
    assert list(found["systems"]) == names and [found["systems"][n]["members"] for n in names][3:] == [[0, 1], [0, 2], [1, 2], [0, 1, 2]]
    calibration, selection = halves["calibration"], halves["selection"]
    for name in ("seed 1", "seeds 0+2"):
        entry = found["systems"][name]
        members = tuple(entry["members"])
        chosen = comparison.choose(comparison.member_probabilities(stored, calibration, members), calibration, drop_floors=False)
        assert entry["setting"] == {"status": chosen["status"], **chosen["params"]}
        direct = metrics.evaluate(comparison.decoded(comparison.member_probabilities(stored, selection, members), chosen["params"]), selection)
        assert entry["selection"]["multi"]["segment_f1_50"]["value"] == pytest.approx(direct["segment_f1_50_multi"])
        assert entry["selection"]["recordings"]["accuracy"]["value"] == pytest.approx(direct["native_accuracy"])
        assert set(entry["classify"]) == {"errors", "ece", "mean_probability"}
    whole = found["systems"]["seeds 0+1+2"]["minus_the_whole_ensemble"]
    assert whole["multi"]["segment_f1_50"]["value"] == 0.0 and whole["multi"]["segment_f1_50"]["interval"] == [0.0, 0.0]
    all_three = found["systems"]["seeds 0+1+2"]["selection"]["multi"]["segment_f1_50"]["value"]
    apart = [name for name in names[:6] if found["systems"][name]["selection"]["multi"]["segment_f1_50"]["value"] != all_three]
    assert apart  # with this much noise some smaller system differs from the three together
    for name in apart:
        mine = found["systems"][name]["selection"]["multi"]["segment_f1_50"]["value"]
        assert found["systems"][name]["minus_the_whole_ensemble"]["multi"]["segment_f1_50"]["value"] == pytest.approx(mine - all_three)
    assert all(isinstance(found["systems"][name]["is_the_published_setting"], bool) for name in names)
    by_size = found["mean_segment_f1_50_multi_by_size"]
    singles = [found["systems"][n]["selection"]["multi"]["segment_f1_50"]["value"] for n in names[:3]]
    assert list(by_size) == ["1", "2", "3"] and by_size["1"] == pytest.approx(np.mean(singles))
    assert by_size["3"] == found["systems"]["seeds 0+1+2"]["selection"]["multi"]["segment_f1_50"]["value"]


def test_the_released_decoder_is_set_against_five_others():
    halves, stored = development(np.random.default_rng(3), noise=1.5)
    found = ablations.other_decoders(stored, halves, draws=60, seed=5)
    selection = halves["selection"]
    published = metrics.evaluate(comparison.decoded(comparison.member_probabilities(stored, selection, MEMBERS), B3), selection)
    assert found["released"]["setting"] == dict(B3)
    assert found["released"]["selection"]["multi"]["segment_f1_50"]["value"] == pytest.approx(published["segment_f1_50_multi"])
    assert list(found["others"]) == [NONE, MODE, MEAN, DURATION, VITERBI, ablations.NO_VIEW_TEST]
    assert found["others"][NONE]["chosen"] is None  # nothing is chosen for it
    raw = metrics.evaluate(comparison.likeliest(comparison.member_probabilities(stored, selection, MEMBERS)), selection)
    assert found["others"][NONE]["selection"]["multi"]["segment_f1_50"]["value"] == pytest.approx(raw["segment_f1_50_multi"])
    for name, entry in found["others"].items():
        difference = entry["released_minus_this"]["multi"]["segment_f1_50"]
        mine, theirs = found["released"]["selection"]["multi"]["segment_f1_50"]["value"], entry["selection"]["multi"]["segment_f1_50"]["value"]
        assert difference["value"] == pytest.approx(mine - theirs) and difference["level"] == 95.0
        if name == ablations.NO_VIEW_TEST:  # the released decoder without one of its parts is described and gets no word
            assert "word" not in entry and "decides" not in entry and set(entry["chosen"]["value"]) >= {"half_window", "threshold", "nms_radius"}
            cal = comparison.member_probabilities(stored, halves["calibration"], MEMBERS)
            decoded = {key: ablations.without_the_view_test(p, entry["chosen"]["value"]) for key, p in cal.items()}
            direct = metrics.evaluate(decoded, halves["calibration"])
            assert entry["chosen"]["calibration"]["segment_f1_50_multi"] == pytest.approx(direct["segment_f1_50_multi"])
            continue
        assert entry["decides"]["level"] == 99.0 and entry["decides"]["value"] == pytest.approx(mine - theirs)
        assert entry["word"] == ablations.word(entry["decides"]) and entry["word"] in (AHEAD, BEHIND, NOT_SHOWN)
        if name != NONE:
            calibration = halves["calibration"]
            again = ablations.choose_number(name, comparison.member_probabilities(stored, calibration, MEMBERS), calibration)
            assert entry["chosen"] == again
            direct = metrics.evaluate(ablations.decode_with(name, again["value"], comparison.member_probabilities(stored, selection, MEMBERS)), selection)
            assert entry["selection"]["multi"]["segment_f1_50"]["value"] == pytest.approx(direct["segment_f1_50_multi"])


def test_without_its_view_test_the_decoder_cuts_at_every_peak():
    prob = np.concatenate([rows_of([0] * 20, sure=0.9), rows_of([0] * 20, sure=0.5)])  # the same view throughout, surer in the first half
    setting = {"half_window": 5, "threshold": 0.01, "nms_radius": 5}
    from echoview import decoder

    assert decoder.detect_boundaries(prob, 5, 0.01, 5).size == 0  # the released decoder: both sides name the same view, so no cut
    peaks = decoder.non_max_suppression(decoder.boundary_scores(prob, 5), 0.01, 5)
    assert peaks.size >= 1 and ablations.without_the_view_test(prob, setting).tolist() == [0] * 40  # the pieces still take their likeliest view
    two = np.concatenate([rows_of([1] * 20), rows_of([3] * 10)])
    assert ablations.without_the_view_test(two, setting).tolist() == [1] * 20 + [3] * 10
    assert ablations.without_the_view_test(two, {**setting, "threshold": 10.0}).tolist() == [1] * 30  # no peak that high: one piece


def test_the_released_decoder_at_every_setting_of_its_grid():
    halves, stored = development(np.random.default_rng(4), noise=0.8)
    found = ablations.decoder_settings(stored, halves)
    assert len(found["settings"]) == len(calibrate.HALF_WINDOWS) * len(calibrate.QUANTILES) == found["selection_segment_f1_50_multi"]["settings"]
    probabilities = comparison.member_probabilities(stored, halves["selection"], MEMBERS)
    third = found["settings"][3]
    direct = metrics.evaluate(calibrate.decode_all(third, probabilities), halves["selection"])
    assert third["order"] == 3 and third["selection"] == {key: direct[key] for key in calibrate.REPORTED}
    scores = [entry["selection"]["segment_f1_50_multi"] for entry in found["settings"]]
    summary = found["selection_segment_f1_50_multi"]
    assert (summary["lowest"], summary["highest"], summary["median"]) == (min(scores), max(scores), pytest.approx(float(np.median(scores))))
    published = metrics.evaluate(calibrate.decode_all(B3, probabilities), halves["selection"])["segment_f1_50_multi"]
    assert found["published_selection"] == pytest.approx(published)
    assert summary["within_0.01_of_the_published"] == sum(abs(s - published) <= 0.01 for s in scores) <= summary["within_0.02_of_the_published"]
    assert summary["within_0.02_of_the_published"] == sum(abs(s - published) <= 0.02 for s in scores)
    assert summary["above_the_published"] == sum(s > published for s in scores)
    assert summary["above_the_published"] + sum(s <= published for s in scores) == len(scores)
    feasible = [entry for entry in found["settings"] if entry["feasible_on_calibration"]]
    assert found["feasible_on_calibration"]["settings"] == len(feasible)
    for entry in found["settings"]:
        cal = entry["calibration"]
        assert entry["feasible_on_calibration"] == (cal["native_macro_f1"] >= 0.980 and (cal["boundary_recall_multi"] or 0) >= 0.95)
    exchanged = found["halves_exchanged"]
    again = comparison.choose(probabilities, halves["selection"], drop_floors=False)
    assert exchanged["setting"] == again["params"] and exchanged["status"] == again["status"]
    other_half = comparison.member_probabilities(stored, halves["calibration"], MEMBERS)
    for name, setting in (("chosen_on_selection", again["params"]), ("published", B3)):
        direct = metrics.evaluate(calibrate.decode_all(setting, other_half), halves["calibration"])
        assert exchanged["calibration"][name] == {key: direct[key] for key in calibrate.REPORTED}


def test_one_temperature_is_fitted_on_one_half_and_reported_on_the_other(monkeypatch):
    halves = two_halves()
    both = (*halves["calibration"], *halves["selection"])
    TEMPERATURES = ablations.TEMPERATURES
    unsure = {key: value / 8.0 for key, value in logits_from(both, np.random.default_rng(5), noise=0.05).items()}  # right, and far from sure
    found = ablations.one_temperature(unsure, halves, draws=80, seed=5)
    assert found["chosen"] == min(ablations.TEMPERATURES) and len(found["calibration_curve"]) == len(ablations.TEMPERATURES)
    losses = [entry["log_loss"] for entry in found["calibration_curve"]]
    assert losses == sorted(losses)  # every lower temperature makes right answers surer
    before, after = found["selection"]["as published"], found["selection"]["with the temperature"]
    assert before["temperature"] == 1.0 and after["temperature"] == found["chosen"] and before["errors"] == after["errors"] == 0
    assert after["log_loss"]["value"] < before["log_loss"]["value"] and after["mean_probability"] > before["mean_probability"]
    difference = found["with_minus_published"]["log_loss"]
    assert difference["value"] == pytest.approx(after["log_loss"]["value"] - before["log_loss"]["value"]) and difference["interval"][1] < 0
    assert found["with_minus_published"]["brier"]["value"] == pytest.approx(after["brier"]["value"] - before["brier"]["value"])
    assert found["word"] == CANDIDATE
    natives = benchmark.of_kind(halves["selection"], "native")
    direct = ablations.recording_probabilities(unsure, natives, found["chosen"])
    assert direct.shape == (len(natives), 5) and after["mean_probability"] == pytest.approx(direct.max(1).mean())
    monkeypatch.setattr(ablations, "TEMPERATURES", (1.0,))  # nothing but the published probabilities to choose from
    same = ablations.one_temperature(unsure, halves, draws=40, seed=5)
    assert same["chosen"] == 1.0 and same["word"] == NOT_SUPPORTED and same["with_minus_published"]["log_loss"]["value"] == 0.0
    monkeypatch.setattr(ablations, "TEMPERATURES", (0.5, 1.0, 2.0))
    wrong = {key: np.roll(value, 1, axis=2) * 3 for key, value in unsure.items()}  # every recording sure of a wrong view: the higher temperature wins
    assert ablations.one_temperature(wrong, halves, draws=40, seed=5)["chosen"] == 2.0
    # the fit reads the calibration half alone: there the recordings are right and unsure, on the selection half they are sure and wrong
    cal_ids = {s.id for s in halves["calibration"]}
    two_kinds = {key: (unsure[key] if key in cal_ids else wrong[key]) for key in unsure}
    fitted = ablations.one_temperature(two_kinds, halves, draws=40, seed=5)
    assert fitted["chosen"] == 0.5 and fitted["word"] == NOT_SUPPORTED  # sharper probabilities make sure-and-wrong recordings cost more
    assert fitted["with_minus_published"]["log_loss"]["value"] > 0
    # no more recordings may be wrong: one recording has five windows that lean to another view and three that are sure of its own
    monkeypatch.setattr(ablations, "TEMPERATURES", TEMPERATURES)
    turned = dict(unsure)
    first = benchmark.of_kind(halves["selection"], "native")[0]
    view = int(first.labels[0])
    lean, sure = np.full(5, -8.0), np.full(5, -8.0)
    lean[(view + 1) % 5], lean[view], sure[view] = 1.0, 0.0, 8.0  # 0.73 against 0.27 as published, 0.97 against 0.03 at a temperature of 0.3
    turned[first.id] = np.stack([np.tile(lean, (3, 1))] * 5 + [np.tile(sure, (3, 1))] * 3)
    one_more = ablations.one_temperature(turned, halves, draws=80, seed=5)
    before, after = one_more["selection"]["as published"]["errors"], one_more["selection"]["with the temperature"]["errors"]
    assert (before, after) == (0, 1) and one_more["with_minus_published"]["log_loss"]["interval"][1] < 0 and one_more["word"] == NOT_SUPPORTED
    # the interval of the log loss must lie wholly below zero
    real = ablations.classify_read.mean_interval
    monkeypatch.setattr(ablations.classify_read, "mean_interval", lambda values, counts: {**real(values, counts), "interval": [-0.2, 0.05]})
    assert ablations.one_temperature(unsure, halves, draws=40, seed=5)["word"] == NOT_SUPPORTED
    monkeypatch.setattr(ablations.classify_read, "mean_interval", lambda values, counts: {**real(values, counts), "interval": [-0.2, -0.05]})
    assert ablations.one_temperature(unsure, halves, draws=40, seed=5)["word"] == CANDIDATE


def test_the_command_holds_the_logits_to_the_published_number_and_writes_once(tmp_path, monkeypatch, capsys):
    halves, stored = development(np.random.default_rng(6), noise=0.6)
    np.savez(tmp_path / "dev.npz", **stored)
    monkeypatch.setattr(ablations.benchmark, "load", lambda: halves)
    monkeypatch.setattr(ablations.benchmark, "load_test", lambda: pytest.fail("this study does not read the test benchmark"))
    selection = halves["selection"]
    f1 = metrics.evaluate(comparison.decoded(comparison.member_probabilities(stored, selection, MEMBERS), B3), selection)["segment_f1_50_multi"]
    out = tmp_path / "stored.json"
    monkeypatch.setattr(comparison, "PUBLISHED_SELECTION_F1", round(f1, 4) + 0.01)
    with pytest.raises(SystemExit, match="must give"):
        ablations.main(["--logits", str(tmp_path / "dev.npz"), "--draws", "40", "--out", str(out)])
    assert not out.exists()
    monkeypatch.setattr(comparison, "PUBLISHED_SELECTION_F1", round(f1, 4))
    assert ablations.main(["--logits", str(tmp_path / "dev.npz"), "--draws", "40", "--out", str(out)]) == 0 and "wrote" in capsys.readouterr().out
    result = json.loads(out.read_text())
    assert result["benchmark"] == "development" and result["draws"] == 40 and result["sha256"] == {"logits": comparison.file_sha256(tmp_path / "dev.npz")}
    assert set(result) >= {"ensemble_size", "decoders", "decoder_settings", "temperature", "published_selection_f1", "sequences"}
    assert result["sequences"] == {half: len(sequences) for half, sequences in halves.items()}
    names = {s.id for half in halves.values() for s in half} | {f.video for half in halves.values() for s in half for f in s.fragments}
    assert not any(name in out.read_text() for name in names) and not re.search(r"NaN|Infinity", out.read_text())  # pooled numbers only
    with pytest.raises(SystemExit, match="does not replace"):
        ablations.main(["--logits", str(tmp_path / "dev.npz"), "--draws", "40", "--out", str(out)])
    np.savez(tmp_path / "few.npz", **{key: value for key, value in list(stored.items())[1:]})
    with pytest.raises(SystemExit, match="lacks the logits of 1 of the"):
        ablations.main(["--logits", str(tmp_path / "few.npz"), "--draws", "40", "--out", str(tmp_path / "never.json")])
    assert ROUTER == "EchoRouter" and not (tmp_path / "never.json").exists()


def test_the_plan_names_the_grids_the_temperatures_and_the_conditions():
    """docs/ABLATIONS.md was written before the results: what it fixes is what the code holds."""
    from pathlib import Path

    from echoview.evaluation import stress

    plan = (Path(__file__).parents[1] / "docs" / "ABLATIONS.md").read_text()

    def listed(values) -> str:
        return ", ".join(f"{v:g}" for v in values)

    assert f"the samples in that window: {listed(ablations.GRIDS[MODE])}" in plan and ablations.GRIDS[MEAN] == ablations.GRIDS[MODE]
    assert f"the samples a piece must last: {listed(ablations.GRIDS[DURATION])}" in plan
    assert f"that cost: {listed(ablations.GRIDS[VITERBI])}" in plan
    assert "0.30, 0.35, and so on up to 2.00" in plan and ablations.TEMPERATURES[:2] == (0.3, 0.35) and ablations.TEMPERATURES[-1] == 2.0
    assert len(ablations.TEMPERATURES) == 35 and ablations.DECODERS_LEVEL == 99.0 and "this interval is taken at 99%" in plan
    assert ablations.COMPARED == (NONE, MODE, MEAN, DURATION, VITERBI) and "Five comparisons are made" in plan
    named = {"gamma 0.5, 0.7, 1.5 and 2.0": ("gamma 0.5", "gamma 0.7", "gamma 1.5", "gamma 2.0"),
             "40 grey levels darker and 40 brighter": ("darker by 40", "brighter by 40"), "contrast 0.5 and 1.5": ("contrast 0.5", "contrast 1.5"),
             "a spread of 10 and of 25 grey levels": ("noise 10", "noise 25"), "a spread of 0.25 about 1": ("speckle 0.25",),
             "Gaussian blur of 1 and of 2 pixels": ("blur 1 pixel", "blur 2 pixels"), "JPEG at quality 30 and 10": ("JPEG quality 30", "JPEG quality 10"),
             "112 and 64 pixels wide": ("112 pixels", "64 pixels"), "mirrored left to right": ("mirrored",), "enlarged 1.2 times": ("zoomed in 1.2",),
             "0.8 times its size": ("zoomed out 0.8",), "rotated by 10 and by 20 degrees": ("rotated 10 degrees", "rotated 20 degrees")}
    for phrase in named:
        assert phrase in plan, phrase
    assert sorted(name for names in named.values() for name in names) == sorted(stress.IMAGE) and len(stress.IMAGE) == 22
    assert stress.TIME == ("still frame", "shuffled frames", "played backwards", "half the frame rate") and all(name.split()[0] in plan for name in stress.TIME)
    assert stress.CLOSE == 1e-3 and "to 0.001 in every probability" in plan
