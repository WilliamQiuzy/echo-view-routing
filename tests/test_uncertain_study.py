"""The study that chooses the uncertain rule: the band, the score of the tests for whole pieces, and the decision."""
from __future__ import annotations

import json

import numpy as np
import pytest

from echoview import uncertain
from echoview.evaluation import benchmark, uncertain_study
from echoview.reference import B3


def flat(view: int, top: float, second: tuple[int, float] | None = None) -> np.ndarray:
    row = np.full(5, 0.0)
    row[view] = top
    if second is not None:
        row[second[0]] = second[1]
    row[row == 0] = (1.0 - row.sum()) / (row == 0).sum()
    return row


def decoded(name: str, middle: np.ndarray, shift: int = 0) -> uncertain_study.Decoded:
    """A multi stream of three pieces of 3 s (views 0, 2, 4) from its own three recordings. `middle` is the probability
    of every sample of the second piece; `shift` moves the system's first change later than the true one."""
    fragments = tuple(benchmark.Fragment(f"{name}_{k}", view, 1.0, tuple(range(30))) for k, view in enumerate((0, 2, 4)))
    prob = np.stack([flat(0, 0.9)] * (30 + shift) + [middle] * (30 - shift) + [flat(4, 0.9)] * 30)
    return uncertain_study.Decoded(benchmark.Sequence(name, "multi", fragments), prob, np.stack([prob, prob, prob]))


CLEAR = flat(2, 0.9)  # read right, with a wide margin
WRONG = flat(3, 0.5, (2, 0.35))  # read as view 3 by a margin of 0.15: a wrong piece
NARROW = flat(2, 0.5, (3, 0.35))  # read right by the same narrow margin


def test_the_mask_from_the_parts_is_the_mask_of_the_rule():
    rng = np.random.default_rng(1)
    fragments = tuple(benchmark.Fragment(f"v{k}", int(view), 1.0, tuple(range(40))) for k, view in enumerate((0, 3, 2, 4)))
    sequence = benchmark.Sequence("s", "multi", fragments)
    prob = rng.dirichlet(np.ones(5) * 0.4, size=160)
    seeds = np.stack([rng.dirichlet(np.ones(5) * 0.4, size=160) for _ in range(3)])
    d = uncertain_study.Decoded(sequence, prob, seeds)
    for band, f_low, f_high, margin, with_seeds in ((0, 1.0, 1.0, 0.0, False), (3, 1.5, 1.0, 0.0, False), (2, 1.0, 3.0, 0.2, False), (4, 2.0, 1.25, 0.4, True)):
        rule = uncertain.Rule(band, f_low, f_high, margin, with_seeds)
        marked = uncertain.mark(prob, B3, rule, seeds)
        assert np.array_equal(d.mask(rule), marked.mask) and np.array_equal(d.decided, marked.decided)
        assert np.array_equal(d.band(rule), uncertain.mark(prob, B3, uncertain.Rule(band=band)).mask)


def test_step_1_takes_the_smallest_band_that_holds_nine_changes_in_ten():
    half = [decoded(f"s{k}", CLEAR, shift=1) for k in range(9)] + [decoded("late", CLEAR, shift=4)]
    step = uncertain_study.band_for(half)
    assert step["matched_changes"] == 20  # two changes per stream; the second is always in place
    assert step["share_within"][0] == pytest.approx(0.5) and step["share_within"][1] == pytest.approx(0.95) and step["band"] == 1
    assert step["view_changes"] == 20
    unmatched = uncertain_study.band_for([decoded(f"s{k}", CLEAR, shift=12) for k in range(4)])
    assert (unmatched["view_changes"], unmatched["matched_changes"], unmatched["band"]) == (8, 4, 0)  # 1.2 s late is too far to be matched
    far = [decoded(f"s{k}", CLEAR, shift=9) for k in range(4)]  # matched, but 0.9 s away: no band up to 0.8 s holds nine in ten
    assert uncertain_study.band_for(far)["band"] is None and "stopped" in uncertain_study.choose(far)


def test_step_2_scores_wrong_pieces_declined_against_right_ones():
    half = [decoded(f"w{k}", WRONG) for k in range(5)] + [decoded(f"c{k}", CLEAR) for k in range(40)]
    off = uncertain_study.score(half, uncertain.Rule())
    assert (off["wrong_declined"], off["right_declined"], off["score"], off["points"], off["uncertain_samples"]) == (0, 0, 0.0, 0, 0)
    margin = uncertain_study.score(half, uncertain.Rule(margin=0.2))
    assert (margin["wrong_declined"], margin["wrong_declined_recordings"], margin["right_declined"], margin["score"]) == (5, 5, 0, 5.0)
    assert margin["points"] == 25
    assert margin["uncertain_samples"] == 5 * 30
    choice = uncertain_study.choose(half)
    assert choice["band"] == 0 and choice["settings"] == 300 and len(choice["all_settings"]) == 300
    assert choice["goes_on"] and choice["candidate"]["wrong_declined"] == 5 and choice["candidate"]["score"] == 5.0
    assert choice["candidate"]["rule"]["margin"] == 0.2 and choice["candidate"]["uncertain_samples"] == 150  # the smallest rule with that score
    json.dumps(choice)  # nothing in it that a file cannot hold


def test_a_candidate_below_the_bar_does_not_go_on():
    nothing_wrong = uncertain_study.choose([decoded(f"c{k}", CLEAR) for k in range(20)])
    assert not nothing_wrong["goes_on"] and "no setting has a score above 0" in nothing_wrong["why_not"]
    too_few = uncertain_study.choose([decoded(f"w{k}", WRONG) for k in range(2)] + [decoded(f"c{k}", CLEAR) for k in range(20)])
    assert not too_few["goes_on"] and too_few["candidate"]["score"] == 2.0 and "the bar is 4 from 3" in too_few["why_not"][0]


def test_step_3_releases_the_tests_only_if_the_selection_half_bears_them_out():
    calibration = [decoded(f"w{k}", WRONG) for k in range(5)] + [decoded(f"c{k}", CLEAR) for k in range(40)]
    choice = uncertain_study.choose(calibration)
    borne_out = uncertain_study.final([decoded(f"x{k}", WRONG) for k in range(6)] + [decoded(f"y{k}", CLEAR) for k in range(40)], choice, draws=300)
    assert borne_out["decision"]["passed"] and borne_out["decision"]["released_rule"]["margin"] == 0.2
    assert borne_out["candidate"]["pieces"]["all"]["beyond_band"]["wrong_declined"] == {"pieces": 6, "recordings": 6}
    assert borne_out["candidate"]["each_test"]["margin"] == {"samples": 180, "wrong": 180, "share_wrong": 1.0}
    assert borne_out["candidate"]["samples"]["multi"]["ratios"]["error_added"] == 1.0
    assert borne_out["published"]["samples"]["multi"]["counts"]["uncertain"] == 0 and len(borne_out["bands"]) == 9
    # a selection half on which the narrow margins belong to pieces that are read right
    not_borne_out = uncertain_study.final([decoded(f"x{k}", NARROW) for k in range(6)] + [decoded(f"y{k}", CLEAR) for k in range(40)], choice, draws=300)
    assert not not_borne_out["decision"]["passed"] and not_borne_out["decision"]["score"] == pytest.approx(-6 / 5)
    assert not_borne_out["decision"]["released_rule"] == {"band": 0, "f_low": 1.0, "f_high": 1.0, "margin": 0.0, "seeds": False}
    stopped = uncertain_study.final(calibration, uncertain_study.choose([decoded(f"c{k}", CLEAR) for k in range(20)]), draws=50)
    assert "candidate" not in stopped and stopped["decision"]["tests_for_whole_pieces"] == "not adopted on the calibration half"
    json.dumps(borne_out)


def test_a_tie_in_the_score_is_a_tie_whatever_floating_point_says(monkeypatch):
    """1 - 1/5 and 3 - 11/5 are the same score. As floats they are 0.8 and 0.7999999999999998, and the larger setting
    would win by that. Counted in whole numbers they tie, and the tie goes to the fewer uncertain samples."""
    assert 1 - 1 / 5 != 3 - 11 / 5

    def fake(half, rule):
        if rule == uncertain.Rule(0, margin=0.1):
            return uncertain_study.row(rule, 1, 1, 1, 500)
        if rule == uncertain.Rule(0, margin=0.3):
            return uncertain_study.row(rule, 3, 3, 11, 100)
        return uncertain_study.row(rule, 0, 0, 0, 0)

    monkeypatch.setattr(uncertain_study, "score", fake)
    choice = uncertain_study.choose([decoded(f"c{k}", CLEAR) for k in range(4)])
    assert choice["candidate"]["rule"]["margin"] == 0.3 and choice["candidate"]["points"] == 4 and choice["candidate"]["uncertain_samples"] == 100
    # with the same number of uncertain samples the earlier setting of the grid wins

    def twins(half, rule):
        same = rule.margin in (0.2, 0.4) and rule.f_low == rule.f_high == 1.0 and not rule.seeds
        return uncertain_study.row(rule, 1, 1, 0, 50) if same else uncertain_study.row(rule, 0, 0, 0, 0)

    monkeypatch.setattr(uncertain_study, "score", twins)
    assert uncertain_study.choose([decoded(f"c{k}", CLEAR) for k in range(4)])["candidate"]["rule"]["margin"] == 0.2


def test_a_rule_outside_the_study_is_refused_by_name():
    d = decoded("s", CLEAR)
    with pytest.raises(ValueError, match="not one of the study's settings"):
        d.mask(uncertain.Rule(margin=0.25))
    with pytest.raises(ValueError, match="not one of the study's settings"):
        d.band(uncertain.Rule(band=9))


@pytest.fixture()
def halves(monkeypatch):
    """The command line on two small synthetic halves instead of the development benchmark."""
    data = {"calibration": [decoded(f"w{k}", WRONG) for k in range(5)] + [decoded(f"c{k}", CLEAR) for k in range(40)],
            "selection": [decoded(f"x{k}", WRONG) for k in range(6)] + [decoded(f"y{k}", CLEAR) for k in range(40)]}
    monkeypatch.setattr(uncertain_study, "load", lambda path, half: data[half])


def test_the_two_commands_write_their_files_whole_and_once(tmp_path, halves, capsys):
    logits, choice, report = tmp_path / "dev_logits.npz", tmp_path / "choice.json", tmp_path / "report.json"
    logits.write_bytes(b"logits")
    assert uncertain_study.main(["choose", "--logits", str(logits), "--out", str(choice), "--draws", "50"]) == 0
    assert json.loads(choice.read_text())["goes_on"] and not list(tmp_path.glob("*.partial"))
    assert json.loads(capsys.readouterr().out)["candidate"]["rule"]["margin"] == 0.2
    before = choice.read_bytes()
    with pytest.raises(SystemExit, match="not replaced without --force"):  # the choice step 3 follows is not made twice by accident
        uncertain_study.main(["choose", "--logits", str(logits), "--out", str(choice)])
    with pytest.raises(SystemExit, match="one of the input files"):
        uncertain_study.main(["choose", "--logits", str(logits), "--out", str(logits), "--force"])
    with pytest.raises(SystemExit, match="one of the input files"):
        uncertain_study.main(["report", "--logits", str(logits), "--choice", str(choice), "--out", str(choice), "--force"])
    assert choice.read_bytes() == before and logits.read_bytes() == b"logits"
    assert uncertain_study.main(["report", "--logits", str(logits), "--choice", str(choice), "--out", str(report), "--draws", "50"]) == 0
    assert json.loads(report.read_text())["decision"]["passed"]
    assert uncertain_study.main(["choose", "--logits", str(logits), "--out", str(choice), "--draws", "50", "--force"]) == 0
    with pytest.raises(SystemExit):  # argparse refuses it
        uncertain_study.main(["choose", "--logits", str(logits), "--out", str(tmp_path / "other.json"), "--draws", "0"])


def test_report_refuses_a_choice_file_that_is_not_one(tmp_path, halves):
    logits, out = tmp_path / "dev_logits.npz", tmp_path / "report.json"
    cases = {"missing.json": None, "text.json": "not json", "other.json": json.dumps({"a": 1}),
             "stopped.json": json.dumps({"step1": {}, "stopped": "no band"}),
             "band.json": json.dumps({"step1": {}, "band": 9, "goes_on": False}),
             "rule.json": json.dumps({"step1": {}, "band": 3, "goes_on": True, "candidate": {"rule": {"band": 3, "margin": 0.25}}})}
    for name, content in cases.items():
        path = tmp_path / name
        if content is not None:
            path.write_text(content)
        with pytest.raises(SystemExit) as stop:
            uncertain_study.main(["report", "--logits", str(logits), "--choice", str(path), "--out", str(out)])
        assert isinstance(stop.value.code, str) and not out.exists(), name


def test_a_logits_file_that_cannot_be_read_ends_with_a_message(tmp_path):
    bad = tmp_path / "dev_logits.npz"
    with pytest.raises(SystemExit, match="cannot read the logits file"):
        uncertain_study.load(bad, "calibration")
    bad.write_bytes(b"PK\x03\x04 not a zip file")
    with pytest.raises(SystemExit, match="cannot read the logits file"):
        uncertain_study.load(bad, "calibration")
    np.savez(bad, something=np.zeros(3))
    with pytest.raises(SystemExit, match="lacks the logits"):
        uncertain_study.load(bad, "calibration")
