"""The numbers added after a review of the read of docs/FOLLOWUP.md, section 9: each follows from the read's output alone."""
from __future__ import annotations

import json
from pathlib import Path
from statistics import NormalDist

import pytest

from echoview.evaluation import followup, followup_review
from echoview.evaluation.scorecard import NEAR

DOCS = Path(__file__).resolve().parents[1] / "docs"
CONFIRMATION = DOCS / "followup" / "confirmation.json"
REVIEW = DOCS / "followup" / "confirmation_review.json"


def multi(mean: float, matched: int, changes: int) -> dict:
    return {"cut_distance_mean_s": {"value": mean, "support": float(matched)}, "cut_matched_share": {"value": matched / changes, "support": float(changes)}}


def test_a_view_change_without_a_cut_counts_as_the_radius_of_the_matching():
    assert followup_review.RADIUS_S == NEAR / 10.0 == 1.0  # the matching of scorecard.py, in seconds
    assert followup_review.unmatched_at_the_radius(multi(0.1, 8, 10)) == pytest.approx((8 * 0.1 + 2 * 1.0) / 10)
    assert followup_review.unmatched_at_the_radius(multi(0.25, 10, 10)) == pytest.approx(0.25)  # every change matched: the mean itself
    assert followup_review.unmatched_at_the_radius(multi(0.0, 0, 4)) == pytest.approx(1.0)


def test_counts_that_do_not_fit_each_other_are_refused():
    row = multi(0.1, 8, 10)
    row["cut_matched_share"]["value"] = 0.7  # 8 of 10 is not 0.7
    with pytest.raises(ValueError, match="do not fit"):
        followup_review.unmatched_at_the_radius(row)


@pytest.mark.parametrize(("only_arm", "only_published", "p"), [
    (0, 1, 1.0), (1, 2, 1.0), (0, 0, 1.0), (5, 5, 1.0), (0, 6, 2 / 64), (6, 0, 2 / 64), (1, 9, 2 * 11 / 1024), (2, 8, 2 * 56 / 1024)])
def test_the_exact_test_of_the_recordings_that_only_one_reads_right(only_arm, only_published, p):
    assert followup_review.sign_test(only_arm, only_published) == pytest.approx(p)
    assert followup_review.sign_test(only_published, only_arm) == pytest.approx(p)  # two-sided


def test_the_sign_test_is_exact_for_many_recordings():
    assert followup_review.sign_test(700, 400) < 1e-18  # whole-number arithmetic: no overflow, no rounding to zero
    assert followup_review.sign_test(600, 600) == 1.0


def test_the_sign_test_takes_counts_only():
    for bad in ((-1, 2), (1.5, 2), (True, 2)):
        with pytest.raises(ValueError, match="counts"):
            followup_review.sign_test(*bad)


@pytest.mark.parametrize("level", [95.0, 97.5])
def test_two_units_at_once_add_their_variances(level):
    z = NormalDist().inv_cdf(0.5 + level / 200)
    decides = {"value": -0.05, "level": level, "streams": [-0.05 - z * 0.003, -0.05 + z * 0.003], "examinations": [-0.05 - z * 0.004, -0.05 + z * 0.004]}
    low, high = followup_review.both_units(decides)
    assert (low, high) == pytest.approx((-0.05 - z * 0.005, -0.05 + z * 0.005))  # 3, 4 and 5
    wider = followup_review.both_units({**decides, "examinations": decides["streams"]})
    assert wider[1] - wider[0] == pytest.approx(2 * z * 0.003 * 2 ** 0.5)  # two equal units: wider than either by the root of two


def test_the_half_widths_are_added_in_quadrature_around_the_value():
    assert followup_review.both_units({"value": 1.0, "level": 95.0, "streams": [0.7, 1.3], "examinations": [0.6, 1.4]}) == pytest.approx([0.5, 1.5])
    lopsided = followup_review.both_units({"value": 1.0, "level": 95.0, "streams": [0.9, 1.5], "examinations": [0.2, 1.0]})
    assert lopsided == pytest.approx([0.5, 1.5])  # the widths count, not where the two intervals lie


def test_an_interval_that_is_missing_gives_none():
    assert followup_review.both_units({"value": -0.05, "level": 95.0, "streams": None, "examinations": [-0.06, -0.04]}) is None
    assert followup_review.both_units({"value": -0.05, "level": 95.0, "streams": [-0.06, -0.04], "examinations": None}) is None
    assert followup_review.both_units({"value": -0.05, "level": 95.0, "streams": [-0.06, -0.04]}) is None


def test_the_intervals_of_the_descriptions_are_counted():
    differences = {"arm": {"multi": {"a": {"value": 1.0, "level": 95.0, "interval": [0.5, 1.5]},     # excludes zero
                                     "b": {"value": 0.0, "level": 95.0, "interval": [-0.5, 0.5]},    # includes zero
                                     "c": {"value": -1.0, "level": 95.0, "interval": [-1.5, 0.0]},   # ends at zero: does not exclude it
                                     "d": {"value": 2.0, "level": 95.0, "interval": None}},           # no interval
                           "recordings": {"e": {"value": -1.0, "level": 95.0, "interval": [-2.0, -0.1]}},
                           "recordings_only_one_reads_right": {"only_the_arm": 0, "only_the_published_system": 1}}}
    assert followup_review.intervals(differences) == {"differences_with_an_interval": 4, "of_them_exclude_zero": 2, "level": 95.0}
    other = {"arm": {"multi": {"a": {"value": 1.0, "level": 97.5, "interval": [0.5, 1.5]}, "b": {"value": 1.0, "level": 95.0, "interval": [0.5, 1.5]}}}}
    assert followup_review.intervals(other) == {"differences_with_an_interval": 1, "of_them_exclude_zero": 1, "level": 95.0}  # one level at a time
    assert followup_review.intervals(other, level=97.5)["differences_with_an_interval"] == 1


def test_the_distance_above_the_floor_of_each_grid():
    rows = {"EchoRouter": {"multi": multi(0.15, 9, 10)}, "every 0.2 s": {"multi": multi(0.12, 9, 10)}, "every 0.1 s": {"multi": multi(0.08, 9, 10)},
            "Perfect on the 0.5 s grid": {"multi": multi(0.12, 10, 10)}, "Perfect on the 0.2 s grid": {"multi": multi(0.05, 10, 10)}}
    above = followup_review.above_the_floor(rows)
    assert {name: entry["matched_cuts"] for name, entry in above.items()} == pytest.approx({"EchoRouter": 0.03, "every 0.2 s": 0.07, "every 0.1 s": 0.08})
    # with every view change counted: nine matched cuts at the mean and one view change at 1 s, minus the floor
    assert above["EchoRouter"]["every_view_change"] == pytest.approx((9 * 0.15 + 1.0) / 10 - 0.12)
    assert above["every 0.2 s"]["every_view_change"] == pytest.approx((9 * 0.12 + 1.0) / 10 - 0.05)
    assert above["every 0.1 s"]["every_view_change"] == pytest.approx((9 * 0.08 + 1.0) / 10)
    for entry in above.values():  # an arm that misses view changes is further above the floor than its matched cuts say
        assert entry["every_view_change"] > entry["matched_cuts"]


def test_a_floor_must_match_every_view_change():
    rows = {"EchoRouter": {"multi": multi(0.15, 9, 10)}, "every 0.2 s": {"multi": multi(0.12, 9, 10)}, "every 0.1 s": {"multi": multi(0.08, 9, 10)},
            "Perfect on the 0.5 s grid": {"multi": multi(0.12, 9, 10)}, "Perfect on the 0.2 s grid": {"multi": multi(0.05, 10, 10)}}
    with pytest.raises(ValueError, match="no floor"):
        followup_review.above_the_floor(rows)


def test_the_review_holds_every_number_for_both_steps_and_names_its_source():
    result = json.loads(CONFIRMATION.read_text())
    review = followup_review.review(result)
    assert review["source_sha256"] is None  # a dict has no digest; the command adds the file's
    assert set(review["cut_distance_with_unmatched_changes_at_1s"]) == {"EchoRouter", "every 0.2 s", "every 0.1 s"}
    assert set(review["recordings_exact_test"]) == set(review["cut_distance_both_units"]) == {"every 0.2 s", "every 0.1 s"}
    # the order of the three rows is the order of the plain means: the selection in the matched cuts does not make it
    penalised = review["cut_distance_with_unmatched_changes_at_1s"]
    assert penalised["every 0.1 s"] < penalised["every 0.2 s"] < penalised["EchoRouter"]
    for arm in ("every 0.2 s", "every 0.1 s"):
        assert review["recordings_exact_test"][arm]["p_two_sided"] == 1.0  # no direction in the recordings that change
        assert review["cut_distance_both_units"][arm]["interval"][1] < 0  # (a) also holds when both units vary at once
        only = result["differences_from_published"][arm]["recordings_only_one_reads_right"]
        assert review["recordings_exact_test"][arm] == {**only, "p_two_sided": 1.0}


def test_the_stored_review_is_what_the_command_gives_for_the_read(tmp_path, capsys):
    out = tmp_path / "review.json"
    assert followup_review.main([str(CONFIRMATION), "--out", str(out)]) == 0
    assert json.loads(out.read_text()) == json.loads(REVIEW.read_text())
    assert json.loads(REVIEW.read_text())["source_sha256"] == followup.CONFIRMATION_SHA256  # the output of the one read
    assert [p.name for p in tmp_path.iterdir()] == ["review.json"]
    assert "not planned" in capsys.readouterr().out


def test_the_command_refuses_a_file_that_is_not_the_read(tmp_path):
    other = tmp_path / "other.json"
    other.write_text(CONFIRMATION.read_text().replace('"seed": 20261123', '"seed": 1'))
    with pytest.raises(SystemExit, match="not the output of the read"):
        followup_review.main([str(other), "--out", str(tmp_path / "review.json")])
    with pytest.raises(SystemExit, match="is not a file"):
        followup_review.main([str(tmp_path / "missing.json"), "--out", str(tmp_path / "review.json")])
    assert not (tmp_path / "review.json").exists()


def test_the_command_never_replaces_a_file(tmp_path):
    out = tmp_path / "review.json"
    out.write_text("kept")
    with pytest.raises(SystemExit, match="exists; it is not replaced"):
        followup_review.main([str(CONFIRMATION), "--out", str(out)])
    assert out.read_text() == "kept" and [p.name for p in tmp_path.iterdir()] == ["review.json"]


def test_without_a_named_output_of_the_read_nothing_is_computed(tmp_path, monkeypatch):
    monkeypatch.setattr(followup, "CONFIRMATION_SHA256", None)
    with pytest.raises(SystemExit, match="names no output of the read"):
        followup_review.main([str(CONFIRMATION), "--out", str(tmp_path / "review.json")])
    assert not list(tmp_path.iterdir())
