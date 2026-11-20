"""The uncertain output: which samples of a decoded stream the system declines to name."""
from __future__ import annotations

import numpy as np
import pytest

from echoview import decoder, uncertain
from echoview.reference import B3


def stream(*pieces: tuple[int, dict[int, float]]) -> np.ndarray:
    """Per-sample probabilities of a stream made of pieces: (length, {view: probability}); the rest is spread evenly."""
    rows = []
    for length, views in pieces:
        row = np.full(5, (1.0 - sum(views.values())) / (5 - len(views)))
        for view, probability in views.items():
            row[view] = probability
        rows.extend([row] * length)
    return np.asarray(rows)


def test_the_band_marks_k_samples_on_each_side_of_every_cut():
    labels = np.array([0] * 10 + [1] * 10 + [4] * 3)
    assert np.flatnonzero(uncertain.band_mask(labels, 2)).tolist() == [8, 9, 10, 11, 18, 19, 20, 21]
    assert not uncertain.band_mask(labels, 0).any()
    assert np.flatnonzero(uncertain.band_mask(labels, 4)).tolist() == [*range(6, 14), *range(16, 23)]  # cut short by the end of the stream
    assert not uncertain.band_mask(np.zeros(30, int), 5).any()  # no cut, no band: the ends of a stream are not cuts


def test_a_weak_cut_is_one_that_another_threshold_would_not_make_or_would_add():
    # a clear change at 40, and a faint one at 80: the last piece leans to view 3 only slightly
    prob = stream((40, {0: 0.9}), (40, {2: 0.6, 3: 0.3}), (40, {2: 0.4, 3: 0.5}))
    scores = decoder.boundary_scores(prob, 7)
    faint, clear = float(scores[80]), float(scores[40])
    assert faint < clear / 5
    setting = {"half_window": 7, "threshold": faint * 0.9, "nms_radius": 5}  # the faint change is just cut
    base = decoder.decode(prob, **setting)
    assert base[39] == 0 and base[41] == 2 and base[81] == 3
    assert not uncertain.weak_cut_mask(prob, base, setting, 1.0, 1.0).any()  # both sides switched off
    higher = uncertain.weak_cut_mask(prob, base, setting, 1.0, 1.5)  # at 1.35 x faint the faint cut is not made
    assert higher[:40].sum() == 0 and higher[40:].any() and not higher[40:].all()  # one of the two merged pieces changes its name
    setting = {"half_window": 7, "threshold": faint * 1.1, "nms_radius": 5}  # now the faint change is just missed
    base = decoder.decode(prob, **setting)
    assert base[81] == base[41]
    lower = uncertain.weak_cut_mask(prob, base, setting, 1.5, 1.0)  # a lower threshold would cut it
    assert lower.any() and not lower[:40].any()
    again = decoder.decode(prob, 7, setting["threshold"] / 1.5, 5)
    assert np.array_equal(lower, again != base)


def test_a_segment_without_a_clear_winner_is_uncertain_as_a_whole():
    prob = stream((20, {0: 0.9}), (20, {2: 0.5, 3: 0.4}), (20, {4: 0.8}))
    labels = np.array([0] * 20 + [2] * 20 + [4] * 20)
    assert not uncertain.margin_mask(prob, labels, 0.0).any() and not uncertain.margin_mask(prob, labels, 0.05).any()
    marked = uncertain.margin_mask(prob, labels, 0.2)  # 0.5 against 0.4 is a margin of 0.1
    assert marked[20:40].all() and marked.sum() == 20
    assert uncertain.margin_mask(prob, labels, 0.9).all()


def test_a_segment_that_one_seed_would_name_differently_is_uncertain():
    agree = stream((20, {0: 0.9}), (20, {2: 0.6, 3: 0.3}))
    differs = stream((20, {0: 0.8}), (20, {2: 0.4, 3: 0.5}))
    labels = np.array([0] * 20 + [2] * 20)
    assert not uncertain.seed_mask(np.stack([agree, agree, agree]), labels).any()
    marked = uncertain.seed_mask(np.stack([agree, differs, agree]), labels)
    assert marked[20:].all() and not marked[:20].any()


def test_short_labelled_runs_beside_uncertain_samples_are_not_kept():
    mask = np.zeros(40, bool)
    mask[[3, 4, 8, 9, 20, 21, 30]] = True  # labelled runs: 0-2 (at the start), 5-7, 10-19, 22-29, 31-39 (at the end)
    tidy = uncertain.tidy(mask)
    assert np.flatnonzero(tidy).tolist() == [*range(0, 10), 20, 21, 30]  # the two runs of 3 go, also the one at the start
    assert uncertain.tidy(np.zeros(10, bool)).sum() == 0 and uncertain.tidy(np.ones(10, bool)).all()
    assert uncertain.tidy(np.zeros(3, bool)).sum() == 0  # a short stream without an uncertain sample is left alone
    stub = np.zeros(12, bool)
    stub[:9] = True
    assert uncertain.tidy(stub).all()  # three labelled samples at the end, beside uncertain ones


def test_mark_joins_the_tests_and_leaves_the_published_labels_alone():
    prob = stream((40, {0: 0.9}), (40, {2: 0.6, 3: 0.3}), (40, {2: 0.4, 3: 0.5}))
    scores = decoder.boundary_scores(prob, 7)
    setting = {"half_window": 7, "threshold": float(scores[80]) * 0.9, "nms_radius": 5}
    base = decoder.decode(prob, **setting)
    seeds = np.stack([prob, prob, prob])
    off = uncertain.mark(prob, setting, uncertain.OFF, seeds)
    assert not off.mask.any() and np.array_equal(off.labels, base) and np.array_equal(off.decided, base)
    rule = uncertain.Rule(band=3, f_low=1.0, f_high=1.5, margin=0.0, seeds=False)
    out = uncertain.mark(prob, setting, rule, seeds)
    expected = uncertain.tidy(uncertain.band_mask(base, 3) | uncertain.weak_cut_mask(prob, base, setting, 1.0, 1.5))
    assert np.array_equal(out.mask, expected) and out.mask[37:43].all()
    assert np.array_equal(out.decided, base)  # what a forced answer would have been
    assert (out.labels[out.mask] == uncertain.UNCERTAIN).all() and np.array_equal(out.labels[~out.mask], base[~out.mask])
    with pytest.raises(ValueError, match="seeds"):
        uncertain.mark(prob, setting, uncertain.Rule(seeds=True), None)
    with pytest.raises(ValueError):
        uncertain.Rule(band=-1)
    with pytest.raises(ValueError):
        uncertain.Rule(f_high=0.5)


def test_the_published_decoder_setting_is_what_mark_decodes_with():
    rng = np.random.default_rng(3)
    prob = rng.dirichlet(np.ones(5) * 0.3, size=200)
    out = uncertain.mark(prob, B3, uncertain.Rule(band=2))
    assert np.array_equal(out.decided, decoder.decode(prob, B3["half_window"], B3["threshold"], B3["nms_radius"]))
    assert uncertain.segments(out.labels)[0][0] == 0 and {view for _, _, view in uncertain.segments(out.labels)} <= {-1, 0, 1, 2, 3, 4}


def test_a_rule_with_values_that_mean_nothing_is_refused():
    for bad in ({"band": -1}, {"band": 2.5}, {"band": True}, {"f_low": 0.5}, {"f_low": float("nan")}, {"f_high": float("inf")},
                {"margin": 1.5}, {"margin": float("nan")}, {"seeds": 1}):
        with pytest.raises(ValueError, match="a rule needs"):
            uncertain.Rule(**bad)
    assert uncertain.Rule(band=np.int64(3), f_low=2, margin=0.5, seeds=True).band == 3  # whole numbers of any kind pass


def test_the_seeds_test_needs_the_probabilities_of_every_seed_for_every_sample():
    labels = np.zeros(40, int)
    one_seed = np.full((40, 5), 0.2)
    with pytest.raises(ValueError, match="seeds, 40 samples, views"):
        uncertain.seed_mask(one_seed, labels)  # (samples, views): the seeds are missing
    with pytest.raises(ValueError, match="seeds, 40 samples, views"):
        uncertain.seed_mask(np.stack([one_seed[:30]] * 3), labels)  # too short


def test_the_released_rule_is_the_one_the_study_decided_on():
    import json
    from pathlib import Path

    import echoview
    from echoview.evaluation import uncertain_study

    rule = uncertain.released()
    uncertain_study.on_grid(rule)
    decided = json.loads((Path(__file__).parents[1] / "docs" / "uncertain" / "report.json").read_text())["decision"]
    assert decided["passed"] and rule == uncertain.Rule(**decided["released_rule"])
    assert rule == uncertain.Rule(band=3, f_low=2.0, f_high=1.25, margin=0.1, seeds=False)
    assert "uncertain" in echoview.__all__ and echoview.uncertain.released() == rule  # as the README uses it


def test_the_released_rule_declines_three_samples_on_each_side_of_a_clear_cut():
    marked = uncertain.mark(stream((40, {0: 0.97}), (40, {2: 0.97})), B3, uncertain.released())
    assert np.flatnonzero(marked.mask).tolist() == [37, 38, 39, 40, 41, 42]
