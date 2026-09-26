"""Threshold rules and score diagnostics for selective routing (echo_routing.evaluate.risk_control)."""
from __future__ import annotations

import numpy as np
import pytest

from echo_routing.evaluate.risk_control import (aurc, cine_counts, fit_temperature, oracle_coverage, select_classwise,
                                                select_threshold_ucb, temperature_scale, threshold_grid)
from echo_routing.evaluate.bootstrap import resample_weights
from echo_routing.temporal.decode_select import select_threshold


def test_temperature_one_is_identity_and_large_flattens():
    p = np.array([[0.7, 0.2, 0.1], [0.1, 0.1, 0.8]])
    assert temperature_scale(p, 1.0) == pytest.approx(p)
    assert temperature_scale(p, 4.0).max(1).max() < p.max(1).max()
    assert temperature_scale(p, 4.0).sum(1) == pytest.approx(np.ones(2))


def test_fit_temperature_softens_overconfident_probabilities():
    rng = np.random.default_rng(0)
    n, c = 4000, 5
    truth = rng.integers(0, c, n)
    pred = np.where(rng.random(n) < 0.7, truth, (truth + 1) % c)          # 70% correct ...
    p = np.full((n, c), 0.01 / (c - 1)); p[np.arange(n), pred] = 0.99       # ... but 99% confident
    assert fit_temperature(p, truth) > 1.5


def test_cine_counts_by_hand():
    scores = np.array([0.9, 0.8, 0.1, 0.95]); wrong = np.array([0, 1, 0, 0], bool); cine = np.array(["a", "a", "b", "b"])
    ids, acc, wr = cine_counts(scores, wrong, cine, np.array([0.5, 0.85]))
    assert list(ids) == ["a", "b"]
    assert acc.tolist() == [[2, 1], [1, 1]] and wr.tolist() == [[1, 0], [0, 0]]


def _world(seed=0, n_cines=60, per=40):
    rng = np.random.default_rng(seed)
    cine = np.repeat(np.arange(n_cines), per)
    hard = rng.random(n_cines) < 0.2                                  # a few hard cines carry most errors
    wrong = rng.random(cine.size) < np.where(hard[cine], 0.5, 0.02)
    scores = np.clip(rng.normal(np.where(wrong, 0.55, 0.8), 0.15), 0, 1)
    return scores, wrong, cine


def test_ucb_rule_is_at_least_as_conservative_as_point_rule():
    scores, wrong, cine = _world()
    taus = threshold_grid(scores)
    ids, acc, wr = cine_counts(scores, wrong, cine, taus)
    point = select_threshold(scores, wrong, np.ones(scores.size), 0.05)
    ucb = select_threshold_ucb(acc, wr, taus, 0.05, resample_weights(len(ids), 500, np.random.default_rng(1)))
    assert point is not None and ucb is not None and ucb >= point - 1e-12


def test_ucb_rule_without_resampling_matches_point_rule_on_grid():
    scores, wrong, cine = _world(seed=2)
    taus = threshold_grid(scores, n=10**6)                            # every unique score
    ids, acc, wr = cine_counts(scores, wrong, cine, taus)
    ucb = select_threshold_ucb(acc, wr, taus, 0.05, np.ones((1, len(ids))))
    assert ucb == pytest.approx(select_threshold(scores, wrong, np.ones(scores.size), 0.05))


def test_aurc_and_oracle_coverage():
    wrong = np.array([0, 0, 0, 1, 1], bool)
    good = np.array([0.9, 0.8, 0.7, 0.2, 0.1]); bad = good[::-1].copy()
    assert aurc(good, wrong) < aurc(bad, wrong)
    assert aurc(good, np.zeros(5, bool)) == 0
    assert oracle_coverage(good, wrong, 0.0) == pytest.approx(0.6)
    assert oracle_coverage(np.array([-np.inf, 0.5]), np.array([0, 0], bool), 0.05) == pytest.approx(0.5)


def test_classwise_thresholds():
    scores, wrong, cine = _world(seed=3)
    pred = np.where(np.arange(scores.size) % 2 == 0, 0, 1)
    wrong = wrong.copy(); wrong[pred == 1] = True                      # class 1 is always wrong: it can never be accepted
    taus = select_classwise(scores, wrong, pred, cine, 0.05, rule="point")
    assert set(taus) == {0, 1} and taus[1] is None and taus[0] is not None
    ucb = select_classwise(scores, wrong, pred, cine, 0.05, rule="ucb", n_boot=300, rng=np.random.default_rng(4))
    assert ucb[1] is None and (ucb[0] is None or ucb[0] >= taus[0] - 1e-12)
