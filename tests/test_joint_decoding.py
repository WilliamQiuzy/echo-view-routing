"""echo_routing/temporal/joint_decoding.py: the dynamic programme against exhaustive search, lambda = 0 as J0, the
alignment of b_t with a change between t-1 and t, the tie rules and input checks. Synthetic data only."""
import itertools

import numpy as np
import pytest

from echo_routing.temporal.baselines.b2_hmm import viterbi
from echo_routing.temporal.joint_decoding import joint_viterbi, joint_viterbi_grid, log_terms, path_score

GRID = [(lam, kap) for lam in (0.0, 0.25, 0.5, 1.0, 2.0) for kap in (0.0, 1.0, 2.0, 4.0, 8.0)]


def _random(t: int, c: int, seed: int):
    rng = np.random.default_rng(seed)
    return rng.dirichlet(np.full(c, 0.7), size=t), rng.uniform(0.0, 1.0, size=t)


def _brute(prob, b, lam, kap):
    logp, lb, l1b = log_terms(prob, b); t, c = logp.shape
    paths = np.array(list(itertools.product(range(c), repeat=t)))
    change = paths[:, 1:] != paths[:, :-1]
    score = logp[np.arange(t)[None], paths].sum(1) + lam * np.where(change, lb[None, 1:], l1b[None, 1:]).sum(1) - kap * change.sum(1)
    return paths[score.argmax()], float(score.max())


@pytest.mark.parametrize("t,c", [(1, 5), (2, 5), (4, 5), (5, 5), (7, 3), (8, 2)])
def test_dynamic_programme_equals_exhaustive_search(t, c):
    for seed in range(6):
        prob, b = _random(t, c, 100 * t + seed)
        got = joint_viterbi_grid(prob, b, GRID)
        for g, (lam, kap) in enumerate(GRID):
            best, score = _brute(prob, b, lam, kap)
            assert got[g].tolist() == best.tolist(), (t, c, seed, lam, kap)
            assert path_score(got[g], prob, b, lam, kap) == pytest.approx(score, abs=1e-9)


def test_grid_equals_single_settings():
    prob, b = _random(40, 5, 7)
    grid = joint_viterbi_grid(prob, b, GRID)
    for g, (lam, kap) in enumerate(GRID):
        assert np.array_equal(grid[g], joint_viterbi(prob, b, lam, kap))


def test_lambda_zero_ignores_b_and_equals_the_hmm_viterbi():
    prob, b = _random(60, 5, 3); other = np.random.default_rng(9).uniform(size=60)
    for kap in (0.0, 1.0, 2.0, 4.0, 8.0):
        j0 = joint_viterbi(prob, b, 0.0, kap)
        assert np.array_equal(j0, joint_viterbi(prob, other, 0.0, kap))          # exact: b never enters
        log_trans = np.where(np.eye(5, dtype=bool), 0.0, -kap)
        assert np.array_equal(j0, viterbi(log_terms(prob, b)[0], log_trans, np.zeros(5)))
    assert np.array_equal(joint_viterbi(prob, b, 0.0, 0.0), prob.argmax(1))      # no switch cost: per-sample argmax


def test_switch_lands_on_the_boundary_index():
    prob = np.array([[0.95, 0.05]] * 3 + [[0.5, 0.5]] * 2 + [[0.05, 0.95]] * 3)
    for spike in (3, 4, 5):                      # the class evidence allows a change at 3, 4 or 5 at equal cost
        b = np.full(8, 0.01); b[spike] = 0.99
        path = joint_viterbi(prob, b, 1.0, 0.0)
        assert (np.flatnonzero(np.diff(path)) + 1).tolist() == [spike]       # y_spike != y_{spike-1}


def test_strong_boundary_evidence_keeps_a_change_that_the_switch_cost_alone_removes():
    prob = np.array([[0.9, 0.1]] * 6 + [[0.3, 0.7]] * 2 + [[0.9, 0.1]] * 6)
    b = np.full(14, 0.01); b[6] = b[8] = 0.99
    assert np.array_equal(joint_viterbi(prob, b, 0.0, 4.0), np.zeros(14, int))      # J0: the short run is not worth two switches
    assert (np.flatnonzero(np.diff(joint_viterbi(prob, b, 2.0, 4.0))) + 1).tolist() == [6, 8]
    weak = np.full(14, 0.01)
    assert np.array_equal(joint_viterbi(prob, weak, 2.0, 4.0), np.zeros(14, int))  # weak boundaries make switching dearer


def test_ties_prefer_staying_and_the_lowest_class():
    flat = np.full((6, 3), 1 / 3)
    assert joint_viterbi(flat, np.full(6, 0.5), 1.0, 0.0).tolist() == [0] * 6      # every step ties: stay, final state 0
    assert joint_viterbi(flat, np.full(6, 0.5), 0.0, 0.0).tolist() == [0] * 6


def test_clipping_keeps_zero_probabilities_finite():
    prob = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]); b = np.array([0.0, 1.0, 0.0])
    path = joint_viterbi(prob, b, 2.0, 1.0)
    assert path.tolist() == _brute(prob, b, 2.0, 1.0)[0].tolist() == [0, 1, 1]       # leaving at b = 0 costs 2 log(1e-6)
    assert np.isfinite(path_score(path, prob, b, 2.0, 1.0))


def test_input_checks():
    prob, b = _random(5, 5, 1)
    with pytest.raises(ValueError):
        joint_viterbi(prob, b[:4], 1.0, 1.0)
    with pytest.raises(ValueError):
        joint_viterbi(prob, b, -0.5, 1.0)
    with pytest.raises(ValueError):
        joint_viterbi(np.where(prob > 0.5, np.nan, prob), b, 1.0, 1.0)
    with pytest.raises(ValueError):
        joint_viterbi(prob, b + 1.0, 1.0, 1.0)
    with pytest.raises(ValueError):
        joint_viterbi_grid(prob, b, [])
    with pytest.raises(ValueError):
        joint_viterbi(np.where(np.arange(5) == 0, 1.1, prob), b, 1.0, 1.0)      # p above 1
    with pytest.raises(ValueError):
        joint_viterbi(np.full((3, 128), 1 / 128), np.full(3, 0.5), 1.0, 1.0)   # int8 back-pointers hold at most 127 classes
    assert joint_viterbi(np.minimum(prob + 1e-7, 1.0 + 1e-6), b, 1.0, 1.0).shape == (5,)   # rounding just above 1 is tolerated
