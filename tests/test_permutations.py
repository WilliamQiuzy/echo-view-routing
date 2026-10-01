"""Within-fragment reorderings (fixed seeds) and deviation-structure statistics."""
from __future__ import annotations

import numpy as np
import pytest

from echo_routing.realvideo.permutations import error_runs, lag1_autocorrelation, order_index


def test_true_and_reverse():
    assert order_index(5, "true")[0].tolist() == [0, 1, 2, 3, 4] and not order_index(5, "true")[1]
    idx, changed = order_index(4, "reverse")
    assert idx.tolist() == [3, 2, 1, 0] and changed


def test_random_is_a_seeded_permutation_of_the_same_samples():
    a, _ = order_index(30, "random", seed=3, recipe_id="r1", frag_index=0)
    b, _ = order_index(30, "random", seed=3, recipe_id="r1", frag_index=0)
    c, _ = order_index(30, "random", seed=4, recipe_id="r1", frag_index=0)
    d, _ = order_index(30, "random", seed=3, recipe_id="r1", frag_index=1)
    assert sorted(a.tolist()) == list(range(30)) and np.array_equal(a, b)
    assert not np.array_equal(a, c) and not np.array_equal(a, d)


def test_block_order_keeps_blocks_and_the_trailing_partial_block():
    idx, changed = order_index(23, "block", seed=1, recipe_id="r", frag_index=0, block=5)
    assert changed and sorted(idx.tolist()) == list(range(23))
    assert idx[-3:].tolist() == [20, 21, 22]  # the partial block stays at the end
    for k in range(4):  # every full block is contiguous and in order
        blk = idx[k * 5:(k + 1) * 5]
        assert blk.tolist() == list(range(blk[0], blk[0] + 5)) and blk[0] % 5 == 0


def test_short_fragment_is_not_perturbed_by_blocks():
    idx, changed = order_index(9, "block", seed=1, recipe_id="r", frag_index=0, block=5)
    assert not changed and idx.tolist() == list(range(9))


def test_name_order_sorts_ids_as_text():
    ids = np.array([0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30])
    idx, changed = order_index(len(ids), "name", ids=ids)
    assert ids[idx].tolist() == [0, 12, 15, 18, 21, 24, 27, 3, 30, 6, 9] and changed
    with pytest.raises(ValueError):
        order_index(3, "name")
    with pytest.raises(ValueError):
        order_index(3, "shuffle")


def test_error_runs_and_autocorrelation():
    assert error_runs(np.array([0, 1, 1, 0, 2, 0, 0, 3, 3, 3]), np.zeros(10, int)).tolist() == [2, 1, 3]
    smooth = np.eye(2)[np.r_[np.zeros(20, int), np.ones(20, int)]]
    alternating = np.eye(2)[np.arange(40) % 2]
    assert lag1_autocorrelation(smooth) > 0.8 and lag1_autocorrelation(alternating) < -0.8
    assert np.isnan(lag1_autocorrelation(np.ones((10, 2)) / 2))


def test_audited_b3_alias_matches_b3_with_the_same_setting():
    from echo_routing.temporal.baselines.registry import AUDITED_B3, decode_with_prob
    rng = np.random.default_rng(0)
    prob = rng.dirichlet(np.ones(5), size=80)
    cfg = {"decode": {"b3": dict(AUDITED_B3), "b3a": dict(AUDITED_B3)}}
    assert np.array_equal(decode_with_prob("b3a", prob, cfg)[0], decode_with_prob("b3", prob, cfg)[0])
    assert np.array_equal(decode_with_prob("b3a", prob, {"decode": {}})[0], decode_with_prob("b3", prob, cfg)[0])
