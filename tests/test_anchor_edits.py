import importlib
import itertools

import numpy as np
import pytest


def api():
    return importlib.import_module('echo_routing.temporal.anchor_edits')


def test_maximal_disagreement_components_do_not_merge_or_expand_and_use_all_classes():
    m = api()
    base = np.array([0, 0, 1, 2, 2, 3, 4, 4])
    raw = np.array([0, 1, 2, 2, 3, 3, 0, 1])
    assert m.components(base, raw) == [(1, 3), (4, 5), (6, 8)]
    assert m.components(base, base) == []
    assert m.components(np.zeros(200, int), np.full(200, 4)) == [(0, 200)]
    for bad in [[], [5], [-1], [1.0], [[0]]]:
        with pytest.raises(ValueError):
            m.components(np.asarray(bad), np.asarray(bad))
    with pytest.raises(ValueError):
        m.components(base, raw[:-1])


def test_one_raw_supported_patch_noop_ties_and_guard_veto_fallback():
    m = api()
    base = np.array([0, 0, 0, 1, 1])
    raw = np.array([1, 1, 0, 1, 0])
    result = m.decode(base, raw, [2., 1.], guarded=False)
    assert result['accepted'] == (0, 2)
    assert result['prediction'].tolist() == [1, 1, 0, 1, 1]
    guarded = m.decode(base, raw, [2., 1.], guarded=True)
    assert guarded['accepted'] == (4, 5)
    assert guarded['vetoed'] == [(0, 2)]
    assert m.majority(guarded['prediction']) == m.majority(base) == 0
    assert m.decode(base, raw, [1., 1.])['accepted'] == (0, 2)
    for scores in [[0., 0.], [-1., 0.]]:
        assert m.decode(base, raw, scores)['accepted'] is None
        np.testing.assert_array_equal(m.decode(base, raw, scores)['prediction'], base)
    np.testing.assert_array_equal(m.decode(base, base, [])['prediction'], base)
    with pytest.raises(ValueError):
        m.decode(base, raw, [np.nan, 1.])
    with pytest.raises(ValueError):
        m.decode(base, raw, [1.])
    with pytest.raises(ValueError):
        m.apply_action(base, raw, (0, 1))
    assert np.all(guarded['prediction'][base == raw] == base[base == raw])


def test_majority_guard_exhaustive_five_class_ties_and_native_harm_counterexample():
    m = api()
    for b in itertools.product(range(5), repeat=3):
        base = np.array(b)
        for r in itertools.product(range(5), repeat=3):
            raw = np.array(r)
            spans = m.components(base, raw)
            result = m.decode(base, raw, np.arange(len(spans), 0, -1), guarded=True)
            assert m.majority(result['prediction']) == m.majority(base)
    # A tie is won by class order, exactly as the historical native metric.
    assert m.majority(np.array([4, 1, 4, 1])) == 1
    # The guard also preserves Base's wrong majority and vetoes a perfect fix.
    wrong = m.decode(np.zeros(3, int), np.ones(3, int), [1.], guarded=True)
    assert wrong['accepted'] is None
    # Majority preservation can still add native frame errors and internal cuts.
    base = np.zeros(7, int)
    raw = np.array([0, 0, 1, 1, 0, 0, 0])
    out = m.decode(base, raw, [1.], guarded=True)['prediction']
    assert m.majority(out) == 0 and np.count_nonzero(out != base) == 2
    assert np.count_nonzero(np.diff(out)) == 2


def test_single_action_utility_has_exact_pooled_f1_sign_with_formal_matching():
    m = api()
    # One unchanged context stream makes this a pooled, not per-stream identity.
    context = (2, 3, 4)
    for truth in itertools.product(range(2), repeat=4):
        y = np.array(truth)
        for b in itertools.product(range(2), repeat=4):
            base = np.array(b)
            bm, bp, bt = m.counts(base, y)
            old_m, old_p, old_t = bm + context[0], bp + context[1], bt + context[2]
            f = 2 * old_m / (old_p + old_t)
            for r in itertools.product(range(2), repeat=4):
                raw = np.array(r)
                for span in m.components(base, raw):
                    target = m.utility(base, raw, span, y, f)
                    nm, npred, nt = m.counts(m.apply_action(base, raw, span), y)
                    new_f = 2 * (nm + context[0]) / (npred + context[1] + nt + context[2])
                    expected = (npred + context[1] + nt + context[2]) * (new_f - f)
                    assert target['utility'] == pytest.approx(expected, abs=1e-14)
                    assert target['delta_M'] == nm - bm
                    assert target['delta_P'] == npred - bp

