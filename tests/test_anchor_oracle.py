import importlib
import itertools
from fractions import Fraction

import numpy as np
import pytest


def api():
    return importlib.import_module('echo_routing.temporal.anchor_oracle')


def test_exact_fractional_action_bound_matches_cartesian_enumeration_and_ties():
    m = api()
    rng = np.random.default_rng(59)
    for _ in range(60):
        options = []
        for length in [4, 5, 6]:
            base, raw, truth = [rng.integers(0, 3, length) for _ in range(3)]
            options.append(m.action_options(base, raw, truth, guarded=False))
        cartesian = list(itertools.product(*options))
        exact = max(Fraction(2 * sum(x['M'] for x in rows), sum(x['P'] + x['T'] for x in rows)) for rows in cartesian)
        result = m.fractional_oracle(options)
        assert Fraction(result['numerator'], result['denominator']) == exact
    tied = [[{'M': 1, 'P': 1, 'T': 1, 'span': None}, {'M': 1, 'P': 1, 'T': 1, 'span': (0, 1)}]]
    assert m.fractional_oracle(tied)['choices'] == [0]
    with pytest.raises(ValueError):
        m.fractional_oracle([[{'M': 0, 'P': 0, 'T': 0, 'span': None}]])


def test_oracle_domains_raw_only_guarded_subset_and_union_is_not_joint_reachability():
    m = api()
    base = np.zeros(7, int)
    raw = np.array([0, 1, 0, 0, 1, 0, 0])
    truth = raw.copy()
    options = m.action_options(base, raw, truth, guarded=False)
    guarded = m.action_options(base, raw, truth, guarded=True)
    assert len(options) == len(guarded) == 3
    assert [x['span'] for x in options] == [None, (1, 2), (4, 5)]
    result = m.fractional_oracle([options])
    assert result['numerator'] < result['denominator']
    coverage = m.missed_coverage(base, raw, truth, guarded=False)
    assert coverage['candidate_union_recovered_true_units'] >= max(x['M'] for x in options)
    assert coverage['union_is_jointly_reachable_claim'] is False
    # Perfect whole-record correction is outside the protected action domain.
    assert len(m.action_options(np.zeros(3, int), np.ones(3, int), np.ones(3, int), guarded=True)) == 1
