"""Synthetic structural contracts; no producer/head import or NN/GT work."""
import ast
import itertools
from pathlib import Path
import unittest

import numpy as np

try:
    from echo_routing.temporal.event_risk_cost_bounds import (
        axis_bounds, aggregate_bounds, observed_geometry,
    )
except ModuleNotFoundError as error:
    if error.name != 'echo_routing.temporal.event_risk_cost_bounds':
        raise
    axis_bounds = aggregate_bounds = observed_geometry = None


def original_window_centres():
    # Execute ONLY this original pure function, avoiding windowed's torch import
    # through features.extract. This is a software fixture, not source inference.
    path = Path(__file__).resolve().parents[1] / 'echo_routing/temporal/windowed.py'
    tree = ast.parse(path.read_text())
    function = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == 'window_centres')
    ns = {'np': np}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), ns)
    return ns['window_centres']


def raw_on_windows(T, centre_labels):
    centres = original_window_centres()(T, 5)
    assignment = np.abs(np.arange(T)[:, None] - centres[None, :]).argmin(1)
    return np.asarray(centre_labels, dtype=np.int64)[assignment]


class StructuralCostBoundsTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(callable(axis_bounds), 'new independent structural module is not implemented')

    def test_exact_window_count_and_edge_short_Base_runs(self):
        centres = original_window_centres()
        for T in list(range(1, 140)) + [199, 200, 201, 700, 701, 1024]:
            bound = axis_bounds(T)
            self.assertEqual(bound['W'], len(centres(T, 5)))
            self.assertEqual(bound['N'], 1)
            self.assertEqual(bound['N1'], 1)
            self.assertEqual(bound['T2'], T * T)
            self.assertEqual(bound['C'], (T + 1) // 2 + T)
            self.assertEqual(bound['TC'], T * bound['C'])
            self.assertEqual(bound['K'], 2 * T)
            self.assertEqual(bound['KT'], 2 * T * T)
            self.assertEqual(bound['Rb'], 1 + (T + 4) // 6)
        base = np.array([0] + [1] * 6 + [2], dtype=np.int64)
        result = observed_geometry(base, raw_on_windows(8, [0, 1]), np.arange(8, dtype=np.int64) % 5)
        self.assertEqual(result['Rb'], 3)  # BOTH axis-edge runs may be length1.

    def test_empty_axis_rejected_empty_aggregate_zero_and_exact_sum(self):
        for bad in [True, False, 0, -1, 1.5, '2', None, np.inf]:
            with self.assertRaises(ValueError):
                axis_bounds(bad)
        zero = aggregate_bounds([])
        self.assertEqual(zero['N'], 0)
        self.assertEqual(zero['max_T'], 0)
        lengths = [1, 2, 17, 701]
        total = aggregate_bounds(lengths)
        for field in axis_bounds(1):
            self.assertEqual(total[field], sum(axis_bounds(T)[field] for T in lengths))
        self.assertEqual(total['max_T'], 701)

    def test_exhaustive_short_valid_grammar_all_J_changes_and_dedup_G(self):
        for T in range(1, 7):
            W = axis_bounds(T)['W']
            bases = [np.zeros(T, np.int64)]
            bases += [np.r_[np.zeros(c, np.int64), np.ones(T - c, np.int64)] for c in range(1, T)]
            for base in bases:
                for centres in itertools.product(range(2), repeat=W):
                    raw_a = raw_on_windows(T, centres)
                    for js in itertools.product(range(2), repeat=T):
                        actual = observed_geometry(base, raw_a, np.array(js, np.int64))
                        bound = axis_bounds(T)
                        for field in ['C', 'TC', 'K', 'KT', 'CR', 'Rb']:
                            self.assertLessEqual(actual[field], bound[field])
                            self.assertLessEqual(actual['all_' + field] if 'all_' + field in actual else actual[field], bound[field])
                        self.assertEqual(actual['all_C'], actual['C'] + actual['G_rejected'])

    def test_long_K_and_small_changed_K_can_still_require_full_TC(self):
        T = 701
        base = np.zeros(T, np.int64)
        a = base.copy()
        j = np.ones(T, np.int64)
        long = observed_geometry(base, a, j)
        self.assertEqual(long['all_K'], T)
        self.assertEqual(long['all_KT'], T * T)
        self.assertEqual(long['C'], 0)  # Full-axis majority guard rejects it.
        j = np.arange(T, dtype=np.int64) % 2
        small = observed_geometry(base, a, j)
        self.assertEqual(small['K'], T // 2)
        self.assertEqual(small['C'], T // 2)
        self.assertEqual(small['TC'], T * (T // 2))
        self.assertGreater(small['TC'], small['K'])

    def test_random_valid_Base_and_nearest_window_A08_with_perframe_J(self):
        rng = np.random.default_rng(6262)
        for T in [7, 8, 17, 32, 64, 127, 201]:
            for repeat in range(4):
                # Start at1: no false minimum edge-run-length assumption.
                cuts = list(range(1 + repeat % 3, T, 6 + repeat))
                base = np.empty(T, np.int64)
                for i, (s, e) in enumerate(zip([0] + cuts, cuts + [T])):
                    base[s:e] = i % 5
                a = raw_on_windows(T, rng.integers(0, 5, axis_bounds(T)['W']))
                j = rng.integers(0, 5, T, dtype=np.int64)
                snapshots = [v.copy() for v in (base, a, j)]
                result = observed_geometry(base, a, j)
                for field in ['C', 'TC', 'K', 'KT', 'CR', 'Rb']:
                    self.assertLessEqual(result[field], axis_bounds(T)[field])
                for before, after in zip(snapshots, [base, a, j]):
                    np.testing.assert_array_equal(before, after)

    def test_invalid_source_structure_and_reference_input_fail_closed(self):
        a = np.zeros(17, np.int64)
        with self.assertRaises(ValueError):
            observed_geometry(np.arange(17, dtype=np.int64) % 2, a, a)
        bad_a = a.copy()
        bad_a[1] = 1  # Not a nearest-centre allocation boundary.
        with self.assertRaises(ValueError):
            observed_geometry(a, bad_a, a)
        with self.assertRaises(ValueError):
            observed_geometry(a, a[:-1], a)
        with self.assertRaises(TypeError):
            observed_geometry(a, a, a, truth=a)


if __name__ == '__main__':
    unittest.main()
