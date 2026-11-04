"""A62 observable-context contracts, entirely synthetic and without NN imports."""
import unittest
import numpy as np

from echo_routing.temporal.proposal_actions import canonical, Action
from echo_routing.temporal.baselines.b3_js_divergence import boundary_scores
try:
    from echo_routing.temporal.event_risk_features import (
        context_features, fit_context_normalizer, normalize_contexts, event_matrix,
    )
except ModuleNotFoundError as error:
    if error.name != 'echo_routing.temporal.event_risk_features':
        raise
    context_features = fit_context_normalizer = normalize_contexts = event_matrix = None


def onehot(labels):
    return np.eye(5)[np.asarray(labels)]


class EventContextFeaturesTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(callable(context_features), 'A62 context features are not implemented yet')

    def test_whole_anchor_run_ordered_probability_bins_and_geometry(self):
        base = np.zeros(16, dtype=np.int64)
        a = onehot(base)
        a[:2] = [.2, .8, 0, 0, 0]
        j = onehot([1] * 8 + [0] * 8)
        action = canonical(base, 0, 2, [1, 1], old=True)
        snapshots = [x.copy() for x in [a, j, base]]
        c = context_features(a, j, base, action)
        self.assertEqual(c.shape, (96,))
        self.assertEqual(c.dtype, np.float64)
        expected_a = np.array([[.2, .8, 0, 0, 0]] + [[1, 0, 0, 0, 0]] * 7)
        expected_j = np.array([[0, 1, 0, 0, 0]] * 4 + [[1, 0, 0, 0, 0]] * 4)
        np.testing.assert_allclose(c[:40].reshape(8, 5), expected_a)
        np.testing.assert_allclose(c[40:80].reshape(8, 5), expected_j)
        np.testing.assert_allclose(c[80:86], [np.log1p(16), np.log1p(2), np.log1p(2), np.log1p(16), 0, .125])
        np.testing.assert_allclose(c[86:93], [0, 0, np.log1p(1), np.log1p(16), np.log1p(16), .125, 1])
        np.testing.assert_allclose(c[93:95], [.8, .8])
        self.assertAlmostEqual(c[95], boundary_scores(a, 7)[2])
        for value, original in zip([a, j, base], snapshots):
            np.testing.assert_array_equal(value, original)

    def test_run_merge_counts_endpoints_and_unchanged_interior(self):
        base = np.array([0] * 4 + [1] * 4 + [0] * 6 + [2] * 6)
        a, j = onehot(base), onehot([0] * 15 + [2] * 5)
        action = canonical(base, 4, 15, [0] * 11, jepa=True)
        c = context_features(a, j, base, action)
        np.testing.assert_allclose(c[80:86], [np.log1p(20), np.log1p(5), np.log1p(11), np.log1p(20), .2, .75])
        np.testing.assert_allclose(c[86:93], [np.log1p(2), np.log1p(3), np.log1p(1), np.log1p(4), np.log1p(6), 1 / 6, 5 / 11])
        np.testing.assert_allclose(c[93:95], [1, 1])

    def test_short_axis_empty_bins_are_zero_and_endpoints_are_finite(self):
        base = np.array([0, 0, 0])
        action = canonical(base, 2, 3, [1], old=True)
        c = context_features(onehot(base), onehot(base), base, action)
        np.testing.assert_array_equal(c[:40].reshape(8, 5).sum(1), [0, 0, 1, 0, 0, 1, 0, 1])
        np.testing.assert_array_equal(c[:40], c[40:80])
        self.assertTrue(np.isfinite(c).all())
        self.assertEqual(c[95], 0)

    def test_observation_validation_and_no_reference_label_argument(self):
        base = np.array([0, 0, 0])
        action = canonical(base, 0, 1, [1])
        a = onehot(base)
        with self.assertRaises(ValueError):
            context_features(a[:-1], a, base, action)
        bad = a.copy()
        bad[0] *= .1
        with self.assertRaises(ValueError):
            context_features(bad, a, base, action)
        with self.assertRaises(ValueError):
            context_features(a, a, base, Action(0, 1, (0,)))
        with self.assertRaises(TypeError):
            context_features(a, a, base, action, truth=np.array([4, 4, 4]))

    def test_R1_duplicate_and_R2_FIT_only_normalization_are_matched_205(self):
        x = np.stack([np.arange(109), np.arange(109) + 2]).astype(float)
        ctx = np.stack([np.arange(96) * 10, np.arange(96) * 10 + 4]).astype(float)
        old = {'mean': np.zeros(109), 'std': np.ones(109)}
        norm = fit_context_normalizer(ctx)
        r1 = event_matrix(x, None, 'R1', old)
        r2 = event_matrix(x, ctx, 'R2', old, norm)
        self.assertEqual(r1.shape, (2, 205))
        self.assertEqual(r2.shape, (2, 205))
        self.assertEqual(r1.dtype, np.float32)
        self.assertEqual(r2.dtype, np.float32)
        np.testing.assert_array_equal(r1[:, :109], r2[:, :109])
        np.testing.assert_array_equal(r1[:, 109:], r1[:, :96])
        np.testing.assert_array_equal(r2[:, 109:], np.array([[-1] * 96, [1] * 96], np.float32))
        with self.assertRaises(ValueError):
            event_matrix(x, ctx, 'R1', old, norm)
        with self.assertRaises(ValueError):
            event_matrix(x, None, 'R2', old, norm)
        with self.assertRaises(ValueError):
            event_matrix(x, ctx, 'R2', old)
        with self.assertRaises(ValueError):
            event_matrix(x, ctx, 'new_arm', old, norm)

    def test_normalizer_floor_and_empty_cache_contract(self):
        values = np.zeros((2, 96))
        values[:, 0] = [0, 4]
        norm = fit_context_normalizer(values)
        self.assertEqual(norm['std'][0], 2)
        np.testing.assert_array_equal(norm['std'][1:], np.full(95, 1e-6))
        np.testing.assert_array_equal(normalize_contexts(values, norm)[:, 0], [-1, 1])
        old = {'mean': np.zeros(109), 'std': np.ones(109)}
        self.assertEqual(event_matrix(np.empty((0, 109)), None, 'R1', old).shape, (0, 205))
        self.assertEqual(event_matrix(np.empty((0, 109)), np.empty((0, 96)), 'R2', old, norm).shape, (0, 205))
        with self.assertRaises(ValueError):
            fit_context_normalizer(np.empty((0, 96)))
        with self.assertRaises(ValueError):
            fit_context_normalizer(np.zeros((2, 95)))
        bad = {'mean': np.zeros(96), 'std': np.zeros(96)}
        with self.assertRaises(ValueError):
            normalize_contexts(values, bad)


if __name__ == '__main__':
    unittest.main()

class EndpointCostEquivalenceTests(unittest.TestCase):
    def test_endpoint_only_JSD_matches_original_full_axis_bits_without_scanning_all_scores(self):
        import importlib
        from unittest.mock import patch
        from echo_routing.temporal.proposal_actions import canonical
        from echo_routing.temporal.baselines import b3_js_divergence as b3
        m=importlib.import_module('echo_routing.temporal.event_risk_features')
        rng=np.random.default_rng(62);original_js=b3.js_divergence
        for n in [1,2,3,8,17,64,701,1024]:
            a=rng.uniform(.01,1,(n,5));a/=a.sum(1,keepdims=True);j=a[:,::-1].copy();base=np.arange(n)%5
            original=b3.boundary_scores(a,7)
            ends=sorted({0,min(1,n),n//2,max(0,n-1),n})
            for s in ends:
                for e in ends:
                    if s>=e:continue
                    action=canonical(base,s,e,((base[s:e]+1)%5).tolist(),jepa=True)
                    calls=[]
                    def tracked(*args):calls.append(1);return original_js(*args)
                    with patch.object(b3,'js_divergence',side_effect=tracked),patch.object(m,'js_divergence',side_effect=tracked,create=True):
                        actual=m.context_features(a,j,base,action)[95]
                    expected=max(float(original[p]) if 0<p<n else 0. for p in [s,e])
                    self.assertEqual(np.float64(actual).tobytes(),np.float64(expected).tobytes())
                    self.assertLessEqual(len(calls),2,'computing every B3 score per candidate violates endpoint-only cost contract')

if __name__=='__main__':unittest.main()
