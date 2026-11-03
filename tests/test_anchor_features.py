import importlib

import numpy as np
import pytest


def api():
    return importlib.import_module('echo_routing.temporal.anchor_features')


def test_exact98_feature_order_regions_source_swap_and_empty_edges():
    m = api()
    p = np.eye(5)[[0, 1, 2, 3, 4]]
    j = np.full((5, 5), .2)
    base = np.array([0, 0, 0, 3, 4])
    a = m.candidate_features(p, j, base, (1, 3), 'A')
    h = m.candidate_features(p, j, base, (1, 3), 'J')
    assert a.shape == h.shape == (98,)
    np.testing.assert_allclose(a[:3], np.log1p([1, 2, 2]))
    np.testing.assert_array_equal(a[3:8], [1, 0, 0, 0, 0])
    np.testing.assert_array_equal(a[8:13], [0, .5, .5, 0, 0])
    np.testing.assert_array_equal(a[13:23], [1, 0, 0, 0, 0, 0, 0, 0, 1, 0])
    np.testing.assert_array_equal(a[23:25], [1, 1])
    assert a[25] == 2
    np.testing.assert_array_equal(a[:62], h[:62])
    np.testing.assert_array_equal(a[26:62], a[62:])
    np.testing.assert_array_equal(h[62:], m.source_features(j, (1, 3)))
    # Per-region 5mean, 5adjacent-difference, entropy, margin.
    np.testing.assert_array_equal(a[26:31], [1, 0, 0, 0, 0])
    np.testing.assert_array_equal(a[31:36], np.zeros(5))
    assert a[36] == 0 and a[37] == 1
    np.testing.assert_array_equal(a[43:48], [0, 1, 1, 0, 0])
    edge = m.source_features(p, (0, 5))
    np.testing.assert_array_equal(edge[:12], np.zeros(12))
    np.testing.assert_array_equal(edge[24:], np.zeros(12))
    assert h[72] == pytest.approx(np.log(5))
    with pytest.raises(ValueError):
        m.candidate_features(p * 2, j, base, (1, 3), 'J')
    with pytest.raises(ValueError):
        m.candidate_features(p, j[:-1], base, (1, 3), 'J')


def test_fit_only_normalizer_is_fixed_finite_and_has_std_floor():
    m = api()
    x = np.arange(3 * 98, dtype=np.float32).reshape(3, 98)
    x[:, 0] = 7
    norm = m.fit_normalizer(x)
    assert norm['std'][0] == 1e-6
    np.testing.assert_allclose(m.normalize(x, norm).mean(0), 0, atol=1e-6)
    fixed = {k: v.copy() for k, v in norm.items()}
    result = m.normalize(np.full((1, 98), 20.), norm)
    assert result.shape == (1, 98)
    for k in norm:
        np.testing.assert_array_equal(norm[k], fixed[k])
    for bad in [np.zeros((0, 98)), np.zeros((3, 97)), np.full((3, 98), np.nan)]:
        with pytest.raises(ValueError):
            m.fit_normalizer(bad)
