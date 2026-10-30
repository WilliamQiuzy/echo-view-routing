import numpy as np
import pytest


def test_fixed_pair_plan_and_shuffled_labels_are_matched_and_balanced():
    from echo_routing.temporal.pw_pilot import pair_plan, fit_labels
    p = pair_plan()
    assert p.shape == (600, 2) and set(p.ravel()) == set(range(19))
    assert np.array_equal(p, pair_plan())
    labels = fit_labels()
    assert labels['L'].shape == labels['S'].shape == (19, 2)
    assert np.all(labels['L'].sum(1) == 1) and np.all(labels['S'].sum(1) == 1)
    assert np.count_nonzero(np.any(labels['L'] != labels['S'], axis=1)) == 9


def test_metrics_cutoff_zero_and_auc_ties_are_explicit():
    from echo_routing.temporal.pw_pilot import metrics
    result = metrics(np.zeros(4))
    assert result == {'correct': 2, 'total': 4, 'accuracy': .5, 'AUROC': .5, 'pair_order_points': 1., 'pairs': 2}
    result = metrics(np.array([-2., 3., -1., 2.]))
    assert result['correct'] == 4 and result['AUROC'] == 1 and result['pair_order_points'] == 2
    with pytest.raises(ValueError): metrics(np.array([0., 1., np.nan, 2.]))


def test_geometry_control_zero_variance_is_finite_and_uses_fit_only_scaling():
    from echo_routing.temporal.pw_pilot import fit_geometry, geometry_scores
    x = np.array([[1., 3.], [1., 5.], [1., 2.], [1., 6.]])
    control = fit_geometry(x)
    assert control['scale'][0] == 1.
    assert np.isfinite(geometry_scores(x, control)).all()
    assert np.array_equal(geometry_scores(np.zeros((2, 2)), {'mean': np.zeros(2), 'scale': np.ones(2), 'coef': np.zeros(3)}), np.zeros(2))
    assert np.array_equal(control['mean'], x.mean(0))
