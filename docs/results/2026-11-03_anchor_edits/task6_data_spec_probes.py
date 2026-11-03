"""A59 complete-seed qualification probes; synthetic arrays only, no optimization."""
import importlib
import numpy as np
import pytest
from echo_routing.temporal.anchor_features import fit_normalizer
from echo_routing.temporal.anchor_runtime import prepare_training
from echo_routing.temporal.anchor_training import draw_plan


@pytest.mark.parametrize('missing', ['both', 'one'])
def test_complete_plan_recount_rejects_missing_declared_seed(missing):
    m = importlib.import_module('echo_routing.temporal.anchor_data_checks')
    qs = [{'id': 'native', 'kind': 'native', 'labels': np.zeros(7, int)},
          {'id': 'multi', 'kind': 'multi', 'labels': np.array([0, 1, 1, 0, 0, 0, 0])}]
    a = [np.eye(5)[[0, 1, 0, 0, 0, 0, 0]], np.eye(5)[[0, 1, 1, 0, 0, 0, 0]]]
    b = [np.zeros(7, int), np.zeros(7, int)]
    data = prepare_training(qs, a, [np.full((7, 5), .2)] * 2, b)
    norms = {arm: fit_normalizer(data['features'][arm]) for arm in ['A', 'J']}
    plans = {} if missing == 'both' else {2026110359: draw_plan(1, 1, 2026110359)}
    with pytest.raises(ValueError, match='seed|plan|complete'):
        m.validate(data, qs, a, b, norms, plans)
