import copy
import importlib

import numpy as np
import pytest

from echo_routing.temporal.anchor_features import fit_normalizer
from echo_routing.temporal.anchor_runtime import prepare_training
from echo_routing.temporal.anchor_training import draw_plan


def test_actual_training_artifact_recounts_targets_scalers_and_full_seed_plan():
    m = importlib.import_module('echo_routing.temporal.anchor_data_checks')
    qs = [{'id': 'n', 'kind': 'native', 'labels': np.zeros(7, int)},
          {'id': 'c', 'kind': 'multi', 'labels': np.array([0, 1, 1, 0, 0, 0, 0])}]
    a = [np.eye(5)[[0, 1, 0, 0, 0, 0, 0]], np.eye(5)[[0, 1, 1, 0, 0, 0, 0]]]
    base = [np.zeros(7, int), np.zeros(7, int)]
    data = prepare_training(qs, a, [np.full((7, 5), .2)] * 2, base)
    norms = {arm: fit_normalizer(data['features'][arm]) for arm in ['A', 'J']}
    plans = {seed: draw_plan(1, 1, seed) for seed in [2026110359, 2026110360]}
    assert m.validate(data, qs, a, base, norms, plans)['all_candidate_targets_independently_recounted']
    for fault in ['target', 'normalizer', 'plan', 'candidate_span', 'seed_missing_both', 'seed_missing_one', 'seed_extra', 'seed_alias']:
        bad, normalizers, draw = copy.deepcopy(data), copy.deepcopy(norms), copy.deepcopy(plans)
        if fault == 'target':
            bad['targets'][0] += 1
        elif fault == 'normalizer':
            normalizers['J']['std'][0] += 1
        elif fault == 'plan':
            draw[2026110359]['native'][0, 0] = 1
        elif fault == 'candidate_span':
            bad['candidate_rows'][0]['span'] = (0, 1)
        elif fault == 'seed_missing_both':
            draw = {}
        elif fault == 'seed_missing_one':
            draw.pop(2026110360)
        elif fault == 'seed_extra':
            draw[2026110361] = draw_plan(1, 1, 2026110361)
        else:
            draw['2026110359'] = draw[2026110359]
        with pytest.raises(ValueError):
            m.validate(bad, qs, a, base, normalizers, draw)
