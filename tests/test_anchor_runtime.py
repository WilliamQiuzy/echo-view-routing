import importlib
import inspect

import numpy as np
import torch


class Scores(torch.nn.Module):
    def __init__(self, scores):
        super().__init__()
        self.values = torch.tensor(scores)
        self.calls = 0
    def forward(self, x):
        self.calls += 1
        return self.values


def test_one_shared_score_pass_policies_have_no_metadata_or_truth_interface():
    m = importlib.import_module('echo_routing.temporal.anchor_runtime')
    assert list(inspect.signature(m.infer).parameters) == ['model', 'normalizer', 'prob_a', 'prob_j', 'base', 'arm']
    base = np.array([0, 0, 0, 1, 1])
    a = np.eye(5)[[1, 1, 0, 1, 0]]
    j = np.full_like(a, .2)
    norm = {'mean': np.zeros(98), 'std': np.ones(98)}
    model = Scores([2., 1.])
    out = m.infer(model, norm, a, j, base, 'A')
    assert model.calls == 1
    assert out['U']['accepted'] == (0, 2) and out['G']['accepted'] == (4, 5)
    changed_j = m.infer(model, norm, a, np.eye(5)[[4, 3, 2, 1, 0]], base, 'A')
    for policy in ['U', 'G']:
        np.testing.assert_array_equal(out[policy]['prediction'], changed_j[policy]['prediction'])
    empty = Scores([])
    no_action = m.infer(empty, norm, np.eye(5)[base], j, base, 'A')
    assert empty.calls == 0 and no_action['G']['accepted'] is None


def test_training_preparation_uses_one_fixed_multi_Ffit_and_shared_candidate_axes():
    m = importlib.import_module('echo_routing.temporal.anchor_runtime')
    qs = [{'id': 'native', 'kind': 'native', 'labels': np.zeros(7, int)},
          {'id': 'multi', 'kind': 'multi', 'labels': np.array([0, 1, 1, 0, 0, 0, 0])}]
    base = [np.zeros(7, int), np.zeros(7, int)]
    a = [np.eye(5)[[0, 1, 0, 0, 0, 0, 0]], np.eye(5)[[0, 1, 1, 0, 0, 0, 0]]]
    j = [np.full((7, 5), .2)] * 2
    value = m.prepare_training(qs, a, j, base)
    assert value['F_fit'] == .5
    assert value['features']['A'].shape == value['features']['J'].shape == (2, 98)
    np.testing.assert_array_equal(value['features']['A'][:, :62], value['features']['J'][:, :62])
    assert value['kind_indices']['native'].tolist() == [0]
    assert value['kind_indices']['constructed'].tolist() == [1]
    assert value['targets'][1] == 3.
    assert value['candidate_rows'][0]['sequence_index'] == 0
    assert value['candidate_rows'][0]['feature_positions'] == 7
