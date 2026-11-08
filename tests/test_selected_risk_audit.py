"""Fabricated trace/state contracts only: no data, NN forward or SGD."""
import copy
import importlib
import os
import random

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
import pytest

m = importlib.import_module('echo_routing.temporal.selected_risk_audit')
SEED = 2026110461
IDENTITY = {k: str(i)*64 for i, k in enumerate(['code', 'config', 'sources', 'input', 'draw_plan', 'normalizer', 'target_scales'])}
UTILITY_NAMES = {'network.0.weight', 'network.0.bias', 'network.2.weight', 'network.2.bias'}
RISK_NAMES = {f'{part}.{layer}.{field}' for part in ['position', 'candidate'] for layer in ['0', '2'] for field in ['weight', 'bias']}


def api(name):
    assert callable(getattr(m, name, None)), 'A63 audit missing: ' + name
    return getattr(m, name)


def plans(role):
    from echo_routing.temporal.safety_training import draw_plan
    return draw_plan(1 if role == 'utility' else 2, 1 if role == 'utility' else 2, SEED, role)


@pytest.fixture(scope='module')
def auxiliary():
    from echo_routing.temporal.proposal_training import lr_factor
    data = dict(groups=[np.array([0, 1]), np.array([2, 3])],
                kind_groups={'native': np.array([0]), 'constructed': np.array([1])},
                kind_indices={'native': np.array([0, 1]), 'constructed': np.array([2, 3])},
                targets=np.array([[1., 2.], [1., 0.], [-1., -2.], [1., 2.]]),
                position_offsets=np.array([0, 1, 3, 6, 10]))
    all_rows = {}
    for role in ['utility', 'risk']:
        plan = plans(role); rows = []
        for u, picks in enumerate(plan['indices']):
            if role == 'utility':
                idx = [0, 1]*32 + [2, 3]*32
                queries = [0]*32 + [1]*32
                details = {'physical_groups': 64, 'physical_unequal_pairs': 160,
                           'pointwise_MSE': .1, 'pairwise_logistic': .2}
                names = UTILITY_NAMES
            else:
                idx = picks[:32].tolist() + (picks[32:]+2).tolist(); queries = []
                details = {'physical_changed_positions': sum(i+1 for i in idx), 'position_CE': .1, 'candidate_BCE': .2}
                names = RISK_NAMES
            rows.append(dict(u=u, identity=dict(IDENTITY), loss=.3, lr=.001*lr_factor(u),
                             gradient_norm=.2, finite_gradients=True,
                             nonzero_gradient_tensors={n: u > 0 for n in names},
                             candidate_indices=idx, query_sequence_indices=queries,
                             source_kind_blocks=[0, 1], physical_candidates=len(idx),
                             changed_position_exposures=sum(i+1 for i in idx),
                             optimizer_update_wall_seconds=.001, **details))
        all_rows[role] = rows
    return data, all_rows


@pytest.mark.parametrize('role', ['utility', 'risk'])
def test_full_aux_trace_recounts_queries_candidates_all_positions_and_pairs(auxiliary, role):
    data, rows = auxiliary
    result = api('audit_aux_trace')(rows[role], data, plans(role), SEED, role, IDENTITY)
    assert result['physical_updates'] == 6000
    assert result['candidate_exposures'] == (768000 if role == 'utility' else 384000)
    assert result['query_group_exposures'] == (384000 if role == 'utility' else 0)
    if role == 'utility':
        assert result['changed_position_exposures'] == 1920000
        assert result['unequal_pair_exposures'] == 960000
    assert all(result['all_tensor_gradient_liveness_over_history'].values())
    assert result['no_model_forward_during_audit'] is True


@pytest.mark.parametrize('change', ['partial_list', 'K', 'pairs', 'identity', 'missing_gradient', 'bool_cursor', 'fake_loss'])
def test_aux_does_not_accept_completion_counters_in_place_of_physical_graph(auxiliary, change):
    data, all_rows = auxiliary; rows = list(all_rows['utility']); rows[7] = copy.deepcopy(rows[7]); r = rows[7]
    if change == 'partial_list': r['candidate_indices'] = r['candidate_indices'][:-1]
    if change == 'K': r['changed_position_exposures'] -= 1
    if change == 'pairs': r['physical_unequal_pairs'] = 0
    if change == 'identity': r['identity']['input'] = 'f'*64
    if change == 'missing_gradient': r['nonzero_gradient_tensors'].pop('network.0.weight')
    if change == 'bool_cursor': r['u'] = True
    if change == 'fake_loss': r['loss'] = .01
    with pytest.raises(ValueError): api('audit_aux_trace')(rows, data, plans('utility'), SEED, 'utility', IDENTITY)


def test_aux_empty_fallback_and_caller_missing_identity_are_rejected(auxiliary):
    data, rows = auxiliary; bad = plans('utility'); bad['fallback'] = True
    with pytest.raises(ValueError): api('audit_aux_trace')(rows['utility'], data, bad, SEED, 'utility', IDENTITY)
    with pytest.raises(ValueError): api('audit_aux_trace')(rows['utility'], data, plans('utility'), SEED, 'utility', None)


@pytest.fixture(scope='module')
def event():
    from echo_routing.temporal.proposal_training import lr_factor
    rng = np.random.default_rng(63)
    draws = np.concatenate([rng.integers(2, size=(6000, 32)), rng.integers(2, 4, size=(6000, 32))], axis=1)
    y = np.array([0, 1, 0, 1]); rows = []
    for u, idx in enumerate(draws):
        rows.append(dict(u=u, identity=dict(IDENTITY), loss=.25, candidate_BCE=.25, lr=.001*lr_factor(u),
                         gradient_norm=.3, finite_gradients=True, nonzero_gradient_tensors={n: u > 0 for n in UTILITY_NAMES},
                         physical_candidates=64, candidate_indices=idx.tolist(), source_kind_blocks=[0, 1],
                         unsafe_targets=int(y[idx].sum()), optimizer_update_wall_seconds=.001))
    return rows, draws, {'native': np.array([0, 1]), 'constructed': np.array([2, 3])}, y


def test_formal_trace_uses_global_matched_draws_and_counts_real_exposure(event):
    result = api('audit_event_trace')(*event, IDENTITY)
    assert result['physical_updates'] == 6000 and result['candidate_exposures'] == 384000
    assert result['native_candidate_exposures'] == result['constructed_candidate_exposures'] == 192000
    assert result['unsafe_candidate_exposures'] == int(event[-1][event[1]].sum())


@pytest.mark.parametrize('change', ['short', 'unsafe', 'zero_gradient', 'mixed_in_native', 'bad_expected_identity'])
def test_event_fixed_physical_trace_and_each_tensor_liveness_are_required(event, change):
    rows, draws, pools, y = event; rows = list(rows); rows[3] = copy.deepcopy(rows[3]); expected = IDENTITY
    if change == 'short': rows = rows[:-1]
    if change == 'unsafe': rows[3]['unsafe_targets'] += 1
    if change == 'zero_gradient':
        rows = [{**r, 'nonzero_gradient_tensors': {**r['nonzero_gradient_tensors'], 'network.0.bias': False}} for r in rows]
    if change == 'mixed_in_native': draws = draws.copy(); draws[0, 0] = 2
    if change == 'bad_expected_identity': expected = None
    with pytest.raises(ValueError): api('audit_event_trace')(rows, draws, pools, y, expected)


def fabricated_states(role):
    """A parameter holder whose forward raises; manually filled Adam, zero SGD."""
    import torch
    from echo_routing.temporal import safety_training as s, event_risk_model as e

    class Layer(torch.nn.Module):
        def __init__(self, n, out):
            super().__init__(); self.weight = torch.nn.Parameter(torch.zeros(out, n)); self.bias = torch.nn.Parameter(torch.zeros(out))

    class Holder(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.seed = SEED
            self.arm, self.role = ('S-L', 'utility') if role == 'utility' else ('C0', 'candidate_unsafe_event')
            self.network = torch.nn.Sequential(Layer(109 if role == 'utility' else 205, 32), torch.nn.Identity(), Layer(32, 1))

        def forward(self, *_): raise AssertionError('audit fixtures must never execute a neural forward')

    model = Holder(); backend = s if role == 'utility' else e; opt = backend.optimizer(model)
    norms = {name: {'mean': np.zeros(n), 'std': np.ones(n)} for name, n in
             ([('candidate', 109), ('position', 47)] if role == 'utility' else [('candidate', 109), ('context', 96)])}
    initial = copy.deepcopy(model.state_dict()); states = {}
    for cursor in [2000, 4000, 6000]:
        with torch.no_grad():
            for p in model.parameters(): p.fill_(cursor*1e-6)
        opt.param_groups[0]['lr'] = .001*s.lr_factor(cursor-1)
        for p in model.parameters():
            opt.state[p] = {'step': torch.tensor(float(cursor)), 'exp_avg': torch.full_like(p, .01), 'exp_avg_sq': torch.full_like(p, .001)}
        args = (model, opt, cursor, IDENTITY, norms, 2.) if role == 'utility' else (model, opt, cursor, IDENTITY, norms)
        states[cursor] = backend.checkpoint(*args)
    states['latest'] = copy.deepcopy(states[6000])
    return model, opt, states, norms, initial


@pytest.mark.parametrize('role', ['utility', 'event'])
def test_qualified_fullstates_all_Adam_norm_RNG_identity_without_forward(role):
    model, opt, states, norms, initial = fabricated_states(role)
    if role == 'utility':
        out = api('validate_aux_states')(model, opt, states, IDENTITY, {'normalizers': norms, 'scale': 2.}, initial)
    else:
        out = api('validate_event_states')(model, opt, states, IDENTITY, norms, initial)
    assert out['fullstates'] == 4 and out['latest_final_complete_state_equal'] and out['all_active_tensors_drifted']
    assert out['no_model_forward_during_audit'] is True


@pytest.mark.parametrize('change', ['latest_RNG', 'missing_milestone', 'Adam_step', 'normalizer', 'no_drift', 'zero_moments'])
def test_aux_fullstate_tampering_is_an_audit_failure_not_a_negative_model_result(change):
    import torch
    model, opt, states, norms, initial = fabricated_states('utility')
    if change == 'latest_RNG': states['latest']['rng']['torch_CPU'] = states['latest']['rng']['torch_CPU'].clone(); states['latest']['rng']['torch_CPU'][0] ^= 1
    if change == 'missing_milestone': del states[4000]
    if change == 'Adam_step': next(iter(states[2000]['optimizer']['state'].values()))['step'].fill_(1999)
    if change == 'normalizer': states[4000]['normalizers']['candidate']['mean'][0] = 1
    if change == 'no_drift': initial = copy.deepcopy(states[6000]['model'])
    if change == 'zero_moments':
        for s in [states[2000], states[4000], states[6000], states['latest']]: next(iter(s['optimizer']['state'].values()))['exp_avg_sq'].zero_()
    with pytest.raises(ValueError): api('validate_aux_states')(model, opt, states, IDENTITY, {'normalizers': norms, 'scale': 2.}, initial)
