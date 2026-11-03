import copy
import importlib
import os

import numpy as np
import pytest
import torch


def api():
    return importlib.import_module('echo_routing.temporal.anchor_training')


def equal(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return a == b


def batch():
    generator = torch.Generator().manual_seed(59)
    return [(torch.randn(32, 98, generator=generator), torch.randn(32, generator=generator) + kind)
            for kind in range(2)]


NORM = {'mean': np.zeros(98), 'std': np.ones(98)}
IDENTITY = {k: str(i) * 64 for i, k in enumerate(['code', 'config', 'sources', 'input', 'draw_plan', 'normalizer'], 1)}


def test_3201_matched_initialization_zero_noop_and_later_hidden_liveness_cpu_only():
    m = api()
    a, j = m.UtilityRegressor('A', 2026110359), m.UtilityRegressor('J', 2026110359)
    assert sum(p.numel() for p in a.parameters()) == 3201
    assert equal(a.state_dict(), j.state_dict())
    assert set(dict(a.named_parameters())) == {'network.0.weight', 'network.0.bias', 'network.2.weight', 'network.2.bias'}
    x, _ = batch()[0]
    assert torch.count_nonzero(a(x)) == 0
    opt = m.optimizer(a)
    before = a.network[0].weight.detach().clone()
    first = m.step(a, opt, batch(), 0)
    assert not first['nonzero_gradient_tensors']['network.0.weight']
    records = [first] + [m.step(a, opt, batch(), u) for u in range(1, 4)]
    assert all(any(row['nonzero_gradient_tensors'][name] for row in records) for name, _ in a.named_parameters())
    assert not torch.equal(before, a.network[0].weight)
    assert all(float(opt.state[p]['step']) == 4 and opt.state[p]['exp_avg_sq'].count_nonzero() > 0
               for p in a.parameters())
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized()


def test_shared_kind_balanced_draw_plan_and_empty_kind_contract():
    m = api()
    plan = m.draw_plan(7, 13, 2026110359)
    repeat = m.draw_plan(7, 13, 2026110359)
    assert plan['native'].shape == plan['constructed'].shape == (6000, 32)
    assert plan['native'].min() == 0 and plan['native'].max() == 6
    assert plan['constructed'].min() == 0 and plan['constructed'].max() == 12
    np.testing.assert_array_equal(plan['native'], repeat['native'])
    np.testing.assert_array_equal(plan['constructed'], repeat['constructed'])
    empty = m.draw_plan(0, 13, 2026110359)
    assert empty['native'].shape == (6000, 0) and empty['constructed'].shape == (6000, 32)
    assert m.draw_plan(0, 0, 2026110359)['no_action_domain']
    model = m.UtilityRegressor('A', 59)
    opt = m.optimizer(model)
    # q starts at zero; two kind means have equal weight regardless of targets.
    row = m.step(model, opt, [(torch.zeros(32, 98), torch.ones(32)),
                              (torch.zeros(32, 98), torch.full((32,), 3.))], 0)
    assert row['loss'] == pytest.approx(5.)
    model = m.UtilityRegressor('A', 59)
    row = m.step(model, m.optimizer(model), [None, (torch.zeros(32, 98), torch.ones(32))], 0)
    assert row['loss'] == 1. and row['physical_candidates'] == 32
    with pytest.raises(ValueError):
        m.step(model, m.optimizer(model), [None, None], 0)


def test_exact_repeat_2plus2_atomic_disk_resume_and_all_rng(tmp_path):
    m = api()
    def run(stop, value=None):
        model = m.UtilityRegressor('J', 2026110359)
        opt = m.optimizer(model)
        start = 0 if value is None else m.restore(model, opt, value, IDENTITY, NORM)
        for u in range(start, stop):
            m.step(model, opt, batch(), u)
        return m.checkpoint(model, opt, stop, IDENTITY, NORM)
    expected, repeat = run(4), run(4)
    assert equal(expected['model'], repeat['model']) and equal(expected['optimizer'], repeat['optimizer'])
    path = tmp_path / 'latest.pt'
    m.atomic_save(run(2), path)
    resumed = run(4, torch.load(path, weights_only=True))
    assert equal(expected['model'], resumed['model']) and equal(expected['optimizer'], resumed['optimizer'])
    assert set(resumed['rng']) == {'torch_CPU', 'numpy', 'python'}
    assert not list(tmp_path.glob('*.tmp'))


@pytest.mark.parametrize('fault', ['cursor', 'recipe', 'moment_shape', 'moment_negative', 'moment_nan', 'model_key', 'source', 'normalizer'])
def test_resume_rejects_cursor_recipe_moments_and_input_binding(fault):
    m = api()
    model = m.UtilityRegressor('A', 59)
    opt = m.optimizer(model)
    m.step(model, opt, batch(), 0)
    value = m.checkpoint(model, opt, 1, IDENTITY, NORM)
    wrong = copy.deepcopy(value)
    state = next(iter(wrong['optimizer']['state'].values()))
    if fault == 'cursor':
        wrong['update'] = 2
        wrong['optimizer']['param_groups'][0]['lr'] = .00002
    elif fault == 'recipe':
        wrong['optimizer']['param_groups'][0]['weight_decay'] = .2
    elif fault == 'moment_shape':
        state['exp_avg_sq'] = torch.zeros(1)
    elif fault == 'moment_negative':
        state['exp_avg_sq'].view(-1)[0] = -1
    elif fault == 'moment_nan':
        state['exp_avg'].view(-1)[0] = float('nan')
    elif fault == 'model_key':
        wrong['model'].pop('network.0.bias')
    elif fault == 'source':
        wrong['identity']['sources'] = 'a' * 64
    elif fault == 'normalizer':
        wrong['normalizer']['std'][0] = 2
    fresh = m.UtilityRegressor('A', 59)
    with pytest.raises(ValueError):
        m.restore(fresh, m.optimizer(fresh), wrong, IDENTITY, NORM)


def test_binding_rejects_parameter_replacement_and_changed_recipe():
    m = api()
    model = m.UtilityRegressor('J', 59)
    opt = m.optimizer(model)
    assert m.optimizer_binding(model, opt)
    opt.param_groups[0]['eps'] = 1e-6
    assert not m.optimizer_binding(model, opt)
    opt = m.optimizer(model)
    model.network[0].weight = torch.nn.Parameter(model.network[0].weight.detach().clone())
    assert not m.optimizer_binding(model, opt)


def test_pending_claim_and_trace_latest_mismatch_never_silently_replay(tmp_path):
    m = api()
    journal = m.UpdateJournal(tmp_path, IDENTITY)
    assert journal.resume_cursor() == 0
    journal.begin(0)
    with pytest.raises(ValueError, match='pending|ambiguous'):
        m.UpdateJournal(tmp_path, IDENTITY)
    with pytest.raises(ValueError):
        journal.begin(0)
    journal.finish({'u': 0, 'loss': 1.})
    # A real physical update after the latest checkpoint is not silently repeated.
    with pytest.raises(ValueError, match='checkpoint|cursor'):
        journal.resume_cursor()
    model = m.UtilityRegressor('A', 59)
    opt = m.optimizer(model)
    m.step(model, opt, batch(), 0)
    value = m.checkpoint(model, opt, 1, IDENTITY, NORM)
    m.atomic_save(value, tmp_path / 'latest.pt')
    assert journal.resume_cursor() == 1
    changed = {**IDENTITY, 'config': 'f' * 64}
    with pytest.raises(ValueError):
        m.UpdateJournal(tmp_path, changed)
