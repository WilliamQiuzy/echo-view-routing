"""A47 contract tests: data identity, strict sources, actual head and chain rule."""
import copy
import importlib
import importlib.util

import numpy as np
import pytest
import torch
from torch.nn import functional as F


def core():
    name = 'echo_routing.temporal.clinical_pretraining'
    assert importlib.util.find_spec(name) is not None, 'A47 core is not implemented'
    torch.set_num_threads(2)
    return importlib.import_module(name)


def test_marker_reference_preserves_slot_fragment_and_gamma():
    cp = core()
    rows = [[('left', i, 1.) for i in range(16)],
            [('right', i + 100, .75) for i in range(16)]]
    assert cp.marker_refs(rows, [0, 15]) == [('left', 0, 1.), ('right', 115, .75)]
    with pytest.raises(ValueError): cp.marker_refs(rows, [16, 1])
    with pytest.raises(ValueError): cp.marker_refs(rows, [1])


def test_strict_source_extraction_rejects_missing_extra_shape_nonfinite():
    cp = core()
    expected = {'features.0.weight': torch.zeros(2, 3), 'classifier.0.weight': torch.ones(2)}
    p = {'weights': {'encoder.encoder.model.' + k: v + 1 for k, v in expected.items()} |
                    {'heads.LVOTDiam.weight': torch.zeros(1, 2)}}
    assert set(cp.extract_image_state(p, expected, 'P')) == set(expected)
    g = dict(expected) | {'classifier.2.weight': torch.zeros(1000, 768),
                          'classifier.2.bias': torch.zeros(1000)}
    assert set(cp.extract_image_state(g, expected, 'G')) == set(expected)
    for bad in [dict(p['weights'], **{'encoder.encoder.model.extra': torch.zeros(1)}),
                {'encoder.encoder.model.features.0.weight': torch.zeros(2, 3)},
                dict(p['weights'], **{'encoder.encoder.model.features.0.weight': torch.zeros(3, 2)}),
                dict(p['weights'], **{'encoder.encoder.model.features.0.weight': torch.full((2,3), float('nan'))})]:
        with pytest.raises(ValueError): cp.extract_image_state({'weights': bad}, expected, 'P')
    with pytest.raises(ValueError): cp.extract_image_state(g | {'unexpected': torch.zeros(1)}, expected, 'G')


def fixture_batch(dtype=torch.float64):
    torch.manual_seed(47)
    x = torch.randn(8, 768, 7, 7, dtype=dtype)
    e = torch.randn(8, 512, dtype=dtype)
    w = torch.tensor([.25, .75, 0., 1., .3, .7, .6, .4], dtype=dtype)
    slots = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    return x, e, w, slots


def test_head_parameter_binding_init_and_batch_independence():
    cp = core()
    m = cp.ClinicalRepair().double().eval()
    x, e, _, _ = fixture_batch()
    assert sum(p.numel() for p in m.parameters()) == 197521
    assert torch.equal(m(x, e), torch.full((8,), -2., dtype=torch.float64))
    with torch.no_grad(): m.fusion[2].weight.fill_(.01)
    assert torch.allclose(m(x,e), torch.cat([m(x[i:i+1],e[i:i+1]) for i in range(8)]), atol=1e-12, rtol=1e-12)


def test_whole_run_objective_matches_hand_calculation():
    cp = core()
    scores = torch.tensor([-1., 2., 99., -2., 3., 1., -.5, .5], dtype=torch.float64)
    _, _, w, slots = fixture_batch()
    loss, r = cp.loss_from_scores(scores, w, slots)
    direct = torch.stack([scores[slots==i].dot(w[slots==i]) for i in range(4)])
    assert torch.equal(r, direct)
    assert torch.allclose(loss, torch.logaddexp(torch.tensor(0.), -direct[0]) +
                          torch.logaddexp(torch.zeros(3), direct[1:]).mean())
    with pytest.raises(ValueError): cp.loss_from_scores(scores, w * 2, slots)


def test_full_gradient_matches_individual_window_j_transpose_v_including_zero():
    cp = core()
    m = cp.ClinicalRepair().double().eval()
    x, e, w, slots = fixture_batch()
    with torch.no_grad(): m.fusion[2].weight.fill_(.02)
    s = m(x,e); loss, _ = cp.loss_from_scores(s,w,slots)
    loss.backward(); full = torch.cat([p.grad.flatten() for p in m.parameters()])
    z = s.detach().requires_grad_(); ll, _ = cp.loss_from_scores(z,w,slots)
    v = torch.autograd.grad(ll,z)[0]; assert v[2] == 0
    m.zero_grad(set_to_none=True)
    for i in range(8): m(x[i:i+1],e[i:i+1]).backward(v[i:i+1])
    individual = torch.cat([p.grad.flatten() for p in m.parameters()])
    assert torch.linalg.vector_norm(full-individual) / torch.linalg.vector_norm(full) < 1e-12
    assert all(p.grad is not None for p in m.parameters())


def test_six_updates_repeat_forced_equal_features_and_resume():
    cp = core()
    x,e,w,slots=fixture_batch(torch.float32)
    def fresh():
        torch.manual_seed(947)
        m=cp.ClinicalRepair().eval()
        return m,torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001)
    def steps(m,o,n):
        for _ in range(n):
            o.zero_grad(set_to_none=True);loss,_=cp.loss_from_scores(m(x,e),w,slots)
            loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1);o.step()
    m,o=fresh();initial=copy.deepcopy(m.state_dict());steps(m,o,6)
    for kind in ['repeat','forced_equal','resume']:
        mm,oo=fresh()
        if kind=='resume':
            steps(mm,oo,3);state=copy.deepcopy((mm.state_dict(),oo.state_dict()))
            mm,oo=fresh();mm.load_state_dict(state[0]);oo.load_state_dict(state[1]);steps(mm,oo,3)
        else: steps(mm,oo,6)
        assert all(torch.equal(v,mm.state_dict()[k]) for k,v in m.state_dict().items())
    assert all(not torch.equal(v,initial[k]) for k,v in m.state_dict().items())
    assert {int(v['step']) for v in o.state.values()} == {6}


def test_one_way_deployment_strict_zero_and_non_a4c_invariance():
    cp = core()
    base=np.array([0,1,2,2,2,3,4])
    score=np.array([np.nan,np.nan,-.01,0.,.01,np.nan,np.nan])
    assert np.array_equal(cp.deploy(base,score), [0,1,2,2,3,3,4])
    assert np.array_equal(base,[0,1,2,2,2,3,4])
    with pytest.raises(ValueError): cp.deploy(base,np.full(7,np.nan))


def test_independent_functional_auditor_matches_trained_head():
    cp=core()
    name='scripts.audit_clinical_pretrained_a5c'
    assert importlib.util.find_spec(name) is not None, 'A47 independent auditor is not implemented'
    audit=importlib.import_module(name)
    torch.manual_seed(47);m=cp.ClinicalRepair().double();x,e,_,_=fixture_batch()
    with torch.no_grad():m.fusion[2].weight.fill_(.01)
    assert torch.allclose(m(x,e),audit.functional_model(m.state_dict(),x,e),atol=1e-12,rtol=1e-12)


def test_actual_release_task_identity_keeps_documentation_discrepancy():
    cp=core()
    assert hasattr(cp,'clinical_head_names'), 'actual checkpoint task-head verification missing'
    names=['LVOTDiam_head','AVStructure_head']+[f'other_{i}_head' for i in range(38)]
    weights={n+'.1.weight':torch.zeros(1,768) for n in names}
    assert cp.clinical_head_names(weights)==sorted(names)
    with pytest.raises(ValueError):cp.clinical_head_names(dict(list(weights.items())[:-1]))
