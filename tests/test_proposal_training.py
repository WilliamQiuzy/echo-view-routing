import importlib
import copy
import numpy as np
import torch
import pytest


def api():return importlib.import_module('echo_routing.temporal.proposal_training')


def fixture():
    rng=np.random.default_rng(60)
    return [(torch.from_numpy(rng.normal(size=(32,109)).astype(np.float32)),torch.from_numpy(rng.normal(size=(32,3)).astype(np.float32))) for _ in range(2)]


def identity():return {k:str(i)*64 for i,k in enumerate(['code','config','sources','input','draw_plan','normalizer','target_scales'],1)}


def test3619_shared_zero_init_three_output_equal_kind_and_output_MSE():
    m=api();a=m.ProposalRegressor('A',2026110360);j=m.ProposalRegressor('J',2026110360)
    assert sum(p.numel() for p in a.parameters())==3619
    assert all(torch.equal(a.state_dict()[k],v) for k,v in j.state_dict().items())
    assert torch.equal(a(torch.zeros(4,109)),torch.zeros(4,3))
    opt=m.optimizer(a);batches=fixture();row=m.step(a,opt,batches,0)
    expected=np.mean([float(np.mean(y.numpy()**2)) for x,y in batches])
    assert np.isclose(row['loss'],expected,rtol=1e-6) and row['physical_candidates']==64
    assert row['nonzero_gradient_tensors']['network.0.weight'] is False
    live=dict(row['nonzero_gradient_tensors'])
    for u in range(1,4):
        r=m.step(a,opt,batches,u)
        live={k:v or r['nonzero_gradient_tensors'][k] for k,v in live.items()}
    assert all(live.values()) and all(float(v['step'])==4 for v in opt.state.values())
    opt.param_groups[0]['weight_decay']=.02
    with pytest.raises(ValueError):m.step(a,opt,batches,4)


def test_shared_fixed_domain60_plans_empty_kind_and_no_fake_updates():
    m=api();a=m.draw_plan(3,5,2026110360);b=m.draw_plan(3,5,2026110360)
    assert all(np.array_equal(a[k],b[k]) for k in ['native','constructed'])
    assert np.array_equal(a['native'],np.random.default_rng([2026110360,60,0]).integers(3,size=(6000,32),dtype=np.int64))
    assert m.draw_plan(0,5,2026110360)['native'].shape==(6000,0)
    model=m.ProposalRegressor('A',2026110360)
    with pytest.raises(ValueError):m.step(model,m.optimizer(model),[None,None],0)
    assert m.draw_plan(0,0,2026110360)['no_action_domain']


def test_full_model_Adam_scales_RNG_disk_resume_and_durable_pending_refusal(tmp_path,monkeypatch):
    m=api();norm={'mean':np.zeros(109),'std':np.ones(109)};scales=np.array([1.,2.,3.]);batches=fixture();ids=identity()
    def run(stop,payload=None):
        model=m.ProposalRegressor('J',2026110360);opt=m.optimizer(model)
        start=0 if payload is None else m.restore(model,opt,payload,ids,norm,scales)
        for u in range(start,stop):m.step(model,opt,batches,u)
        return m.checkpoint(model,opt,stop,ids,norm,scales)
    expected=run(4);prefix=run(2);path=tmp_path/'resume.pt';m.atomic_save(prefix,path)
    resumed=run(4,torch.load(path,weights_only=True))
    assert all(torch.equal(expected['model'][k],v) for k,v in resumed['model'].items())
    assert all(torch.equal(expected['optimizer']['state'][k]['exp_avg_sq'],v['exp_avg_sq']) for k,v in resumed['optimizer']['state'].items())
    bad=copy.deepcopy(prefix);bad['update']=3
    with pytest.raises(ValueError):run(4,bad)
    model=m.ProposalRegressor('J',2026110360)
    with pytest.raises(ValueError):m.restore(model,m.optimizer(model),prefix,ids,norm,np.ones(3))
    events=[];real=m.fsync_directory
    def sync(folder):
        events.append((tmp_path/'journal'/'pending.json').exists());real(folder)
    monkeypatch.setattr(m,'fsync_directory',sync)
    journal=m.UpdateJournal(tmp_path/'journal',ids);journal.begin(0)
    assert events[-1] is True
    with pytest.raises(ValueError):m.UpdateJournal(tmp_path/'journal',ids)
