import importlib
import numpy as np
import torch
import pytest


def api():return importlib.import_module('echo_routing.temporal.safety_training')


def test_pairwise_all_unordered_unequal_including_fixed_noop_and_group_weight():
    m=api();z=torch.tensor([2.,-1.,.25],requires_grad=True);targets=torch.tensor([2.,2.,-1.],dtype=torch.float64)
    p,l,n=m.utility_losses(z,targets,[2,1],True)
    expected_point=(((z[:2]-targets[:2])**2).mean()+((z[2:]-targets[2:])**2).mean())/2
    # first query excludes tie(2,2), compares each to fixedzero; second compares negative target tozero.
    expected_rank=(torch.nn.functional.softplus(-z[:2]).mean()+torch.nn.functional.softplus(z[2:]).mean())/2
    assert n==3 and torch.allclose(p,expected_point.float()) and torch.allclose(l,expected_rank)
    p0,l0,n0=m.utility_losses(torch.tensor([0.,0.]),torch.tensor([0.,0.]),[2],True)
    assert p0==0 and l0==0 and n0==0
    assert m.utility_losses(z,targets,[2,1],False)[1]==0


def test_shared_seed_architecture_zero_outputs_risk_loss_and_gross_aggregation():
    m=api();heads=[m.Utility(arm,2026110461) for arm in ['U-P','U-L','S-P','S-L']]
    assert all(sum(p.numel() for p in h.parameters())==3553 for h in heads)
    assert all(torch.equal(v,heads[0].state_dict()[k]) for h in heads for k,v in h.state_dict().items())
    assert torch.equal(heads[0](torch.ones(3,109)),torch.zeros(3,1))
    risk=m.Risk(2026110461);assert sum(p.numel() for p in risk.parameters())==5188
    pos=torch.tensor([[0.,0.,0.],[2.,0.,0.],[0.,2.,0.]])
    ce,bce=m.risk_losses(pos,torch.tensor([0,0,1]),torch.tensor([0.,0.]),torch.tensor([0.,1.]),[1,2])
    wanted=(torch.nn.functional.cross_entropy(pos[:1],torch.tensor([0]))+torch.nn.functional.cross_entropy(pos[1:],torch.tensor([0,1])))/2
    assert torch.allclose(ce,wanted) and torch.isclose(bce,torch.tensor(np.log(2),dtype=torch.float32))
    out=m.risk_summary(pos,torch.tensor(0.))
    assert np.isclose(out['predicted_H']+out['predicted_R']+out['predicted_neutral'],3)
    assert 0<=out['r']<=1 and out['pFP']==.5
    with pytest.raises(ValueError):m.Utility('U-P',True)
    with pytest.raises(ValueError):m.draw_plan(1,1,'02026110461','utility')
    a=m.draw_plan(3,0,2026110461,'utility');b=m.draw_plan(3,0,2026110461,'risk')
    assert a['indices'].shape==(6000,64) and a['source_kinds']==[0,0] and a['fallback']
    assert not np.array_equal(a['indices'],b['indices'])


def test_real_steps_repeat6_diskresume3plus3_exact_optimizer_identity(tmp_path):
    m=api();identity={k:'a'*64 for k in ['code','config','sources','input','draw_plan','normalizer','target_scales']}
    norms={'candidate':{'mean':np.zeros(109),'std':np.ones(109)},'position':{'mean':np.zeros(47),'std':np.ones(47)}}
    rng=np.random.default_rng(3);x=torch.tensor(rng.normal(size=(4,109)),dtype=torch.float32)
    px=torch.tensor(rng.normal(size=(7,47)),dtype=torch.float32)
    ub={'x':x,'target':torch.tensor([2.,-1.,0.,-2.]),'lengths':[2,2]}
    rb={'x':x,'positions':px,'labels':torch.tensor([0,1,2,0,1,2,0]),'fp':torch.tensor([0.,1.,0.,1.]),'lengths':[1,2,1,3]}
    for role in ['utility','risk']:
        def model():return m.Utility('S-L',2026110461) if role=='utility' else m.Risk(2026110461)
        full=model();opt=m.optimizer(full)
        for i in range(6):m.step(full,opt,ub if role=='utility' else rb,i,role)
        saved=m.checkpoint(full,opt,6,identity,norms,2.)
        broken=dict(saved);broken['update']=5
        with pytest.raises(ValueError):m.restore(model(),m.optimizer(model()),broken,identity,norms,2.)
        partial=model();po=m.optimizer(partial)
        journal=m.UpdateJournal(tmp_path/role,identity)
        for i in range(3):
            journal.begin(i);row=m.step(partial,po,ub if role=='utility' else rb,i,role)
            journal.finish(row,m.checkpoint(partial,po,i+1,identity,norms,2.))
        resumed=model();ro=m.optimizer(resumed);v=torch.load(tmp_path/role/'latest.pt',weights_only=True)
        assert m.restore(resumed,ro,v,identity,norms,2.)==3
        for i in range(3,6):m.step(resumed,ro,ub if role=='utility' else rb,i,role)
        assert all(torch.equal(p,resumed.state_dict()[k]) for k,p in full.state_dict().items())
        assert journal.resume_cursor()==3
        journal.begin(3)
        with pytest.raises(ValueError):m.UpdateJournal(tmp_path/role,identity)
