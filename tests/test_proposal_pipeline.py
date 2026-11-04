import json
import numpy as np
import pytest
import torch
from echo_routing.temporal.proposal_training import ProposalRegressor,optimizer,draw_plan
from echo_routing.temporal.proposal_loop import train_one
from echo_routing.temporal.proposal_policy_cache import PolicyCache
from echo_routing.temporal.proposal_runtime import infer


def identity():return {k:str(i+1)*64 for i,k in enumerate(['code','config','sources','input','draw_plan','normalizer','target_scales'])}

def test_loop_three_target_scale_and_durable_inner_no_replay(tmp_path):
    rng=np.random.default_rng(1);features=rng.normal(size=(4,109));target=rng.normal(size=(4,3))*[1,10,3]
    data={'features':{'A':features,'J':features},'targets':target,'kind_indices':{'native':np.array([0,1]),'constructed':np.array([2,3])},'candidate_rows':[{'feature_positions':8}]*4}
    norm={'mean':np.zeros(109),'std':np.ones(109)};scales=np.array([1,10,3]);plan=draw_plan(2,2,2026110360)
    def make():m=ProposalRegressor('A',2026110360);return m,optimizer(m)
    m,o=make();seen=[]
    def inner(*args):
        assert (tmp_path/'resume/inner_pending.json').is_file();seen.append(args[1]);return None
    train_one(m,o,data,plan,norm,scales,identity(),tmp_path/'resume',stop=2,latest_every=1,milestones=(2,4),on_inner=inner)
    m,o=make();train_one(m,o,data,plan,norm,scales,identity(),tmp_path/'resume',stop=4,latest_every=1,milestones=(2,4),on_inner=inner)
    assert seen==[2,4]
    a,ao=make();train_one(a,ao,data,plan,norm,scales,identity(),tmp_path/'straight',stop=4,latest_every=1,milestones=())
    assert all(torch.equal(v,a.state_dict()[k]) for k,v in m.state_dict().items())
    raw=json.loads((tmp_path/'straight/physical_updates.jsonl').read_text().splitlines()[0])
    idx=np.r_[data['kind_indices']['native'][plan['native'][0]],data['kind_indices']['constructed'][plan['constructed'][0]]]
    assert np.isclose(raw['loss'],np.mean((target[idx]/scales)**2))
    def broken(*args):raise RuntimeError('synthetic interrupted inner')
    b,bo=make()
    with pytest.raises(RuntimeError):train_one(b,bo,data,plan,norm,scales,identity(),tmp_path/'broken',stop=2,latest_every=1,milestones=(2,),on_inner=broken)
    with pytest.raises(ValueError):train_one(b,bo,data,plan,norm,scales,identity(),tmp_path/'broken',stop=4,latest_every=1,milestones=(2,),on_inner=inner)


def test_three_policy_cache_empty_prefix_and_source_identity(tmp_path):
    b=[np.zeros(20,int),np.array([0]*8+[1]*4+[0]*8)]
    a=[np.eye(5)[x]*.8+.04 for x in b];j=[a[0],np.eye(5)[np.array([0]*8+[2]*4+[0]*8)]*.8+.04]
    norm={'mean':np.zeros(109),'std':np.ones(109)}
    class Count(torch.nn.Module):
        def __init__(self):super().__init__();self.calls=0
        def forward(self,x):self.calls+=1;return torch.tensor([[1.,-1.,-1.]]).repeat(len(x),1)
    model=Count();folder=tmp_path/'cache';identity={'code':'a'*64,'source':'b'*64}
    cache=PolicyCache(folder,identity,b,[p.argmax(1) for p in a],[p.argmax(1) for p in j])
    cache.run_one(0,lambda:infer(model,None,None,a[0],j[0],b[0],'J'));assert model.calls==0
    cache=PolicyCache(folder,identity,b,[p.argmax(1) for p in a],[p.argmax(1) for p in j])
    cache.run_one(1,lambda:infer(model,norm,np.ones(3),a[1],j[1],b[1],'J'));assert model.calls==1
    readonly=PolicyCache(folder,identity,b,[p.argmax(1) for p in a],[p.argmax(1) for p in j],readonly=True)
    assert readonly.load(0)['outputs'].shape==(0,3) and readonly.load(1)['R']['accepted'] is not None
    with pytest.raises(ValueError):readonly.run_one(0,lambda:(_ for _ in ()).throw(AssertionError('no replay')))
    with pytest.raises(ValueError):PolicyCache(folder,{'code':'c'*64},b,[p.argmax(1) for p in a],[p.argmax(1) for p in j],readonly=True)
