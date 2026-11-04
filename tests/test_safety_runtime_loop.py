import importlib
import numpy as np
import pytest
import torch


def test_empty_G_inference_never_calls_model_or_nonempty_normalizer():
    m=importlib.import_module('echo_routing.temporal.safety_runtime')
    class Forbidden:
        def __call__(self,*a):raise AssertionError('empty G model forward')
    p=np.tile([.8,.05,.05,.05,.05],(12,1));b=np.zeros(12,int)
    r=m.infer(Forbidden(),Forbidden(),None,2.,p,p,b,tau=0.)
    assert r['score_matrix'].shape==(0,6) and all(np.array_equal(r[k]['prediction'],b) for k in ['E','N','C'])


def test_loop_callback_ambiguous_claim_stops_without_replay(tmp_path):
    m=importlib.import_module('echo_routing.temporal.safety_loop')
    from echo_routing.temporal.safety_training import Utility,optimizer,draw_plan
    data={'features':np.ones((2,109)),'position_features':np.ones((2,47)),'position_labels':np.array([0,1]),'position_offsets':np.arange(3),'targets':np.array([[1.,1.],[-1.,-2.]]),'effects':[{'dFP':0},{'dFP':1}], 'normalizers':{'candidate':{'mean':np.zeros(109),'std':np.ones(109)},'position':{'mean':np.zeros(47),'std':np.ones(47)}},'scale':2.,'groups':[np.array([0]),np.array([1])],'kind_groups':{'native':np.array([0]),'constructed':np.array([1])},'kind_indices':{'native':np.array([0]),'constructed':np.array([1])}}
    plan=draw_plan(1,1,2026110461,'utility');identity={k:'a'*64 for k in ['code','config','sources','input','draw_plan','normalizer','target_scales']}
    model=Utility('U-P',2026110461);opt=optimizer(model);calls=[]
    def fail(*args):calls.append(1);raise RuntimeError('deliberate inner interruption')
    with pytest.raises(RuntimeError):m.train_one(model,opt,data,plan,identity,tmp_path,stop=2,latest_every=1,milestones=(2,),on_inner=fail)
    assert len(calls)==1 and (tmp_path/'inner_pending.json').exists()
    with pytest.raises(ValueError):m.train_one(model,opt,data,plan,identity,tmp_path,stop=2,latest_every=1,milestones=(2,),on_inner=fail)
    assert len(calls)==1 and all(float(opt.state[p]['step'])==2 for p in model.parameters())
