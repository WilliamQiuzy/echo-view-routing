import importlib,json,copy
import numpy as np
import pytest
import torch
from echo_routing.temporal.safety_training import Utility,Risk,optimizer,draw_plan

def mod():
 try:m=importlib.import_module('echo_routing.temporal.selected_risk_training')
 except ModuleNotFoundError:m=None
 assert m is not None,'new complement-only aux trainer absent'
 return m

def data():
 rng=np.random.default_rng(8);norms={'candidate':{'mean':np.zeros(109),'std':np.ones(109)},'position':{'mean':np.zeros(47),'std':np.ones(47)}}
 return {'normalizers':norms,'scale':1.,'features':rng.normal(size=(4,109)),'position_features':rng.normal(size=(8,47)),'position_labels':np.array([0,1,2,0,1,2,0,1]),'position_offsets':np.array([0,2,4,6,8]),'targets':np.array([[1.,1.],[-1.,-2.],[.5,.5],[-.2,-1.2]]),'effects':[{'dFP':0},{'dFP':1},{'dFP':0},{'dFP':1}],'groups':[np.array([0,1]),np.array([2,3])],'kind_indices':{'native':np.array([0,1]),'constructed':np.array([2,3])},'kind_groups':{'native':np.array([0]),'constructed':np.array([1])}}

def test_complement_sources_actual_refs_no_pool_fallback():
 m=mod();groups=[{'id':'own_0','refs':[('own_0',0,1.)]}];held={'held'}
 assert m.validate_training_sources(groups,{'own'},held)
 with pytest.raises(ValueError):m.validate_training_sources([{'id':'mix','refs':[('held_0',0,1.)]}],{'own'},held)
 d=data();d['kind_groups']['native']=np.array([],np.int64)
 with pytest.raises(ValueError):m.aux_plans(d,2026110461)

def test_both_aux_roles_two_updates_and_one_plus_one_fullstate_restore(tmp_path):
 m=mod();d=data();identity={k:'1'*64 for k in ['code','config','sources','input','draw_plan','normalizer','target_scales']}
 for role in ['utility','risk']:
  model=Utility('S-L',2026110461) if role=='utility' else Risk(2026110461);opt=optimizer(model);plans=m.aux_plans(d,model.seed);p=plans[role]
  direct=m.train_aux_one(model,opt,d,p,identity,tmp_path/(role+'direct'),stop=2,latest_every=1,milestones=())
  a=Utility('S-L',2026110461) if role=='utility' else Risk(2026110461);ao=optimizer(a)
  m.train_aux_one(a,ao,d,p,identity,tmp_path/(role+'split'),stop=1,latest_every=1,milestones=())
  b=Utility('S-L',2026110461) if role=='utility' else Risk(2026110461);bo=optimizer(b)
  resumed=m.train_aux_one(b,bo,d,p,identity,tmp_path/(role+'split'),stop=2,latest_every=1,milestones=())
  assert direct['actual_Adam_update']==resumed['actual_Adam_update']==2 and resumed['new_updates_this_process']==1
  for k,v in model.state_dict().items():assert torch.equal(v,b.state_dict()[k])
  (tmp_path/(role+'split')/'pending.json').write_text('{}')
  with pytest.raises(ValueError):m.train_aux_one(b,bo,d,p,identity,tmp_path/(role+'split'),stop=2)

def test_cuda_guard_precedes_any_model_or_journal_access(monkeypatch,tmp_path):
 m=mod();monkeypatch.setattr(torch.cuda,'is_initialized',lambda:True)
 with pytest.raises(ValueError):m.train_aux_one(None,None,None,None,None,tmp_path/'never')
 assert not (tmp_path/'never').exists()
