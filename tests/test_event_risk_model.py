import importlib,copy
import numpy as np
import pytest
import torch

def api():return importlib.import_module('echo_routing.temporal.event_risk_model')

def fixture():
 g=torch.Generator().manual_seed(9);x=torch.randn(64,205,generator=g);y=(x[:,0]+x[:,1]>.2).float();return {'x':x,'y':y}

def identity():return {k:'a'*64 for k in ['code','config','sources','input','draw_plan','normalizer','target_scales']}

def norms():return {'candidate':{'mean':np.zeros(109),'std':np.ones(109)},'context':{'mean':np.zeros(96),'std':np.ones(96)}}

def test_6625_matched_init_zero_output_BCE_and_live_repeat_diskresume(tmp_path):
 m=api();torch.set_num_threads(1);a=m.EventRisk('R1',2026110461);other=m.EventRisk('R2',2026110461)
 assert sum(p.numel() for p in a.parameters())==6625
 assert all(torch.equal(v,other.state_dict()[k]) for k,v in a.state_dict().items())
 assert torch.equal(a(fixture()['x']),torch.zeros(64))
 opt=m.optimizer(a);first=m.step(a,opt,fixture(),0)
 assert first['physical_candidates']==64 and first['candidate_BCE']==pytest.approx(np.log(2)) and first['finite_gradients']
 assert not first['nonzero_gradient_tensors']['network.0.weight']
 m.step(a,opt,fixture(),1);state=m.checkpoint(a,opt,2,identity(),norms())
 repeated=m.EventRisk('R1',2026110461);ro=m.optimizer(repeated)
 for u in range(2):m.step(repeated,ro,fixture(),u)
 assert all(torch.equal(v,repeated.state_dict()[k]) for k,v in a.state_dict().items())
 resume=m.EventRisk('R1',2026110461);so=m.optimizer(resume);m.step(resume,so,fixture(),0);path=tmp_path/'state.pt';m.atomic_save(path,m.checkpoint(resume,so,1,identity(),norms()));loaded=torch.load(path,weights_only=True)
 assert m.restore(resume,so,loaded,identity(),norms())==1;m.step(resume,so,fixture(),1)
 assert all(torch.equal(v,resume.state_dict()[k]) for k,v in a.state_dict().items())
 bad=copy.deepcopy(state);bad['update']=3
 with pytest.raises(ValueError):m.restore(resume,so,bad,identity(),norms())
 bad=copy.deepcopy(state);bad['normalizers']['context']['std'][0]=2
 with pytest.raises(ValueError):m.restore(resume,so,bad,identity(),norms())
 opt.param_groups[0]['weight_decay']=0
 with pytest.raises(ValueError):m.step(a,opt,fixture(),2)
 assert all(float(v['step'])==2 for v in so.state.values())

def test_reject_foreign_model_optimizer_missing_params_seed_alias_and_nonbinary():
 m=api();a=m.EventRisk('R1',2026110462);b=m.EventRisk('R1',2026110462)
 with pytest.raises(ValueError):m.step(a,m.optimizer(b),fixture(),0)
 with pytest.raises(ValueError):m.EventRisk('R1','2026110461')
 with pytest.raises(ValueError):m.EventRisk('R3',2026110461)
 batch=fixture();batch['y'][0]=.5
 with pytest.raises(ValueError):m.step(a,m.optimizer(a),batch,0)
