import importlib,json
import numpy as np
import pytest
import torch

def test_fixed_draw_physical_updates_and_pending_checkpoint_rejection(tmp_path):
 m=importlib.import_module('echo_routing.temporal.event_risk_loop')
 from echo_routing.temporal.event_risk_model import EventRisk,optimizer
 from echo_routing.temporal.safety_training import draw_plan
 g=torch.Generator().manual_seed(4);x=torch.randn(4,205,generator=g);d={'x':{'R1':x,'R2':x.clone()},'y':torch.tensor([0.,1.,0.,1.]),'kind_indices':{'native':np.array([0,2]),'constructed':np.array([1,3])},'normalizers':{'candidate':{'mean':np.zeros(109),'std':np.ones(109)},'context':{'mean':np.zeros(96),'std':np.ones(96)}}}
 identity={k:'c'*64 for k in ['code','config','sources','input','draw_plan','normalizer','target_scales']};a=EventRisk('R1',2026110461);o=optimizer(a);plan=draw_plan(2,2,2026110461,'risk')
 v=m.train_one(a,o,d,plan,identity,tmp_path/'model',stop=2,latest_every=1,milestones=())
 assert v['committed_physical_updates']==2 and v['physical_candidate_exposures']==128 and v['actual_Adam_update']==2
 rows=[json.loads(x) for x in (tmp_path/'model/physical_updates.jsonl').read_text().splitlines()];assert len(rows)==2 and all(len(r['candidate_indices'])==64 for r in rows)
 (tmp_path/'model/pending.json').write_text('{}')
 with pytest.raises(ValueError):m.train_one(a,o,d,plan,identity,tmp_path/'model',stop=2,latest_every=1,milestones=())

def test_existing_CUDA_context_rejected_before_journal_or_model_access(tmp_path,monkeypatch):
 m=importlib.import_module('echo_routing.temporal.event_risk_loop');monkeypatch.setattr(m.torch.cuda,'is_initialized',lambda:True)
 with pytest.raises(ValueError,match='CUDA'):m.train_one(None,None,None,None,None,tmp_path/'never',stop=1)
 assert not (tmp_path/'never').exists()
