"""Synthetic persistence checks; no pretrained model or neural forward."""
import importlib,json
from pathlib import Path
import numpy as np
import pytest
import torch
from torch import nn

class NoForward(nn.Module):
 def __init__(self):super().__init__();self.head=nn.Linear(1,1);self.joint=False;self.feature_contract={'kind':'synthetic persistence fixture'}
 def forward(self,*args):raise AssertionError('no neural forward allowed in persistence test')

def test_milestone_is_immutable_and_matches_restored_tensor_values(tmp_path,monkeypatch):
 a=importlib.import_module('scripts.train_dense_multidepth');m=NoForward();oldread=a.read;monkeypatch.setattr(a,'read',lambda p:{'target_encoder_sha256':'source'} if Path(p).name=='source.json' else {'git_commit':'future'} if Path(p).name=='stage_freeze.json' else oldread(p));original=a.atomic_torch_save;calls=[]
 def save(value,path):calls.append(str(path));return original(value,path)
 monkeypatch.setattr(a,'atomic_torch_save',save);one=a.milestone(m,tmp_path,500,'F-MD',57);two=a.milestone(m,tmp_path,500,'F-MD',57)
 assert one==two and len(calls)==1
 m.head.weight.data.add_(1)
 with pytest.raises(ValueError):a.milestone(m,tmp_path,500,'F-MD',57)

def test_precheckpoint_failure_keeps_physical_replay_accounting_before_next_forward(tmp_path,monkeypatch):
 a=importlib.import_module('scripts.train_dense_multidepth');monkeypatch.setattr(a,'RUN',tmp_path);monkeypatch.setattr(a.md,'build_model',lambda *args,**kwargs:NoForward());folder=tmp_path/'seed57/F-MD';folder.mkdir(parents=True);(folder/'physical_updates.jsonl').write_text(''.join(json.dumps({'u':u,'attempt':1})+'\n' for u in range(2)))
 class StopBeforeForward:
  def batch(self,*args,**kwargs):raise RuntimeError('expected stop before neural forward')
 cfg={'head_lr':.001,'optimizer':{'betas':[.9,.999],'eps':1e-8,'weight_decay':.05}};plan={'sequence':np.zeros((6000,2),np.int64),'start':np.zeros((6000,2),np.int64)}
 with pytest.raises(RuntimeError,match='expected stop'):a.train_one(cfg,'F-MD',57,[{}],[],StopBeforeForward(),None,plan)
 proof=json.loads((folder/'resume_attempt2.json').read_text());assert proof['checkpoint_next_update']==0 and proof['invalidated_after_checkpoint']==2 and proof['committed_physical_updates']==2
