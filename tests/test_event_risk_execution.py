import importlib
from pathlib import Path
import numpy as np

def test_readonly_audit_score_loading_never_builds_or_forwards_heads(tmp_path,monkeypatch):
 m=importlib.import_module('scripts.event_risk_execution');b=np.zeros(3,int);prob=np.ones((3,5))/5;v={'actions':[],'old_scores':np.empty((0,6)),'q':np.empty((0,2))}
 monkeypatch.setattr(m,'verify_freeze',lambda n:{'objects':{}});monkeypatch.setattr(m,'bank',lambda r:([{'kind':'native','labels':b}], [prob],[prob],[b]))
 calls=[]
 def old(seed,role,**kw):calls.append(kw);return [(np.empty((0,6)),{'scores_sha256':'a'*64})]
 monkeypatch.setattr(m,'old_scores',old);monkeypatch.setattr(m,'verify_pin',lambda p:tmp_path/'normalizers.json');monkeypatch.setattr(m,'sha',lambda p:'a'*64)
 def read(p):
  name=Path(p).name
  if name=='training_inputs.json':return {'normalizers':{'path':'x','sha256':'b'*64}}
  if name=='input_manifest.json':return {'roles':{'CAL':{'sha256':'c'*64}}}
  if name=='model_audit.json':return {'models':[{'seed':2026110461,'arm':a,'path':a,'sha256':'d'*64} for a in ['R1','R2']]}
  return {}
 monkeypatch.setattr(m,'read',read);monkeypatch.setattr(m,'event_heads',lambda s:(_ for _ in ()).throw(AssertionError('readonly model construction')))
 class Cache:
  cursor=1;path=m.RUN/'never_written_toy';journal=tmp_path/'journal';identity='test'
  def __init__(self,*a,**kw):assert kw['readonly']
  def load(self,i):return v
 monkeypatch.setattr(m,'EventCache',Cache)
 values,info=m.score_values(2026110461,'CAL',readonly=True)
 assert values==[v] and info['new_event_MLP_forwards_this_process']==0 and calls==[{'readonly':True}]

def test_config_rejects_unused_training_or_top_level_and_preserves_frozen_contract():
 import copy,json
 m=importlib.import_module('scripts.event_risk_common');cfg=json.loads(m.CFG.read_text());assert m.validate_config(cfg)
 for bad in [{**cfg,'unused_lr':1},{**cfg,'training':{**cfg['training'],'aux_loss':1}}]:
  import pytest
  with pytest.raises(ValueError):m.validate_config(bad)

def test_fresh_outer_head_reader_refuses_before_readonly_adoption(tmp_path,monkeypatch):
 import pytest
 m=importlib.import_module('scripts.event_risk_common');role='a62_confirmation_selection';row={'source':'a08','role':role,'path':'unused','identity':'id','journal_sha256':'a'*64,'input_sha256':'a'*64}
 monkeypatch.setattr(m,'RUN',tmp_path/'run');monkeypatch.setattr(m,'PUB',tmp_path/'pub');monkeypatch.setattr(m,'sha',lambda p:'a'*64);monkeypatch.setattr(m,'sequences',lambda r:[{'refs':[1]}])
 def read(p):return {'source_cache_roles':{},'roles':{role:{'sha256':'a'*64}}} if Path(p).name=='input_manifest.json' else {'complete':True,'new_event_updates_at_production':0,'records':[row]}
 monkeypatch.setattr(m,'read',read);monkeypatch.setattr(m,'ProbabilityCache',lambda *a,**k:(_ for _ in ()).throw(AssertionError('reader opened before adoption')))
 with pytest.raises(ValueError,match='adoption'):m.probability_reader('a08',role)
