import importlib,json
import numpy as np
import pytest

def test_new_joint_q_cache_empty_prefix_and_pending_no_replay(tmp_path):
 m=importlib.import_module('echo_routing.temporal.event_risk_cache');b=np.zeros(5,int);taus={k:.5 for k in ['L0','M0','R1','R2']};refs=[{'source':'frozen','sha256':'a'*64}]
 c=m.EventCache(tmp_path/'cache','identity',[b],[b],[b],taus,refs)
 def predict():
  assert c.pending.exists()
  return {'old_scores':np.empty((0,6)),'q':np.empty((0,2)),'old_reference':refs[0],'seconds':0.}
 c.run_one(0,predict);v=m.EventCache(tmp_path/'cache','identity',[b],[b],[b],taus,refs,readonly=True).load(0)
 assert v['q'].shape==(0,2) and all(v[p]['accepted_index'] is None for p in ['E','N','L0','M0','R1','R2'])
 (tmp_path/'cache/pending.json').write_text('{}')
 with pytest.raises(ValueError):m.EventCache(tmp_path/'cache','identity',[b],[b],[b],taus,refs,readonly=True)

def test_complete_headless_adoption_checks_every_expected_payload_and_production_time(tmp_path):
 m=importlib.import_module('echo_routing.temporal.event_risk_cache')
 from echo_routing.temporal.anchor_cache import ProbabilityCache,sha
 records=[];expected={}
 for source in ['a08','jepa']:
  for role,n in [('CAL_mixed',150),('AUDIT_mixed',150),('a62_confirmation_selection',120),('a62_confirmation_calibration',120)]:
   folder=tmp_path/source/role;c=ProbabilityCache(folder,source+role,[1]*n)
   for i in range(n):c.run_one(i,lambda:np.ones((1,5))/5)
   records.append({'source':source,'role':role,'path':str(folder),'identity':c.identity,'journal_sha256':sha(c.journal),'committed_sequences':n,'complete':True,'input_sha256':'b'*64})
   expected[role]=('b'*64,[1]*n)
 manifest={'complete':True,'new_event_updates_at_production':0,'records':records}
 out=m.adopt_sources(manifest,expected,root=tmp_path,formal_updates=24000)
 assert out['verified_payloads']==1080 and out['new_source_forwards']==0 and out['new_event_updates_at_production']==0 and out['formal_updates_at_binding']==24000
 bad=json.loads(json.dumps(manifest));bad['records']=bad['records'][:-1]
 with pytest.raises(ValueError):m.adopt_sources(bad,expected,root=tmp_path,formal_updates=24000)
