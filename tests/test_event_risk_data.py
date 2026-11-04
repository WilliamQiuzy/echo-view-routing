import importlib,hashlib
import numpy as np
import pytest

def api():return importlib.import_module('echo_routing.temporal.event_risk_data')

def natives():
 out=[]
 for s in range(100):
  for k in range(6+(s<28)):
   name=f'fake_session{s:03d}_{k}';label=k%5
   out.append({'id':name,'kind':'native','labels':np.full(80,label,np.int64),'refs':[(name,t,1.) for t in range(80)]})
 return out

def test_hash_partition_exact50_source_disjoint_and_no_retry():
 m=api();qs=natives();a,b,record=m.partition_inner(qs)
 expected=sorted({q['id'].rsplit('_',1)[0] for q in qs},key=lambda s:(hashlib.sha256(b'A62-inner-v1\0'+s.encode()).digest(),s))
 assert {q['id'].rsplit('_',1)[0] for q in a}==set(expected[:50])
 assert len(a)+len(b)==628 and record['sessions']==[50,50]
 assert not ({q['id'] for q in a}&{q['id'] for q in b})
 assert all(np.bincount([q['labels'][0] for q in v],minlength=5).min()>0 for v in [a,b])
 with pytest.raises(ValueError):m.partition_inner(qs[:-1])
 with pytest.raises(ValueError):m.partition_inner([{**q,'kind':'multi'} for q in qs])

def test_fixed_four_recipe_domains_and_no_cross_group_source():
 m=api();a,b,_=m.partition_inner(natives());roles=['CAL_mixed','AUDIT_mixed','a62_confirmation_selection','a62_confirmation_calibration']
 for i,role in enumerate(roles):
  source=a if i%2==0 else b;count=150 if i<2 else 120
  q,recipes,record=m.construct(source,role)
  assert len(q)==len(recipes)==count and record['RNG_domain']==[2026110463,62,i]
  assert {ref[0] for z in q for ref in z['refs']}<={z['id'] for z in source}
  assert all(len({x['video_id'] for x in r['fragments']})==len(r['fragments']) for r in recipes)
 with pytest.raises(ValueError):m.construct(a,'inner_cal')

def test_unsafe_targets_are_gross_event_not_net_and_exact_original_risk_plan():
 m=api();assert m.unsafe_targets([{'H':2,'R':9,'dFP':0},{'H':0,'R':1,'dFP':1},{'H':0,'R':4,'dFP':-1}]).tolist()==[1.,1.,0.]
 from echo_routing.temporal.safety_training import draw_plan
 original=draw_plan(2,3,2026110461,'risk');p=m.validate_draw_plan(original,2,3,2026110461)
 assert np.array_equal(p['indices'],original['indices'])
 assert m.candidate_indices(p,0,np.array([0,2]),np.array([1,3,4])).shape==(64,)
 changed={**original,'indices':original['indices'].copy()};changed['indices'][0,0]=1-changed['indices'][0,0]
 with pytest.raises(ValueError):m.validate_draw_plan(changed,2,3,2026110461)
