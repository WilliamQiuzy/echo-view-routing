import numpy as np

def test_all156_empty_G_and_readonly_fulltask_composition(monkeypatch):
 import scripts.readout_event_risk as m
 from echo_routing.temporal.event_risk_metrics import TABLES,HALVES,SEEDS,POLICIES,BASE
 b=np.array([0,0,0]);p=np.eye(5)[b];main=[{'id':'native_0','kind':'native','labels':b},{'id':'multi_0','kind':'multi','labels':b},{'id':'pair_0','kind':'pair','labels':b}]
 banks={};ctx={}
 for t in TABLES:
  ctx[t]={}
  for h in HALVES:
   role=m.role_for(t,h);qs=main if t=='main' else [{'id':role,'kind':'multi','labels':b}];banks[role]=(qs,[p]*len(qs),[p]*len(qs),[b]*len(qs));ctx[t][h]={'sequences':main if t=='main' else [main[0],main[2],qs[0]]}
 monkeypatch.setattr(m,'preouter',lambda:{});monkeypatch.setattr(m,'bank',lambda r:banks[r]);monkeypatch.setattr(m,'contexts',lambda:ctx)
 def read(p):
  if p.name=='training_inputs.json':return {'F_fit':.8}
  if p.name=='calibration.json':return {'thresholds':[{'seed':s,'policy':x,'tau':.5} for s in SEEDS for x in ['M0','R1','R2']],'legacy_thresholds':[{'seed':s,'policy':'L0','tau':.1} for s in SEEDS]}
  return {}
 monkeypatch.setattr(m,'read',read);calls=[]
 def scores(s,r,readonly=False):
  calls.append(readonly);v=[{'actions':[],'all_actions':[],'G_rejected':[],'old_scores':np.empty((0,6)),'q':np.empty((0,2))} for q in banks[r][0]]
  return v,{'path':'private','identity':'fixed','journal_sha256':'a'*64,'sequences':len(v)}
 monkeypatch.setattr(m,'score_values',scores);monkeypatch.setattr(m,'evaluate',lambda p,c,b:({'segment_f1_50_multi':1.},{}))
 old=[{'table':t,'half':h,'seed':s,'arm':'S-L','policy':x,'segment_f1_50_multi':1.} for t in TABLES[:-1] for h in HALVES for s in SEEDS for x in ['E','N','C']]
 monkeypatch.setattr(m,'old_records',lambda:{'outer_metrics.json':{'rows':old}})
 rows,pred,*_=m.complete(readonly=True)
 assert len(rows)==156 and all(calls)
 for t in TABLES:
  for h in HALVES:
   for n in pred[t][h]:
    assert np.array_equal(pred[t][h][n]['native_0'],b) and np.array_equal(pred[t][h][n]['pair_0'],b)
 assert all(r['action_statistics_full_task']['accepted_actions']==0 for r in rows if r['role']=='policy')
