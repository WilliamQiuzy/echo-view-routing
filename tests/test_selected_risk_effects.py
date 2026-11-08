import importlib
import numpy as np
from echo_routing.temporal.proposal_actions import Action

def mod():
 try:m=importlib.import_module('echo_routing.temporal.selected_risk_effects')
 except ModuleNotFoundError:m=None
 assert m is not None,'new actual-axis event signature extractor missing'
 return m

def test_actual_pureFP_signature_excludes_unrelated_axis_sources():
 m=mod();b=np.array([0]*6+[1]*2+[0]*6);truth=np.zeros(14,np.int64);q={'id':'axis','kind':'multi','labels':truth,'refs':[('unrelated_0',i,1.) if i not in [6,7] else ('responsible_0',i,.9) for i in range(14)]};ctx=np.zeros(96);ctx[91]=.5
 r=m.record(q,b,Action(7,8,(2,),False,True),ctx,D=True,X=False)
 assert r['H']==0 and r['R']==0 and r['dFP']==1 and r['responsible_sessions']==['responsible']
 assert r['unsafe'] and r['event_signature'] and r['new_FP16_segments']==2

def test_safe_J_recovery_fullaxis_counts_and_shortloss_target_unchanged():
 m=mod();b=np.zeros(14,np.int64);truth=np.array([0]*6+[3]*2+[0]*6);q={'id':'axis','kind':'multi','labels':truth,'refs':[('source_0',i,1.) for i in range(14)]};ctx=np.zeros(96);ctx[91]=2/14
 r=m.record(q,b,Action(6,8,(3,3),False,True),ctx,D=True,X=True)
 assert r['H']==0 and r['R']==2 and r['dFP']==-1 and not r['unsafe'] and r['short16_lost']==0 and r['Jexclusive_short16_recovered']==3
 assert r['event_signature'] is None and r['draw_signature'] and r['min_overlap']==2/14
