import importlib
import numpy as np
from echo_routing.temporal.proposal_actions import canonical
from echo_routing.temporal.event_risk_policy import decode

def test_fewer_cuts_can_lose_shortunit_and_unsafe_event_stays_visible():
 m=importlib.import_module('echo_routing.temporal.event_risk_analysis');b=np.r_[np.zeros(30,int),np.ones(10,int),np.zeros(30,int)];a=canonical(b,30,40,np.zeros(10,int),jepa=True);old=np.array([[1,0,10,0,.01,.01]]);q=np.array([[.1,.2]]);v={**decode(b,[a],old,q[:,0],q[:,1],{k:.5 for k in ['L0','M0','R1','R2']}),'actions':[a],'all_actions':[a],'G_rejected':[],'old_scores':old,'q':q}
 result=m.summaries([{'id':'fake','kind':'multi','labels':b}],[b],[v],.8)
 assert set(result)=={'E','N','L0','M0','R1','R2'}
 for r in result.values():
  assert r['accepted_actions']==1 and r['accepted_effect_totals']['H']==10 and r['accepted_effect_totals']['delta_P']==-2
  assert r['accepted_short16_true_units_lost']==1 and r['accepted_new_JEPA_short16_true_units_recovered']==0
  assert r['whole_axis_accepted_actions']==0 and r['event_risk_all_candidates']['R1']['count']==1
