import importlib
import numpy as np
import pytest
from echo_routing.temporal.proposal_actions import canonical

def api():return importlib.import_module('echo_routing.temporal.event_risk_policy')

def test_fixed_positive_top_N_unchanged_q_only_strict_C_no_fallback():
 m=api();b=np.zeros(10,int);actions=[canonical(b,1,2,[1]),canonical(b,5,6,[2])];scores=np.array([[3,.1,.9,0,.1,.2],[2,.2,.8,0,.1,.1]])
 out=m.decode(b,actions,scores,[.8,.01],[.7,.01],{'L0':.5,'M0':.15,'R1':.5,'R2':.6})
 assert all(out[k]['accepted_index']==0 for k in ['E','N','L0'])
 assert all(out[k]['accepted_index'] is None for k in ['M0','R1','R2'])
 scores[0,1:3]=[.9,.1]
 out=m.decode(b,actions,scores,[0,0],[0,0],{'L0':1,'M0':1,'R1':1,'R2':1})
 assert out['E']['accepted_index']==0 and all(out[k]['accepted_index'] is None for k in ['N','L0','M0','R1','R2'])
 z=m.decode(b,[],np.empty((0,6)),[],[],{'L0':0,'M0':0,'R1':0,'R2':0});assert all(v['accepted_index'] is None for v in z.values())

def test_same_CAL_population_first_unsafe_ties_saturation_and_audit_no_tau_fit():
 m=api();rows=[{'N':True,'unsafe':True,'r':.3,'q1':.2,'q2':.9},{'N':True,'unsafe':False,'r':.2,'q1':.2,'q2':.1},{'N':False,'unsafe':True,'r':0,'q1':0,'q2':0}]
 value=m.calibrate(rows);assert value['M0']['tau']==.3 and value['R1']['tau']==.2 and value['R1']['accepted']==0 and value['R2']['accepted']==1
 assert value['N_admitted']==2 and value['unsafe_N_top']==1 and value['calibration_limited']
 assert all(m.calibrate([])[k]['tau']==0 for k in ['M0','R1','R2'])
 assert all(m.calibrate([{'N':True,'unsafe':False,'r':1.,'q1':1.,'q2':1.}])[k]['tau']==1 for k in ['M0','R1','R2'])
 with pytest.raises(ValueError):m.calibrate(rows,population='AUDIT')
