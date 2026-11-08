import importlib
import numpy as np
import pytest
from echo_routing.temporal.proposal_actions import Action

def mod():
 try:m=importlib.import_module('echo_routing.temporal.selected_risk_policy')
 except ModuleNotFoundError:m=None
 assert m is not None,'A63 same-top/production lifecycle guard absent'
 return m

def test_strict_q_same_N_top_no_fallback_and_empty():
 m=mod();base=np.zeros(8,dtype=np.int64);actions=[Action(2,3,(1,),False,True),Action(4,5,(2,),False,True)];c6=np.array([[2.,1,2,0,.1,.2],[1.,0,2,0,.1,.1]])
 out=m.decode(base,actions,c6,np.array([[.5,.2],[.1,.1]]),{'C0':.5,'C1':.3})
 assert out['C0']['accepted_index'] is None and out['C1']['accepted_index']==0
 c6[0,1]=3;out=m.decode(base,actions,c6,np.zeros((2,2)),{'C0':1.,'C1':1.})
 assert out['C0']['accepted_index'] is None and out['C1']['accepted_index'] is None
 out=m.decode(base,[],np.empty((0,6)),np.empty((0,2)),{'C0':0.,'C1':0.})
 assert out['C0']['top_index'] is None

def test_calibration_strict_first_unsafe_Ntop_not_all_candidates():
 m=mod();rows=[{'N_admitted':False,'q':None,'H':999,'dFP':1},{'N_admitted':True,'q':.2,'H':1,'dFP':0},{'N_admitted':True,'q':.2,'H':0,'dFP':0},{'N_admitted':True,'q':.1,'H':0,'dFP':0}]
 a=m.calibrate(rows);assert a['tau']==.2 and a['accepted']==1 and a['unsafe_tied_at_threshold']==1 and a['total_axes']==4
 assert m.calibrate([{'N_admitted':False,'q':None,'H':0,'dFP':0}])['tau']==0
 assert m.calibrate([{'N_admitted':True,'q':0.,'H':0,'dFP':0}])['tau']==1

def test_confirm_requires_exact_both_seed_coverage_and_aux_audit():
 m=mod();assert m.production_phase('FIT',None,None)=={'roles':['F0_mixed','F1_mixed'],'payloads':6000,'aux_updates_at_production':0,'formal_updates_at_production':0}
 with pytest.raises(ValueError):m.production_phase('CONFIRM',{'passed':False},None)
 cov={'passed':True,'seeds':{'2026110461':{'passed':True},'2026110462':{'passed':True}}};audit={'passed':True,'aux_optimizer_updates':48000,'physical_models':8,'fullstate_count':32}
 assert m.production_phase('CONFIRM',cov,audit)['payloads']==480
 cov['seeds']['2026110462']['passed']=False
 with pytest.raises(ValueError):m.production_phase('CONFIRM',cov,audit)
