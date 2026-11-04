import importlib
import numpy as np
import torch

def test_q_heads_shared_source_inputs_and_empty_domains_skip_forward():
 m=importlib.import_module('echo_routing.temporal.event_risk_runtime')
 from echo_routing.temporal.event_risk_model import EventRisk
 b=np.zeros(7,int);pa=np.tile([.9,.025,.025,.025,.025],(7,1));pj=pa.copy();pj[2]=[.025,.9,.025,.025,.025];norm={'candidate':{'mean':np.zeros(109),'std':np.ones(109)},'context':{'mean':np.zeros(96),'std':np.ones(96)}};heads={k:EventRisk(k,2026110461) for k in ['R1','R2']};old=np.array([[.3,.1,.9,0.,.01,.1]])
 v=m.infer_events(heads,norm,pa,pj,b,old,{'source':'fixture'});assert v['q'].shape==(1,2) and np.array_equal(v['q'],[[.5,.5]]) and v['new_event_MLP_forwards']==2 and v['old_fixed_MLP_forwards']==0
 for h in heads.values():h.forward=lambda x:(_ for _ in ()).throw(AssertionError('empty NN'))
 v=m.infer_events(heads,norm,pa,pa,b,np.empty((0,6)),{'source':'fixture'});assert v['q'].shape==(0,2) and v['new_event_MLP_forwards']==0
