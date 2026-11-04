import importlib
import numpy as np
import pytest
import torch


def test_actual_trace_exposure_cannot_substitute_candidate_for_query_or_skip_pairs():
    m=importlib.import_module('echo_routing.temporal.safety_audit')
    from echo_routing.temporal.safety_training import Utility,optimizer,draw_plan,step
    from echo_routing.temporal.safety_data import prepare_training,tensor_data,batch
    qs=[{'kind':'native','labels':np.zeros(20,int)},{'kind':'multi','labels':np.r_[np.zeros(10,int),np.ones(10,int)]}]
    bs=[np.zeros(20,int),np.zeros(20,int)];a=[np.tile([.7,.1,.1,.05,.05],(20,1)) for _ in qs];j=[p.copy() for p in a];j[0][4:6]=[.1,.7,.1,.05,.05];j[1][10:]=[.1,.7,.1,.05,.05]
    d=prepare_training(qs,a,j,bs);p=draw_plan(1,1,2026110461,'utility');values,exposure=batch(d,tensor_data(d),p,0,'utility','U-L')
    model=Utility('U-L',2026110461);row={**step(model,optimizer(model),values,0,'utility'),**exposure}
    assert m.validate_trace([row],d,p,'utility','U-L',1)['updates']==1
    wrong={**row,'candidate_indices':row['candidate_indices'][:-1]}
    with pytest.raises(ValueError):m.validate_trace([wrong],d,p,'utility','U-L',1)
    wrong={**row,'physical_unequal_pairs':0}
    with pytest.raises(ValueError):m.validate_trace([wrong],d,p,'utility','U-L',1)
