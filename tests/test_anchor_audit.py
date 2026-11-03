import copy
import numpy as np
import pytest
from echo_routing.temporal.anchor_audit import validate_trace, exact_models
from echo_routing.temporal.anchor_training import lr_factor


def test_exact_matrix_and_prefrozen_draw_trace_reject_missing_extra_or_drift():
    rows=[{'seed':s,'arm':a} for s in [2026110359,2026110360] for a in ['A','J']]
    assert exact_models(rows)
    for bad in [rows[:-1],rows+[rows[0]],rows[:3]+[{'seed':1,'arm':'J'}]]:
        with pytest.raises(ValueError): exact_models(bad)
    data={'kind_indices':{'native':np.arange(2),'constructed':np.arange(2,4)},
          'candidate_rows':[{'feature_positions':i+1} for i in range(4)]}
    plan={k:np.array([[0,1],[1,0]]) for k in ['native','constructed']}
    trace=[]
    for u in range(2):
        trace.append({'u':u,'identity':{'x':'a'},'candidate_indices':np.r_[plan['native'][u],plan['constructed'][u]+2].tolist(),
         'physical_feature_positions':10,'physical_candidates':4,'lr':.001*lr_factor(u),'loss':1.,'gradient_norm':1.,
         'finite_gradients':True,'nonzero_gradient_tensors':{'a':u==1}})
    assert validate_trace(trace,data,plan,{'x':'a'},updates=2)['a']
    for key,value in [('u',2),('lr',.7),('candidate_indices',[0,0,2,3]),('physical_feature_positions',9),('finite_gradients',False)]:
        bad=copy.deepcopy(trace);bad[1][key]=value
        with pytest.raises(ValueError):validate_trace(bad,data,plan,{'x':'a'},updates=2)
