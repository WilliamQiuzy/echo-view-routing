import copy
import pytest
from echo_routing.temporal import safety_rle_budget as m


def test_observable_geometry_keeps_original_bounds_and_complete_roles():
    roles={r:{'sequences':1,'positions':10,'max_length':10,'sha256':r} for r in m.OUTER_ROLES}
    g={r:dict(N=1,T=10,C=2,C2=4,TC=20,all_C=3,all_TC=30,K=4,CR=6,max_C=2,max_T=10,native_N=0,native_T=0,native_T2=0,input_sha256=r) for r in roles}
    value={'passed':True,'new_optimizer_updates':0,'new_neural_forwards_in_this_cost_stage':0,'A61_outer_head_outcomes_used':False,'outer_GT_values_used':False,'observable_geometry_all10roles':g}
    assert m.observed(roles,value)['selection']['CR']==6
    for key,val in [('CR',41),('C2',5),('K',21),('input_sha256','wrong')]:
        bad=copy.deepcopy(value);bad['observable_geometry_all10roles']['selection'][key]=val
        with pytest.raises(ValueError):m.observed(roles,bad)
    bad=copy.deepcopy(value);bad['observable_geometry_all10roles'].pop('a61_confirmation_selection')
    with pytest.raises(ValueError):m.observed(roles,bad)


def test_rle_factorization_retains_pair_run_and_volume_costs():
    low=dict(C=100,all_C=150,C2=10000,TC=70000,all_TC=105000,CR=400,descriptor_max_seconds=1.,oracle_max_seconds=2.)
    high={**low,'CR':140000,'descriptor_max_seconds':10.,'oracle_max_seconds':20.}
    long=dict(T=701,K=175,descriptor_max_seconds=.01,oracle_max_seconds=.02)
    timing={'empty_descriptor':{'seconds':.001},'long_701':long,'dense_16':{'oracle_max_seconds':.002}}
    rle={'low_RLE_dense':low,'retained_high_RLE_dense':high,'all_unequal_pair_kernel':{'C2':10000,'seconds':.1}}
    d,o=m.densities(timing,rle)
    v=dict(N=2,T=1402,C=10,all_C=20,C2=100,TC=7000,all_TC=14000,K=350,CR=40)
    base=d(v)
    for key in ['N','T','C','all_C','C2','TC','all_TC','K','CR']:
        more={**v,key:v[key]*2};assert d(more)>base
    assert o({**v,'CR':80})>o(v)
    bad=copy.deepcopy(rle);bad['low_RLE_dense']['descriptor_max_seconds']=float('nan')
    with pytest.raises(ValueError):m.densities(timing,bad)
