import numpy as np
import pytest


def test_selected_union_keeps_same_top_and_never_falls_back():
    from echo_routing.temporal.selected_risk_resume_supply import selected_indices
    value={'actions':[0,1], 'D':{'top_index':0,'N_admitted':False},'X':{'top_index':1,'N_admitted':True}}
    assert selected_indices(value)=={1:(False,True)}
    value['D']={'top_index':1,'N_admitted':True}
    assert selected_indices(value)=={1:(True,True)}
    value['X']={'top_index':None,'N_admitted':False};value['D']={'top_index':0,'N_admitted':False}
    assert selected_indices(value)=={}


def test_native_frozen_c6_reuses_global_candidate_order_without_inference():
    from echo_routing.temporal.selected_risk_resume_supply import native_C6
    all_scores=np.arange(24,dtype=np.float64).reshape(4,6)
    old={'groups':[np.array([0,1]),np.array([],dtype=np.int64),np.array([2,3])]}
    out=native_C6(all_scores,old,[2,0,1],[2,2,0])
    assert np.array_equal(out[0],all_scores[2:]) and out[2].shape==(0,6)
    with pytest.raises(ValueError):native_C6(all_scores,old,[0],[1])


def test_effect_lookup_is_closed_until_actual_git_selector_seal():
    from echo_routing.temporal.selected_risk_resume_supply import require_selector_seal
    with pytest.raises(ValueError):require_selector_seal({'passed':True},'a'*64)
    good={'passed':True,'complete':True,'cache_manifest_sha256':'a'*64,'effect_lookup_before_seal':False,'aux_optimizer_updates':48000,'formal_optimizer_updates':0,'actual_Git_object_proof':{'path':'proof','sha256':'b'*64}}
    assert require_selector_seal(good,'a'*64)
    good['effect_lookup_before_seal']=True
    with pytest.raises(ValueError):require_selector_seal(good,'a'*64)
