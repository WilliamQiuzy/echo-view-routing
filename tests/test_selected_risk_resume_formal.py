import numpy as np
import pytest


def test_formal_constructor_is_closed_before_full_coverage(monkeypatch):
    from echo_routing.temporal.selected_risk_resume_formal import new_head
    class Forbidden:
        def __init__(self,*a,**k):raise AssertionError('production C0/C1 instantiated during preparation')
    monkeypatch.setattr('echo_routing.temporal.event_risk_model.EventRisk',Forbidden)
    for cov in [None,{'passed':True},{'passed':False}]:
        with pytest.raises(ValueError):new_head(2026110461,'C0',cov)


def test_explicit_matched_draws_are_global_and_keep_kind_blocks():
    from echo_routing.temporal.selected_risk_resume_formal import validate_matched_draws
    a=np.tile(np.r_[np.zeros(32,dtype=np.int64),np.ones(32,dtype=np.int64)],(6000,1));p={'native':np.array([0]),'constructed':np.array([1,2])}
    b=a.copy();b[:,48:]=2
    assert validate_matched_draws({'C0':a,'C1':b},p,3)
    bad=b.copy();bad[:,0]=1
    with pytest.raises(ValueError):validate_matched_draws({'C0':a,'C1':bad},p,3)
    bad=b.copy();bad[:,33]=2
    with pytest.raises(ValueError):validate_matched_draws({'C0':a,'C1':bad},p,3)
    with pytest.raises(ValueError):validate_matched_draws({'C0':a.astype(float),'C1':b},p,3)


def test_formal_target_is_original_event_not_acceptance_safe_or_net():
    from echo_routing.temporal.selected_risk_resume_formal import unsafe_target
    assert unsafe_target({'H':1,'R':20,'dFP':-2})==1
    assert unsafe_target({'H':0,'R':20,'dFP':1})==1
    assert unsafe_target({'H':0,'R':0,'dFP':0,'short16_lost':9})==0
