import importlib
import inspect
import numpy as np
import pytest


def api():return importlib.import_module('echo_routing.temporal.proposal_actions')


def test_jepa_class_absent_from_raw_creates_new_action_and_old_policy_stays_base():
    m=api();base=np.zeros(6,int);raw=base.copy();jepa=np.array([0,0,1,1,0,0])
    actions=m.propose(base,raw,jepa)
    assert len(actions)==1 and actions[0].key==(2,4,(1,1)) and actions[0].jepa and not actions[0].old
    eligible,rejected=m.partition(base,actions);assert len(eligible)==1 and not rejected
    for policy in ['E','R']:
        out=m.decode(base,eligible,np.array([[1.,0.,0.]]),policy)
        assert np.array_equal(out['prediction'],jepa) and out['accepted']==actions[0]
    assert np.array_equal(m.decode(base,eligible,np.array([[1.,0.,0.]]),'O')['prediction'],base)


def test_canonical_trim_keeps_unchanged_interior_and_deduplicates_provenance():
    m=api();base=np.array([0,1,0,1,0]);raw=np.array([0,2,2,1,0])
    actions=m.propose(base,raw,raw)
    assert len(actions)==1 and actions[0].old and actions[0].jepa
    assert actions[0].key==(1,3,(2,2))
    base=np.array([0,1,0,2,0]);actions=m.propose(base,base,np.zeros(5,int))
    assert len(actions)==1 and actions[0].key==(1,4,(0,0,0))
    assert np.count_nonzero(m.apply(base,actions[0])!=base)==2
    assert not m.propose(base,base,base)


def test_no_length_filter_and_global_majority_can_still_allow_frame_and_cut_harm():
    m=api()
    for n in [1,257]:
        base=np.zeros(n,int);actions=m.propose(base,base,np.ones(n,int))
        assert len(actions)==1 and actions[0].key==(0,n,(1,)*n)
        eligible,rejected=m.partition(base,actions);assert not eligible and rejected==actions
    base=np.array([0,0,0,1,2]);jepa=np.array([0,0,3,1,2])
    actions=m.propose(base,base,jepa);eligible,_=m.partition(base,actions)
    pred=m.decode(base,eligible,np.array([[1.,0.,0.]]),'E')['prediction']
    assert np.bincount(pred,minlength=5).argmax()==np.bincount(base,minlength=5).argmax()
    assert np.count_nonzero(pred!=base)==1 and np.count_nonzero(pred[1:]!=pred[:-1])>np.count_nonzero(base[1:]!=base[:-1])
    assert set(inspect.signature(m.decode).parameters)=={'base','actions','outputs','policy'}


def test_one_action_canonical_ties_strict_utility_and_zero_signed_risk_thresholds():
    m=api();base=np.zeros(7,int);jepa=np.array([0,1,0,0,2,0,0]);actions,_=m.partition(base,m.propose(base,base,jepa))
    out=m.decode(base,actions,np.array([[1.,-1.,0.],[1.,0.,-1.]]),'R')
    assert out['accepted']==actions[0] and np.count_nonzero(out['prediction']!=base)==1
    out=m.decode(base,actions,np.array([[2.,0.,.001],[1.,0.,0.]]),'R')
    assert out['accepted']==actions[1]
    assert m.decode(base,actions,np.zeros((2,3)),'E')['accepted'] is None
    with pytest.raises(ValueError):m.decode(base,actions,np.array([[1.,0.,np.nan],[1.,0.,0.]]),'R')
