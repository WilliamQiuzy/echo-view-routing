import importlib
import dataclasses
import numpy as np
import pytest
from echo_routing.temporal.proposal_actions import Action,propose


def api():return importlib.import_module('echo_routing.temporal.proposal_features')


def test_exact109_common69_and_source_evidence_use_only_changed_positions():
    m=api();base=np.array([0,1,0,2,0]);action=propose(base,base,np.zeros(5,int))[0]
    a=np.tile([.6,.1,.1,.1,.1],(5,1));j=np.tile([.2,.5,.1,.1,.1],(5,1))
    fa=m.candidate_features(a,j,base,action,'A');fj=m.candidate_features(a,j,base,action,'J')
    assert fa.shape==(109,) and fa.dtype==np.float64 and np.array_equal(fa[:69],fj[:69])
    assert np.array_equal(fa[29:69],fa[69:]) and not np.array_equal(fj[29:69],fj[69:])
    assert np.isclose(fa[26],np.log1p(2)) and np.isclose(fa[27],2/3) and fa[28]==2/5
    assert np.allclose(fa[65:68],[.5,.5,.5])
    changed_origin=dataclasses.replace(action,old=True,jepa=False)
    assert np.array_equal(fa,m.candidate_features(a,j,base,changed_origin,'A'))
    assert np.array_equal(fa,m.candidate_features(a,np.full((5,5),.2),base,action,'A'))


def test_signed_three_targets_use_full_matcher_and_neighbor_spurious_short_segments():
    m=api();base=np.zeros(6,int);action=Action(2,4,(1,1),jepa=True)
    truth=np.array([0,0,1,1,0,0])
    assert np.array_equal(m.targets(base,action,truth,.5),[5,-2,-1])
    assert np.array_equal(m.targets(base,action,base,.5),[-3,2,3])
    base=np.zeros(20,int);action=Action(10,11,(1,),jepa=True)
    # One newly wrong position creates two unmatched short units, not just the changed singleton.
    assert np.array_equal(m.targets(base,action,base,.5),[-1,1,2])


def test_shared_target_RMS_has_no_centering_zero_stays_zero_and_fit_normalizer_is109():
    m=api();y=np.array([[2,6,0],[-2,-2,0]],np.float64)
    scales=m.fit_target_scales(y)
    assert np.allclose(scales,[2,np.sqrt(20),1])
    assert np.array_equal(m.scale_targets(np.zeros((2,3)),scales),np.zeros((2,3),np.float32))
    assert np.array_equal(m.physical_outputs(np.zeros((2,3)),scales),np.zeros((2,3)))
    assert np.allclose(m.physical_outputs(m.scale_targets(y,scales),scales),y,atol=1e-6)
    x=np.zeros((2,109));x[1]=2
    norm=m.fit_normalizer(x)
    assert np.array_equal(norm['mean'],np.ones(109)) and np.array_equal(norm['std'],np.ones(109))
    assert m.normalize(x,norm).dtype==np.float32
    with pytest.raises(ValueError):m.fit_normalizer(np.zeros((2,98)))
