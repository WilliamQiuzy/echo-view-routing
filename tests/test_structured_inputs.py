import importlib
import numpy as np
import pytest

def test_canonical_stride_two_uses_original_endpoint_marker_and_no_labels():
    m=importlib.import_module('echo_routing.temporal.structured_inputs')
    assert m.windows(5,2)['centre'].tolist()==[1,3]
    assert m.windows(1,2)['centre'].tolist()==[0]
    w=m.windows(19,2)
    for centre,indices,marker in zip(w['centre'],w['samples'],w['marker']):
        assert len(indices)==16 and indices[marker]==centre and indices.min()>=0 and indices.max()<19
    with pytest.raises(ValueError):m.windows(0,2)

def test_private_source_roles_cover_all_footprints_and_reject_cross_role():
    m=importlib.import_module('echo_routing.temporal.structured_inputs')
    q={'id':'one','kind':'native','refs':[('one',0,1.)],'labels':np.array([0])};mixed={'id':'bank','kind':'mixed','refs':[('one',0,1.)],'labels':np.array([0])}
    assert m.verify_roles({'fit':[q,mixed],'inner':[{**q,'id':'two','refs':[('two',0,1.)]}]})['cross_role_sources']==0
    with pytest.raises(ValueError):m.verify_roles({'fit':[q],'inner':[q]})
    with pytest.raises(ValueError):m.verify_roles({'fit':[q,{**mixed,'refs':[('outside',0,1.)]}]})

def test_overlap_oracle_is_only_position_accuracy_bound_and_axis_checks_fail_closed():
    m=importlib.import_module('echo_routing.temporal.structured_inputs');a=np.array([0,1,1,2]);b=np.array([1,1,0,2]);y=np.array([0,1,0,1]);r=m.error_overlap(a,b,y,np.ones(4,bool));assert r=={'positions':4,'both_correct':1,'left_only_correct':1,'right_only_correct':1,'both_wrong':1,'position_oracle_correct':3}
    with pytest.raises(ValueError):m.error_overlap(a,b[:-1],y,np.ones(4,bool))
    p=np.full((4,5),.2,np.float32);m.check_probabilities(p,4)
    for bad in [p[:3],p*2,np.full((4,5),np.nan)]:
        with pytest.raises(ValueError):m.check_probabilities(bad,4)
