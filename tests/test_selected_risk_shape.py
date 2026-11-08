import pytest
from scripts.probe_selected_risk_shape import shapes

def manifest():return {'roles':{'F0_mixed':{'max_length':370},'F1_mixed':{'max_length':383}}}
def profile():return {'old_FIT_maxima':{'native':{'C':9,'K':86,'T':1081}},'max_training_C':16}

def test_kind_specific_shape_retains_whole_axis_T_and_complete_G_maxima():
    assert shapes(profile(),manifest())=={'native_C':9,'mixed_C':16,'native_K':86,'mixed_K':383,'native_axis_T_retained':1081}

def test_shape_refuses_invalid_or_truncated_measured_counts():
    p=profile();p['old_FIT_maxima']['native']['K']=-1
    with pytest.raises(ValueError):shapes(p,manifest())
    p=profile();p['max_training_C']=1.5
    with pytest.raises(ValueError):shapes(p,manifest())

def test_timed_batch_uses_frozen_tensor_gather_with_complete_kind_draws():
    import scripts.probe_selected_risk_shape as m
    from echo_routing.temporal.safety_data import batch
    f=getattr(m,'synthetic_training_data',None);assert callable(f),'cached tensor preparation must be separate from timed complete batch gather'
    data,tensors,plans=f({'native_C':2,'mixed_C':3,'native_K':1,'mixed_K':4,'native_axis_T_retained':10})
    u,e=batch(data,tensors,plans['utility'],0,'utility','S-L');assert len(u['lengths'])==64 and u['lengths'][:32]==[2]*32 and u['lengths'][32:]==[3]*32 and len(u['x'])==160
    r,e=batch(data,tensors,plans['risk'],0,'risk','risk');assert r['lengths']==[1]*32+[4]*32 and len(r['positions'])==160
    assert not plans['utility']['fallback'] and not plans['risk']['fallback']
