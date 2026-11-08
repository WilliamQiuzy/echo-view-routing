import importlib
import numpy as np
import pytest

def mod():
    try:m=importlib.import_module('echo_routing.temporal.selected_risk_inputs')
    except ModuleNotFoundError:m=None
    assert m is not None,'A63 immutable native alignment/minimal data representation absent'
    return m

def test_native_subset_requires_original_axis_refs_and_truth_exact():
    m=mod();old=[{'id':'own_0','kind':'native','refs':[('own_0',0,1.)],'labels':np.array([2])},{'id':'mixed','kind':'multi','refs':[('own_0',0,1.)],'labels':np.array([2])}]
    q={k:v.copy() if isinstance(v,np.ndarray) else v for k,v in old[0].items()};assert m.original_native_indices(old,[q])==[0]
    q['refs']=[('own_0',0,.9)]
    with pytest.raises(ValueError):m.original_native_indices(old,[q])
    q['refs']=old[0]['refs'];q['labels']=np.array([3])
    with pytest.raises(ValueError):m.original_native_indices(old,[q])

def test_redundant_position_groups_are_only_omitted_serialized_field():
    m=mod();x=np.ones((2,47));d={'position_features':x,'position_groups':[x[:1],x[1:]],'normalizers':{'position':{'mean':x.mean(0)}}}
    v=m.minimal_aux_data(d);assert set(v)=={'position_features','normalizers'} and v['position_features'] is x
    assert 'position_groups' in d and np.array_equal(v['normalizers']['position']['mean'],np.ones(47))
