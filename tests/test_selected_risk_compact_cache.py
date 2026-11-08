import importlib
import numpy as np
import pytest

def module():
    try:m=importlib.import_module('echo_routing.temporal.selected_risk_compact_cache')
    except ModuleNotFoundError:m=None
    assert m is not None,'lossless action-digest and shared-feature supply cache absent'
    return m

def value(u=1.):return {'D_C6':np.array([[u,.2,.7,.1,.1,.2]]),'X_C6':np.array([[u,.1,.8,.1,.1,.1]]),'x109':np.zeros((1,109)),'context96':np.zeros((1,96))}

def test_all_scores_and_bitwise_features_reappear_from_immutable_reference(tmp_path):
    m=module();b=np.zeros(8,np.int64);j=b.copy();j[3]=1
    first=m.CompactSupplyCache(tmp_path/'first','first',[b],[b],[j]);first.run_one(0,lambda:value());ref=m.CompactSupplyCache(tmp_path/'first','first',[b],[b],[j],readonly=True)
    second=m.CompactSupplyCache(tmp_path/'second','second',[b],[b],[j],feature_reference=ref);second.run_one(0,lambda:value(2.));out=second.load(0)
    assert np.array_equal(out['x109'],ref.load(0)['x109']) and out['D_C6'][0,0]==2. and out['D']['N_admitted']
    assert second.file(0).stat().st_size<=m.payload_upper(1,reference=True) and first.file(0).stat().st_size<=m.payload_upper(1,reference=False)
    assert second.file(0).stat().st_size<first.file(0).stat().st_size
    with np.load(second.file(0),allow_pickle=False) as z:assert 'x109' not in z.files and 'context96' not in z.files and 'actions_SHA' in z.files
    with pytest.raises(ValueError):m.CompactSupplyCache(tmp_path/'bad','bad',[b],[b],[j],feature_reference=first)

def test_reference_mismatch_retains_pending_and_never_silently_replaces(tmp_path):
    m=module();b=np.zeros(8,np.int64);j=b.copy();j[3]=1;f=m.CompactSupplyCache(tmp_path/'f','f',[b],[b],[j]);f.run_one(0,lambda:value());ref=m.CompactSupplyCache(tmp_path/'f','f',[b],[b],[j],readonly=True);s=m.CompactSupplyCache(tmp_path/'s','s',[b],[b],[j],feature_reference=ref)
    def wrong():v=value();v['x109'][0,0]=-0.;return v
    with pytest.raises(ValueError):s.run_one(0,wrong)
    assert s.pending.exists() and s.cursor==0

def test_empty_axis_is_committed_without_any_head_replay(tmp_path):
    m=module();b=np.zeros(4,np.int64);f=m.CompactSupplyCache(tmp_path/'f','f',[b],[b],[b]);f.run_one(0,lambda:{'D_C6':np.empty((0,6)),'X_C6':np.empty((0,6)),'x109':np.empty((0,109)),'context96':np.empty((0,96))});assert f.load(0)['D']['top_index'] is None
