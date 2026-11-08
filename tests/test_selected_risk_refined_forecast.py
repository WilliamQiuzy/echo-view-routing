import importlib,copy
import pytest

def module():
    try:m=importlib.import_module('echo_routing.temporal.selected_risk_refined_forecast')
    except ModuleNotFoundError:m=None
    assert m is not None,'prospective lossless full-pass refined estimator absent'
    return m

def fixture():
    base={'scope':{'aux_updates':48000,'formal_updates':24000,'outer_rows':238},'phases':{'aux48000_updates':1000.,'formal24000_updates':200.,'resource_OS_independent_audit_publication':3600.,'backend_TC_and_CR':500.},'CPU_seconds':5300.,'remaining_wall_seconds':12500.,'storage_components':{'current_namespace_plus_admin_upper':400000000,'features_joint_cache_and_event_copies':90000000,'retained_fullstates_atomic_upper':12500000,'72000_trace_rows_empirical_index_aware_forecast':120000000,'12_draw_arrays_plus_atomic_temp':39936000,'selected_digest_records_4096_bytes_each':53000000,'teacher_payload_journal_atomic_upper':65000000,'closing_reserve':128000000},'projected_geometry':{k:{'N':1500,'C':4000,'K':10000,'T':200000} for k in ['F0_mixed','F1_mixed']},'GPU_seconds':4795.}
    profile={'geometry':{'native':{'N':2736,'C':265,'K':2285,'T':171302}}};shape={'training':{'utility_update_seconds':.002,'risk_update_seconds':.003},'trace_ratio_forecast':.5,'trace_profiles':[{'max_line_bytes':1000,'archive_and_exact_roundtrip_seconds':1.}]};sessions=['source'+str(i) for i in range(402)]
    return base,profile,shape,sessions

def test_scientific_scope_and_nontraining_phases_remain_while_closing_is_contained():
    m=module();b,p,s,ss=fixture();original=copy.deepcopy(b);out=m.refine_complete(b,p,s,ss)
    assert b==original and out['scope']==b['scope'] and out['phases']['backend_TC_and_CR']==500
    assert out['phases']['resource_OS_independent_audit_publication']==3600 and out['phases']['formal24000_updates']==200
    assert out['phases']['aux48000_updates']==pytest.approx(240.)
    assert out['remaining_wall_seconds']==pytest.approx(out['CPU_seconds']-3600+7200)
    assert out['storage_components']['closing_reserve']==128000000 and out['storage_components']['current_namespace_plus_admin_upper']==400000000
    assert out['storage_components']['selected_typed_records_atomic_upper']<53000000

def test_ratio_floor_and_original_full_auxiliary_counters_cannot_be_reduced():
    m=module();b,p,s,ss=fixture();s['trace_ratio_forecast']=.49
    with pytest.raises(ValueError):m.refine_complete(b,p,s,ss)
