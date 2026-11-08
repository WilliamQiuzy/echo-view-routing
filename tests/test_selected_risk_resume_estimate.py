import pytest


def test_completed_work_is_subtracted_only_after_verified_whole_stage():
    from echo_routing.temporal.selected_risk_resume_estimate import remove_complete
    work={'GPU_seconds':100.,'CPU_seconds':5000.,'remaining_wall_seconds':8600.,'source_FIT_GPU_forecast':50.,'source_CONFIRM_GPU_forecast':10.,'phases':{'teacher_CPU_including_entry_load_IO':100.,'aux48000_updates':400.,'formal24000_updates':60.,'resource_OS_independent_audit_publication':3600.}}
    out=remove_complete(work,{'source_FIT':{'complete':True,'phase':'FIT','committed_payloads':6000,'aux_updates_at_production':0,'formal_updates_at_production':0}})
    assert out['GPU_seconds']==50 and out['CPU_seconds']==4950 and work['GPU_seconds']==100
    with pytest.raises(ValueError):remove_complete(work,{'source_FIT':{'complete':True,'committed_payloads':5999}})
    with pytest.raises(ValueError):remove_complete(work,{'audit_aux':{'passed':True}})


def test_completed_aux_does_not_erase_closing_floor_or_other_io():
    from echo_routing.temporal.selected_risk_resume_estimate import remove_complete
    work={'GPU_seconds':100.,'CPU_seconds':5000.,'remaining_wall_seconds':8600.,'phases':{'aux48000_updates':400.,'fullstate768_write_read':15.,'resource_OS_independent_audit_publication':3600.}}
    out=remove_complete(work,{'audit_aux':{'passed':True,'aux_optimizer_updates':48000,'physical_models':8,'fullstate_count':32}})
    assert out['CPU_seconds']==4600 and out['remaining_wall_seconds']==8200
    assert out['phases']['fullstate768_write_read']==15
    assert out['phases']['resource_OS_independent_audit_publication']==3600


def test_fit_risk_diagnostic_complete_cost_and_codec_domain():
    from echo_routing.temporal.selected_risk_resume_estimate import fit_diagnostic_cost
    from echo_routing.temporal.selected_risk_resume_storage import storage_bound
    import numpy as np
    # C3, two nonempty axes, three total axes; both seeds, all four q columns.
    v=fit_diagnostic_cost(3,2,3)
    assert v['storage_upper_bytes']==6014188
    assert v['CPU_seconds']==120 and v['remaining_wall_seconds']==120 and v['GPU_seconds']==0
    assert v['storage_components']=={'all_score_artifacts_and_receipts':4800,'global_exposure_membership_offsets_and_headers':8364,'one_atomic_extra_copy':2000000,'pending_room':1024,'public_FIT_aggregate_server_and_local_copy':4000000}
    assert fit_diagnostic_cost(0,0,0)['storage_upper_bytes']==6009232
    assert fit_diagnostic_cost(0,0,0)['storage_components']['one_atomic_extra_copy']==2000000
    for c in [0,1,16,86,383,15823]:
        b=storage_bound({'identity':'a'*64,'q':np.zeros((c,4),np.float64)})
        assert b['artifact_bytes']<=640+32*c
    for args in [(True,1,1),(2,3,3),(-1,0,0),(2,1,0)]:
        with pytest.raises(ValueError):fit_diagnostic_cost(*args)


def test_new_teacher_fp32_layout_matches_qualified_probability_cache(tmp_path):
    from echo_routing.temporal.selected_risk_resume_estimate import teacher_future_fp32_bound
    from echo_routing.temporal.anchor_cache import ProbabilityCache
    import numpy as np
    lengths=[1,32,383,1081];roles=[{'positions':t,'sequences':1,'max_length':t} for t in lengths]
    assert teacher_future_fp32_bound(roles)==1184464
    total=0
    for producer in ['a08','jepa']:
        for t in lengths:
            cache=ProbabilityCache(tmp_path/(producer+str(t)),'a'*64,[t]);pending_sizes=[]
            def predict(t=t):
                pending_sizes.append(cache.pending.stat().st_size)
                return np.full((t,5),.2,np.float64)
            cache.run_one(0,predict)
            with np.load(cache.file(0),allow_pickle=False) as z:
                assert z.files==['prob'] and z['prob'].dtype==np.float32 and z['prob'].nbytes==20*t
            files=[p for p in cache.path.iterdir() if p.is_file()]
            assert sum(p.stat().st_size for p in files)-20*t+max(pending_sizes)<=4096
            total+=sum(p.stat().st_size for p in files)
    assert total+2*20*max(lengths)<=teacher_future_fp32_bound(roles)


def test_materialized_aux_obligations_preserve_remaining_joint_targets_and_one_draw_atomic():
    from echo_routing.temporal.selected_risk_resume_estimate import remaining_storage_components
    import copy
    work={'storage_upper_bytes':1000+300+39936000,'storage_components':{'current_namespace_plus_admin_upper':1000,'minimal_aux_plus_compact_shared_cache_phase_peak':300,'12_draw_arrays_plus_atomic_temp':39936000,'closing_reserve':128000000},'feature_layout_components':{'minimal_aux_datasets':100,'complete_two_seed_compact_cache':180,'formal_targets_only_no_persistent205':20}}
    # The scalar total below deliberately omits unrelated components; the helper
    # changes only named future obligations and never rebuilds current/spend.
    original=copy.deepcopy(work)
    assert remaining_storage_components(work,set())==work and work==original
    after=remaining_storage_components(work,{'prepare_aux'})
    assert after['storage_components']['minimal_aux_plus_compact_shared_cache_phase_peak']==200
    assert after['storage_components']['12_draw_arrays_plus_atomic_temp']==13*6000*64*8
    assert after['storage_components']['current_namespace_plus_admin_upper']==1000
    assert after['storage_components']['closing_reserve']==128000000
    assert after['storage_upper_bytes']==work['storage_upper_bytes']-100
    sealed=remaining_storage_components(work,{'prepare_aux','seal_selectors','prepare_formal'})
    assert sealed['storage_components']['minimal_aux_plus_compact_shared_cache_phase_peak']==0
    assert sealed['storage_components']['12_draw_arrays_plus_atomic_temp']==6000*64*8
    assert work==original
    import pytest
    with pytest.raises(ValueError):remaining_storage_components(work,{'prepare_formal'})


def test_storage_credit_needs_closed_actual_stage_not_a_passed_flag(monkeypatch):
    import echo_routing.temporal.selected_risk_resume_estimate as m
    class Context:
        def read(self,pin):return {'arguments':{},'predecessors':{'prepare_aux':{'payload_pin':{'path':'fake','sha256':'a'*64}}}}
        _bound_runtime_spec_pin={'path':'spec','sha256':'b'*64}
    import pytest
    with pytest.raises(ValueError):m.verify_materialized_storage(Context(),None,{'prepare_aux':{'passed':True,'folds':[{'held_fold':0},{'held_fold':1}]}},[])


def test_materialized_stage_requires_exact_successful_pinned_child_output():
    import json,hashlib,pytest
    from echo_routing.temporal.selected_risk_resume_estimate import _closed_storage_stage
    pin={'path':'docs/results/new/aux_training_inputs.json','sha256':'a'*64}
    log=json.dumps({'stage':'prepare_aux','output':pin,'stage_returned_success':True,'runtime_authorization_sha256':'b'*64}).encode()
    logpin={'path':'runs/new/stage.stdout','sha256':hashlib.sha256(log).hexdigest()}
    class C:
        owner_sha='b'*64
        def raw(self,p):
            assert p==logpin and hashlib.sha256(log).hexdigest()==p['sha256']
            return log
    spec={'predecessors':{'prepare_aux':{'payload_pin':pin}}};verified={'prepare_aux':{'passed':True,'folds':[{'held_fold':0},{'held_fold':1}]}}
    row={'stage':'prepare_aux','model_pid':123,'stage_exit_status':0,'observed_worker_exit_code':0,'inner_model_observed_OS_exit_code':0,'whole_child_accounting_complete':True,'actual_child_directly_waited':True,'runtime_authorization_sha256':'b'*64,'log_pins':[logpin]}
    assert _closed_storage_stage(C(),spec,verified,[row],'prepare_aux') is row
    with pytest.raises(ValueError):_closed_storage_stage(C(),spec,verified,[{**row,'inner_model_observed_OS_exit_code':7}],'prepare_aux')
    spec['predecessors']['prepare_aux']['payload_pin']={**pin,'sha256':'c'*64}
    with pytest.raises(ValueError):_closed_storage_stage(C(),spec,verified,[row],'prepare_aux')


def test_formal_trace_projection_verifies_all_qualified_source_bytes():
    from pathlib import Path
    import hashlib,pytest
    from echo_routing.temporal.selected_risk_resume_estimate import formal_trace_projection
    root=Path(__file__).resolve().parents[1]
    class C:
        def __init__(self):self.seen=[]
        def raw(self,pin):
            self.seen.append(pin)
            raw=(root/pin['path']).read_bytes()
            assert hashlib.sha256(raw).hexdigest()==pin['sha256']
            return raw
    geometry={'roles':{str(i):{'geometry':{'C':v}} for i,v in zip(['F0_native','F1_native','F0_mixed','F1_mixed'],[130,135,7000,8558])}}
    ctx=C();v=formal_trace_projection(ctx,geometry)
    assert v['candidate_upper']==15823 and v['per_head_raw_bytes']==9276000 and len(ctx.seen)==3
    class Tampered(C):
        def raw(self,pin):raise ValueError('source bytes changed')
    with pytest.raises(ValueError):formal_trace_projection(Tampered(),geometry)
