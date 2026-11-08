import pytest


def test_only_complete_known_CPU_final_cap_attempt_can_be_adopted():
    from echo_routing.temporal.selected_risk_resume_source_adoption import validate_original_attempt
    p={'stage':'source_FIT','stage_exit_status':1,'observed_worker_exit_code':1,'inner_model_observed_OS_exit_code':0,'guardian_termination_reason':'actual_final_cap_exceeded','guardian_error':None,'whole_child_accounting_complete':True,'actual_child_directly_waited':True,'unresolved_worker_uncertainty':False,'current_remote_caps_unknown':False,'runtime_authorization_sha256':'a'*64,'runtime_spec_sha256':'b'*64,'measured_CPU_core_seconds':4968.,'full_process_wall_seconds':3575.,'sampled_tracked_storage_upper_bytes':719200000,'finished_epoch':150.,'GPU_job':True,'CUDA_context_occupied':True}
    s={'stage':'source_FIT','owner':{'sha256':'a'*64},'arguments':{'stage_CPU_upper_seconds':4500.,'stage_wall_upper_seconds':4500.}}
    assert validate_original_attempt(p,s,'b'*64,{'deadline_epoch':200})
    for delta in [{'inner_model_observed_OS_exit_code':1},{'guardian_termination_reason':'guardian_exception'},{'measured_CPU_core_seconds':6001.},{'full_process_wall_seconds':4501.},{'unresolved_worker_uncertainty':True},{'finished_epoch':201.},{'sampled_tracked_storage_upper_bytes':1300000001}]:
        with pytest.raises(ValueError):validate_original_attempt({**p,**delta},s,'b'*64,{'deadline_epoch':200})


def test_geometry_requires_new_actual_adoption_and_no_neural_replay():
    from echo_routing.temporal.selected_risk_resume_contract import stage_graph,_successful
    g=stage_graph()
    assert len(g)==33 and 'source_FIT_adoption' in g['observe_geometry']['requires']
    assert g['source_FIT_adoption']['requires']==['source_FIT'] and g['source_FIT_adoption']['GPU_source_only'] is False
    p={'passed':True,'complete':True,'committed_payloads':6000,'original_source_stage_exit_status':1,'original_teacher_inner_OS_exit_code':0,'original_source_stage_cap_failure_preserved':True,'new_teacher_forwards':0,'new_optimizer_updates':0,'cache_writes':0}
    _successful('source_FIT_adoption',p)
    with pytest.raises(ValueError):_successful('source_FIT_adoption',{**p,'new_teacher_forwards':1})


def test_adoption_success_flag_cannot_replace_its_own_actual_parent_OS0():
    import json,hashlib
    from echo_routing.temporal.selected_risk_resume_estimate import _closed_storage_stage
    pin={'path':'docs/results/new/source_FIT_adoption.abc.json','sha256':'a'*64}
    payload={'passed':True,'complete':True,'committed_payloads':6000,'original_source_stage_exit_status':1,'original_teacher_inner_OS_exit_code':0,'original_source_stage_cap_failure_preserved':True,'new_teacher_forwards':0,'new_optimizer_updates':0,'cache_writes':0}
    raw=json.dumps({'stage':'source_FIT_adoption','output':pin,'stage_returned_success':True,'runtime_authorization_sha256':'b'*64}).encode();log={'path':'runs/new/adopt.stdout','sha256':hashlib.sha256(raw).hexdigest()}
    class C:
        owner_sha='b'*64
        def raw(self,p):assert p==log;return raw
    spec={'predecessors':{'source_FIT_adoption':{'payload_pin':pin}}};row={'stage':'source_FIT_adoption','model_pid':123,'stage_exit_status':1,'observed_worker_exit_code':1,'inner_model_observed_OS_exit_code':0,'whole_child_accounting_complete':True,'actual_child_directly_waited':True,'unresolved_worker_uncertainty':False,'runtime_authorization_sha256':'b'*64,'log_pins':[log]}
    with pytest.raises(ValueError):_closed_storage_stage(C(),spec,{'source_FIT_adoption':payload},[row],'source_FIT_adoption')
    row.update(stage_exit_status=0,observed_worker_exit_code=0)
    assert _closed_storage_stage(C(),spec,{'source_FIT_adoption':payload},[row],'source_FIT_adoption') is row
