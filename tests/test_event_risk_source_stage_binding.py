import copy
import pytest


def fixture():
    roles=['CAL_mixed','AUDIT_mixed','a62_confirmation_selection','a62_confirmation_calibration'];counts=[150,150,120,120]
    records=[{'source':s,'role':r,'committed_sequences':n,'committed_logical_neural_units':n,'new_neural_units':n,'resumed_committed_prefix_sequences':0,'journal_sha256':str(k),'complete':True} for s in ['a08','jepa'] for k,(r,n) in enumerate(zip(roles,counts))]
    producer={'complete':True,'source_production_stage':'sources','source_freeze_git':'a'*40,'new_event_updates_at_production':0,'records':records};identity={'git_commit':'a'*40}
    row={'stage':'producer','receipt_id':'producer:1','measurement_host':'toy_host','model_pid':12,'supervisor_pid':11,'process_start_ticks':42,'observed_worker_exit_code':0,'child_exit_code_directly_waited':True,'full_process_wall_seconds':3.,'measured_CPU_core_seconds':2.,'GPU_job':True,'CUDA_context_occupied':True,'termination_reason':None}
    return producer,identity,[row],{'producer:1':{'path':'docs/results/2026-11-04_anchor_context_risk/process_producer_1.json','sha256':'rawbytes'}}


def test_exact_logical_to_actual_binding_counts_gpu_once():
    from echo_routing.temporal.event_risk_source_stage import build,validate
    from scripts.package_event_risk import source_costs
    p,i,rows,pins=fixture();b=build(p,'manifestbytes',i,'identitybytes',rows,pins)
    assert validate(b,p,'manifestbytes',i,'identitybytes',rows,pins)
    costs=source_costs(p,rows,b,'manifestbytes',i,'identitybytes',pins)
    assert costs['source_production_stage']=='sources' and costs['actual_supervisor_stage']=='producer'
    assert costs['actual_occupied_GPU_wall_seconds']==3. and costs['actual_CPU_seconds']==2.
    assert costs['committed_teacher_payloads']==1080
    with pytest.raises(ValueError):source_costs(p,rows)
    changed=copy.deepcopy(rows);changed[0]['process_start_ticks']+=1
    with pytest.raises(ValueError):validate(b,p,'manifestbytes',i,'identitybytes',changed,pins)


def test_binding_rejects_changed_units_journal_and_double_stage_accounting():
    from echo_routing.temporal.event_risk_source_stage import build,validate
    p,i,rows,pins=fixture();b=build(p,'manifestbytes',i,'identitybytes',rows,pins)
    for key in ['journal_sha256','committed_logical_neural_units']:
        changed=copy.deepcopy(p);changed['records'][0][key]='changed'
        with pytest.raises(ValueError):validate(b,changed,'manifestbytes',i,'identitybytes',rows,pins)
    extra={**rows[0],'stage':'sources','receipt_id':'sources:1'}
    with pytest.raises(ValueError):build(p,'manifestbytes',i,'identitybytes',rows+[extra],pins)
