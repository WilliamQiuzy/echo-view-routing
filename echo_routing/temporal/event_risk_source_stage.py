"""Exact accounting-only sources→producer binding; never rename raw evidence."""
import copy


def build(producer,manifest_sha,identity,identity_sha,receipts,attempt_pins):
    if producer.get('complete') is not True or producer.get('source_production_stage')!='sources' or producer.get('new_event_updates_at_production')!=0 or producer.get('source_freeze_git')!=identity['git_commit']:raise ValueError('exact completed original logical source manifest/Git required')
    records=producer['records'];roles={'CAL_mixed':150,'AUDIT_mixed':150,'a62_confirmation_selection':120,'a62_confirmation_calibration':120}
    if len(records)!=8 or {(r['source'],r['role']) for r in records}!={(s,r) for s in ['a08','jepa'] for r in roles} or any(not r['complete'] or r['committed_sequences']!=roles[r['role']] for r in records):raise ValueError('all1080 original source payloads must be complete')
    if any(r['stage']=='sources' for r in receipts):raise ValueError('logical and physical source attempts cannot both be counted')
    attempts=[r for r in receipts if r['stage']=='producer'];ids=[r['receipt_id'] for r in attempts]
    if not attempts or len(set(ids))!=len(ids) or set(attempt_pins)!=set(ids) or any(not r['GPU_job'] for r in attempts):raise ValueError('exact unique actual producer attempts and raw byte pins required')
    fields=['receipt_id','measurement_host','model_pid','supervisor_pid','process_start_ticks','full_process_wall_seconds','measured_CPU_core_seconds','GPU_job','CUDA_context_occupied','observed_worker_exit_code','child_exit_code_directly_waited']
    if any(any(k not in r for k in fields) for r in attempts):raise ValueError('preserve actual parent OS/PID/cost provenance including unknowns')
    return {'schema':'A62_source_stage_binding_v1','logical_source_production_stage':'sources','actual_supervisor_stage':'producer','producer_manifest_sha256':manifest_sha,'producer_identity_sha256':identity_sha,'producer_source_git':identity['git_commit'],'producer_records':copy.deepcopy(records),'actual_attempts':copy.deepcopy(attempts),'actual_attempt_file_pins':copy.deepcopy(attempt_pins),'raw_manifest_and_ledger_not_renamed_or_overwritten':True,'accounting_alias_only_no_new_forward':True}


def validate(binding,producer,manifest_sha,identity,identity_sha,receipts,attempt_pins):
    if binding!=build(producer,manifest_sha,identity,identity_sha,receipts,attempt_pins):raise ValueError('actual source stage binding/provenance changed')
    return True
