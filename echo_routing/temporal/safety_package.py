"""Separate committed cache geometry, invocation suffix work and actual attempts."""
import math


def source_costs(records,geometry,ledger,production_stage='cache_outer'):
    keys=[(r['source'],r['role']) for r in records]
    if len(keys)!=len(set(keys)) or set(keys)!=set(geometry):raise ValueError('exact complete source-role geometry coverage required')
    units={'a08':0,'jepa':0};calls={'a08':0,'jepa':0};last={'a08':0,'jepa':0}
    for r in records:
        source=r['source'];counts=geometry[(source,r['role'])]
        if source not in units or not r['complete'] or r['committed_sequences']!=len(counts) or any(type(n)!=int or n<=0 for n in counts):raise ValueError('complete committed source sequences/positive exact neural geometry required')
        total=sum(counts);new=r['new_neural_units']
        if type(new)!=int or not 0<=new<=total:raise ValueError('last-invocation suffix units must fit committed geometry')
        units[source]+=total;last[source]+=new;batch=32 if source=='a08' else 2;calls[source]+=sum((n+batch-1)//batch for n in counts)
    if production_stage not in ['cache_outer','producer_precache']:raise ValueError('declared actual source production stage required')
    attempts=[r for r in ledger if r['stage']==production_stage]
    adoption=[r for r in ledger if r['stage']=='cache_outer'] if production_stage=='producer_precache' else []
    if len({r['receipt_id'] for r in attempts+adoption})!=len(attempts+adoption):raise ValueError('production/adoption receipts must be charged exactly once')
    if not attempts or len({r['receipt_id'] for r in attempts})!=len(attempts):raise ValueError('actual source attempt receipts required exactly once')
    for r in attempts+adoption:
        if any(not isinstance(r[k],(float,int)) or not math.isfinite(r[k]) or r[k]<0 for k in ['full_process_wall_seconds','measured_CPU_core_seconds']):raise ValueError('finite actual source attempt costs required')
    exact=len(attempts)==1 and attempts[0]['observed_worker_exit_code']==0 and attempts[0].get('child_exit_code_directly_waited') is True and not attempts[0].get('termination_reason') and not attempts[0].get('guardian_exception') and not attempts[0].get('OS_status_inferred_from_manifest',False) and units==last
    return {'source_production_stage':production_stage,'read_only_adoption_CPU_seconds':sum(r['measured_CPU_core_seconds'] for r in adoption),'read_only_adoption_GPU_context_wall_seconds':sum(r['full_process_wall_seconds'] for r in adoption if r['GPU_job'] or r['CUDA_context_occupied']),'read_only_adoption_attempts':len(adoption),'read_only_adoption_new_source_forward_calls':0,'committed_unique_neural_units':units,'source_python_model_forward_calls_for_committed_sequences':calls,'last_manifest_invocation_new_neural_units':last,'recorded_source_attempts':len(attempts),'source_cache_GPU_jobwall_seconds':sum(r['full_process_wall_seconds'] for r in attempts if r['GPU_job'] or r['CUDA_context_occupied']),'source_cache_CPU_seconds':sum(r['measured_CPU_core_seconds'] for r in attempts),'source_attempt_OS_statuses':[{'attempt':r['attempt'],'observed_worker_exit_code':r['observed_worker_exit_code'],'child_exit_code_directly_waited':r.get('child_exit_code_directly_waited'),'OS_status_inferred_from_manifest':r.get('OS_status_inferred_from_manifest',False)} for r in attempts],'physical_call_count_exact_for_recorded_attempts':exact,'actual_total_source_model_forward_calls':calls if exact else None,'failed_or_interrupted_attempt_forward_counts_not_inferred_from_committed_cache':not exact,'all_actual_source_attempt_time_charged_once':True,'committed_geometry_is_not_a_reexecution_or_failed_attempt_count':True}
