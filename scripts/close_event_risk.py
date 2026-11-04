#!/usr/bin/env python3
"""Queue-held A62 resource snapshot; full science or explicit failed/incomplete mode."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import sys,time,resource,socket,json,subprocess,fcntl
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.event_risk_common import *
from scripts.event_risk_execution import preouter
from echo_routing.temporal.event_risk_budget import totals,PRIOR_STORAGE_UPPER_BYTES,DEADLINE_EPOCH,CPU_CAP_SECONDS,GPU_CAP_SECONDS,STORAGE_CAP_BYTES
from echo_routing.temporal.structured_guard import storage_bytes,process_sample
from echo_routing.temporal.anchor_cache import durable_directory


def completion_state(science_complete=True):return {'server_scientific_collection_complete':bool(science_complete),'scientific_incomplete':not science_complete,'resource_snapshot_collected':True,'caller_final_OS_receipt_pending':True,'protected_root_public_integration_pending':True,'overall_complete':False,'all_authorized_stages_complete':False,'phase':'server_science_collected_pending_caller_and_protected_public_integration' if science_complete else 'failed_or_incomplete_science_resource_snapshot_pending_caller_and_root'}


def validate_science(pub):
    pub=Path(pub);m=read(pub/'models_manifest.json');s=read(pub/'summary.json');o=read(pub/'action_space_diagnostic.json');n=read(pub/'native_attribution.json');r=read(pub/'regression_summary.json');f=read(pub/'figure_receipt.json')
    if (m['formal_optimizer_updates'],len(m['models']),len(m['fullstates']),len(m['inner_records']),len(m['thresholds']),s['metric_rows'],s['paired_comparison_count'],o['new_full_oracle_count'],o['readonly_old_oracle_count'],o['support_count'],n['cell_count'],n['base_reference_count'],r['record_count'],len(f['artifacts']))!=(24000,4,16,26,6,156,240,6,30,48,24,2,16,18):raise ValueError('complete original authorized scientific matrix required even when result negative')
    return True


def main():
    p=execution_parser(__doc__);p.add_argument('--incomplete',action='store_true');p.add_argument('--failure-reason');a=p.parse_args()
    if not authorize(a):return 0
    started=time.time();durable_directory(RUN);lock=(RUN/'exclusive_supervisor.lock').open('a+')
    try:fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise ValueError('samequeue still occupied; no closure')
    if (PUB/'resources.json').exists() or (PUB/'terminal_exit_receipt.json').exists():raise ValueError('one collector only; retain earlier evidence')
    rows=actual_receipts();workers=[r for r in rows if r.get('child_exit_code_directly_waited')];spent=totals(rows);admissions=[read(p) for p in PUB.glob('*_admission_v*.json')]
    if a.incomplete and (not a.failure_reason or not (any(r['observed_worker_exit_code'] not in [None,0] or r.get('termination_reason') for r in workers) or any(not r['all_caps_fit'] for r in admissions) or time.time()>=DEADLINE_EPOCH)):raise ValueError('explicit incomplete mode requires retained actual failure/admission/deadline evidence; never a shortcut for the authorized matrix')
    active=int((RUN/'active_process.json').exists());active_claim=read(RUN/'active_process.json') if active else None
    if active_claim and active_claim.get('model_pid'):
        sample=process_sample(active_claim['model_pid'])
        if sample and sample['start_ticks']==active_claim.get('process_start_ticks'):raise ValueError('unclosed owned active claim is still live; cannot finalize resource costs')
    pending=[str(p.relative_to(RUN)) for p in RUN.rglob('pending*.json')]
    if not a.incomplete and (active or pending):raise ValueError('active/pending operations require explicit reconciliation')
    freeze=verify_freeze('incomplete_terminal_freeze.json' if a.incomplete else 'terminal_freeze.json');contracts=[];contract_errors=[]
    for p in sorted(RUN.glob('*_freeze.json')):
        try:v=verify_freeze(p.name);contracts.append({'path':str(p.relative_to(ROOT)),'sha256':sha(p),'git_commit':v['git_commit'],'objects':len(v['objects'])})
        except Exception as e:
            if not a.incomplete:raise
            contract_errors.append({'path':str(p.relative_to(ROOT)),'error':repr(e)})
    missing=[]
    if not a.incomplete:
        verify_prior_science();preouter();validate_science(PUB);visual=read(PUB/'root_figure_visual_QA.json');science=read(PUB/'independent_scientific_review.json')
        if not visual['passed'] or visual['figure_receipt_sha256']!=sha(PUB/'figure_receipt.json') or not science['approved'] or science['summary_sha256']!=sha(PUB/'summary.json'):raise ValueError('actual fixed-figure visualQA and independent bounded scientific review required')
        for r in read(PUB/'figure_receipt.json')['artifacts']:
            if sha(PUB/r['path'])!=r['sha256']:raise ValueError('actual PNG/PDF/CSV bytes changed')
    else:missing=[name for name in ['models_manifest.json','calibration.json','outer_metrics.json','summary.json','action_space_diagnostic.json','native_attribution.json','regression_summary.json','package_receipt.json','figure_receipt.json','root_figure_visual_QA.json','independent_scientific_review.json'] if not (PUB/name).exists()]
    live=[];statuses=[]
    for r in workers:
        sample=process_sample(r['model_pid']) if r['model_pid'] else None;matching=sample and sample['start_ticks']==r['process_start_ticks']
        if matching:live.append(r['model_pid'])
        statuses.append({k:r.get(k) for k in ['receipt_id','model_pid','process_start_ticks','observed_worker_exit_code','child_exit_code_directly_waited','termination_reason']})
    if live:raise ValueError('recorded owned workers still live; cannot finalize resource costs')
    try:gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True,timeout=5);owners=[int(x.strip()) for x in gpu.splitlines() if x.strip()]
    except FileNotFoundError:owners=[]
    owned_live=[pid for pid in owners if any(pid==r['model_pid'] and (sample:=process_sample(pid)) and sample['start_ticks']==r['process_start_ticks'] for r in workers)]
    if os.getpid() in owners:owned_live.append(os.getpid())
    if owned_live:raise ValueError('owned A62 CUDA context remains')
    du=storage_bytes([RUN,PUB]);upper=max([du]+[r.get('maximum_sampled_tracked_storage_upper_bytes',0) for r in rows]);finish=time.time();sb=resource.getrusage(resource.RUSAGE_SELF);cb=resource.getrusage(resource.RUSAGE_CHILDREN)
    # Fresh standalone collector: complete lifetime CPU includes imports/startup.
    selfcpu=sb.ru_utime+sb.ru_stime;childcpu=cb.ru_utime+cb.ru_stime;cpu=selfcpu+childcpu;state=completion_state(not a.incomplete);newcpu=spent['new_measured_CPU_seconds']+cpu;newgpu=spent['new_GPU_wall_seconds'];capsfit=newcpu<=CPU_CAP_SECONDS and newgpu<=GPU_CAP_SECONDS and upper<=STORAGE_CAP_BYTES and PRIOR_STORAGE_UPPER_BYTES+upper<=40000000000 and finish<=DEADLINE_EPOCH
    resources={**state,'passed':True,'actual_caps_fit':None if active else capsfit,'unclosed_active_claim':active_claim,'unrecorded_active_claim_CPU_seconds':None,'CPU_and_GPU_known_totals_are_lower_bounds_if_unclosed_active_claim':bool(active),'failure_reason':a.failure_reason,'missing_scientific_artifacts':missing,'recorded_receipts':rows,'server_worker_statuses':statuses,'all_recorded_worker_OS_statuses_directly_waited':all(r['observed_worker_exit_code'] is not None and r['child_exit_code_directly_waited'] for r in workers),'unknown_worker_OS_statuses':sum(r['observed_worker_exit_code'] is None for r in workers)+active,'collector_pid':os.getpid(),'collector_start_ticks':process_sample(os.getpid())['start_ticks'],'SELF_OS_exit_code':None,'collector_CPU_seconds_through_snapshot':cpu,'collector_CPU_scope':'whole fresh process lifetime SELF+waitedCHILDREN through snapshot, including imports/startup','collector_SELF_CPU_seconds':selfcpu,'collector_waited_probe_CPU_seconds':childcpu,'collector_main_started_epoch':started,'collector_snapshot_epoch':finish,'collector_final_write_exit_tail_not_yet_measured':True,'incremental_GPU_seconds':newgpu,'incremental_CPU_seconds':newcpu,'cumulative_GPU_hours':spent['cumulative_GPU_hours'],'cumulative_CPU_accounting_hours':19.78341831+newcpu/3600,'incremental_tracked_storage_upper_bytes':upper,'cumulative_tracked_storage_upper_bytes':PRIOR_STORAGE_UPPER_BYTES+upper,'measured_du_plus105MB_upper_bytes':du,'observed_du_bytes':du-105000000,'storage_scope':'max actual du plus105MB margin / observed stage peak; rejected and unexecuted forecasts are not actual spend','prospective_storage_bounds_not_charged_as_actual':[{k:r.get(k) for k in ['stage','version','all_caps_fit','forecast_cumulative_tracked_storage_upper_bytes_including_reserve']} for r in admissions],'actual_GPU_owners':owners,'A62_GPU_owners':owned_live,'foreign_GPU_work_not_charged_or_killed':True,'same_queue_lock_held_during_collection':True,'current_contracts':contracts,'contract_verification_errors':contract_errors,'immutable_prior_science_and_stage_Git_proofs_verified':not a.incomplete and not contract_errors,'unknown_coordination_model_reasoning_sourcecommit_tool_tail_preserved_not_zero':True,'deadline_epoch':DEADLINE_EPOCH,'terminal_freeze_git':freeze['git_commit']}
    write(PUB/'resources.json',resources);write(PUB/'terminal_exit_receipt.json',{**state,'passed':True,'actual_caps_fit':capsfit,'resources_sha256':sha(PUB/'resources.json'),'collector_pid':os.getpid(),'collector_start_ticks':resources['collector_start_ticks'],'collector_SELF_OS_exit_code':None,'recorded_owned_worker_live':[],'A62_GPU_owners':[],'foreign_GPU_owners':owners,'pending_count':len(pending),'pending_paths_preserved':pending,'active_count':active,'active_claim_preserved':bool(active),'observed_worker_statuses':statuses,'actual_collector_exit_must_be_observed_by_parent':True});write(RUN/'runtimecoordination.json',{**state,'collector_pid':os.getpid(),'epoch':time.time()});print({'server_scientific_resource_snapshot':not a.incomplete,'resource_snapshot_collected':True,'actual_caps_fit':capsfit,'overall_complete':False,'resources_sha256':sha(PUB/'resources.json'),'terminal_sha256':sha(PUB/'terminal_exit_receipt.json')},flush=True);return 0
if __name__=='__main__':raise SystemExit(main())
