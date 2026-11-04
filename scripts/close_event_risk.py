#!/usr/bin/env python3
"""Queue-held A62 scientific/server resource snapshot; caller/root still pending."""
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


def completion_state():return {'server_scientific_collection_complete':True,'caller_final_OS_receipt_pending':True,'protected_root_public_integration_pending':True,'overall_complete':False,'all_authorized_stages_complete':False,'phase':'server_science_collected_pending_caller_and_protected_public_integration'}


def validate_science(pub):
    pub=Path(pub);m=read(pub/'models_manifest.json');s=read(pub/'summary.json');o=read(pub/'action_space_diagnostic.json');n=read(pub/'native_attribution.json');r=read(pub/'regression_summary.json');f=read(pub/'figure_receipt.json')
    if (m['formal_optimizer_updates'],len(m['models']),len(m['fullstates']),len(m['inner_records']),len(m['thresholds']),s['metric_rows'],s['paired_comparison_count'],o['new_full_oracle_count'],o['readonly_old_oracle_count'],o['support_count'],n['cell_count'],n['base_reference_count'],r['record_count'],len(f['artifacts']))!=(24000,4,16,26,6,156,240,6,30,48,24,2,16,18):raise ValueError('complete original authorized scientific matrix required even when result negative')
    return True


def main():
    p=execution_parser(__doc__);a=p.parse_args()
    if not authorize(a):return 0
    started=time.time();sa=resource.getrusage(resource.RUSAGE_SELF);ca=resource.getrusage(resource.RUSAGE_CHILDREN);durable_directory(RUN);lock=(RUN/'exclusive_supervisor.lock').open('a+')
    try:fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise ValueError('samequeue still occupied; no closure')
    if (PUB/'resources.json').exists() or (PUB/'terminal_exit_receipt.json').exists():raise ValueError('one collector only; retain earlier evidence')
    if (RUN/'active_process.json').exists() or list(RUN.rglob('pending.json')) or list(RUN.rglob('pending_update.json')):raise ValueError('active/pending operations require explicit reconciliation')
    freeze=verify_freeze('terminal_freeze.json');contracts=[]
    for p in sorted(RUN.glob('*_freeze.json')):
        v=verify_freeze(p.name);contracts.append({'path':str(p.relative_to(ROOT)),'sha256':sha(p),'git_commit':v['git_commit'],'objects':len(v['objects'])})
    verify_prior_science();preouter();validate_science(PUB)
    visual=read(PUB/'root_figure_visual_QA.json');science=read(PUB/'independent_scientific_review.json')
    if not visual['passed'] or visual['figure_receipt_sha256']!=sha(PUB/'figure_receipt.json') or not science['approved'] or science['summary_sha256']!=sha(PUB/'summary.json'):raise ValueError('actual fixed-figure visualQA and independent bounded scientific review required')
    fig=read(PUB/'figure_receipt.json')
    for r in fig['artifacts']:
        if sha(PUB/r['path'])!=r['sha256']:raise ValueError('actual PNG/PDF/CSV bytes changed')
    rows=actual_receipts();spent=totals(rows);workers=[r for r in rows if r.get('child_exit_code_directly_waited')];live=[];statuses=[]
    for r in workers:
        sample=process_sample(r['model_pid']) if r['model_pid'] else None;matching=sample and sample['start_ticks']==r['process_start_ticks']
        if matching:live.append(r['model_pid'])
        statuses.append({k:r.get(k) for k in ['receipt_id','model_pid','process_start_ticks','observed_worker_exit_code','child_exit_code_directly_waited','termination_reason']})
    if live:raise ValueError('recorded owned workers still live')
    try:gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True,timeout=5);owners=[int(x.strip()) for x in gpu.splitlines() if x.strip()]
    except FileNotFoundError:owners=[]
    owned={r['model_pid'] for r in workers if r['model_pid']};owned_live=[pid for pid in owners if any(pid==r['model_pid'] and (sample:=process_sample(pid)) and sample['start_ticks']==r['process_start_ticks'] for r in workers)]
    if os.getpid() in owners:owned_live.append(os.getpid())
    if owned_live:raise ValueError('owned A62 CUDA context remains')
    du=storage_bytes([RUN,PUB]);upper=max([du]+[r.get('maximum_sampled_tracked_storage_upper_bytes',0) for r in rows]+[read(p)['remaining_complete_workload']['storage_upper_bytes'] for p in PUB.glob('*_admission_v*.json')]);finish=time.time();sb=resource.getrusage(resource.RUSAGE_SELF);cb=resource.getrusage(resource.RUSAGE_CHILDREN);selfcpu=sb.ru_utime+sb.ru_stime-sa.ru_utime-sa.ru_stime;childcpu=cb.ru_utime+cb.ru_stime-ca.ru_utime-ca.ru_stime;cpu=selfcpu+childcpu;state=completion_state();newcpu=spent['new_measured_CPU_seconds']+cpu;newgpu=spent['new_GPU_wall_seconds']
    if newcpu>CPU_CAP_SECONDS or newgpu>GPU_CAP_SECONDS or upper>STORAGE_CAP_BYTES or PRIOR_STORAGE_UPPER_BYTES+upper>40000000000 or finish>DEADLINE_EPOCH:raise ValueError('actual original incremental/cumulative caps or original deadline exceeded')
    resources={**state,'passed':True,'recorded_receipts':rows,'server_worker_statuses':statuses,'all_recorded_worker_OS_statuses_directly_waited':all(r['observed_worker_exit_code'] is not None and r['child_exit_code_directly_waited'] for r in workers),'unknown_worker_OS_statuses':sum(r['observed_worker_exit_code'] is None for r in workers),'collector_pid':os.getpid(),'collector_start_ticks':process_sample(os.getpid())['start_ticks'],'SELF_OS_exit_code':None,'collector_CPU_seconds_through_snapshot':cpu,'collector_SELF_CPU_seconds':selfcpu,'collector_waited_probe_CPU_seconds':childcpu,'collector_started_epoch':started,'collector_snapshot_epoch':finish,'collector_final_write_exit_tail_not_yet_measured':True,'incremental_GPU_seconds':newgpu,'incremental_CPU_seconds':newcpu,'cumulative_GPU_hours':spent['cumulative_GPU_hours'],'cumulative_CPU_accounting_hours':19.78341831+newcpu/3600,'incremental_tracked_storage_upper_bytes':upper,'cumulative_tracked_storage_upper_bytes':PRIOR_STORAGE_UPPER_BYTES+upper,'measured_tracked_bytes':du,'actual_GPU_owners':owners,'A62_GPU_owners':owned_live,'foreign_GPU_work_not_charged_or_killed':True,'same_queue_lock_held_during_collection':True,'current_contracts':contracts,'immutable_prior_science_and_stage_Git_proofs_verified':True,'unknown_coordination_model_reasoning_sourcecommit_tool_tail_preserved_not_zero':True,'deadline_epoch':DEADLINE_EPOCH,'terminal_freeze_git':freeze['git_commit']}
    write(PUB/'resources.json',resources);write(PUB/'terminal_exit_receipt.json',{**state,'passed':True,'resources_sha256':sha(PUB/'resources.json'),'collector_pid':os.getpid(),'collector_start_ticks':resources['collector_start_ticks'],'collector_SELF_OS_exit_code':None,'recorded_owned_worker_live':[],'A62_GPU_owners':[],'foreign_GPU_owners':owners,'pending_count':0,'active_count':0,'observed_worker_statuses':statuses,'actual_collector_exit_must_be_observed_by_parent':True});write(RUN/'runtimecoordination.json',{**state,'collector_pid':os.getpid(),'epoch':time.time()});print({'server_scientific_resource_snapshot':True,'overall_complete':False,'resources_sha256':sha(PUB/'resources.json'),'terminal_sha256':sha(PUB/'terminal_exit_receipt.json')},flush=True);return 0
if __name__=='__main__':raise SystemExit(main())
