#!/usr/bin/env python3
"""A62 aggregate-only six scientific figure families and complete public tables."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import sys,json,csv,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.event_risk_common import *
from echo_routing.temporal.event_risk_figures import render
from echo_routing.temporal.event_risk_budget import totals


def export(path,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('x',newline='') as h:
        w=csv.DictWriter(h,fieldnames=fields);w.writeheader()
        for r in rows:w.writerow({k:json.dumps(v,sort_keys=True,allow_nan=False) if isinstance(v,(dict,list)) else v for k,v in r.items()})


def source_costs(producer,receipts):
    stage=producer['source_production_stage'];attempts=[r for r in receipts if r['stage']==stage];ids=[r['receipt_id'] for r in attempts]
    if not attempts or len(ids)!=len(set(ids)):raise ValueError('all actual source attempts exactly once required')
    complete=all(r['observed_worker_exit_code']==0 and r.get('child_exit_code_directly_waited') and not r.get('termination_reason') for r in attempts)
    return {'source_production_stage':stage,'production_event_updates':0,'later_adoption_event_updates':24000,'actual_occupied_GPU_wall_seconds':sum(r['full_process_wall_seconds'] for r in attempts if r['GPU_job'] or r['CUDA_context_occupied']),'actual_CPU_seconds':sum(r['measured_CPU_core_seconds'] for r in attempts),'attempts':[{k:r.get(k) for k in ['receipt_id','model_pid','process_start_ticks','observed_worker_exit_code','child_exit_code_directly_waited','full_process_wall_seconds','measured_CPU_core_seconds']} for r in attempts],'committed_teacher_payloads':sum(r['committed_sequences'] for r in producer['records']),'committed_logical_neural_units':{s:sum(r['committed_logical_neural_units'] for r in producer['records'] if r['source']==s) for s in ['a08','jepa']},'actual_forward_units_across_all_attempts':{s:sum(r['new_neural_units'] for r in producer['records'] if r['source']==s) for s in ['a08','jepa']} if len(attempts)==1 and complete and not any(r['resumed_committed_prefix_sequences'] for r in producer['records']) else None,'multiple_failed_or_resumed_actual_forward_units_unknown_not_imputed_from_geometry':not(len(attempts)==1 and complete),'cached_head_latency_is_full_system_latency':False,'operational_all_policies_use_A08_and_JEPA_producers':True}


def main():
    p=execution_parser(__doc__);p.add_argument('--attempt',default='final_v1');a=p.parse_args()
    if not authorize(a):return 0
    freeze=verify_freeze('package_freeze.json');summary=read(PUB/'summary.json');outer=read(PUB/'outer_metrics.json');inner=read(PUB/'inner_metrics.json');cal=read(PUB/'calibration.json');oracle=read(PUB/'action_space_diagnostic.json');native=read(PUB/'native_attribution.json');reg=read(PUB/'regression_summary.json');costs=actual_receipts();spent=totals(costs)
    if (outer['metric_rows'],inner['metric_rows'],summary['paired_comparison_count'],oracle['new_full_oracle_count'],oracle['readonly_old_oracle_count'],oracle['support_count'],native['cell_count'],native['base_reference_count'],reg['record_count'])!=(156,26,240,6,30,48,24,2,16):raise ValueError('complete182/240/6+30/48/24+2/16 before package')
    curves=[]
    for row in read(PUB/'model_audit.json')['models']:
        trace=ROOT/row['path'];trace=trace.parent/'physical_updates.jsonl';raw=[json.loads(x) for x in trace.read_text().splitlines()]
        if len(raw)!=6000:raise ValueError('complete6000 physical BCE trace required')
        for offset in range(0,6000,100):block=raw[offset:offset+100];curves.append({'seed':row['seed'],'arm':row['arm'],'update':offset+len(block),'loss':sum(r['loss'] for r in block)/len(block),'mean_of_actual_last100_updates':True})
    directory=PUB/'figure_attempts'/a.attempt
    if directory.exists():raise ValueError('retain previous figure attempt bytes; choose new explicit attempt')
    resources={'incremental_GPU_seconds':spent['new_GPU_wall_seconds'],'incremental_CPU_seconds':spent['new_measured_CPU_seconds'],'incremental_tracked_storage_upper_bytes':max(r.get('maximum_sampled_tracked_storage_upper_bytes',0) for r in costs)}
    pins=render(directory,outer['rows'],summary['paired_comparisons'],inner['rows'],cal,reg['records'],curves,resources)
    # Renderer returns a receipt, with paths relative to its attempt root.
    for artifact in pins['artifacts']:artifact['path']=str((directory/artifact['path']).relative_to(PUB))
    pins.update(summary_sha256=sha(PUB/'summary.json'),outer_metrics_sha256=sha(PUB/'outer_metrics.json'),inner_metrics_sha256=sha(PUB/'inner_metrics.json'),regression_sha256=sha(PUB/'regression_summary.json'),native_sha256=sha(PUB/'native_attribution.json'),package_freeze_git=freeze['git_commit'],actual_visual_QA_passed=False)
    write(PUB/'figure_receipt.json',pins)
    export(PUB/'metrics.csv',outer['rows']);export(PUB/'inner_metrics.csv',inner['rows']);export(PUB/'paired_comparisons.csv',summary['paired_comparisons']);export(PUB/'full_oracles.csv',oracle['new_oracle_domains']+oracle['readonly_A61_reference_domains']);export(PUB/'N_top_support.csv',oracle['fixed_N_top_support']);export(PUB/'native_cubes.csv',native['cells']+native['base_references']);export(PUB/'development_regression.csv',reg['records']);export(PUB/'training_curves.csv',curves)
    write(PUB/'inference_costs.json',{'teacher_production':source_costs(read(PUB/'producer_manifest.json'),costs),'source_adoption':read(PUB/'source_adoption.json'),'fixed_head_cache_cells':cal['fixed_head_cache_records']+outer['new_fixed_head_cache_cells'],'event_head_cache_cells':cal['q_cache_records']+outer['joint_q_cache_cells'],'logical6policies_share_same_C6_q_no_repeat_teacher':True,'cached_timings_exclude_fullsystem_frozen_producer_cost':True,'physical_MLP_counts_are_not_patient_or_logical_policy_counts':True});write(PUB/'package_receipt.json',{'passed':True,'new_model_or_teacher_forwards':0,'new_optimizer_updates':0,'six_families_png_pdf_csv':18,'figure_receipt_sha256':sha(PUB/'figure_receipt.json'),'actual_visual_QA_pending':True,'source_costs_sha256':sha(PUB/'inference_costs.json'),'freeze_git':freeze['git_commit'],'epoch':time.time()});from scripts.report_event_risk import report
    report();print({'figures':18,'all_tables_exported':True,'visual_QA_pending':True},flush=True);return 0
if __name__=='__main__':raise SystemExit(main())
