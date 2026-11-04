#!/usr/bin/env python3
"""A60 terminal closure under same exclusive queue; actual receipts, no reset."""
import os,sys,time,json,resource,socket,subprocess
_START=time.time()
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.proposal_common import RUN,PUB,CFG,read,write,sha,verify_freeze,verify_pin,actual_receipts,digest
from echo_routing.temporal.proposal_terminal import validate_history
from echo_routing.temporal.proposal_resources import ExclusiveQueue,totals,PRIOR_STORAGE_UPPER_BYTES,DEADLINE_EPOCH


def main():
    with ExclusiveQueue(RUN/'exclusive_supervisor.lock'):
        if (PUB/'resources.json').exists() or (PUB/'terminal_exit_receipt.json').exists():raise ValueError('terminal evidence already exists;never overwrite a completed or partial closure')
        names=['fullFIT_check_freeze.json','formal_admission_freeze.json',
               'formal_freeze.json','audit_freeze.json','outer_admission_freeze.json','outer_freeze.json','readout_freeze.json','aggregate_freeze.json',
               'oracle_freeze.json','package_freeze.json','native_attribution_freeze.json','terminal_freeze.json']
        contracts={name:verify_freeze(name) for name in names}
        historical=read(PUB/'historical_source_version_verification.json');oldidentity=read(PUB/'preparation_identity_v3.json')
        divergences=[p for p,h in oldidentity['objects'].items() if sha(ROOT/p)!=h]
        validate_history(historical,oldidentity,sha(PUB/'preparation_identity_v3.json'),digest(oldidentity['objects']),divergences)
        for name in ['preparation_freeze.json','training_prepare_freeze.json']:
            if sha(RUN/name)!=sha(PUB/'preparation_identity_v3.json'):raise ValueError('historical server freeze itself changed')
        if (RUN/'active_process.json').exists():raise ValueError('active queue claim prevents terminal closure')
        records=[json.loads(line) for line in (RUN/'processes.jsonl').read_text().splitlines()]
        local=read(PUB/'local_cpu_receipts.json')
        for path,expected in local['file_pins'].items():
            if sha(ROOT/path)!=expected:raise ValueError('local CPU receipt changed')
        actual=totals(actual_receipts())
        unknown_worker_OS_statuses=sum(r.get('observed_worker_exit_code') is None for r in records)
        directly_waited=all(r.get('child_exit_code_directly_waited') is True
                            and r.get('observed_worker_exit_code') is not None
                            and not r.get('OS_status_inferred_from_manifest',False) for r in records)
        pids={r[key] for r in records for key in ['model_pid','supervisor_pid'] if isinstance(r.get(key),int)}
        if any(Path(f'/proc/{pid}').exists() for pid in pids):raise ValueError('recorded execution/guardian PID still exists')
        gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
        if gpu:raise ValueError('GPU owner still present')
        pending=list(RUN.rglob('*pending*.json'))
        if pending:raise ValueError('ambiguous update/readout claim still present')
        summary=read(PUB/'summary.json');oracle=read(PUB/'action_space_diagnostic.json');models=read(PUB/'models_manifest.json');native=read(PUB/'native_attribution.json')
        if (summary['formal_models']!=4 or summary['formal_optimizer_updates']!=24000 or summary['metric_rows']!=124
            or summary['paired_comparison_count']!=144 or oracle['domain_count']!=32 or oracle['overlapping_old_A59_oracles_exactly_reproduced']!=12 or not models['passed'] or not native['passed'] or native['cell_count']!=24 or native['source_outer_metrics_sha256']!=sha(PUB/'outer_metrics.json')):
            raise ValueError('all authorized scientific stages must complete')
        cfg=read(CFG);pins=[cfg['source']['a08'],cfg['source']['jepa']]+[
              {'path':m['path'],'sha256':m['sha256']} for row in models['models'] for m in row['milestones']]
        for pin in pins:verify_pin(pin)
        apparent=sum(int(line.split()[0]) for line in subprocess.check_output(['du','-sb',str(RUN),str(PUB)],text=True).splitlines());allocated=sum(int(line.split()[0]) for line in subprocess.check_output(['du','-sB1',str(RUN),str(PUB)],text=True).splitlines())
        codepaths=[p for folder,pattern in [('scripts','*proposal*.py'),('echo_routing/temporal','proposal*.py'),('tests','test_proposal*.py')] for p in (ROOT/folder).glob(pattern)]
        codebytes=sum(p.stat().st_size for p in codepaths)
        latest=max(p.stat().st_size for p in RUN.glob('models/seed*/*/latest.pt'))
        storage=max(read(PUB/'formal_admission.json')['incremental_tracked_path_peak_upper_bytes'],read(PUB/'outer_admission.json')['incremental_tracked_path_peak_upper_bytes'],max(apparent,allocated)+latest+codebytes+115_000_000)
        own=resource.getrusage(resource.RUSAGE_SELF);children=resource.getrusage(resource.RUSAGE_CHILDREN)
        cpu=own.ru_utime+own.ru_stime+children.ru_utime+children.ru_stime
        cumulativeCPU=actual['cumulative_CPU_accounting_hours']+cpu/3600
        cumulativeGPU=actual['cumulative_GPU_hours'];now=time.time()
        if cumulativeGPU>24 or cumulativeCPU>72 or PRIOR_STORAGE_UPPER_BYTES+storage>40e9 or now>DEADLINE_EPOCH:
            raise ValueError('inherited cumulative caps exceeded')
        value={'passed':True,'all_authorized_A60_stages_complete':True,'no_remaining_authorized_model_training_or_readout':True,
            'scientific_summary_sha256':sha(PUB/'summary.json'),'action_space_diagnostic_sha256':sha(PUB/'action_space_diagnostic.json'),
            'models_manifest_sha256':sha(PUB/'models_manifest.json'),'native_attribution_sha256':sha(PUB/'native_attribution.json'),'source_frozen_objects_verified':len(contracts['outer_freeze.json']['objects']),
            'all_current_contracts_verified':names,'historical_preparation_source_versions_verified_against_actual_Git_receipt':historical,'source_and_all12milestone_weight_files_verified':len(pins),
            'new_GPU_job_hours':actual['new_GPU_wall_seconds']/3600,'cumulative_GPU_job_hours':cumulativeGPU,
            'new_summed_measured_CPU_receipt_hours_including_collector':(actual['new_measured_CPU_seconds']+cpu)/3600,
            'cumulative_summed_measured_CPU_receipt_hours':actual['cumulative_measured_CPU_hours']+cpu/3600,
            'prior_completed_CPU_allowance_hours':actual['prior_CPU_allowance_hours'],
            'new_CPU_allowance_hours':0,'cumulative_CPU_receipt_plus_accounting_allowances_hours':cumulativeCPU,
            'retained_tracked_paths_apparent_bytes':apparent,'retained_tracked_paths_allocated_bytes':allocated,
            'new_source_code_and_test_file_bytes':codebytes,'largest_atomic_checkpoint_bytes':latest,
            'new_tracked_storage_upper_bytes':storage,'prior_tracked_storage_upper_bytes':PRIOR_STORAGE_UPPER_BYTES,
            'cumulative_tracked_storage_upper_bytes':PRIOR_STORAGE_UPPER_BYTES+storage,
            'storage_scope':'conservative tracked-path upper from component forecast/currentretained+atomic/reportmargin;not whole-filesystem peak',
            'actual_totals':{k:v for k,v in actual.items() if k!='receipt_records'},
            'local_cpu_receipts_sha256':sha(PUB/'local_cpu_receipts.json'),'process_ledger_sha256':sha(RUN/'processes.jsonl'),
            'all_recorded_model_and_supervisor_PIDs_absent':True,'recorded_execution_PID_count':len(pids),
            'GPU_compute_processes_absent':True,'pending_claims':0,'same_queue_lock_held_through_checks_and_writes':True,
            'new_worker_OS_statuses_directly_waited_not_inferred_from_manifests':directly_waited,
            'unknown_server_worker_OS_status_count':unknown_worker_OS_statuses,
            'local_SELF_collection_snapshots_have_null_OS_status_not_failed_or_unknown_neural_jobs':True,
            'server_failed_or_terminated_stage_count':sum(r['observed_worker_exit_code'] not in [0,None] or bool(r.get('termination_reason')) for r in records),
            'local_nonzero_software_check_receipts_including_expected_RED':sum(r['observed_worker_exit_code'] not in [0,None] for r in local['receipts']),
            'local_SELF_snapshot_null_OS_receipts':sum(r['observed_worker_exit_code'] is None for r in local['receipts']),
            'CPU_scope':'sum measured SELF+waitedCHILDREN receipts plus entry complements and inherited completedallowances;legacy provenance/overlap and knownpre-entrysetup CPU gaps disclosed;not unique wholehost CPU',
            'known_unmetered_pre_entry_setup_failures':read(PUB/'prelaunch_failures.json'),
            'entry_CPU_receipts_count':len(list(PUB.glob('entry_CPU_*.json'))),
            'entry_CPU_complements_are_not_second_worker_jobs':True,
            'prior_A55_A56_and_A57_A58_cost_receipts_retained_separately':True,'all_inherited_caps_passed':True,
            'earlier_deadline_epoch':DEADLINE_EPOCH,'closure_epoch':now,
            'collector':{'measurement_host':socket.gethostname(),'model_pid':os.getpid(),'started_epoch':_START,
                         'CPU_SELF_seconds':own.ru_utime+own.ru_stime,'CPU_waited_CHILDREN_seconds':children.ru_utime+children.ru_stime,
                         'CPU_seconds_through_snapshot':cpu,'observed_worker_exit_code':None,
                         'scope':'running collector SELF + waited probes; final JSON bookkeeping aftersnapshot unmetered; actual OS exit observed by SSH caller separately'},
            'root_final_integration_coordination_tail_separate':True}
        write(PUB/'resources.json',value)
        write(PUB/'terminal_exit_receipt.json',{'passed':True,'all_workers_absent':True,'GPU_idle':True,'pending_claims':0,
            'same_queue_lock_held':True,'worker_OS_exit_statuses':[{'stage':r['stage'],'attempt':r['attempt'],
                'model_pid':r['model_pid'],'observed_worker_exit_code':r['observed_worker_exit_code'],
                'child_exit_code_directly_waited':r.get('child_exit_code_directly_waited'),
                'OS_status_inferred_from_manifest':r.get('OS_status_inferred_from_manifest',False)} for r in records],
            'all_new_worker_statuses_directly_waited':directly_waited,'unknown_server_worker_OS_status_count':unknown_worker_OS_statuses,'resource_receipt_sha256':sha(PUB/'resources.json'),'epoch':now})
        write(RUN/'runtimecoordination.json',{'phase':'closed_complete','model_pid':None,'formal_optimizer_updates':24000,
             'metric_rows':124,'paired_comparisons':144,'oracle_cells':32,'all_authorized_stages_complete':True,
             'next_action':'root selective public integration and delete only capstone-a60;no automatic A61',
             'resource_receipt_sha256':sha(PUB/'resources.json'),'epoch':now})
        print(json.dumps({'closed':True,'new_GPU_job_hours':value['new_GPU_job_hours'],'cumulative_GPU_job_hours':cumulativeGPU,
              'cumulative_CPU_accounting_hours':cumulativeCPU,'cumulative_tracked_storage_upper_bytes':PRIOR_STORAGE_UPPER_BYTES+storage}),flush=True)

if __name__=='__main__':main()
