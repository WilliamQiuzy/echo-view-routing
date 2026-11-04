#!/usr/bin/env python3
"""A61 terminal collector: same queue, actual costs/exits and full history proof."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,time,json,resource,socket,subprocess
_START=time.time()
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.safety_common import RUN,PUB,CFG,read,write,sha,verify_freeze,verify_pin,actual_receipts
from echo_routing.temporal.safety_history import EXPECTED,validate as validate_history
from echo_routing.temporal.safety_engineering_history import EXPECTED as ENGINEERING_EXPECTED,validate as validate_engineering
from echo_routing.temporal.safety_terminal import validate_science,worker_status,process_relation,collector_state
from echo_routing.temporal.safety_resources import ExclusiveQueue,totals,PRIOR_STORAGE_UPPER_BYTES,DEADLINE_EPOCH


def sample(pid):
    path=Path('/proc')/str(pid)/'stat'
    if not path.exists():return None
    raw=path.read_text();fields=raw[raw.rfind(')')+2:].split();ticks=int(fields[19]);boot=int(next(x.split()[1] for x in Path('/proc/stat').read_text().splitlines() if x.startswith('btime ')))
    return {'start_ticks':ticks,'start_epoch':boot+ticks/os.sysconf('SC_CLK_TCK')}


def main():
    with ExclusiveQueue(RUN/'exclusive_supervisor.lock'):
        if (PUB/'resources.json').exists() or (PUB/'terminal_exit_receipt.json').exists():raise ValueError('terminal evidence exists; no overwrite/repeated collector')
        names=['cost_probe_freeze.json','formal_admission_freeze.json','producer_precache_admission_freeze.json','producer_precache_freeze.json','role_cost_freeze.json','formal_measured_admission_freeze.json','formal_freeze.json','audit_freeze.json','outer_measured_admission_freeze.json','outer_freeze.json','readout_freeze.json','aggregate_freeze.json','oracle_freeze.json','native_freeze.json','package_freeze.json','figure_repair_freeze.json','terminal_freeze.json']
        contracts={n:verify_freeze(n) for n in names}
        history_pin=read(PUB/'historical_Git_proof_identity.json');proof=read(verify_pin(history_pin));raw={n:(PUB/(n+'.json')).read_bytes() for n in EXPECTED}
        for n in EXPECTED:
            freeze={'preparation_identity':'preparation_freeze.json','training_prepare_identity':'training_prepare_freeze.json','fullFIT_check_identity':'fullFIT_check_freeze.json'}[n]
            if sha(RUN/freeze)!=EXPECTED[n]['identity_sha256']:raise ValueError('original executed server contract changed')
        historical=validate_history(proof,raw,ROOT)
        engineering_pin=read(PUB/'engineering_Git_proof_identity.json');engineering_proof=read(verify_pin(engineering_pin));engineering_raw={n:(PUB/(n+'.json')).read_bytes() for n in ENGINEERING_EXPECTED}
        for n,freeze in [('component_cost_identity','component_cost_freeze.json'),('rle_cost_identity','rle_cost_freeze.json')]:
            if sha(RUN/freeze)!=ENGINEERING_EXPECTED[n]['identity_sha256']:raise ValueError('executed cost freeze identity changed')
        engineering=validate_engineering(engineering_proof,engineering_raw,ROOT);write(PUB/'engineering_source_version_verification.json',engineering)
        write(PUB/'historical_source_version_verification.json',historical)
        if (RUN/'active_process.json').exists():raise ValueError('active queue claim prevents closure')
        records=[json.loads(x) for x in (RUN/'processes.jsonl').read_text().splitlines()];actual=totals(actual_receipts());local=read(PUB/'local_cpu_receipts.json');status=worker_status(records);processes=[]
        for row in records:
            for key in ['model_pid','supervisor_pid']:
                pid=row.get(key)
                if not isinstance(pid,int):continue
                observed=sample(pid);relation=process_relation(row,pid,observed);processes.append({'stage':row['stage'],'pid':pid,'role':key,'relation':relation,'sample':observed})
                if relation=='possibly_same_owned_live_process':raise ValueError('recorded A61 worker/supervisor still live or ambiguous')
        gpu=[int(x.strip()) for x in subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).splitlines() if x.strip()]
        if any(p['pid'] in gpu and p['relation']=='possibly_same_owned_live_process' for p in processes):raise ValueError('owned GPU context remains')
        if list(RUN.rglob('*pending*.json')):raise ValueError('ambiguous pending update/readout claim remains')
        summary=read(PUB/'summary.json');oracle=read(PUB/'action_space_diagnostic.json');native=read(PUB/'native_attribution.json');models=read(PUB/'models_manifest.json');validate_science(summary,oracle,native,models)
        if native['source_outer_metrics_sha256']!=sha(PUB/'outer_metrics.json') or native['source_summary_sha256']!=sha(PUB/'summary.json') or oracle['summary_sha256']!=sha(PUB/'summary.json'):raise ValueError('native/oracle/scientific summary identity changed')
        figures=read(PUB/'figure_receipt.json')
        if not figures['passed'] or figures['summary_sha256']!=sha(PUB/'summary.json') or figures['action_space_diagnostic_sha256']!=sha(PUB/'action_space_diagnostic.json'):raise ValueError('complete fixed figures must bind sealed summaries')
        for p,h in figures['figures'].items():verify_pin({'path':p,'sha256':h})
        cfg=read(CFG);pins=[cfg['source']['a08'],cfg['source']['jepa']]
        for row in models['models']:
            pins.extend({'path':m['path'],'sha256':m['sha256']} for m in row['milestones']);pins.append({'path':str(Path(row['path']).parent/'latest.pt'),'sha256':row['latest_sha256']})
        for pin in pins:verify_pin(pin)
        apparent=sum(int(x.split()[0]) for x in subprocess.check_output(['du','-sb',str(RUN),str(PUB)],text=True).splitlines());allocated=sum(int(x.split()[0]) for x in subprocess.check_output(['du','-sB1',str(RUN),str(PUB)],text=True).splitlines());code=[p for folder,pattern in [('scripts','*safety*.py'),('echo_routing/temporal','safety*.py'),('tests','test_safety*.py')] for p in (ROOT/folder).glob(pattern)];codebytes=sum(p.stat().st_size for p in code);largest=max(p.stat().st_size for p in RUN.glob('models/seed*/*/latest.pt'))
        storage=max(read(PUB/'formal_admission.json')['incremental_tracked_path_peak_upper_bytes'],read(PUB/'outer_admission.json')['incremental_tracked_path_peak_upper_bytes'],max(apparent,allocated)+largest+codebytes+115_000_000)
        own=resource.getrusage(resource.RUSAGE_SELF);children=resource.getrusage(resource.RUSAGE_CHILDREN);cpu=own.ru_utime+own.ru_stime+children.ru_utime+children.ru_stime;now=time.time();cpuaccount=actual['cumulative_CPU_accounting_hours']+cpu/3600
        if actual['cumulative_GPU_hours']>24 or cpuaccount>72 or PRIOR_STORAGE_UPPER_BYTES+storage>40e9 or now>DEADLINE_EPOCH:raise ValueError('inherited cumulative caps/deadline exceeded')
        value={'passed':True,**collector_state(),'no_remaining_authorized_model_training_or_readout':True,'scientific_summary_sha256':sha(PUB/'summary.json'),'action_space_diagnostic_sha256':sha(PUB/'action_space_diagnostic.json'),'models_manifest_sha256':sha(PUB/'models_manifest.json'),'native_attribution_sha256':sha(PUB/'native_attribution.json'),'all_current_contracts_verified':names,'historical_source_versions':historical,'executed_engineering_source_versions':engineering,'source_and_all40_model_state_weight_files_verified':len(pins),'new_GPU_job_hours':actual['new_GPU_wall_seconds']/3600,'cumulative_GPU_job_hours':actual['cumulative_GPU_hours'],'new_summed_measured_CPU_receipt_hours_including_collector':(actual['new_measured_CPU_seconds']+cpu)/3600,'cumulative_summed_measured_CPU_receipt_hours':actual['cumulative_measured_CPU_hours']+cpu/3600,'prior_completed_CPU_allowance_hours':actual['prior_CPU_allowance_hours'],'new_CPU_allowance_hours':0,'cumulative_CPU_receipt_plus_accounting_allowances_hours':cpuaccount,'retained_tracked_paths_apparent_bytes':apparent,'retained_tracked_paths_allocated_bytes':allocated,'new_source_code_and_test_file_bytes':codebytes,'largest_atomic_checkpoint_bytes':largest,'new_tracked_storage_upper_bytes':storage,'prior_tracked_storage_upper_bytes':PRIOR_STORAGE_UPPER_BYTES,'cumulative_tracked_storage_upper_bytes':PRIOR_STORAGE_UPPER_BYTES+storage,'storage_scope':'conservative tracked-path component forecast/currentretained+atomic/reportmargin;not whole-filesystem peak','actual_totals':{k:v for k,v in actual.items() if k!='receipt_records'},'local_cpu_receipts_sha256':sha(PUB/'local_cpu_receipts.json'),'process_ledger_sha256':sha(RUN/'processes.jsonl'),'recorded_process_identity_checks':processes,'all_recorded_owned_model_and_supervisor_processes_absent':True,'GPU_compute_processes_absent':not bool(gpu),'foreign_GPU_processes_may_remain_and_are_not_A61_cost':gpu,'A61_GPU_contexts_absent':True,'pending_claims':0,'same_queue_lock_held_through_checks_and_writes':True,**status,'server_failed_or_terminated_stage_count':sum(r.get('observed_worker_exit_code') not in [0,None] or bool(r.get('termination_reason')) for r in records),'local_nonzero_software_check_receipts_including_expected_RED':sum(r['observed_worker_exit_code'] not in [0,None] for r in local['receipts']),'CPU_scope':'summed measured SELF+waitedCHILDREN+entrycomplements;inherited allowance/provenance/overlap and knownpre-entrysetup gaps retained;not unique wholehost CPU','known_unmetered_setup_metadata_gaps':read(PUB/'predeployment_metadata_failure.json'),'additional_unmetered_administrative_metadata_gaps':read(PUB/'late_administrative_extract_failure.json'),'prior_A60_admin_tail_and_all_review_receipts_kept_once':True,'all_inherited_caps_passed':True,'earlier_deadline_epoch':DEADLINE_EPOCH,'closure_epoch':now,'collector':{'measurement_host':socket.gethostname(),'model_pid':os.getpid(),'started_epoch':_START,'CPU_SELF_seconds':own.ru_utime+own.ru_stime,'CPU_waited_CHILDREN_seconds':children.ru_utime+children.ru_stime,'CPU_seconds_through_snapshot':cpu,'observed_worker_exit_code':None,'scope':'running SELF+waited probes;actual final OS from caller receipt;postsampleJSON/admin tail separate'},'root_final_integration_coordination_tail_separate':True}
        write(PUB/'resources.json',value);write(PUB/'terminal_exit_receipt.json',{'passed':True,**collector_state(),'all_recorded_owned_workers_absent':True,'A61_GPU_contexts_absent':True,'whole_GPU_idle':not bool(gpu),'pending_claims':0,'same_queue_lock_held':True,'worker_OS_exit_statuses':[{'stage':r['stage'],'attempt':r['attempt'],'model_pid':r.get('model_pid'),'observed_worker_exit_code':r.get('observed_worker_exit_code'),'child_exit_code_directly_waited':r.get('child_exit_code_directly_waited'),'OS_status_inferred_from_manifest':r.get('OS_status_inferred_from_manifest',False)} for r in records],**status,'resource_receipt_sha256':sha(PUB/'resources.json'),'epoch':now})
        write(RUN/'runtimecoordination.json',{**collector_state(),'model_pid':None,'formal_optimizer_updates':60000,'metric_rows':314,'paired_comparisons':240,'oracle_cells':60,'native_cells':56,'overall_root_integration_and_caller_OS_completion_still_required':True,'next_action':'root protected exactpublic integration and delete onlycapstone-a61;noautomaticA62','resource_receipt_sha256':sha(PUB/'resources.json'),'epoch':now})
        print(json.dumps({'server_closed':True,'overall_complete':False,'cumulative_GPU_job_hours':actual['cumulative_GPU_hours'],'cumulative_CPU_accounting_hours':cpuaccount,'cumulative_tracked_storage_upper_bytes':PRIOR_STORAGE_UPPER_BYTES+storage}),flush=True)
if __name__=='__main__':main()
