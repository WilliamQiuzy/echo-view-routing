#!/usr/bin/env python3
"""Attach a resource-only GPU-context guard to the existing CPU formal worker."""
import os,sys,time,json,resource,signal,fcntl
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.structured_common import RUN,PUB,CFG,read,write,sha,verify_freeze
from scripts.supervise_structured_evidence import remaining_limits
from echo_routing.temporal.structured_context import classify_context_records
from echo_routing.temporal.structured_guard import process_sample
from echo_routing.temporal.structured_budget import caps_fit

def main():
    freeze=verify_freeze('context_guard_freeze.json');claim=read(RUN/'active_process.json');assert claim['stage']=='formal' and claim['model_pid']==317679 and claim['supervisor_pid']==317671;pid=claim['model_pid'];sample=process_sample(pid);assert sample and Path(f"/proc/{claim['supervisor_pid']}").exists();ticks=sample['start_ticks'];raw=[json.loads(s) for s in (RUN/'processes.jsonl').read_text().splitlines()];rows=classify_context_records(raw);budget=read(CFG)['budget'];limits=remaining_limits(budget,[{**r,'process_CPU_seconds':r['process_CPU_seconds']+r.get('unmeasured_completed_CPU_accounting_allowance_seconds',0)} for r in rows],claim['start_epoch'],True)
    with (RUN/'exclusive_supervisor.lock').open('a+') as lock:
        try:fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:pass
        else:fcntl.flock(lock.fileno(),fcntl.LOCK_UN);raise RuntimeError('primary supervisor does not hold expected exclusive queue')
    base=read(PUB/'admission.json');prior=read(PUB/'formal_runtime_admission.json');formal_upper=prior['head_training_upper_seconds_with_full_journal_digest_atomic_overhead']+base['head_MAP_seconds_upper']+base['atomic_checkpoint_seconds_upper']+600;head_check_wall=sum(r['full_process_wall_seconds'] for r in raw if r['stage']=='head_check');gpu=prior['forecast_new_GPU_hours_including2hreserve']+(formal_upper+head_check_wall)/3600;cpu=prior['forecast_new_CPU_receipt_hours_plus_completed_allowances']+120/3600;end=prior['forecast_end_epoch']+120;passed=caps_fit(budget,gpu,cpu,prior['forecast_new_tracked_peak_upper_bytes'],end);assert passed;write(PUB/'formal_context_GPU_admission.json',{'all_caps_fit':passed,'freeze_git':freeze['git_commit'],'original_runtime_admission_sha256':sha(PUB/'formal_runtime_admission.json'),'model_backend_unchanged':'cpu','model_and_optimizer_moments_verified_CPU_at_update1500':True,'formal_GPU_context_observed_MiB':426,'cause':'installedAdamWacceleratorgraphcapturehealthcheckqueriescurrentCUDAstream','GPU_charge':'conservativefullhead_checkandformalprocessintervals;rawCPUbackendreceiptsretained;noextraGPUkernelclaim','head_check_GPU_charge_seconds_conservative_same_code_path':head_check_wall,'full_formal_context_GPU_seconds_upper':formal_upper,'new_GPU_forecast_hours_including2hreserve':gpu,'cumulative_GPU_forecast_hours_includingreserve':budget['prior_GPUjob_h']+gpu,'cumulative_CPU_receipt_plus_allowances_forecast_hours':budget['prior_summed_measured_CPUreceipt_h']+budget['prior_setup_budget_allowance_h']+cpu,'forecast_cumulative_tracked_peak_upper_bytes':prior['forecast_cumulative_tracked_peak_upper_bytes'],'forecast_end_epoch':end,'extra_guard_CPU_reserve_seconds_not_actual_cost':120,'CPUandGPUoccupationare_same_job_not_additional_wall_stage':True,'no_model_recipe_data_seed_or_checkpoint_change':True,'epoch':time.time()});adopt=time.time();write(RUN/'context_guard_claim.json',{'guardian_model_pid':os.getpid(),'formal_model_pid':pid,'formal_start_epoch':claim['start_epoch'],'formal_start_ticks':ticks,'primary_supervisor_pid':claim['supervisor_pid'],'adopted_epoch':adopt,'same_primary_exclusive_queue_owner_verified':True,'second_execution_queue':False});reason=None;lastwrite=0
    while True:
        now=time.time();sample=process_sample(pid)
        if sample is None or sample['state']=='Z':break
        if sample['start_ticks']!=ticks:raise RuntimeError('formal PID reused')
        if now-claim['start_epoch']>=min(limits['GPU_seconds']-10,limits['wall_seconds']-10):reason='context_GPU_or_inherited_deadline_limit'
        if not Path(f"/proc/{claim['supervisor_pid']}").exists():reason='primary_supervisor_lost'
        if now-lastwrite>=10:write(RUN/'context_guard_heartbeat.json',{'guardian_model_pid':os.getpid(),'formal_model_pid':pid,'formal_start_ticks':ticks,'primary_supervisor_pid':claim['supervisor_pid'],'full_formal_GPU_occupied_elapsed_seconds':now-claim['start_epoch'],'remaining_original_GPU_bound_seconds':limits['GPU_seconds']-(now-claim['start_epoch']),'epoch':now,'model_updates_or_forwards_by_guard':0});lastwrite=now
        if reason:os.kill(pid,signal.SIGTERM);break
        time.sleep(.5)
    for _ in range(60):
        if (RUN/'process_formal_1.json').exists():break
        time.sleep(.5)
    row=read(RUN/'process_formal_1.json');assert row['model_pid']==pid and row['start_epoch']==claim['start_epoch'];own=resource.getrusage(resource.RUSAGE_SELF);children=resource.getrusage(resource.RUSAGE_CHILDREN);receipt={'passed':reason is None and row['exit_code']==0 and row['termination_reason'] is None,'guardian_model_pid':os.getpid(),'measurement_host':claim['measurement_host'],'formal_model_pid':pid,'formal_start_epoch':claim['start_epoch'],'formal_start_ticks':ticks,'primary_supervisor_pid':claim['supervisor_pid'],'adopted_epoch':adopt,'end_epoch':time.time(),'formal_GPU_occupied_full_process_seconds':row['full_process_wall_seconds'],'raw_formal_receipt_sha256':sha(RUN/'process_formal_1.json'),'process_CPU_seconds':own.ru_utime+own.ru_stime+children.ru_utime+children.ru_stime,'CPU_SELF_seconds':own.ru_utime+own.ru_stime,'CPU_waited_CHILDREN_seconds':children.ru_utime+children.ru_stime,'CPU_scope':'fullattachedresourceguardSELFpluswaitedchildren;noformalCPUduplication;postreceiptbookkeepingunmetered','observed_formal_worker_exit_code':row['exit_code'],'observed_guardian_OS_exit_code':None,'termination_reason':reason,'no_worker_restart_or_model_update_by_guard':True,'same_primary_queue_owner_preserved':True,'GPU_context_charge_overlay_not_second_GPU_interval':True};write(RUN/'context_guard_receipt.json',receipt);write(PUB/'context_guard_receipt.json',receipt);print(json.dumps(receipt),flush=True);assert receipt['passed']
if __name__=='__main__':main()
