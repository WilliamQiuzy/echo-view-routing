#!/usr/bin/env python3
"""Adopt the existing immutable cache PID after a disclosed du/ENOENT guard failure."""
import os,sys,time,json,socket,resource,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.structured_common import RUN,PUB,CFG,read,write,sha,verify_freeze
from scripts.supervise_structured_evidence import remaining_limits
from echo_routing.temporal.dense_resource_accounting import terminal_queue_lock
from echo_routing.temporal.structured_guard import process_sample,storage_bytes

def main():
    freeze=verify_freeze('adoption_freeze.json')
    with terminal_queue_lock(RUN/'exclusive_supervisor.lock'):
        original=read(RUN/'active_process.json');pid=original['model_pid'];assert original['stage']=='cache_fit' and original['GPU_job'];assert not Path(f"/proc/{original['supervisor_pid']}").exists();sample=process_sample(pid);assert sample is not None and sample['state']!='Z';start_ticks=sample['start_ticks'];cmd=Path(f'/proc/{pid}/cmdline').read_bytes();assert b'produce_structured_cache.py\x00fit' in cmd
        own0=resource.getrusage(resource.RUSAGE_SELF);child0=resource.getrusage(resource.RUSAGE_CHILDREN);adopted=time.time();budget=read(CFG)['budget'];rows=[json.loads(s) for s in (RUN/'processes.jsonl').read_text().splitlines()];limits=remaining_limits(budget,rows,original['start_epoch'],True);reason=None;last_disk=0;maxthreads=sample['threads'];last=sample;proof={'original_claim':original,'guardian_pid':os.getpid(),'adopted_epoch':adopted,'worker_start_ticks':start_ticks,'measurement_host':socket.gethostname(),'frozen_git':freeze['git_commit'],'failure':'GNUduENOENTonatomicallyremovedpendingfile','source_producer_unchanged':True,'no_worker_restart':True,'CPU_accounting':'worker/proc samples plus guardian SELF and waited guards; original dead supervisor CPU unavailable, separate completed accounting allowance'};write(RUN/'cache_guard_adoption.json',proof)
        while True:
            now=time.time();sample=process_sample(pid)
            if sample is None or sample['state']=='Z':break
            if sample['start_ticks']!=start_ticks:raise RuntimeError('adopted PID identity changed')
            last=sample;maxthreads=max(maxthreads,sample['threads']);own=resource.getrusage(resource.RUSAGE_SELF);children=resource.getrusage(resource.RUSAGE_CHILDREN);guardianCPU=own.ru_utime+own.ru_stime-own0.ru_utime-own0.ru_stime+children.ru_utime+children.ru_stime-child0.ru_utime-child0.ru_stime
            if now-original['start_epoch']>=min(28800,limits['GPU_seconds']-10,limits['wall_seconds']-10):reason='adopted_full_job_GPU_or_deadline_limit'
            if sample['CPU_seconds']+guardianCPU+900>=limits['CPU_seconds']-10:reason='adopted_CPU_limit_including_disclosed_uncertainty_allowance'
            if now-last_disk>=10:
                try:storage=storage_bytes([RUN,PUB])
                except Exception as error:reason='storage_guard_fail_closed:'+repr(error)
                else:
                    if storage>=min(budget['new_peak_bytes'],budget['cumulative_peak_upper_bytes']-budget['prior_tracked_storage_upper_bytes']):reason='adopted_storage_upper_limit'
                last_disk=now;write(RUN/'cache_guard_heartbeat.json',{'guardian_pid':os.getpid(),'worker_pid':pid,'worker_start_ticks':start_ticks,'original_start_epoch':original['start_epoch'],'epoch':now,'elapsed_full_GPU_job_seconds':now-original['start_epoch'],'sampled_worker_CPU_seconds':sample['CPU_seconds'],'guardian_CPU_seconds':guardianCPU,'storage_guard_active':True,'remaining_original_GPU_bound_seconds':limits['GPU_seconds']-(now-original['start_epoch'])})
            if reason:
                os.kill(pid,signal.SIGTERM);time.sleep(2)
                live=process_sample(pid)
                if live and live['state']!='Z' and live['start_ticks']==start_ticks:os.kill(pid,signal.SIGKILL)
                break
            time.sleep(.2)
        end=time.time();own=resource.getrusage(resource.RUSAGE_SELF);children=resource.getrusage(resource.RUSAGE_CHILDREN);guardianCPU=own.ru_utime+own.ru_stime-own0.ru_utime-own0.ru_stime+children.ru_utime+children.ru_stime-child0.ru_utime-child0.ru_stime;success=reason is None and (PUB/'cache_fit_manifest.json').exists() and read(PUB/'cache_fit_manifest.json')['passed'];record={**original,'stage':'cache_fit','attempt':1,'guardian_pid':os.getpid(),'guardian_adoption_epoch':adopted,'worker_start_ticks':start_ticks,'end_epoch':end,'full_process_wall_seconds':end-original['start_epoch'],'process_CPU_seconds':last['CPU_seconds']+guardianCPU,'worker_sampled_CPU_seconds':last['CPU_seconds'],'guardian_measured_CPU_seconds':guardianCPU,'CPU_scope':'sampled_worker_lifetime_plus_measured_guardian_SELF/CHILDREN; lost_original_supervisor_and_lastpolltail_not_in_measured_sum','unmeasured_completed_CPU_accounting_allowance_seconds':900,'CPU_tail_max_observed_threads':maxthreads,'CPU_sampling_interval_seconds':.2,'prior_supervisor_CPU_provenance_missing':True,'GPU_job':True,'exit_code':0 if success else None,'child_exit_code_directly_waited':False,'completion_verified_by_frozen_complete_cache_manifest':success,'timed_out':reason is not None,'termination_reason':reason,'no_restarted_or_replayed_worker':True,'adoption_receipt_sha256':sha(RUN/'cache_guard_adoption.json')};write(RUN/'process_cache_fit_1.json',record)
        with (RUN/'processes.jsonl').open('a') as f:f.write(json.dumps(record)+'\n');f.flush();os.fsync(f.fileno())
        (RUN/'active_process.json').unlink();write(PUB/'cache_guard_adoption.json',{**proof,'private_claim_paths_exported':False,'final_receipt':record});print(json.dumps(record),flush=True);assert success
if __name__=='__main__':main()
