"""One foreground continuation child under the original shared kernel flock.

RuntimeState supplies independently verified raw inputs, live receipts, storage
and a freshly recomputed complete remaining forecast. This module never creates
a second queue or starts a daemon. Every created child is directly waited, even
when a cap or exceptional path fails; foreign processes are never signalled.
"""
import math
import os
import resource
import socket
import subprocess
import time
import uuid
from pathlib import Path
from echo_routing.temporal.selected_risk_resume_contract import stage_graph


def start_ticks(pid):
    try:return int(Path(f'/proc/{pid}/stat').read_text().rsplit(') ',1)[1].split()[19])
    except (FileNotFoundError,ProcessLookupError):return None


def _stop_own(child,ticks):
    if child.poll() is None:
        if ticks is None or start_ticks(child.pid)!=ticks:raise RuntimeError('owned child PID/start identity lost; never signal a reused PID')
        child.terminate()
        try:return child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            if start_ticks(child.pid)!=ticks:raise RuntimeError('owned child identity changed before kill')
            child.kill()
    return child.wait()


def execute(state,stage):
    if stage not in stage_graph():raise ValueError('exact continuation stage basename required before queue/log writes')
    state.authorize();nonce=uuid.uuid4().hex;start=time.time();cpu0=state.usage();child=None;code=None;reason=None;error=None;intent=None;ticks=None;live=None;maxstorage=None;bounds=None;waited=False;final_unknown=False
    gpu=stage in ['source_FIT','source_CONFIRM']
    # The existing queue path is opened without O_CREAT by RuntimeState. All
    # live inventory/admission/intent operations occur under this one lock.
    with state.queue() as lock_fd:
        try:
            state.authorize();live=state.inventory();bounds=state.admission();maxstorage=state.storage()
            if bounds.get('all_caps_fit') is not True or maxstorage>bounds['actual_storage_cap'] or gpu and live['GPU_owners']:raise ValueError('fresh complete-work caps/current storage/GPU ownership do not admit this stage')
            for key in ['stage_CPU_limit','stage_wall_limit']:
                if isinstance(bounds.get(key),bool) or not isinstance(bounds.get(key),(int,float)) or not math.isfinite(bounds[key]) or bounds[key]<=0:raise ValueError('finite positive separately admitted stage limits required')
            intent=state.claim(stage,nonce,live)
            command=state.command(stage,intent,lock_fd)
            def child_limit():
                limit=max(1,math.ceil(bounds['stage_CPU_limit']));resource.setrlimit(resource.RLIMIT_CPU,(limit,limit+1))
            with state.logs(stage,nonce) as (stdout,stderr):
                child=subprocess.Popen(command,stdout=stdout,stderr=stderr,pass_fds=(lock_fd,),preexec_fn=child_limit,**state.popen_options(stage) if hasattr(state,'popen_options') else {})
                ticks=start_ticks(child.pid)
                while child.poll() is None:
                    elapsed=time.time()-start;cpu=state.usage()-cpu0
                    maxstorage=max(maxstorage,state.storage());state.authorize()
                    if elapsed>bounds['stage_wall_limit'] or cpu>bounds['stage_CPU_limit'] or maxstorage>bounds['actual_storage_cap']:
                        reason='actual_cap_reached';code=_stop_own(child,ticks);waited=True;break
                    time.sleep(.05)
                if not waited:code=child.wait();waited=True
        except BaseException as exc:
            error=type(exc).__name__+': '+str(exc);reason=reason or ('prelaunch_refused' if child is None else 'guardian_exception')
            if child is not None and not waited:
                try:code=_stop_own(child,ticks);waited=True
                except BaseException as stop_error:
                    error+='; '+type(stop_error).__name__+': '+str(stop_error)
                    # Popen.wait remains mandatory even if signalling was refused.
                    code=child.wait();waited=True
        finally:
            end=time.time();cpu=max(0.,state.usage()-cpu0)
            if maxstorage is not None:
                try:maxstorage=max(maxstorage,state.storage())
                except BaseException as exc:error=(error or '')+'; final storage unavailable: '+type(exc).__name__;reason=reason or 'final_resource_unknown';final_unknown=True
            if child is not None and bounds is not None and (cpu>bounds['stage_CPU_limit'] or end-start>bounds['stage_wall_limit'] or maxstorage is not None and maxstorage>bounds['actual_storage_cap']):reason='actual_final_cap_exceeded'
            if child is not None and ticks is None:reason=reason or 'owned_child_start_unobserved'
            valid=code==0 and reason is None and waited
            row={'receipt_id':'resume:'+stage+':'+nonce,'stage':stage,'nonce':nonce,'measurement_host':socket.gethostname(),'boot_id':None if live is None else live['boot_id'],'supervisor_pid':os.getpid(),'supervisor_start_ticks':None if live is None else live['guardian_start_ticks'],'model_pid':None if child is None else child.pid,'model_start_ticks':ticks,'started_epoch':start,'finished_epoch':end,'full_process_wall_seconds':end-start,'measured_CPU_core_seconds':cpu,'CPU_measurement_scope':'guardian SELF plus all directly waited children in measured region; startup/write/exit complement separate','GPU_job':bool(gpu and child is not None),'CUDA_context_occupied':bool(gpu and child is not None),'GPU_charge_scope':'one full owned source-job guardian interval; startup/exit included conservatively once','observed_worker_exit_code':code,'actual_child_directly_waited':waited,'SELF_OS_exit_code':None,'termination_reason':reason,'guardian_error':error,'stage_exit_status':0 if valid else 1,'sampled_tracked_storage_upper_bytes':maxstorage,'initial_GPU_owners':None if live is None else live['GPU_owners'],'current_remote_caps_unknown':final_unknown,'overall_complete':False}
            return state.record(row,intent)
