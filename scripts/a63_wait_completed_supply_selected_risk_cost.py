#!/usr/bin/env python3
"""One original-queue, CPU-only cost attempt; directly wait the full process tree."""
import os,time
ENTRY_EPOCH=time.time();ENTRY_MONOTONIC=time.monotonic()
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import signal
import socket
import stat
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from echo_routing.temporal.selected_risk_resume_budget import forecast
from echo_routing.temporal.selected_risk_resume_contract import RUN
from scripts.a63_probe_completed_supply_selected_risk_cost import SOURCE,WAITER,LIMITS,PURPOSE,pin,setup,_write,files,file_manifest,cleanup_owned,cuda_observation,CUDA_MARKER


def probe_admission(spent,current,now,owner):
 return forecast(spent,0,600+3600,current+80000000+128000000,now,600+7200,owner,closing=False)


def ticks(pid):
 try:
  s=Path(f'/proc/{pid}/stat').read_text();return int(s[s.rfind(')')+2:].split()[19])
 except (OSError,ValueError,IndexError):return None


def boot():return Path('/proc/sys/kernel/random/boot_id').read_text().strip()


def validate_lock(ctx,lock_fd,current_spec_pin):
 from echo_routing.temporal.selected_risk_resume_runtime import SHARED_QUEUE
 if sys.platform!='linux' or type(lock_fd) is not int:raise ValueError('actual server/inherited original cost queue lock required')
 a=os.fstat(lock_fd);b=os.stat(ctx.root/SHARED_QUEUE,follow_symlinks=False)
 if not stat.S_ISREG(a.st_mode) or (a.st_dev,a.st_ino)!=(b.st_dev,b.st_ino):raise ValueError('same original ordinary shared queue required')
 fcntl.flock(lock_fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
 ip={'path':os.environ.get('A63_COST_PROBE_INTENT_PATH'),'sha256':os.environ.get('A63_COST_PROBE_INTENT_SHA')};intent=ctx.read(ip)
 current_spec=ctx.read(current_spec_pin)
 if intent.get('probe_spec_pin')!=current_spec_pin or any(intent.get(k)!=current_spec.get(k) for k in ['source_runtime_spec_pin','source_Git_contract_pin','purpose','limits']):raise ValueError('exact current probe spec/source/contract/limits intent required')
 if intent['runtime_authorization_sha256']!=ctx.owner_sha or intent['boot_id']!=boot() or ticks(intent['parent_pid'])!=intent['parent_start_ticks']:raise ValueError('current actual sameowner cost-parent intent required')
 p=os.getpid();found=False
 for _ in range(5):
  if p==intent['parent_pid']:found=True;break
  try:s=Path(f'/proc/{p}/stat').read_text();p=int(s[s.rfind(')')+2:].split()[1])
  except (OSError,ValueError,IndexError):break
 if not found:raise ValueError('inherited lock may only serve the actual owned cost process tree')
 return intent


def inventory(ctx,state,source):
 from scripts.a63_wait_resume_selected_risk import _no_unclosed_callers
 if boot()!=source['expected_boot_id']:raise ValueError('fresh boot differs from source/owner observation')
 old=ctx.root/'runs/a63_selected_action_risk_v1/active_process.json'
 if old.exists() or old.is_symlink():raise ValueError('old active claim requires independent reconciliation')
 _no_unclosed_callers(ctx);rows=state._worker_rows();closed={r.get('nonce'):r for r in rows if r.get('nonce')}
 for p in (ctx.root/RUN/'launches').glob('*.intent.json'):
  if p.is_symlink():raise ValueError('owned launch symlink')
  raw=p.read_bytes();v=json.loads(raw);row=closed.get(v['nonce'])
  if row is None or row.get('source_intent_sha256')!=hashlib.sha256(raw).hexdigest():raise ValueError('unclosed original launch claim')
  if row.get('model_pid') is not None and ticks(row['model_pid'])==row.get('model_start_ticks'):raise ValueError('old owned worker still live')
 for p in (ctx.root/RUN/'cost_only_synthetic_cache_probe').glob('*/outer_intent.json'):
  if p.is_symlink() or not p.with_name('outer_receipt.json').is_file():raise ValueError('previous cost-only attempt unclosed; no replay')
 result=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],check=True,capture_output=True,text=True,timeout=5)
 return {'GPU_owners':[int(x.strip()) for x in result.stdout.splitlines() if x.strip()],'boot_id':boot(),'claims_closed':True}


def _tree_cpu(pid):
 """Only owned descendants: live SELF plus CPU of children already waited."""
 try:
  s=Path(f'/proc/{pid}/stat').read_text();x=s[s.rfind(')')+2:].split();hz=os.sysconf('SC_CLK_TCK');total=sum(int(x[k]) for k in [11,12,13,14])/hz
  children=Path(f'/proc/{pid}/task/{pid}/children').read_text().split()
 except (FileNotFoundError,ProcessLookupError):return 0.
 except (OSError,ValueError,IndexError) as exc:raise ValueError('owned CPU-stat observation unavailable') from exc
 return total+sum(_tree_cpu(int(p)) for p in children)


def stop_owned(child,start,outer):
    """Stop only the still-matching direct child/group, then reap it once.

    Termination can strand unwaited grandchildren: returned CPU is a lower
    bound and never advertised as a complete measurement for future credit.
    """
    leases={child.pid:start}
    def descendants(pid):
        if leases.get(pid) is None or ticks(pid)!=leases[pid]:return
        try:children=Path(f'/proc/{pid}/task/{pid}/children').read_text().split()
        except OSError:return
        for p in children:
            if ticks(pid)!=leases[pid]:return
            p=int(p)
            try:
                raw=Path(f'/proc/{p}/stat').read_text();parts=raw[raw.rfind(')')+2:].split()
                if int(parts[1])!=pid:continue
                identity=int(parts[19])
            except (OSError,ValueError,IndexError):continue
            leases[p]=identity;descendants(p)
    descendants(child.pid)
    if start is not None and ticks(child.pid)==start:
        try:
            if outer and os.getpgid(child.pid)==child.pid:os.killpg(child.pid,signal.SIGTERM)
            else:os.kill(child.pid,signal.SIGTERM)
        except ProcessLookupError:pass
    code=usage=None;until=time.monotonic()+2.
    while time.monotonic()<until:
        try:
            p,status,u=os.wait4(child.pid,os.WNOHANG)
            if p:code=os.waitstatus_to_exitcode(status);usage=u;break
        except InterruptedError:continue
        except ChildProcessError:break
        time.sleep(.05)
    # Even after the original leader exits, only captured PID/start leases are
    # signalled; never send to a recycled or foreign process-group number.
    for pid,identity in sorted(leases.items(),reverse=True):
        if identity is not None and ticks(pid)==identity:
            try:os.kill(pid,signal.SIGKILL)
            except ProcessLookupError:pass
    if usage is None:
        while True:
            try:p,status,usage=os.wait4(child.pid,0);code=os.waitstatus_to_exitcode(status);break
            except InterruptedError:continue
            except ChildProcessError:break
    child.returncode=code
    return code,usage,True


def sample_owned_bytes(root):
    """Monitor only: tolerate same-owner cleanup, never other IO errors.

    Strict manifest/cleanup traversal remains unchanged. Every enumerated
    entry is checked with one lstat; symlinks and special files are refused.
    """
    root=Path(root);total=0
    try:mode=root.lstat().st_mode
    except FileNotFoundError:return 0
    if not stat.S_ISDIR(mode):raise ValueError('ordinary owned cost directory required')
    def error(exc):
        if not isinstance(exc,FileNotFoundError):raise exc
    for parent,dirs,names in os.walk(root,followlinks=False,onerror=error):
        for name in dirs+names:
            p=Path(parent)/name
            try:mode=p.lstat()
            except FileNotFoundError:continue
            if stat.S_ISREG(mode.st_mode):total+=mode.st_size
            elif not stat.S_ISDIR(mode.st_mode):raise ValueError('owned monitor symlink/special file refused')
    return total


def run_process(ctx,spec,command,name,lock_fd,*,outer=False):
 base=ctx.output(spec['output_prefix'][len(RUN)+1:]+'/'+name+'.stdout');base.parent.mkdir(parents=True,exist_ok=True)
 err=base.with_suffix('.stderr');started=time.time();peak=0;reason=None;child=None;start=None;usage=None;code=None;uncertain=False
 with base.open('xb') as out,err.open('xb') as errors:
  try:
   child=subprocess.Popen(command,stdout=out,stderr=errors,pass_fds=(lock_fd,),start_new_session=outer,env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1'})
   start=ticks(child.pid)
   while True:
    pid,status,used=os.wait4(child.pid,os.WNOHANG)
    if pid:usage=used;code=os.waitstatus_to_exitcode(status);child.returncode=code;break
    if outer:
     temp=sample_owned_bytes(ctx.root/spec['output_prefix']);peak=max(peak,temp)
     own=resource.getrusage(resource.RUSAGE_SELF);before=resource.getrusage(resource.RUSAGE_CHILDREN);cpu=own.ru_utime+own.ru_stime+before.ru_utime+before.ru_stime+_tree_cpu(child.pid)
     if temp>LIMITS['temp_bytes'] or cpu+1>LIMITS['CPU_seconds'] or time.monotonic()-ENTRY_MONOTONIC>LIMITS['wall_seconds']:raise ValueError('cost_only_actual_limit_exceeded')
    time.sleep(.1)
  except BaseException as exc:
   reason=type(exc).__name__+': '+str(exc);uncertain=True
   if child is not None:code,usage,_=stop_owned(child,start,outer)
 ended=time.time();lower=None if usage is None else usage.ru_utime+usage.ru_stime
 observations=[]
 for line in err.read_text(errors='replace').splitlines():
  if line.startswith(CUDA_MARKER):
   try:observations.append(json.loads(line[len(CUDA_MARKER):]))
   except ValueError:observations.append({'known':False,'initialized':None})
 gpu_bad=any(not o.get('known') or o.get('initialized') is not False for o in observations)
 if gpu_bad:
  print(CUDA_MARKER+json.dumps({'known':all(o.get('known') for o in observations),'initialized':True if any(o.get('initialized') for o in observations) else None}),file=sys.stderr,flush=True)
  uncertain=True;reason=reason or 'unexpected_or_unknown_CUDA_context'
 # Abnormal coordinator exit cannot certify that every descendant was waited.
 if outer and code!=0:uncertain=True
 complete=usage is not None and code is not None and start is not None and not uncertain
 row={'pid':None if child is None else child.pid,'start_ticks':start,'boot_id':boot(),'actual_OS_exit_code':code,'actual_directly_waited':usage is not None,'CPU_core_seconds':lower if complete else None,'known_partial_CPU_core_seconds':lower,'wall_seconds':ended-started,'full_accounting_complete':complete,'raw_stdout_pin':pin(ctx.root,base),'raw_stderr_pin':pin(ctx.root,err),'termination_reason':reason,'unresolved_worker_uncertainty':uncertain,'GPU_context_unknown':gpu_bad or uncertain,'CUDA_context_observed':any(o.get('initialized') is True for o in observations),'sampled_temp_peak_bytes':peak,'sampled_temp_peak_scope':'maximum sampled logical regular bytes in declared synthetic attempt, not physical/global/RAM peak; <=.1s sampling'}
 rp=_write(ctx,spec,name+'_process.json',row)
 return {**row,'raw_receipt_pin':rp}


def execute(ctx,source,spec,binding,spec_pin):
 from echo_routing.temporal.selected_risk_resume_runtime import RuntimeState
 from echo_routing.temporal.selected_risk_resume_contract import digest
 state=RuntimeState(ctx,source,spec['source_runtime_spec_pin'])
 # Only canonical fee inputs are supplemented; the raw scientific spec remains
 # immutable and no scientific command/admission is called by this wrapper.
 if spec.get('additional_administrative_fee_receipts'):
  state.spec={**source,'arguments':{**source['arguments'],'additional_administrative_fee_receipts':spec['additional_administrative_fee_receipts']}}
 with state.queue() as lock_fd:
  live=inventory(ctx,state,source);spent=state.spent();current=state.storage()
  bounds=probe_admission(spent,current,time.time(),ctx.owner)
  admission=_write(ctx,spec,'cost_only_admission.json',bounds)
  if not bounds['all_caps_fit']:raise ValueError('cost-only600/600/80MB plus original closing/cumulative caps refuse measurement')
  intent={'schema':'A63_cost_only_probe_intent_v1','parent_pid':os.getpid(),'parent_start_ticks':ticks(os.getpid()),'boot_id':boot(),'runtime_authorization_sha256':ctx.owner_sha,'probe_spec_pin':spec_pin,'source_runtime_spec_pin':spec['source_runtime_spec_pin'],'source_Git_contract_pin':spec['source_Git_contract_pin'],'purpose':PURPOSE,'limits':LIMITS,'fresh_inventory':live,'started_epoch':ENTRY_EPOCH}
  if intent['parent_start_ticks'] is None:raise ValueError('actual parent start identity unavailable')
  ip=_write(ctx,spec,'outer_intent.json',intent);os.environ['A63_COST_PROBE_INTENT_PATH']=ip['path'];os.environ['A63_COST_PROBE_INTENT_SHA']=ip['sha256']
  command=[sys.executable,str(ctx.root/SOURCE),'--execute','--mode','parent','--probe-spec',spec_pin['path'],'--probe-spec-sha',spec_pin['sha256'],'--owner-sha',ctx.owner_sha,'--lock-fd',str(lock_fd)]
  t=ENTRY_EPOCH;cuda_observation();prechildren=resource.getrusage(resource.RUSAGE_CHILDREN);pre_cpu=prechildren.ru_utime+prechildren.ru_stime;row=run_process(ctx,spec,command,'cost_parent',lock_fd,outer=True);payload_pin=None;error=None
  try:
   output=json.loads(ctx.raw(row['raw_stdout_pin']))
   if set(output)!={'passed','payload_pin'} or output['passed'] is not True:raise ValueError('unique successful parent stdout required')
   payload_pin=output['payload_pin'];payload=ctx.read(payload_pin)
   if payload.get('passed') is not True or payload.get('probe_spec_pin')!=spec_pin:raise ValueError('cost-parent report raw pin/authority differs')
  except (ValueError,KeyError) as exc:error=str(exc)
  # Failure-only cleanup is restricted to the predeclared synthetic file list.
  # Unknown files are never removed, and failure always remains failure.
  if error is not None:
   planfile=ctx.root/spec['output_prefix']/'owned_plan.json';folder=ctx.root/spec['output_prefix']/'payloads'
   if planfile.exists() and folder.exists():
    plan=json.loads(planfile.read_bytes());observed=file_manifest(folder)
    if {r['path'] for r in observed}<=set(plan['allowed_files']):
     fp=_write(ctx,spec,'failed_deleted_file_manifest.private.json',observed);clean=cleanup_owned(folder,observed);_write(ctx,spec,'failed_cleanup.json',{**clean,'deleted_file_manifest_pin':fp})
  ended=time.time();cuda_observation();selfusage=resource.getrusage(resource.RUSAGE_SELF);measured=None if row['CPU_core_seconds'] is None else selfusage.ru_utime+selfusage.ru_stime+pre_cpu+row['CPU_core_seconds'];good=row['actual_OS_exit_code']==0 and row['full_accounting_complete'] and error is None and row['termination_reason'] is None
  total=ended-t
  if measured is None or measured+1>600 or total>600:good=False;error=error or 'cost_only_actual_CPU_or_wall_cap'
  receipt={'schema':'A63_cost_probe_outer_wait4_receipt_v1','receipt_id':'resume_cost_probe:'+spec['nonce'],'nonce':spec['nonce'],'stage':'cost_only_cache_probe','runtime_authorization_sha256':ctx.owner_sha,'probe_spec_pin':spec_pin,'source_runtime_spec_pin':spec['source_runtime_spec_pin'],'source_Git_contract_pin':spec['source_Git_contract_pin'],'cost_parent_pid':row['pid'],'cost_parent_start_ticks':row['start_ticks'],'boot_id':boot(),'actual_child_OS_exit_code':row['actual_OS_exit_code'],'actual_child_directly_waited':row['actual_directly_waited'],'whole_child_accounting_complete':row['full_accounting_complete'],'CPU_parent_SELF_through_sample_seconds':selfusage.ru_utime+selfusage.ru_stime,'CPU_wait4_child_and_descendants_seconds':row['CPU_core_seconds'],'CPU_prelaunch_waited_children_seconds':pre_cpu,'measured_CPU_core_seconds':measured,'SELF_OS_exit_code':None,'entry_wall_scope':'time frozen immediately after os/time imports before other stdlib/project/bootstrap imports; remaining interpreter prologue wall unmeasured; absolute SELF CPU includes it','caller_exit_CPU_allowance_seconds':1.,'successful_stdout_pin':row['raw_stdout_pin'],'raw_stderr_pin':row['raw_stderr_pin'],'payload_pin':payload_pin,'cost_only_admission_pin':admission,'fresh_inventory':live,'started_epoch':t,'finished_epoch':ended,'full_process_wall_seconds':total,'sampled_temp_peak_bytes':row['sampled_temp_peak_bytes'],'actual_temp_peak_upper_bytes':80000000,'temp_peak_scope':row['sampled_temp_peak_scope'],'passed':good,'stage_exit_status':0 if good else (row['actual_OS_exit_code'] or 1),'observed_worker_exit_code':0 if good else (row['actual_OS_exit_code'] or 1),'termination_reason':None if good else error or row['termination_reason'] or 'cost_probe_failed','measurement_kind':'cost_probe_outer_SELF_plus_authoritative_wait4','measurement_host':socket.gethostname(),'model_pid':row['pid'],'model_start_ticks':row['start_ticks'],'model_instances':0,'GPU_job':bool(row['GPU_context_unknown'] or row['CUDA_context_observed']),'CUDA_context_occupied':bool(row['GPU_context_unknown'] or row['CUDA_context_observed']),'GPU_interval_unknown':bool(row['GPU_context_unknown']),'known_partial_CPU_core_seconds':selfusage.ru_utime+selfusage.ru_stime+pre_cpu+(row['known_partial_CPU_core_seconds'] or 0.),'cuda_context_observation':cuda_observation(),'NN_calls':0,'SGD_updates':0,'unresolved_worker_uncertainty':not row['full_accounting_complete'],'covered_charge_groups':[r['path'] for r in [row['raw_receipt_pin']]],'additional_administrative_fee_receipts':state.spec['arguments'].get('additional_administrative_fee_receipts',[])}
  if good:
   receipt['prepare_process_raw_receipt_pin']=payload['prepare_process']['raw_receipt_pin']
   receipt['three_fresh_verifier_raw_receipt_pins']=[r['raw_receipt_pin'] for r in payload['three_fresh_verifier_processes']]
  rp=_write(ctx,spec,'outer_receipt.json',receipt)
  # Same canonical append-only ledger, one whole attempt. Nested verifier
  # records are informational and must never be charged again.
  fd,name=ctx._parent(RUN+'/processes.jsonl',create=True)
  try:
   stream=os.open(name,os.O_WRONLY|os.O_APPEND|os.O_CREAT|os.O_NOFOLLOW,0o600,dir_fd=fd)
   with os.fdopen(stream,'ab') as f:f.write((json.dumps({'receipt_id':receipt['receipt_id'],'receipt':rp},sort_keys=True)+'\n').encode());f.flush();os.fsync(f.fileno())
   os.fsync(fd)
  finally:os.close(fd)
  return {'parent_pin':rp,'payload_pin':payload_pin,'actual_child_OS_exit_code':row['actual_OS_exit_code'],'passed':good}


def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execute',action='store_true');p.add_argument('--probe-spec');p.add_argument('--probe-spec-sha');p.add_argument('--owner-sha');a=p.parse_args(argv)
 if not a.execute:print(json.dumps({'preparation_only':True,'new_process_or_queue':False,'NN':0,'SGD':0}));return 0
 if not all([a.probe_spec,a.probe_spec_sha,a.owner_sha]):p.error('independent exact cost-spec and raw owner pins required')
 if sys.platform!='linux':raise ValueError('real server only; local preparation never launches measurement')
 cuda_observation();spec_pin={'path':a.probe_spec,'sha256':a.probe_spec_sha};ctx,source,spec,binding=setup(ROOT,spec_pin,a.owner_sha)
 out=execute(ctx,source,spec,binding,spec_pin);print(json.dumps(out),flush=True);return 0 if out['passed'] else 1

if __name__=='__main__':raise SystemExit(main())
