#!/usr/bin/env python3
"""Future foreground A63 caller: one authoritative whole-child wait4 receipt.

Inspection is the default. The caller never acquires a queue or restarts work;
the fixed guardian retains the single original flock and scientific caps. Its
mid-region snapshot is informational and covered, never charged a second time.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import socket
import stat
import subprocess
import sys
import time
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from echo_routing.temporal.selected_risk_resume_contract import RUN,PUB,stage_graph


def direct_wait(child,*,wait4=os.wait4):
    """Do not poll/waitpid first (that would lose wait4 usage), replay or signal.

    A catchable caller interruption is recorded, then the existing guardian is
    waited. The guardian alone supervises its own child/caps. A hard caller
    crash leaves durable intent for explicit reconciliation, never auto-replay.
    """
    error=None
    while True:
        try:
            pid,status,usage=wait4(child.pid,0)
            if pid!=child.pid:raise RuntimeError('wait4 returned a foreign child')
            child.returncode=os.waitstatus_to_exitcode(status)
            return child.returncode,usage,error
        except InterruptedError:continue
        except ChildProcessError as exc:return None,None,'ChildProcessError: '+str(exc)
        except (KeyboardInterrupt,SystemExit) as exc:error=type(exc).__name__


def canonical_row(stage,nonce,start,end,guardian_pid,guardian_ticks,code,usage,self_cpu,
                  snapshot,snapshot_pin,intent_pin,spec_pin,owner_sha,error):
    """Whole guardian/descendant CPU once; SELF sample and allowance separate."""
    if stage not in stage_graph() or end<start:raise ValueError('registered ordered caller stage required')
    values=[self_cpu]+([] if usage is None else [usage.ru_utime,usage.ru_stime])
    if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not __import__('math').isfinite(v) or v<0 for v in values):raise ValueError('actual finite nonnegative wait4/SELF quantities required')
    child_cpu=0. if usage is None else usage.ru_utime+usage.ru_stime
    inner_unknown=(snapshot is None or snapshot.get('model_pid') is not None and
                   (snapshot.get('observed_worker_exit_code') is None or snapshot.get('model_start_ticks') is None or snapshot.get('actual_child_directly_waited') is not True))
    unknown=(guardian_pid is not None and (usage is None or code is None) or inner_unknown or
             snapshot is not None and (snapshot.get('current_remote_caps_unknown') is True or
             snapshot.get('stage_exit_status')==0 and snapshot.get('model_pid') is None))
    gpu=unknown or bool(snapshot and (snapshot['GPU_job'] or snapshot['CUDA_context_occupied']))
    good=(guardian_pid is not None and code==0 and snapshot is not None and snapshot.get('stage_exit_status')==0 and snapshot.get('observed_worker_exit_code')==0 and not snapshot.get('termination_reason') and not unknown and error is None)
    row={**({} if snapshot is None else snapshot),'receipt_id':'resume_caller:'+stage+':'+nonce,
      'stage':stage,'nonce':nonce if snapshot is None else snapshot['nonce'],'caller_nonce':nonce,
      'source_intent_sha256':None if snapshot is None else snapshot.get('source_intent_sha256'),
      'caller_intent_sha256':intent_pin['sha256'],'caller_intent':intent_pin,'runtime_spec_sha256':spec_pin['sha256'],
      'runtime_authorization_sha256':owner_sha,'caller_pid':os.getpid(),'guardian_pid':guardian_pid,
      'guardian_start_ticks':guardian_ticks,'model_pid':None if snapshot is None else snapshot.get('model_pid'),
      'model_start_ticks':None if snapshot is None else snapshot.get('model_start_ticks'),
      'measurement_host':socket.gethostname(),'started_epoch':start,'finished_epoch':end,
      'full_process_wall_seconds':end-start,'measured_CPU_core_seconds':None if unknown else self_cpu+child_cpu,
      'known_partial_CPU_core_seconds':self_cpu+child_cpu,'whole_child_accounting_complete':not unknown,
      'measurement_kind':'caller_SELF_plus_authoritative_wait4','caller_SELF_CPU_seconds_through_sample':self_cpu,
      'guardian_and_waited_descendants_wait4_CPU_seconds':child_cpu,'caller_exit_CPU_allowance_seconds':1.,
      'CPU_measurement_scope':'caller SELF entire lifetime through sample + direct wait4 whole guardian/its waited descendants once; caller final receipt/write/exit tail explicit allowance, not measured',
      'caller_exit_allowance_is_not_measured_or_a_proven_upper_for_all_unknown_tool_model_CPU':True,
      'GPU_job':gpu,'CUDA_context_occupied':gpu,
      'GPU_observed_occupied_seconds':None if unknown else (end-start if gpu else 0.),
      'GPU_conservative_charged_seconds':end-start if gpu else 0.,
      'GPU_charge_scope':'one whole attempt interval as conservative occupied-job accounting; no device-utilization or active-kernel-time claim',
      'GPU_interval_unknown':unknown,'inner_worker_OS_unknown':inner_unknown,
      'observed_worker_exit_code':code,'guardian_observed_OS_exit_code':code,
      'inner_model_observed_OS_exit_code':None if snapshot is None else snapshot.get('observed_worker_exit_code'),
      'guardian_termination_reason':None if snapshot is None else snapshot.get('termination_reason'),
      'actual_child_directly_waited':guardian_pid is not None and usage is not None,
      'guardian_snapshot':snapshot_pin,'covered_charge_groups':[] if snapshot is None else [snapshot['receipt_id']],
      'canonical_charge':True,'guardian_snapshot_is_informational_not_additional_charge':True,
      'SELF_OS_exit_code':None,'unresolved_worker_uncertainty':unknown,'caller_error':error,
      'termination_reason':None if good else ('caller_interrupted_or_exception' if error else 'caller_or_guardian_failed_or_unknown'),
      'stage_exit_status':0 if good else 1,'overall_complete':False}
    return row


def _identity():
    from echo_routing.temporal.selected_risk_resume_guardian import start_ticks
    try:return start_ticks(os.getpid()),Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    except OSError:return None,None


def _raw(ctx,path):
    fd,name=ctx._parent(path)
    try:
        leaf=os.open(name,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK,dir_fd=fd)
        with os.fdopen(leaf,'rb') as f:
            if not stat.S_ISREG(os.fstat(f.fileno()).st_mode):raise ValueError('ordinary caller/ledger file required')
            return f.read()
    finally:os.close(fd)


def _rows(ctx,name):
    from echo_routing.temporal.selected_risk_resume_context import strict_json
    path=RUN+'/'+name+'.jsonl'
    if not (ctx.root/path).exists() and not (ctx.root/path).is_symlink():return []
    rows=[]
    for raw in _raw(ctx,path).splitlines():
        v=strict_json(raw)
        if set(v)!={'receipt_id','receipt'}:raise ValueError('canonical immutable receipt pin ledger required')
        payload=ctx.read(v['receipt'])
        if payload['receipt_id']!=v['receipt_id']:raise ValueError('ledger/receipt identity mismatch')
        rows.append((v['receipt'],payload))
    return rows


def _no_unclosed_callers(ctx):
    # Residual-intent detection, not an atomic cross-process reservation. The
    # sole worker dispatches serially; the one original guardian flock plus its
    # caller-ticket inventory rejects competing/unclosed parents before science.
    closed={v['caller_intent_sha256'] for _,v in _rows(ctx,'processes')}
    folder=ctx.root/RUN/'caller_intents'
    if folder.is_symlink():raise ValueError('caller intent directory symlink forbidden')
    for path in folder.glob('*.json') if folder.exists() else []:
        raw=_raw(ctx,str(path.relative_to(ctx.root)))
        if hashlib.sha256(raw).hexdigest() not in closed:raise ValueError('unclosed caller intent: explicit reconciliation, never duplicate or replay')


def _snapshot(ctx,pid,ticks,boot,spec_pin,intent_pin,stage,start,end):
    matches=[]
    for pin,row in _rows(ctx,'guardian_snapshots'):
        if (row.get('supervisor_pid'),row.get('runtime_spec_sha256'),row.get('caller_intent_sha256'),row.get('stage'),row.get('boot_id'),row.get('runtime_authorization_sha256'))!=(pid,spec_pin['sha256'],intent_pin['sha256'],stage,boot,ctx.owner_sha):continue
        if ticks is not None and row.get('supervisor_start_ticks')!=ticks:raise ValueError('direct guardian PID/start differs from own physical child')
        if not start<=row['started_epoch']<=row['finished_epoch']<=end:raise ValueError('guardian snapshot lies outside the caller attempt')
        matches.append((pin,row))
    if len(matches)>1:raise ValueError('ambiguous duplicate matching guardian snapshot')
    return (None,None) if not matches else matches[0]


def _publish(ctx,row,intent_pin,spec_pin):
    """Receipt-only after deadline; never reopens scientific work authority."""
    nonce=row.get('caller_nonce') if isinstance(row,dict) else None
    if not isinstance(nonce,str) or len(nonce)!=32 or any(c not in '0123456789abcdef' for c in nonce) or row.get('stage') not in stage_graph() or row.get('receipt_id')!='resume_caller:'+row['stage']+':'+nonce:
        raise ValueError('exact canonical own stage/receipt nonce required before publication')
    for pin in [ctx.owner_pin,ctx.proposal_pin,ctx.baseline_pin,spec_pin]:ctx.raw(pin)
    intent=ctx.read(intent_pin)
    if intent['caller_pid']!=os.getpid() or intent['caller_nonce']!=row['caller_nonce'] or intent['runtime_spec']!=spec_pin or intent['runtime_authorization_sha256']!=ctx.owner_sha or intent['stage']!=row['stage'] or intent['caller_start_ticks']!=row.get('caller_start_ticks') or intent['boot_id']!=row.get('caller_boot_id'):raise ValueError('own original caller intent/raw authority required for post-stop receipt')
    live_ticks,live_boot=_identity()
    if live_ticks is None or (live_ticks,live_boot)!=(intent['caller_start_ticks'],intent['boot_id']):
        raise ValueError('current physical caller birth/boot must still match own durable intent')
    name='caller_process_'+row['caller_nonce']+'.json';path=PUB+'/'+name
    raw=json.dumps(row,sort_keys=True,separators=(',',':'),allow_nan=False).encode();fd,leaf=ctx._parent(path,create=True);temp='.'+leaf+'.owned-'+uuid.uuid4().hex
    try:
        child=os.open(temp,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600,dir_fd=fd)
        with os.fdopen(child,'wb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
        os.link(temp,leaf,src_dir_fd=fd,dst_dir_fd=fd,follow_symlinks=False);os.fsync(fd);os.unlink(temp,dir_fd=fd);os.fsync(fd)
    finally:os.close(fd)
    pin={'path':path,'sha256':hashlib.sha256(raw).hexdigest()};fd,leaf=ctx._parent(RUN+'/processes.jsonl',create=True)
    try:
        child=os.open(leaf,os.O_WRONLY|os.O_CREAT|os.O_APPEND|os.O_NOFOLLOW,0o600,dir_fd=fd)
        with os.fdopen(child,'ab') as f:f.write(json.dumps({'receipt_id':row['receipt_id'],'receipt':pin},sort_keys=True).encode()+b'\n');f.flush();os.fsync(f.fileno())
        os.fsync(fd)
    finally:os.close(fd)
    return pin


def execute(context,spec,spec_pin,*,spawn=subprocess.Popen,wait4=os.wait4):
    from echo_routing.temporal.selected_risk_resume_guardian import start_ticks
    if sys.platform!='linux':raise ValueError('scientific caller is server-only; local preparation never launches')
    context.revalidate();context.raw(spec_pin);_no_unclosed_callers(context)
    boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    if boot!=spec['expected_boot_id']:raise ValueError('fresh independently supplied boot identity required')
    stage=spec['stage'];nonce=uuid.uuid4().hex;start=time.time();ticks=start_ticks(os.getpid())
    if stage not in stage_graph() or ticks is None:raise ValueError('exact stage and live caller PID/start required')
    own={'caller_pid':os.getpid(),'caller_start_ticks':ticks,'boot_id':boot,'runtime_spec':spec_pin,
         'runtime_authorization_sha256':context.owner_sha,'stage':stage,'caller_nonce':nonce,'started_epoch':start}
    intent=context.write_json('caller_intents/'+stage+'.'+nonce+'.json',own)
    child=None;usage=None;code=None;error=None;child_ticks=None;snapshot=None;snapshot_pin=None
    try:
        env={**os.environ,'A63_RESUME_CALLER_INTENT_PATH':intent['path'],'A63_RESUME_CALLER_INTENT_SHA':intent['sha256'],
             'CUDA_VISIBLE_DEVICES':'0' if stage in ['source_FIT','source_CONFIRM'] else '',
             'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1'}
        command=[sys.executable,str(context.root/'scripts/a63_resume_selected_risk.py'),'--execute','--stage',stage,
                 '--runtime-spec',spec_pin['path'],'--runtime-spec-sha',spec_pin['sha256'],'--owner-sha',context.owner_sha]
        # No shell, second queue, detachment or automatic retry.
        child=spawn(command,cwd=context.root,env=env);child_ticks=start_ticks(child.pid)
        code,usage,error=direct_wait(child,wait4=wait4)
    except BaseException as exc:
        error=type(exc).__name__+': '+str(exc)
        if child is not None and usage is None:
            try:
                code,usage,wait_error=direct_wait(child,wait4=wait4)
                if wait_error:error+='; '+wait_error
            except BaseException as wait_error:error+='; wait unavailable: '+type(wait_error).__name__+': '+str(wait_error)
    end=time.time()
    if child is not None:
        try:snapshot_pin,snapshot=_snapshot(context,child.pid,child_ticks,boot,spec_pin,intent,stage,start,end)
        except BaseException as exc:error=(error or '')+'; snapshot: '+type(exc).__name__+': '+str(exc)
    # Include post-wait snapshot/ledger parsing through the same cost cutoff;
    # source GPU accounting is a conservative whole-attempt interval.
    end=time.time()
    owncpu=resource.getrusage(resource.RUSAGE_SELF);cpu=owncpu.ru_utime+owncpu.ru_stime
    row=canonical_row(stage,nonce,start,end,None if child is None else child.pid,child_ticks,code,usage,cpu,snapshot,snapshot_pin,intent,spec_pin,context.owner_sha,error)
    row['caller_start_ticks']=ticks;row['caller_boot_id']=boot
    receipt=_publish(context,row,intent,spec_pin)
    return {**row,'public_receipt':receipt}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execute',action='store_true');p.add_argument('--stage',required=True,choices=sorted(stage_graph()));p.add_argument('--runtime-spec');p.add_argument('--runtime-spec-sha');p.add_argument('--owner-sha')
    a=p.parse_args(argv)
    if not a.execute:
        print(json.dumps({'preparation_only':True,'execution_authorized':False,'new_child_created':False,'new_runtime_directories_created':False,'stage':a.stage}));return 0
    if not all([a.runtime_spec,a.runtime_spec_sha,a.owner_sha]):p.error('--execute requires independently raw-pinned new owner and runtime spec')
    from echo_routing.temporal.selected_risk_resume_runtime import bootstrap
    context,spec=bootstrap(ROOT,{'path':a.runtime_spec,'sha256':a.runtime_spec_sha},a.owner_sha)
    if spec['stage']!=a.stage:raise ValueError('requested caller stage differs from externally pinned spec')
    result=execute(context,spec,{'path':a.runtime_spec,'sha256':a.runtime_spec_sha});print(json.dumps(result,sort_keys=True),flush=True);return result['stage_exit_status']


if __name__=='__main__':raise SystemExit(main())
