#!/usr/bin/env python3
"""Single durable caller for terminal collector; directly wait and meter tail."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import sys,time,resource,socket,subprocess
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.event_risk_common import *
from echo_routing.temporal.anchor_cache import fsync_directory,durable_directory


def run(command,timeout=600):
    durable_directory(RUN);intent=RUN/'collector_caller_intent.json'
    if intent.exists() or (PUB/'collector_OS_exit_receipt.json').exists() or (PUB/'resources.json').exists():raise ValueError('existing collector/caller evidence cannot be replayed')
    started=time.time();s=resource.getrusage(resource.RUSAGE_SELF);c=resource.getrusage(resource.RUSAGE_CHILDREN);write(intent,{'caller_pid':os.getpid(),'started_epoch':started,'state':'durable_launch_intent'});p=None;code=None;error=None;reason=None
    try:
        with (RUN/'collector.stdout.log').open('x') as out,(RUN/'collector.stderr.log').open('x') as err:
            p=subprocess.Popen(command,cwd=ROOT,stdout=out,stderr=err,env={**os.environ,'CUDA_VISIBLE_DEVICES':''});write(intent,{'caller_pid':os.getpid(),'collector_pid':p.pid,'started_epoch':started,'state':'actual_PID_bound'})
            try:code=p.wait(timeout=timeout)
            except subprocess.TimeoutExpired:reason='collector_timeout';p.kill();code=p.wait()
    except BaseException as e:
        error=repr(e)
        if p is not None:
            if p.poll() is None:p.kill()
            code=p.wait()
    finish=time.time();ss=resource.getrusage(resource.RUSAGE_SELF);cc=resource.getrusage(resource.RUSAGE_CHILDREN);own=ss.ru_utime+ss.ru_stime-s.ru_utime-s.ru_stime;waited=cc.ru_utime+cc.ru_stime-c.ru_utime-c.ru_stime;resources=read(PUB/'resources.json') if (PUB/'resources.json').exists() else None;terminal=read(PUB/'terminal_exit_receipt.json') if (PUB/'terminal_exit_receipt.json').exists() else None;snapshot=resources['collector_CPU_seconds_through_snapshot'] if resources else 0.;passed=code==0 and not error and not reason and resources is not None and terminal is not None and resources['collector_pid']==p.pid and terminal['resources_sha256']==sha(PUB/'resources.json')
    if waited+1e-5<snapshot:passed=False;error='waited collector CPU below its known snapshot'
    value={'passed':passed,'collector_pid':None if p is None else p.pid,'caller_pid':os.getpid(),'caller_SELF_OS_exit_code':None,'observed_collector_OS_exit_code':code,'collector_exit_directly_waited':p is not None,'exception':error,'termination_reason':reason,'started_epoch':started,'finished_epoch':finish,'full_process_wall_seconds':finish-started,'measurement_host':socket.gethostname(),'collector_waited_CPU_seconds':waited,'collector_already_charged_snapshot_CPU_seconds':snapshot,'collector_postsnapshot_CPU_seconds':max(0.,waited-snapshot),'caller_SELF_CPU_seconds':own,'incremental_postsnapshot_and_caller_CPU_seconds':max(0.,waited-snapshot)+own,'resources_sha256':sha(PUB/'resources.json') if resources else None,'terminal_exit_receipt_sha256':sha(PUB/'terminal_exit_receipt.json') if terminal else None,'original_raw_resources_and_terminal_not_modified':True,'overall_complete':False,'protected_root_public_integration_pending':True,'caller_final_write_exit_tail_unmetered':True}
    write(PUB/'collector_OS_exit_receipt.json',value);print(value,flush=True);return 0 if passed else 1


def main():
    a=execution_parser(__doc__).parse_args()
    if not authorize(a):return 0
    verify_freeze('terminal_freeze.json');return run([sys.executable,str(ROOT/'scripts/close_event_risk.py'),'--execute','--approved-plan-sha',CARD_SHA])
if __name__=='__main__':raise SystemExit(main())
