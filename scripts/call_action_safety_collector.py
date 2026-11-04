#!/usr/bin/env python3
"""Single direct OS wait for the queue-owning collector; tail charged separately."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,time,resource,subprocess,socket
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.safety_common import RUN,PUB,read,write,verify_freeze,sha
from echo_routing.temporal.anchor_cache import fsync_directory
from echo_routing.temporal.safety_resources import DEADLINE_EPOCH


def main():
    verify_freeze('terminal_freeze.json');intent=RUN/'collector_launch_intent.json';target=PUB/'collector_OS_exit_receipt.json'
    if intent.exists() or target.exists() or (PUB/'resources.json').exists() or (PUB/'terminal_exit_receipt.json').exists():raise ValueError('existing collector evidence/intent; never duplicate launch')
    start=time.time();limit=min(900.,DEADLINE_EPOCH-start-5.)
    if limit<=0:raise ValueError('original deadline leaves no collector room')
    with intent.open('x') as h:
        h.write('{"state":"durable_prelaunch_intent","caller_pid":'+str(os.getpid())+'}\n');h.flush();os.fsync(h.fileno())
    fsync_directory(RUN);before=resource.getrusage(resource.RUSAGE_CHILDREN);p=None;code=None;reason=None;exception=None
    try:
        with (RUN/'terminal_collector.stdout.log').open('xb') as out,(RUN/'terminal_collector.stderr.log').open('xb') as err:
            p=subprocess.Popen([sys.executable,str(ROOT/'scripts/close_action_safety.py')],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
            write(intent,{'state':'collector_PID_bound','caller_pid':os.getpid(),'collector_pid':p.pid,'started_epoch':start})
            try:code=p.wait(timeout=limit)
            except subprocess.TimeoutExpired:
                reason='original_deadline_or_collector_timeout';p.terminate()
                try:code=p.wait(timeout=5)
                except subprocess.TimeoutExpired:p.kill();code=p.wait()
    except BaseException as e:
        exception=repr(e)
        if p is not None:
            if p.poll() is None:p.kill()
            code=p.wait()
    after=resource.getrusage(resource.RUSAGE_CHILDREN);own=resource.getrusage(resource.RUSAGE_SELF);childCPU=after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime;parentCPU=own.ru_utime+own.ru_stime;snapshot=None
    if p is not None and (PUB/'resources.json').exists():
        snapshot=read(PUB/'resources.json')
        if snapshot['collector']['model_pid']!=p.pid:raise ValueError('resource snapshot is not the directly waited collector')
    accounted=snapshot['collector']['CPU_seconds_through_snapshot'] if snapshot else 0.
    if childCPU<accounted-1e-5:raise ValueError('direct waited total CPU less than collector snapshot')
    tail=max(0.,childCPU-accounted);passed=code==0 and reason is None and exception is None and snapshot is not None and snapshot['passed'] and read(PUB/'terminal_exit_receipt.json')['passed']
    value={'passed':bool(passed),'collector_pid':None if p is None else p.pid,'caller_pid':os.getpid(),'measurement_host':socket.gethostname(),'started_epoch':start,'finished_epoch':time.time(),'observed_collector_OS_exit_code':code,'collector_exit_directly_waited':p is not None,'collector_OS_inferred_from_files_or_disappearance':False,'parent_SELF_OS_exit_code':None,'collector_total_waited_CPU_seconds':childCPU,'collector_CPU_already_in_resource_snapshot':accounted,'collector_postsnapshot_CPU_seconds':tail,'caller_SELF_CPU_seconds_through_sample':parentCPU,'total_incremental_CPU_seconds_after_resource_snapshot':tail+parentCPU,'caller_final_record_write_exit_CPU_not_measured':True,'termination_reason':reason,'exception':exception,'resource_receipt_sha256':sha(PUB/'resources.json') if snapshot else None,'terminal_receipt_sha256':sha(PUB/'terminal_exit_receipt.json') if (PUB/'terminal_exit_receipt.json').exists() else None,'no_new_NN_or_optimizer_updates':True,'overall_complete':False,'protected_root_public_integration_pending':True}
    write(target,value);write(intent,{'state':'direct_wait_receipt_committed','caller_pid':os.getpid(),'collector_pid':None if p is None else p.pid,'observed_collector_OS_exit_code':code,'receipt_sha256':sha(target)})
    if passed:write(RUN/'runtimecoordination.json',{'phase':'server_scientific_closed_caller_OS_observed_public_integration_pending','overall_complete':False,'model_pid':None,'formal_optimizer_updates':60000,'metric_rows':314,'paired_comparisons':240,'oracle_cells':60,'native_cells':56,'collector_directly_waited_OS_exit':code,'caller_receipt_sha256':sha(target),'next_action':'root protected exact-public integration;noautomaticA62','epoch':time.time()})
    print(value,flush=True);return 0 if passed else 1
if __name__=='__main__':raise SystemExit(main())
