#!/usr/bin/env python3
"""Single durable caller for terminal collector; directly wait and meter tail."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import sys,time,resource,socket,subprocess,json,uuid
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.event_risk_common import *
from echo_routing.temporal.structured_guard import process_sample
from echo_routing.temporal.event_risk_terminal_recovery import verify as verify_recovery
from echo_routing.temporal.anchor_cache import fsync_directory,durable_directory


def reserve_intent(path,value):
    with Path(path).open('x') as h:json.dump(value,h,sort_keys=True);h.write('\n');h.flush();os.fsync(h.fileno())
    fsync_directory(Path(path).parent)


def publish_once(path,value):
    path=checked_output(path);temporary=path.with_name(path.name+'.'+str(os.getpid())+'.tmp')
    with temporary.open('x') as h:json.dump(value,h,sort_keys=True,allow_nan=False);h.write('\n');h.flush();os.fsync(h.fileno())
    os.link(temporary,path);fsync_directory(path.parent);temporary.unlink();fsync_directory(path.parent)


def run(command,timeout=600):
    durable_directory(RUN);intent=RUN/'terminal_recovery_v2_caller_intent.json'
    if intent.exists() or (PUB/'collector_recovery_OS_exit_receipt.json').exists() or (PUB/'resources.json').exists():raise ValueError('existing collector/caller evidence cannot be replayed')
    nonce=uuid.uuid4().hex;boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip() if Path('/proc/sys/kernel/random/boot_id').exists() else 'local-toy-non-linux';sample=process_sample(os.getpid());ticks=sample['start_ticks'] if sample else 1
    if boot!='local-toy-non-linux' and sample is None:raise ValueError('real Linux caller start ticks must be observed')
    started=time.time();s=resource.getrusage(resource.RUSAGE_SELF);c=resource.getrusage(resource.RUSAGE_CHILDREN);reserve_intent(intent,{'caller_pid':os.getpid(),'caller_start_ticks':ticks,'boot_id':boot,'nonce':nonce,'started_epoch':started,'state':'durable_launch_intent'});p=None;code=None;error=None;reason=None
    try:
        with (RUN/'terminal_recovery_v2.stdout.log').open('x') as out,(RUN/'terminal_recovery_v2.stderr.log').open('x') as err:
            p=subprocess.Popen(command,cwd=ROOT,stdout=out,stderr=err,env={**os.environ,'CUDA_VISIBLE_DEVICES':''})
            if read(intent).get('nonce')!=nonce or read(intent).get('caller_pid')!=os.getpid():raise ValueError('owned recovery intent changed; preserve concurrent evidence')
            write(intent,{'caller_pid':os.getpid(),'caller_start_ticks':ticks,'collector_pid':p.pid,'collector_start_ticks':(process_sample(p.pid) or {}).get('start_ticks'),'boot_id':boot,'nonce':nonce,'started_epoch':started,'state':'actual_PID_bound'})
            try:code=p.wait(timeout=timeout)
            except subprocess.TimeoutExpired:reason='collector_timeout';p.kill();code=p.wait()
    except BaseException as e:
        error=repr(e)
        if p is not None:
            if p.poll() is None:p.kill()
            code=p.wait()
    finish=time.time();ss=resource.getrusage(resource.RUSAGE_SELF);cc=resource.getrusage(resource.RUSAGE_CHILDREN);own=ss.ru_utime+ss.ru_stime-(0. if __name__=='__main__' else s.ru_utime+s.ru_stime);waited=cc.ru_utime+cc.ru_stime-c.ru_utime-c.ru_stime;resources=read(PUB/'resources.json') if (PUB/'resources.json').exists() else None;terminal=read(PUB/'terminal_exit_receipt.json') if (PUB/'terminal_exit_receipt.json').exists() else None;snapshot=resources['collector_CPU_seconds_through_snapshot'] if resources else 0.;passed=code==0 and not error and not reason and resources is not None and terminal is not None and resources['collector_pid']==p.pid and terminal['resources_sha256']==sha(PUB/'resources.json') and resources.get('passed',True)
    if waited+1e-5<snapshot:passed=False;error='waited collector CPU below its known snapshot'
    value={'passed':passed,'collector_pid':None if p is None else p.pid,'caller_pid':os.getpid(),'caller_SELF_OS_exit_code':None,'observed_collector_OS_exit_code':code,'collector_exit_directly_waited':p is not None,'exception':error,'termination_reason':reason,'started_epoch':started,'finished_epoch':finish,'full_process_wall_seconds':finish-started,'measurement_host':socket.gethostname(),'collector_waited_CPU_seconds':waited,'collector_already_charged_snapshot_CPU_seconds':snapshot,'collector_postsnapshot_CPU_seconds':max(0.,waited-snapshot),'caller_SELF_CPU_seconds':own,'caller_SELF_scope':'whole fresh standalone process including imports through sample' if __name__=='__main__' else 'function region only in isolated software fixture','incremental_postsnapshot_and_caller_CPU_seconds':max(0.,waited-snapshot)+own,'resources_sha256':sha(PUB/'resources.json') if resources else None,'terminal_exit_receipt_sha256':sha(PUB/'terminal_exit_receipt.json') if terminal else None,'original_raw_resources_and_terminal_not_modified':True,'overall_complete':False,'protected_root_public_integration_pending':True,'caller_final_write_exit_tail_unmetered':True}
    publish_once(PUB/'collector_recovery_OS_exit_receipt.json',value);print(value,flush=True);return 0 if passed else 1


def main():
    p=execution_parser(__doc__);p.add_argument('--incomplete',action='store_true');p.add_argument('--failure-reason');a=p.parse_args()
    if not authorize(a):return 0
    if a.incomplete:raise ValueError('this explicit recovery is full-mode only')
    contract=verify_recovery()
    from echo_routing.temporal.event_risk_terminal_recovery import receipts
    from echo_routing.temporal.event_risk_budget import totals,forecast
    costs=receipts();spent=totals(costs);peak=max(r.get('maximum_sampled_tracked_storage_upper_bytes',0) for r in costs)+64000000
    admission=forecast(spent,0.,900.,peak,time.time(),900.);admission.update(scope='terminal recovery only; science already complete; unchanged original reserves and caps',recovery_git=contract['git_commit'])
    publish_once(PUB/'terminal_recovery_admission.json',admission)
    if not admission['all_caps_fit']:raise ValueError('current complete terminal recovery caps do not fit')
    command=[sys.executable,str(ROOT/'scripts/close_event_risk_terminal_recovery.py'),'--execute','--approved-plan-sha',CARD_SHA]
    if a.incomplete:
        if not a.failure_reason:raise ValueError('explicit failed/incomplete reason required')
        command+=['--incomplete','--failure-reason',a.failure_reason]
    return run(command)
if __name__=='__main__':raise SystemExit(main())
