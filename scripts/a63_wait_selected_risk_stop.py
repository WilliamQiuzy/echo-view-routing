#!/usr/bin/env python3
"""One exclusive local parent wait; does not contact the server or own its queue."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['PYTHONDONTWRITEBYTECODE']='1'
import sys,json,time,resource,subprocess,uuid,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

def closing_forecast(spent,local_upper,remote_last_known,now):
    from echo_routing.temporal.selected_risk_budget import forecast
    for v in [local_upper,remote_last_known]:
        if type(v) is not int or v<0:raise ValueError('explicit nonnegative storage counts required')
    # The historical 300MB admin forecast is replaced by root's full local
    # account; it is not added again. Unknown remote final-write bytes retain
    # a separate prospective allowance, not a claimed current du measurement.
    storage=local_upper+remote_last_known+64000000+128000000+64000000
    f=forecast(spent,0.,900.,storage,now,900.)
    if not f['all_caps_fit']:raise ValueError('local closing-only prospective budget cannot fit original caps/window')
    return {'closure_only_all_caps_forecast_fit':True,'forecast':f,'local_storage_upper':local_upper,'last_known_remote_namespace_sample':remote_last_known,'unknown_remote_poststage_allowance':64000000,'original_closing_storage_reserve':128000000,'local_stage_new_byte_allowance':64000000,'GPU_seconds':0,'CPU_seconds':900,'wall_seconds':900,'forecast_not_actual_spend':True,'not_a_fresh_remote_actual_caps_certificate':True,'scientific_queues_remain_stopped':True}

def closing_prelaunch():
    from scripts.selected_risk_common import PUB,read,write,actual_receipts
    from scripts.a63_collect_selected_risk_stop import merge_receipts,recovered_parent_rows
    from echo_routing.temporal.selected_risk_budget import totals
    parent=read(PUB/'late_sourcegate_v3_parent_observation.json')['actual_enclosing_receipt'];rows=merge_receipts(actual_receipts(),recovered_parent_rows(parent))
    root=read(PUB/'root_local_closure_storage_snapshot.json');last=max(r['maximum_sampled_tracked_storage_upper_bytes'] for r in parent['new_process_receipts'])-300000000
    value=closing_forecast(totals(rows),root['local_only_current_plus_prospective_allowance_sum_bytes'],last,time.time());write(PUB/'local_resource_stop_prelaunch.json',value);return value

def run():
    from scripts.selected_risk_common import ROOT,RUN,PUB,CARD_SHA,read,sha,checked_output,fsync_directory
    from scripts.a63_collect_selected_risk_stop import verify_contract
    verify_contract();intent=checked_output(RUN/'local_resource_stop_caller_intent.json');target=checked_output(PUB/'local_resource_stop_OS_receipt.json')
    existing=[target,PUB/'local_resource_stop_OS_receipt.partial.json',PUB/'local_resource_stop.stdout',PUB/'local_resource_stop.stderr',PUB/'resources_resource_stopped.json',PUB/'terminal_resource_stopped.json',PUB/'local_resource_stop_CPU_snapshot.json']
    if any(p.exists() or p.is_symlink() for p in existing):raise ValueError('immutable local stop output exists; no collector replay')
    nonce=uuid.uuid4().hex;intent.parent.mkdir(parents=True,exist_ok=True);start=time.time();p=None;code=None;error=None;snapshot=None
    claim={'nonce':nonce,'caller_PID':os.getpid(),'started_epoch':start,'scope':'local-only no server/queue/model operation'}
    with intent.open('x') as f:json.dump(claim,f,sort_keys=True);f.flush();os.fsync(f.fileno())
    fsync_directory(intent.parent)
    def child_limit():resource.setrlimit(resource.RLIMIT_CPU,(900,905))
    try:
        closing_prelaunch()
        with checked_output(PUB/'local_resource_stop.stdout').open('x') as out,checked_output(PUB/'local_resource_stop.stderr').open('x') as err:
            cmd=[sys.executable,'scripts/a63_collect_selected_risk_stop.py','--execute','--approved-plan-sha',CARD_SHA]
            p=subprocess.Popen(cmd,cwd=ROOT,env={**os.environ,'CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1'},stdout=out,stderr=err,preexec_fn=child_limit)
            if read(intent)!=claim:raise ValueError('own durable local caller intent changed after Popen')
            code=p.wait(timeout=900)
        if code!=0:raise ValueError('actual local collector OS'+str(code))
        snapshot=read(PUB/'local_resource_stop_CPU_snapshot.json')
        if snapshot['process_PID']!=p.pid or snapshot['SELF_OS_exit_code'] is not None:raise ValueError('actual child PID/self-OS snapshot mismatch')
        for n in ['terminal_resource_stopped.json','resources_resource_stopped.json']:
            if read(PUB/n)['overall_complete'] is not False:raise ValueError('local stop cannot preclaim root main completion')
    except BaseException as e:
        error=repr(e)
        if p is not None:
            if p.poll() is None:p.kill()
            code=p.wait()
    # A child can fail after producing an honest snapshot. Deduct that same
    # snapshot in both success and failure paths; never charge it a second time.
    if p is not None and snapshot is None and (PUB/'local_resource_stop_CPU_snapshot.json').exists():
        try:
            candidate=read(PUB/'local_resource_stop_CPU_snapshot.json')
            if candidate['process_PID']!=p.pid or candidate['SELF_OS_exit_code'] is not None:raise ValueError('failed-child snapshot PID/self OS differs')
            snapshot=candidate
        except BaseException as e:error=(error or '')+'; invalid snapshot: '+repr(e)
    a=resource.getrusage(resource.RUSAGE_SELF);b=resource.getrusage(resource.RUSAGE_CHILDREN);selfcpu=a.ru_utime+a.ru_stime;waited=b.ru_utime+b.ru_stime;excluded=0. if snapshot is None else snapshot['whole_fresh_process_CPU_through_snapshot']
    if waited+1e-6<excluded:raise ValueError('actual child wait CPU below frozen collector snapshot')
    result={'passed':error is None and code==0,'scope':'actual local-only collector direct wait; remote collector/current remote state not established','caller_PID':os.getpid(),'collector_PID':None if p is None else p.pid,'observed_collector_OS_exit_code':code,'collector_exit_directly_waited':p is not None,'caller_SELF_OS_exit_code':None,'error':error,'caller_SELF_CPU_seconds':selfcpu,'caller_waited_CHILDREN_CPU_seconds':waited,'collector_CPU_snapshot_subtracted_seconds':excluded,'separate_CPU_increment_outside_collector_snapshot':selfcpu+waited-excluded,'complement_includes_caller_startup_and_child_exit_tail_not_just_postsnapshot':True,'started_epoch':start,'sampled_epoch':time.time(),'intent_nonce':nonce,'intent_sha256':sha(intent),'raw_outputs':{n:sha(PUB/n) for n in ['local_resource_stop_CPU_snapshot.json','resources_resource_stopped.json','terminal_resource_stopped.json'] if (PUB/n).exists()},'overall_complete':False,'finalwrite_exit_tool_model_CPU_unknown':True}
    tmp=checked_output(PUB/'local_resource_stop_OS_receipt.partial.json')
    with tmp.open('x') as f:json.dump(result,f,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.link(tmp,target);fsync_directory(target.parent);print(json.dumps(result),flush=True);return 0 if result['passed'] else 1

def main():
    from scripts.selected_risk_common import execution_parser,authorize
    p=execution_parser(__doc__);a=p.parse_args()
    if not authorize(a):return 0
    return run()
if __name__=='__main__':raise SystemExit(main())
