#!/usr/bin/env python3
"""A58 subsequent stages: same exclusive queue, fail-closed child cleanup and receipts."""
import os,sys,time,json,socket,resource,subprocess,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.structured_common import RUN,PUB,CFG,read,write,verify_freeze
from scripts.supervise_structured_evidence import remaining_limits
from echo_routing.temporal.structured_guard import storage_bytes,process_sample
from echo_routing.temporal.dense_resource_accounting import terminal_queue_lock
COMMANDS={'head_check':('scripts/check_structured_head_pipeline.py','run',900,False),'admission':('scripts/admit_structured_evidence.py','formal',900,False),'formal':('scripts/train_structured_evidence.py','formal',86400,False),'audit':('scripts/report_structured_evidence.py','audit',3600,False),'cache_outer':('scripts/produce_structured_cache.py','outer',21600,True),'readout':('scripts/readout_structured_evidence.py','run',21600,False),'aggregate':('scripts/aggregate_structured_evidence.py','run',14400,False)}

def execute(stage,command,gpu,limit):
    with terminal_queue_lock(RUN/'exclusive_supervisor.lock'):
        active=RUN/'active_process.json'
        if active.exists():raise RuntimeError('existing queue claim; independent closure required')
        budget=read(CFG)['budget'];rows=[json.loads(s) for s in (RUN/'processes.jsonl').read_text().splitlines()];allowance=sum(r.get('unmeasured_completed_CPU_accounting_allowance_seconds',0) for r in rows);budgetrows=[{**r,'process_CPU_seconds':r['process_CPU_seconds']+r.get('unmeasured_completed_CPU_accounting_allowance_seconds',0)} for r in rows];bounds=remaining_limits(budget,budgetrows,time.time(),gpu,closing=stage=='cache_outer');effective=min(limit,int(min(bounds['CPU_seconds'],bounds['wall_seconds'],bounds['GPU_seconds'] if gpu else float('inf'))-10));attempt=1
        while (RUN/f'process_{stage}_{attempt}.json').exists():attempt+=1
        if stage=='formal':assert read(PUB/'formal_runtime_admission.json')['all_caps_fit'];verify_freeze('formal_freeze.json')
        if stage=='cache_outer':verify_freeze('outer_freeze.json')
        if gpu and subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():raise RuntimeError('GPU already has an owner')
        own0=resource.getrusage(resource.RUSAGE_SELF);child0=resource.getrusage(resource.RUSAGE_CHILDREN);start=time.time();reason=None;exception=None
        def bound():resource.setrlimit(resource.RLIMIT_CPU,(effective,effective+5))
        with (RUN/f'{stage}_{attempt}.stdout.log').open('w') as stdout,(RUN/f'{stage}_{attempt}.stderr.log').open('w') as stderr:
            child=subprocess.Popen(command,cwd=ROOT,stdout=stdout,stderr=stderr,preexec_fn=bound);write(active,{'stage':stage,'attempt':attempt,'model_pid':child.pid,'supervisor_pid':os.getpid(),'measurement_host':socket.gethostname(),'start_epoch':start,'GPU_job':gpu,'guardian_version':2});lastdisk=0
            try:
                while True:
                    try:code=child.wait(timeout=1);break
                    except subprocess.TimeoutExpired:pass
                    now=time.time();sample=process_sample(child.pid);own=resource.getrusage(resource.RUSAGE_SELF)
                    if now-start>=effective:reason='effective_stage_or_shared_deadline_wall_limit'
                    if sample and sample['CPU_seconds']+own.ru_utime+own.ru_stime-own0.ru_utime-own0.ru_stime>=bounds['CPU_seconds']-10:reason='worker_and_supervisor_actual_CPU_limit'
                    if now-lastdisk>=10:
                        if storage_bytes([RUN,PUB])>=min(budget['new_peak_bytes'],budget['cumulative_peak_upper_bytes']-budget['prior_tracked_storage_upper_bytes']):reason='new_or_inherited_tracked_storage_upper_limit'
                        lastdisk=now
                    if reason:break
            except BaseException as error:exception=repr(error);reason='guardian_exception_fail_closed'
            finally:
                if child.poll() is None:
                    child.terminate()
                    try:code=child.wait(timeout=5)
                    except subprocess.TimeoutExpired:child.kill();code=child.wait()
                else:code=child.wait()
        end=time.time();own1=resource.getrusage(resource.RUSAGE_SELF);child1=resource.getrusage(resource.RUSAGE_CHILDREN);cc=child1.ru_utime+child1.ru_stime-child0.ru_utime-child0.ru_stime;sc=own1.ru_utime+own1.ru_stime-own0.ru_utime-own0.ru_stime;record={'stage':stage,'attempt':attempt,'model_pid':child.pid,'supervisor_pid':os.getpid(),'measurement_host':socket.gethostname(),'start_epoch':start,'end_epoch':end,'full_process_wall_seconds':end-start,'process_CPU_seconds':cc+sc,'child_CPU_seconds':cc,'supervisor_CPU_seconds_through_receipt_snapshot':sc,'CPU_scope':'waitedworker/guardsubprocessesplusdisjointsupervisorSELF;postreceiptbookkeepingunmeteredsetup','GPU_job':gpu,'exit_code':code,'observed_worker_exit_code':code,'child_exit_code_directly_waited':True,'timed_out':reason is not None,'termination_reason':reason,'guardian_exception':exception,'declared_stage_limit_seconds':limit,'effective_bounded_limit_seconds':effective,'dual_resource_limits_at_launch':bounds,'prior_completed_unmeasured_CPU_accounting_allowance_seconds':allowance,'guard_removed_file_race_safe':True,'max_RSS_KB':child1.ru_maxrss};write(RUN/f'process_{stage}_{attempt}.json',record)
        with (RUN/'processes.jsonl').open('a') as f:f.write(json.dumps(record)+'\n');f.flush();os.fsync(f.fileno())
        active.unlink();return record

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=COMMANDS);a=p.parse_args();script,mode,limit,gpu=COMMANDS[a.stage];r=execute(a.stage,[str(ROOT/'.venv/bin/python'),script,mode],gpu,limit);print(json.dumps(r),flush=True);raise SystemExit(r['exit_code'])
if __name__=='__main__':main()
