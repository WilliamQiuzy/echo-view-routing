#!/usr/bin/env python3
"""One A58 queue with actual host/PID/start receipts, including failed workers."""
import os,sys,time,json,socket,resource,subprocess,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.structured_common import RUN,PUB,CFG,read,write
from echo_routing.temporal.dense_resource_accounting import terminal_queue_lock
COMMANDS={'qualify':('scripts/prepare_structured_evidence.py','source_qualification',600,False),'metadata':('scripts/prepare_structured_evidence.py','metadata',900,False),'diagnosis':('scripts/diagnose_structured_complementarity.py','run',900,False),'check':('scripts/check_structured_evidence.py','run',900,True),'cache_fit':('scripts/cache_structured_evidence.py','fit',28800,True),'formal':('scripts/train_structured_evidence.py','formal',86400,None),'audit':('scripts/report_structured_evidence.py','audit',3600,False),'cache_outer':('scripts/cache_structured_evidence.py','outer',21600,True),'readout':('scripts/readout_structured_evidence.py','run',21600,False),'aggregate':('scripts/report_structured_evidence.py','aggregate',14400,False)}

def remaining_limits(b,rows,now,gpu,closing=False):
    spent_gpu=sum(r['full_process_wall_seconds'] for r in rows if r['GPU_job'])/3600;spent_cpu=sum(r['process_CPU_seconds'] for r in rows)/3600
    available_gpu=min(b['new_GPUjob_h']-spent_gpu,b['cumulative_GPUjob_h']-b['prior_GPUjob_h']-spent_gpu)-(0 if closing else b['closure_GPUreserve_h'])
    available_cpu=min(b['new_CPUcore_h']-spent_cpu-.25,b['cumulative_CPUcore_h']-b['prior_summed_measured_CPUreceipt_h']-b['prior_setup_budget_allowance_h']-spent_cpu-.25)
    wall=min(b['new_deadline'],b['cumulative_deadline'])-now
    if wall<=10 or available_cpu*3600<=10 or (gpu and available_gpu*3600<=10):raise RuntimeError('new or inherited resource/deadline limit leaves no admissible worker')
    return {'GPU_seconds':available_gpu*3600,'CPU_seconds':available_cpu*3600,'wall_seconds':wall,'spent_new_GPU_hours':spent_gpu,'spent_new_CPU_hours':spent_cpu}

def execute(stage,command,gpu,limit,out=RUN):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    with terminal_queue_lock(out/'exclusive_supervisor.lock'):
        active=out/'active_process.json'
        if active.exists():raise RuntimeError('existing queue claim; audit before any subsequent worker')
        budget=read(CFG)['budget'];rows=[json.loads(s) for s in (out/'processes.jsonl').read_text().splitlines()] if (out/'processes.jsonl').exists() else [];bounds=remaining_limits(budget,rows,time.time(),gpu,closing=stage=='cache_outer');effective=min(limit,int(min(bounds['CPU_seconds'],bounds['wall_seconds'],bounds['GPU_seconds'] if gpu else float('inf'))-10))
        attempt=1
        while (out/f'process_{stage}_{attempt}.json').exists():attempt+=1
        own0=resource.getrusage(resource.RUSAGE_SELF);child0=resource.getrusage(resource.RUSAGE_CHILDREN)
        if gpu and subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():raise RuntimeError('GPU is owned; refuse competing GPU worker')
        start=time.time()
        def bound():resource.setrlimit(resource.RLIMIT_CPU,(effective,effective+5))
        with (out/f'{stage}_{attempt}.stdout.log').open('w') as stdout,(out/f'{stage}_{attempt}.stderr.log').open('w') as stderr:
            child=subprocess.Popen(command,cwd=ROOT,stdout=stdout,stderr=stderr,preexec_fn=bound);write(active,{'stage':stage,'attempt':attempt,'model_pid':child.pid,'supervisor_pid':os.getpid(),'measurement_host':socket.gethostname(),'start_epoch':start,'GPU_job':bool(gpu)})
            timeout=False;reason=None;last_disk=start
            while True:
                try:code=child.wait(timeout=1);break
                except subprocess.TimeoutExpired:pass
                now=time.time()
                if now-start>=effective:reason='effective_stage_or_shared_deadline_wall_limit'
                proc=Path(f'/proc/{child.pid}/stat')
                if proc.exists():
                    fields=proc.read_text().split(') ',1)[1].split();cpu=(int(fields[11])+int(fields[12]))/os.sysconf('SC_CLK_TCK');own=resource.getrusage(resource.RUSAGE_SELF);cpu+=own.ru_utime+own.ru_stime-own0.ru_utime-own0.ru_stime
                    if cpu>=bounds['CPU_seconds']-10:reason='actual_running_worker_and_supervisor_CPU_limit'
                if now-last_disk>=10:
                    paths=[str(p) for p in [out,PUB] if p.exists()];storage=sum(int(line.split()[0]) for line in subprocess.check_output(['du','-sB1',*paths],text=True).splitlines())+105_000_000
                    if storage>=min(budget['new_peak_bytes'],budget['cumulative_peak_upper_bytes']-budget['prior_tracked_storage_upper_bytes']):reason='new_or_inherited_tracked_storage_upper_limit'
                    last_disk=now
                if reason:
                    timeout=True;child.terminate()
                    try:code=child.wait(timeout=5)
                    except subprocess.TimeoutExpired:child.kill();code=child.wait()
                    break
        end=time.time();own1=resource.getrusage(resource.RUSAGE_SELF);child1=resource.getrusage(resource.RUSAGE_CHILDREN);cc=child1.ru_utime+child1.ru_stime-child0.ru_utime-child0.ru_stime;sc=own1.ru_utime+own1.ru_stime-own0.ru_utime-own0.ru_stime
        record={'stage':stage,'attempt':attempt,'model_pid':child.pid,'supervisor_pid':os.getpid(),'measurement_host':socket.gethostname(),'start_epoch':start,'end_epoch':end,'full_process_wall_seconds':end-start,'process_CPU_seconds':cc+sc,'child_CPU_seconds':cc,'supervisor_CPU_seconds_through_receipt_snapshot':sc,'CPU_scope':'waitedworker/guardsubprocessesplusdisjointsupervisorSELF;postreceiptbookkeepingunmeteredsetup','GPU_job':bool(gpu),'exit_code':code,'timed_out':timeout,'termination_reason':reason,'declared_stage_limit_seconds':limit,'effective_bounded_limit_seconds':effective,'dual_resource_limits_at_launch':bounds,'max_RSS_KB':child1.ru_maxrss}
        write(out/f'process_{stage}_{attempt}.json',record)
        with (out/'processes.jsonl').open('a') as f:f.write(json.dumps(record)+'\n');f.flush();os.fsync(f.fileno())
        active.unlink();return record

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=COMMANDS);a=p.parse_args();script,mode,limit,gpu=COMMANDS[a.stage]
    if gpu is None:
        admission=read(ROOT/'docs/results/2026-11-03_structured_evidence/admission.json');assert admission['all_caps_fit'];gpu=admission['training_backend']=='cuda'
    r=execute(a.stage,[str(ROOT/'.venv/bin/python'),script,mode],gpu,limit);print(json.dumps(r),flush=True);raise SystemExit(r['exit_code'])
if __name__=='__main__':main()
