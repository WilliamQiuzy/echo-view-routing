#!/usr/bin/env python3
"""Single owned process supervisor; every attempt has a full wall/CPU receipt."""
import argparse
import json
import os
from pathlib import Path
import resource
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/pw_purpose_transfer_live_v1'


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',required=True,choices=('check','structural','train','readout','evaluate','audit_models','audit_final'));a=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'active_process.json').exists():
        old=json.loads((OUT/'active_process.json').read_text())
        try:os.kill(old['model_pid'],0)
        except ProcessLookupError:pass
        else:raise RuntimeError('owned worker still alive; refuse second process')
    attempt=1
    while (OUT/f'process_{a.stage}_{attempt}.json').exists():attempt+=1
    s=time.time();before=resource.getrusage(resource.RUSAGE_CHILDREN);gpu=a.stage in ('check','train','readout','audit_models')
    limits={'check':600,'structural':600,'train':3600,'readout':1200,'evaluate':600,'audit_models':900,'audit_final':900}
    cmd=[str(ROOT/'.venv/bin/python'),'scripts/train_pw_purpose_transfer_live.py','--stage',a.stage]
    if a.stage.startswith('audit_'):cmd=[str(ROOT/'.venv/bin/python'),'scripts/audit_pw_purpose_transfer_live.py','--mode',a.stage.removeprefix('audit_')]
    def bound():resource.setrlimit(resource.RLIMIT_CPU,(limits[a.stage],limits[a.stage]+5))
    with (OUT/f'{a.stage}_{attempt}.stdout.log').open('w') as out,(OUT/f'{a.stage}_{attempt}.stderr.log').open('w') as err:
        p=subprocess.Popen(cmd,cwd=ROOT,stdout=out,stderr=err,preexec_fn=bound)
        (OUT/'active_process.json').write_text(json.dumps({'stage':a.stage,'attempt':attempt,'model_pid':p.pid,'supervisor_pid':os.getpid(),'start_epoch':s,'GPU_job':gpu})+'\n')
        timed_out=False
        try:code=p.wait(timeout=limits[a.stage]+30)
        except subprocess.TimeoutExpired:
            timed_out=True;p.terminate()
            try:code=p.wait(timeout=10)
            except subprocess.TimeoutExpired:p.kill();code=p.wait()
    after=resource.getrusage(resource.RUSAGE_CHILDREN)
    rec={'stage':a.stage,'attempt':attempt,'model_pid':p.pid,'supervisor_pid':os.getpid(),'start_epoch':s,'end_epoch':time.time(),'exit_code':code,'timed_out':timed_out,
         'full_process_wall_seconds':time.time()-s,'process_CPU_seconds':after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime,'GPU_job':gpu,'max_RSS_KB':after.ru_maxrss}
    (OUT/f'process_{a.stage}_{attempt}.json').write_text(json.dumps(rec,indent=2)+'\n')
    with (OUT/'processes.jsonl').open('a') as f:f.write(json.dumps(rec)+'\n')
    (OUT/'active_process.json').unlink();print(json.dumps(rec),flush=True)
    raise SystemExit(code)


if __name__=='__main__':main()
