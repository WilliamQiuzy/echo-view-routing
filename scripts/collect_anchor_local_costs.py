#!/usr/bin/env python3
"""Append all completed local raw software-check receipts exactly once."""
import json,os,socket,time,resource,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];PUB=ROOT/'docs/results/2026-11-03_anchor_edits'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    start=time.time();usage=resource.getrusage(resource.RUSAGE_SELF)
    path=PUB/'local_cpu_receipts.json';old=json.loads(path.read_text());records={r['receipt_id']:r for r in old['receipts']};pins=dict(old['file_pins'])
    for p,v in pins.items():
        if sha(ROOT/p)!=v:raise ValueError('previous raw CPU receipt changed')
    raw=[]
    for p in sorted(PUB.glob('*.json')):
        value=json.loads(p.read_text())
        if (all(k in value for k in ['started_epoch','finished_epoch','full_process_wall_seconds','measured_CPU_core_seconds','observed_worker_exit_code'])
            and value.get('measurement_host')==socket.gethostname() and not value.get('GPU_job',False)):
            raw.append((p,value))
    finish=time.time();after=resource.getrusage(resource.RUSAGE_SELF)
    own={'stage':'local_cost_collection','measurement_host':socket.gethostname(),'model_pid':os.getpid(),'supervisor_pid':None,
         'started_epoch':start,'finished_epoch':finish,'full_process_wall_seconds':finish-start,
         'measured_CPU_core_seconds':after.ru_utime+after.ru_stime-usage.ru_utime-usage.ru_stime,
         'GPU_job':False,'CUDA_context_occupied':False,'observed_worker_exit_code':None,
         'observed_OS_exit_unavailable_for_running_SELF_snapshot':True,
         'scope':'SELF cost of receipt verification/collection; final JSON bookkeeping after snapshot unmetered; no neural work'}
    p=PUB/f'local_cost_collection_{int(start*1000000)}.json';p.write_text(json.dumps(own,sort_keys=True)+'\n');raw.append((p,own))
    for p,value in raw:
        digest=sha(p);relative=str(p.relative_to(ROOT));key='local:'+digest
        if key in records:continue
        if relative in pins and pins[relative]!=digest:raise ValueError('raw receipt identity changed')
        records[key]={'stage':'local_'+value['stage'],'receipt_id':key,**{k:value[k] for k in ['started_epoch','finished_epoch','full_process_wall_seconds',
          'measured_CPU_core_seconds','observed_worker_exit_code']},'measurement_host':value['measurement_host'],'model_pid':value['model_pid'],
          'supervisor_pid':value.get('supervisor_pid'),'GPU_job':False,'CUDA_context_occupied':False,
          'source_receipt':relative,'source_receipt_sha256':digest,'CPU_scope':value.get('scope','local software-check SELF+CHILDREN measured receipt')}
        pins[relative]=digest
    result={**old,'receipts':list(records.values()),'file_pins':pins,'measured_CPU_seconds':sum(r['measured_CPU_core_seconds'] for r in records.values()),
            'test_receipts':len(records),'all_completed_local_raw_checks_including_RED_and_reviewer_merged_once':True,'epoch':time.time()}
    path.write_text(json.dumps(result,sort_keys=True)+'\n')
    print({'receipts':len(records),'measured_CPU_seconds':result['measured_CPU_seconds']})

if __name__=='__main__':main()
