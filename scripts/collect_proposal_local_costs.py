#!/usr/bin/env python3
"""Exact-once A60 raw local/review/read-only-probe measured CPU receipts."""
import json,os,socket,time,resource,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];PUB=ROOT/'docs/results/2026-11-03_jepa_proposals'

def main():
    start=time.time();before=resource.getrusage(resource.RUSAGE_SELF);path=PUB/'local_cpu_receipts.json'
    old=json.loads(path.read_text()) if path.exists() else {'receipts':[],'file_pins':{}}
    records={r['receipt_id']:r for r in old['receipts']};pins=dict(old['file_pins'])
    for p,h in pins.items():
        if hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h:raise ValueError('raw measured receipt changed')
    for p in sorted(PUB.glob('*.json')):
        if p.name=='local_cpu_receipts.json':continue
        value=json.loads(p.read_text());required=['started_epoch','finished_epoch','full_process_wall_seconds','measured_CPU_core_seconds','observed_worker_exit_code']
        if not all(k in value for k in required) or p.name.startswith('process_'):continue
        digest=hashlib.sha256(p.read_bytes()).hexdigest();relative=str(p.relative_to(ROOT));key='external:'+digest
        if key in records:continue
        records[key]={'stage':value.get('stage',p.stem),'receipt_id':key,**{k:value[k] for k in required},'measurement_host':value.get('measurement_host'),'model_pid':value.get('model_pid'),'supervisor_pid':value.get('supervisor_pid'),'GPU_job':False,'CUDA_context_occupied':False,'source_receipt':relative,'source_receipt_sha256':digest,'CPU_scope':value.get('scope',value.get('CPU_scope','measured source receipt'))}
        pins[relative]=digest
    # These two originally measured document-read process fees only have summary provenance.
    p=PUB/'plan_review.json';v=json.loads(p.read_text());digest=hashlib.sha256(p.read_bytes()).hexdigest()
    for i,cpu in enumerate(v['review_process_CPU_seconds']):
        key=f'planreview:{digest}:{i}'
        if key not in records:records[key]={'stage':'plan_review_document_read','receipt_id':key,'started_epoch':v['epoch'],'finished_epoch':v['epoch'],'full_process_wall_seconds':0.,'measured_CPU_core_seconds':cpu,'GPU_job':False,'CUDA_context_occupied':False,'observed_worker_exit_code':None,'measurement_host':None,'model_pid':None,'CPU_scope':'actual process CPU from approved plan-review summary; exact process/wall provenance unavailable;event timestamp is administrative only'}
    pins[str(p.relative_to(ROOT))]=digest
    end=time.time();after=resource.getrusage(resource.RUSAGE_SELF)
    own={'stage':'local_cost_collection','receipt_id':f'collector:{socket.gethostname()}:{os.getpid()}:{start}','measurement_host':socket.gethostname(),'model_pid':os.getpid(),'started_epoch':start,'finished_epoch':end,'full_process_wall_seconds':end-start,'measured_CPU_core_seconds':after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime,'GPU_job':False,'CUDA_context_occupied':False,'observed_worker_exit_code':None,'CPU_scope':'running SELF collection snapshot; final JSON bookkeeping/coordination metadata unmetered; no neural work'}
    records[own['receipt_id']]=own
    temp=path.with_suffix('.json.tmp');temp.write_text(json.dumps({'receipts':list(records.values()),'file_pins':pins,'measured_CPU_seconds':sum(r['measured_CPU_core_seconds'] for r in records.values()),'receipt_count':len(records),'nested_fixture_receipts_not_scanned':True,'root_coordination_overhead_unmetered_separate':True},sort_keys=True)+'\n');temp.replace(path)
    print({'receipt_count':len(records),'CPU_seconds':sum(r['measured_CPU_core_seconds'] for r in records.values())})
if __name__=='__main__':main()
