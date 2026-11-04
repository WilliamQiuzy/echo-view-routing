#!/usr/bin/env python3
"""A61 exact-once measured local checks/review/probe CPU; scopes retained."""
import json,hashlib,os,socket,time,resource
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0,str(ROOT))
from echo_routing.temporal.anchor_cache import atomic_json,sha
PUB=ROOT/'docs/results/2026-11-04_action_safety_ranking'


def main():
    start=time.time();before=resource.getrusage(resource.RUSAGE_SELF);path=PUB/'local_cpu_receipts.json'
    old=json.loads(path.read_text()) if path.exists() else {'receipts':[],'file_pins':{}}
    records={r['receipt_id']:r for r in old['receipts']};pins=dict(old['file_pins'])
    for p,h in pins.items():
        if sha(ROOT/p)!=h:raise ValueError('raw measured local receipt changed')
    for p in sorted(PUB.glob('*.json')):
        if p.name=='local_cpu_receipts.json' or p.name.startswith(('process_','entry_CPU_')):continue
        v=json.loads(p.read_text());required=['started_epoch','finished_epoch','full_process_wall_seconds','measured_CPU_core_seconds','observed_worker_exit_code']
        row=None
        if all(k in v for k in required):row={k:v[k] for k in required};row.update(measurement_host=v.get('measurement_host'),model_pid=v.get('model_pid'),supervisor_pid=v.get('supervisor_pid'))
        elif p.name=='initial_live_probe.json':row={'started_epoch':v['started_epoch'],'finished_epoch':v['ended_epoch'],'full_process_wall_seconds':v['ended_epoch']-v['started_epoch'],'measured_CPU_core_seconds':v['CPU_seconds'],'observed_worker_exit_code':v['observed_SSH_OS_exit'],'measurement_host':v['measurement_host'],'model_pid':v['supervisor_PID'],'supervisor_pid':None}
        else:
            key=next((k for k in ['self_plus_waitedchildren_cpu_seconds_through_sample','rusage_self_plus_waited_children_total_cpu_seconds'] if k in v),None)
            if key is not None:row={'started_epoch':start,'finished_epoch':start,'full_process_wall_seconds':0.,'measured_CPU_core_seconds':v[key],'observed_worker_exit_code':None,'measurement_host':None,'model_pid':None,'supervisor_pid':None,'interval_scope':'actual CPU from read process through sample; missing exact process interval/PID retained, administrative event timestamp only'}
        if row is None:continue
        digest=sha(p);key='external:'+digest
        if key in records:continue
        relative=str(p.relative_to(ROOT));records[key]={'receipt_id':key,'stage':v.get('stage',p.stem),**row,'GPU_job':False,'CUDA_context_occupied':False,'source_receipt':relative,'source_receipt_sha256':digest,'CPU_scope':v.get('scope',v.get('unmetered_limitation',v.get('unmetered_overhead','actual measured CPU; launcher/final receipt-write/model reasoning unmetered')))};pins[relative]=digest
    # Measured A60 post-snapshot tail was explicitly outside inherited resource baseline.
    prior=ROOT/'docs/results/2026-11-03_jepa_proposals/collector_OS_exit_receipt.json';h=sha(prior)
    key='prior_A60_admin_tail:'+h
    if key not in records:records[key]={'receipt_id':key,'stage':'prior_A60_measured_administrative_tail','started_epoch':start,'finished_epoch':start,'full_process_wall_seconds':0.,'measured_CPU_core_seconds':.052479,'observed_worker_exit_code':None,'measurement_host':None,'model_pid':None,'GPU_job':False,'CUDA_context_occupied':False,'CPU_scope':'completed measured collectorposttail .050352 +parent .002127 sec; separate from A60 resource baseline, not a second model job; summary interval provenance retained'}
    pins[str(prior.relative_to(ROOT))]=h
    after=resource.getrusage(resource.RUSAGE_SELF);end=time.time();own={'receipt_id':f'localcollector:{socket.gethostname()}:{os.getpid()}:{start}','stage':'local_CPU_collection','measurement_host':socket.gethostname(),'model_pid':os.getpid(),'started_epoch':start,'finished_epoch':end,'full_process_wall_seconds':end-start,'measured_CPU_core_seconds':after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime,'observed_worker_exit_code':None,'GPU_job':False,'CUDA_context_occupied':False,'CPU_scope':'running SELF snapshot; final JSON/administrative tools separate, no neural work'}
    records[own['receipt_id']]=own
    value={'receipts':list(records.values()),'file_pins':pins,'receipt_count':len(records),'measured_CPU_seconds':sum(r['measured_CPU_core_seconds'] for r in records.values()),'nested_synthetic_test_fixture_receipts_not_doublecounted':True,'root_modelreasoning_and_tool_coordination_unmetered_separate':True}
    atomic_json(path,value);print({'local_receipts':len(records),'measured_CPU_seconds':value['measured_CPU_seconds']})
if __name__=='__main__':main()
