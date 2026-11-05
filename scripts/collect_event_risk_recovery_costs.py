#!/usr/bin/env python3
"""Separate terminal-recovery fee ledger; original pinned ledger never rewritten."""
import hashlib,json,math,os,resource,socket,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from echo_routing.temporal.anchor_cache import atomic_json,sha
PUB=ROOT/'docs/results/2026-11-04_anchor_context_risk'
KEYS=('known_measured_tool_CPU_seconds','known_measured_tool_CPU_seconds_to_snapshot','CPU_SELF_plus_waitedchildren_seconds','measured_SELF_plus_waited_children_CPU_seconds','rusage_self_plus_waited_children_cpu_seconds_through_sample','self_plus_waitedchildren_cpu_seconds_through_sample','rusage_self_plus_waited_children_total_cpu_seconds','CPU_SELF_seconds_through_sample','authorization_write_SELF_CPU_seconds_through_sample','measured_CPU_core_seconds')


def collect(pub=PUB,root=ROOT):
    pub,root=Path(pub),Path(root);start=time.time();before=resource.getrusage(resource.RUSAGE_SELF);path=pub/'local_cpu_receipts_terminal_recovery.json';base=pub/'local_cpu_receipts.json';old=json.loads((path if path.exists() else base).read_text())
    records={r['receipt_id']:r for r in old['receipts']};pins=dict(old['file_pins'])
    if len(records)!=len(old['receipts']):raise ValueError('prior duplicate receipt identities')
    for p,h in pins.items():
        if sha(root/p)!=h:raise ValueError('raw immutable measured receipt changed')
    for p in sorted(pub.glob('*.json')):
        if p.name in ['local_cpu_receipts.json','local_cpu_receipts_terminal_recovery.json'] or p.name.startswith(('process_','entry_CPU_')):continue
        v=json.loads(p.read_text())
        approved_type=(p.name in ['planning_review_CPU.json','execution_authorization.json'] or p.name.startswith(('root_','review_')) or p.name.startswith('a62_') and p.name.endswith('receipt.json') or v.get('durable_claim_before_Popen') is True and isinstance(v.get('command'),list) and 'pytest' in v['command'])
        if not approved_type:continue
        key=next((k for k in KEYS if k in v),None)
        if key is None:continue
        cpu=v[key]
        if isinstance(cpu,bool) or not isinstance(cpu,(int,float)) or not math.isfinite(cpu) or cpu<0:raise ValueError('real finite CPU receipt value required')
        digest=sha(p);relative=str(p.relative_to(root));pins[relative]=digest;identity='external:'+digest
        if identity in records:continue
        if all(k in v for k in ['started_epoch','finished_epoch','full_process_wall_seconds','observed_worker_exit_code']):
            fields={k:v[k] for k in ['started_epoch','finished_epoch','full_process_wall_seconds','observed_worker_exit_code']};fields.update(measurement_host=v.get('measurement_host'),model_pid=v.get('model_pid'),supervisor_pid=v.get('supervisor_pid'))
        else:fields={'started_epoch':start,'finished_epoch':start,'full_process_wall_seconds':0.,'observed_worker_exit_code':None,'measurement_host':None,'model_pid':None,'supervisor_pid':None,'interval_scope':'receipt-based CPU identity; event timestamp not claimed original process interval'}
        records[identity]={'receipt_id':identity,'stage':v.get('stage',p.stem),**fields,'measured_CPU_core_seconds':float(cpu),'GPU_job':False,'CUDA_context_occupied':False,'source_receipt':relative,'source_receipt_sha256':digest,'source_CPU_key':key,'scope':v.get('scope',v.get('charge_scope','actual partial CPU only; model reasoning/tool/final write/exit tail unmetered'))}
    after=resource.getrusage(resource.RUSAGE_SELF);end=time.time();r={'receipt_id':f'localcollector:{socket.gethostname()}:{os.getpid()}:{start}','stage':'local_CPU_collection','measurement_host':socket.gethostname(),'model_pid':os.getpid(),'started_epoch':start,'finished_epoch':end,'full_process_wall_seconds':end-start,'measured_CPU_core_seconds':after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime,'GPU_job':False,'CUDA_context_occupied':False,'observed_worker_exit_code':None,'scope':'SELF through sample; no experiments; final record write/exit tail unmetered'};records[r['receipt_id']]=r
    value={'receipts':list(records.values()),'file_pins':pins,'receipt_count':len(records),'measured_CPU_seconds':sum(x['measured_CPU_core_seconds'] for x in records.values()),'partial_review_CPU_not_model_reasoning':True,'nested_synthetic_test_children_not_doublecounted':True,'prior_A61_costs_not_recharged':True}
    atomic_json(path,value);return value

if __name__=='__main__':
    v=collect();print({'local_receipts':v['receipt_count'],'measured_CPU_seconds':v['measured_CPU_seconds']})
