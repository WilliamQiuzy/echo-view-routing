#!/usr/bin/env python3
"""Account added reporting work without changing the initial closure receipts."""
import sys,json,time,resource,socket,os
_START=time.time()
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.anchor_common import RUN,PUB,read,verify_pin,verify_freeze
from scripts.supervise_anchor_edits import execute,stage_exit_status,write_json
from echo_routing.temporal.anchor_resources import totals


def main():
    verify_freeze('action_lengths_freeze.json')
    additional=read(PUB/'reporting_additional_cpu_receipts.json')
    for pin in additional['input_pins']:verify_pin(pin)
    records=[json.loads(line) for line in (RUN/'processes.jsonl').read_text().splitlines()]
    records+=read(PUB/'local_cpu_receipts.json')['receipts']+additional['receipts']
    result=execute('action_lengths',[str(ROOT/'.venv/bin/python'),'scripts/audit_anchor_action_lengths.py'],
                   RUN,PUB,totals(records),120.,False)
    own=resource.getrusage(resource.RUSAGE_SELF);finish=time.time()
    overhead=own.ru_utime+own.ru_stime-result['measured_supervisor_CPU_seconds']
    write_json(PUB/'reporting_supervisor_overhead.json',{'receipt_id':'reporting_supervisor_overhead:1',
        'stage':'reporting_supervisor_startup_and_reconciliation','measurement_host':socket.gethostname(),'model_pid':os.getpid(),
        'started_epoch':_START,'finished_epoch':finish,'full_process_wall_seconds':finish-_START,
        'measured_CPU_core_seconds':overhead,'GPU_job':False,'CUDA_context_occupied':False,'observed_worker_exit_code':None,
        'scope':'SELF full CPU minus execute guard SELF delta already charged in action_lengths receipt;child/probes NOT counted again;postwrite bookkeeping unmetered',
        'SELF_snapshot_not_observed_OS_exit':True})
    print(json.dumps(result),flush=True)
    return stage_exit_status(result)

if __name__=='__main__':raise SystemExit(main())
