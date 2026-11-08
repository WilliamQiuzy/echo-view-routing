#!/usr/bin/env python3
"""Only reconcile the exact closed cost row during normal residual-caller scan.

The immutable canonical receipt and its fees are not filtered or rewritten in
RuntimeState. The original whole-caller execution/wait/publication is reused.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from echo_routing.temporal.selected_risk_resume_contract import RUN,stage_graph
from scripts import a63_wait_resume_selected_risk as original
SOURCE='scripts/a63_wait_measured_cache_ledger_selected_risk.py'


def validate_cost_row(ctx,pin,row):
    from echo_routing.temporal.selected_risk_resume_guardian import start_ticks
    spec=ctx.read(ctx._bound_runtime_spec_pin)
    if pin!=spec['arguments']['measured_cache_probe_parent'] or ctx.read(pin)!=row:
        raise ValueError('only the independently pinned canonical cost parent is compatible')
    if row.get('schema')!='A63_cost_probe_outer_wait4_receipt_v1' or row.get('stage')!='cost_only_cache_probe' or row.get('runtime_authorization_sha256')!=ctx.owner_sha or row.get('payload_pin')!=spec['arguments']['measured_cache_probe']:
        raise ValueError('exact same-owner cost probe schema/payload required')
    if any(row.get(k) is not True for k in ['passed','actual_child_directly_waited','whole_child_accounting_complete']) or any(row.get(k) is not False for k in ['unresolved_worker_uncertainty','GPU_interval_unknown','CUDA_context_occupied','GPU_job']):
        raise ValueError('unknown or unclosed cost parent cannot be reconciled')
    if any(type(row.get(k)) is not int or row[k]!=0 for k in ['actual_child_OS_exit_code','observed_worker_exit_code','stage_exit_status','NN_calls','SGD_updates','model_instances']) or row.get('termination_reason') is not None:
        raise ValueError('only actual closed zero-NN cost parent OS0 is compatible')
    cpu=row.get('measured_CPU_core_seconds')
    if isinstance(cpu,bool) or not isinstance(cpu,(int,float)) or not math.isfinite(cpu) or cpu<=0:raise ValueError('whole cost CPU remains known and charged')
    probe=ctx.read(row['probe_spec_pin']);prefix=probe['output_prefix']
    limits=probe.get('limits')
    if probe.get('schema')!='A63_full_cache_verifier_cost_probe_spec_v1' or probe.get('purpose')!='cost_only_synthetic_nonadoptable' or limits!={'CPU_seconds':600,'wall_seconds':600,'temp_bytes':80000000} or any(type(x) is not int for x in limits.values()):raise ValueError('fixed independent cost-only schema/purpose/limits required')
    if pin['path']!=prefix+'/outer_receipt.json' or row['receipt_id']!='resume_cost_probe:'+prefix.rsplit('/',1)[-1]:raise ValueError('canonical cost namespace/identity required')
    intent_pin=spec['arguments']['measured_cache_probe_outer_intent']
    if intent_pin['path']!=prefix+'/outer_intent.json':raise ValueError('exact independently pinned cost intent path required')
    intent=ctx.read(intent_pin)
    if row['boot_id']!=spec['expected_boot_id'] or probe['runtime_authorization_sha256']!=ctx.owner_sha or any(row[k]!=probe[k] for k in ['source_runtime_spec_pin','source_Git_contract_pin']):raise ValueError('cost current boot/owner/source bindings differ')
    for key in ['source_runtime_spec_pin','source_Git_contract_pin']:ctx.raw(row[key])
    if intent.get('schema')!='A63_cost_only_probe_intent_v1' or intent.get('purpose')!='cost_only_synthetic_nonadoptable' or intent.get('runtime_authorization_sha256')!=ctx.owner_sha or intent.get('probe_spec_pin')!=row['probe_spec_pin'] or intent.get('source_runtime_spec_pin')!=row['source_runtime_spec_pin'] or intent.get('source_Git_contract_pin')!=row['source_Git_contract_pin'] or intent.get('limits')!=probe['limits'] or intent.get('boot_id')!=row['boot_id']:
        raise ValueError('closed cost raw intent bindings differ')
    for pid,ticks in [(intent['parent_pid'],intent['parent_start_ticks']),(row['model_pid'],row['model_start_ticks'])]:
        if type(pid) is not int or pid<=0 or type(ticks) is not int or ticks<=0 or start_ticks(pid)==ticks:raise ValueError('prior cost process still live or identity unknown')
    return True


def residual_callers(ctx):
    closed=set()
    for pin,row in original._rows(ctx,'processes'):
        h=row.get('caller_intent_sha256')
        if h is None:
            if row.get('schema')!='A63_cost_probe_outer_wait4_receipt_v1' or row.get('stage')!='cost_only_cache_probe':raise ValueError('unknown missing caller intent is not a cost probe')
            validate_cost_row(ctx,pin,row)
        else:
            if not isinstance(h,str) or len(h)!=64 or any(x not in '0123456789abcdef' for x in h):raise ValueError('normal caller raw intent SHA required')
            closed.add(h)
    folder=ctx.root/RUN/'caller_intents'
    if folder.is_symlink():raise ValueError('caller intent directory symlink forbidden')
    for path in folder.glob('*.json') if folder.exists() else []:
        raw=original._raw(ctx,str(path.relative_to(ctx.root)))
        if hashlib.sha256(raw).hexdigest() not in closed:raise ValueError('unclosed caller intent: explicit reconciliation, never duplicate or replay')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execute',action='store_true');p.add_argument('--stage',required=True,choices=sorted(stage_graph()));p.add_argument('--runtime-spec');p.add_argument('--runtime-spec-sha');p.add_argument('--owner-sha');a=p.parse_args(argv)
    if not a.execute:print(json.dumps({'preparation_only':True,'new_child_created':False,'stage':a.stage}));return 0
    if not all([a.runtime_spec,a.runtime_spec_sha,a.owner_sha]):p.error('original owner and exact runtime spec required')
    from echo_routing.temporal.selected_risk_resume_runtime import bootstrap
    from scripts.a63_resume_measured_cache_selected_risk import verify_extra_sources
    from scripts.a63_wait_measured_cache_selected_risk import spawn_guardian
    pin={'path':a.runtime_spec,'sha256':a.runtime_spec_sha};ctx,spec=bootstrap(ROOT,pin,a.owner_sha)
    if spec['stage']!=a.stage:raise ValueError('caller/raw spec stage differs')
    verify_extra_sources(ctx,spec);ctx.verify_stage(a.stage,spec['freeze'],required={SOURCE})
    # Sole worker, no other thread/caller; only the residual scan is adapted.
    prior=original._no_unclosed_callers;original._no_unclosed_callers=residual_callers
    try:result=original.execute(ctx,spec,pin,spawn=lambda command,**kw:spawn_guardian(ROOT,command,**kw))
    finally:original._no_unclosed_callers=prior
    print(json.dumps(result,sort_keys=True),flush=True);return result['stage_exit_status']


if __name__=='__main__':raise SystemExit(main())
