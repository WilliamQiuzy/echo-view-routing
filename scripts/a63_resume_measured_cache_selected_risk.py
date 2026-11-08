#!/usr/bin/env python3
"""Same scientific children, with separately reviewed completed trace/cache admission.

The original prepared/trained sources are immutable. This administrative entry
retains the original guardian, queue, tickets, caps and direct wait contract.
Only admission children use this file; all other scientific children retain the
original CLI. Inspection creates no process or namespace.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from echo_routing.temporal.selected_risk_resume_contract import stage_graph

EXTRA_SOURCES=(
 'echo_routing/temporal/selected_risk_completed_aux_controller.py',
 'scripts/a63_resume_completed_aux_selected_risk.py',
 'scripts/a63_wait_completed_aux_selected_risk.py',
 'scripts/a63_prepare_completed_aux_selected_risk.py',
 'echo_routing/temporal/selected_risk_completed_supply_controller.py',
 'scripts/a63_resume_completed_supply_selected_risk.py',
 'scripts/a63_wait_completed_supply_selected_risk.py',
 'scripts/a63_prepare_completed_supply_selected_risk.py',
 'echo_routing/temporal/selected_risk_measured_cache_controller.py',
 'scripts/a63_resume_measured_cache_selected_risk.py',
 'scripts/a63_wait_measured_cache_selected_risk.py',
 'scripts/a63_prepare_measured_cache_selected_risk.py',
 'scripts/a63_probe_completed_supply_selected_risk_cost.py',
 'scripts/a63_wait_completed_supply_selected_risk_cost.py',
)


def verify_extra_sources(context,spec):
    # The unchanged Context verifies every current byte and actual Git object,
    # as well as its full original required closure and eight historical pins.
    return context.verify_stage(spec['stage'],spec['freeze'],required=set(EXTRA_SOURCES))


def state_class():
    from echo_routing.temporal.selected_risk_measured_cache_controller import MeasuredCacheRuntimeState
    class RoutedState(MeasuredCacheRuntimeState):
        def admission(self):
            self._actual_fresh_admission=None
            value=super().admission()
            # Preserve the actual numeric bounds used by this guardian. Previous
            # canonical rows remain independently stored; never recursively
            # embed them in each new guardian/caller snapshot.
            spent=dict(value['already_spent']);rows=spent.pop('receipt_records')
            raw=json.dumps(rows,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
            spent.update(receipt_records_canonical_json_sha256=hashlib.sha256(raw).hexdigest(),
              receipt_records_count=len(rows),
              receipt_records_scope='canonical original receipt rows are hashed here, not new charges; independently verify original ledger and immutable receipts')
            snapshot={**value,'already_spent':spent}
            encoded=json.dumps(snapshot,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
            if len(encoded)>128*1024:raise ValueError('compact actual admission snapshot exceeds its128KiB metadata bound')
            self._actual_fresh_admission=json.loads(encoded)
            return value

        def record(self,row,intent):
            return super().record({**row,'actual_fresh_completed_aux_admission':getattr(self,'_actual_fresh_admission',None)},intent)

        def command(self,stage,intent,lock_fd):
            command=super().command(stage,intent,lock_fd)
            if stage.startswith('admit_'):
                if command[1]!=str(self.ctx.root/'scripts/a63_resume_selected_risk.py'):
                    raise ValueError('only the exact qualified admission entry may be substituted')
                command[1]=str(self.ctx.root/'scripts/a63_resume_measured_cache_selected_risk.py')
            return command
    return RoutedState


def admission_output(context,spec,spec_pin):
    # Identical public admission schema to the original pure-IO helper, with
    # the independently reviewed controller supplying the complete forecast.
    state=state_class()(context,spec,spec_pin);value=state.admission()
    work=value['remaining_complete_workload'];stage=spec['stage']
    e={k:context.evidence(v) for k,v in spec['predecessors'].items()}
    out={**value,'passed':value['all_caps_fit'],'stage':stage,
      'input_manifest_sha256':spec['metadata']['sha256'],
      'source_phase':'FIT' if stage=='admit_FIT' else 'CONFIRM',
      'runtime_authorization_sha256':context.owner_sha,
      'prepared_data_atomic_bytes_upper':work['prepared_data_atomic_bytes_upper'],
      'formal_data_atomic_bytes_upper':work['formal_data_atomic_bytes_upper'],
      'selected_table_atomic_bytes_upper':work['selected_table_atomic_bytes_upper'],
      'forecast_is_not_actual_spend_or_guarantee':True}
    for key,dep in [('geometry_sha256','observe_geometry'),('producer_FIT_sha256','source_FIT'),
                    ('completed_CONFIRM_sha256','source_CONFIRM'),('prepared_formal_sha256','prepare_formal')]:
        if dep in e:out[key]=e[dep]['payload_sha256']
    pin=context.write_json(stage+'.'+context.runtime_spec_semantics_sha256[:12]+'.json',out,public=True)
    if out['passed'] is not True:
        raise ValueError('full remaining original caps/window do not fit; failed admission retained')
    return pin


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute',action='store_true');p.add_argument('--stage',required=True,choices=sorted(stage_graph()))
    p.add_argument('--runtime-spec');p.add_argument('--runtime-spec-sha');p.add_argument('--owner-sha')
    p.add_argument('--child',action='store_true');p.add_argument('--guardian-intent')
    p.add_argument('--guardian-intent-sha');p.add_argument('--guardian-lock-fd',type=int)
    a=p.parse_args(argv)
    if not a.execute:
        print(json.dumps({'preparation_only':True,'execution_authorized':False,'new_child_created':False,'stage':a.stage}));return 0
    if not all([a.runtime_spec,a.runtime_spec_sha,a.owner_sha]):
        p.error('independent original-window owner and exact raw spec pins are required')
    os.environ['CUDA_VISIBLE_DEVICES']='0' if a.stage in ['source_FIT','source_CONFIRM'] else ''
    for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
    os.environ['PYTHONDONTWRITEBYTECODE']='1'
    from echo_routing.temporal.selected_risk_resume_runtime import bootstrap,validate_child_ticket
    pin={'path':a.runtime_spec,'sha256':a.runtime_spec_sha};ctx,spec=bootstrap(ROOT,pin,a.owner_sha)
    if spec['stage']!=a.stage:raise ValueError('CLI/raw spec stage differs')
    verify_extra_sources(ctx,spec)
    if a.child:
        if not a.stage.startswith('admit_') or not all([a.guardian_intent,a.guardian_intent_sha]) or a.guardian_lock_fd is None:
            p.error('only a ticketed admission child may use this administrative entry')
        validate_child_ticket(ctx,pin,{'path':a.guardian_intent,'sha256':a.guardian_intent_sha},a.guardian_lock_fd,a.stage)
        result=admission_output(ctx,spec,pin)
        print(json.dumps({'stage':a.stage,'output':result,'stage_returned_success':True,'runtime_authorization_sha256':ctx.owner_sha}),flush=True);return 0
    from echo_routing.temporal.selected_risk_resume_guardian import execute
    result=execute(state_class()(ctx,spec,pin),a.stage)
    print(json.dumps(result,sort_keys=True),flush=True);return result['stage_exit_status']


if __name__=='__main__':raise SystemExit(main())
