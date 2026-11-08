#!/usr/bin/env python3
"""Inspect A63 continuation by default; separate new raw owner required to run.

No deployment or execution is authorized by preparing this script. The child
path additionally requires the live original-flock guardian ticket. No A64.
"""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from echo_routing.temporal.selected_risk_resume_contract import stage_graph

IMPLEMENTED=set(stage_graph())



def _adopt(context,spec,inputs):
    from echo_routing.temporal.selected_risk_resume_contract import OLD_MANIFEST
    context.enter('adopt_metadata',spec['freeze'],spec['predecessors'],required={'scripts/a63_resume_selected_risk.py','echo_routing/temporal/selected_risk_resume_inputs.py'})
    from echo_routing.temporal.selected_risk_resume_context import CLOSED_MANIFEST_ALIAS
    publication=context.verify_closed_publication({'path':CLOSED_MANIFEST_ALIAS,'sha256':OLD_MANIFEST})
    m=inputs.manifest
    for p in [m['partition_mapping'],*m['recipes'].values()]:context.raw(p)
    counts={r:len(inputs.sequences(r)) for r in ['FIT_native','F0_native','F1_native','F0_mixed','F1_mixed','a63_confirmation_selection','a63_confirmation_calibration']}
    if [counts[r] for r in ['FIT_native','F0_native','F1_native','F0_mixed','F1_mixed','a63_confirmation_selection','a63_confirmation_calibration']]!=[2736,1366,1370,1500,1500,120,120]:raise ValueError('exact original metadata, no new recipe or partition generation')
    return context.write_json('metadata_adoption.json',{'passed':True,'original_metadata_pin':spec['metadata'],'role_counts':counts,'partition_counts':[201,201],'new_recipes_or_redraws':0,'new_teacher_forwards':0,'aux_optimizer_updates':0,'formal_optimizer_updates':0,'old_publication_manifest_sha256':OLD_MANIFEST,'old_namespaces_written':False,'closed_publication_readonly_verification':publication},public=True)


def _admit(context,spec,spec_pin):
    from echo_routing.temporal.selected_risk_resume_runtime import RuntimeState
    state=RuntimeState(context,spec,spec_pin);value=state.admission();work=value['remaining_complete_workload'];e={k:context.evidence(v) for k,v in spec['predecessors'].items()};stage=spec['stage']
    out={**value,'passed':value['all_caps_fit'],'stage':stage,'input_manifest_sha256':spec['metadata']['sha256'],'source_phase':'FIT' if stage=='admit_FIT' else 'CONFIRM','runtime_authorization_sha256':context.owner_sha,'prepared_data_atomic_bytes_upper':work['prepared_data_atomic_bytes_upper'],'formal_data_atomic_bytes_upper':work['formal_data_atomic_bytes_upper'],'selected_table_atomic_bytes_upper':work['selected_table_atomic_bytes_upper'],'forecast_is_not_actual_spend_or_guarantee':True}
    for key,dep in [('geometry_sha256','observe_geometry'),('producer_FIT_sha256','source_FIT'),('completed_CONFIRM_sha256','source_CONFIRM'),('prepared_formal_sha256','prepare_formal')]:
        if dep in e:out[key]=e[dep]['payload_sha256']
    pin=context.write_json(stage+'.'+context.runtime_spec_semantics_sha256[:12]+'.json',out,public=True)
    if out['passed'] is not True:raise ValueError('full remaining original caps/window do not fit; failed admission retained')
    return pin


def dispatch(context,spec,spec_pin):
    from echo_routing.temporal.selected_risk_resume_seals import VerifiedOuterInputs
    stage=spec['stage'];inputs=VerifiedOuterInputs(context,spec['metadata'],FIT_producer_pin=spec['FIT_producer'],CONFIRM_producer_pin=spec['CONFIRM_producer'])
    if stage=='adopt_metadata':return _adopt(context,spec,inputs)
    if stage.startswith('admit_'):return _admit(context,spec,spec_pin)
    if stage=='source_FIT_adoption':
        from echo_routing.temporal.selected_risk_resume_source_adoption import SourceAdoption
        return SourceAdoption(context,inputs,spec['freeze'],spec['predecessors'],spec['arguments']).source_FIT_adoption()
    if stage in ['seal_aux','seal_selectors','seal_heads','adopt_confirmation']:
        from echo_routing.temporal.selected_risk_resume_seals import Seals
        return getattr(Seals(context,inputs,spec['freeze'],spec['predecessors'],spec['arguments']),stage)()
    if stage in ['source_FIT','source_CONFIRM']:
        from echo_routing.temporal.selected_risk_resume_source import Source
        return Source(context,inputs,spec['freeze'],spec['predecessors']).produce(stage.split('_')[1])
    if stage in ['observe_geometry','prepare_aux','train_aux','audit_aux']:
        from echo_routing.temporal.selected_risk_resume_S import StageS
        return getattr(StageS(context,inputs,spec['freeze'],spec['predecessors']),stage)()
    if stage in ['held_score','coverage']:
        from echo_routing.temporal.selected_risk_resume_supply import Supply
        return getattr(Supply(context,inputs,spec['freeze'],spec['predecessors']),stage)()
    if stage in ['prepare_formal','train_formal','audit_formal']:
        from echo_routing.temporal.selected_risk_resume_formal import Formal
        return getattr(Formal(context,inputs,spec['freeze'],spec['predecessors']),stage)()
    if stage in ['fit_risk_scores','calibrate','outer','aggregate']:
        from echo_routing.temporal.selected_risk_resume_readout import Readout
        return getattr(Readout(context,inputs,spec['freeze'],spec['predecessors']),stage)()
    if stage in ['supports','forensics','package']:
        from echo_routing.temporal.selected_risk_resume_diagnostics import Diagnostics
        return getattr(Diagnostics(context,inputs,spec['freeze'],spec['predecessors']),stage)()
    if stage in ['scientific_review','resource_review','close_negative','close_full']:
        from echo_routing.temporal.selected_risk_resume_finish import Finish
        return getattr(Finish(context,inputs,spec['freeze'],spec['predecessors'],spec['arguments']),stage)()
    raise ValueError('implementation incomplete: no placeholder can execute or claim completion')


def launch(a):
    from echo_routing.temporal.selected_risk_resume_runtime import bootstrap,RuntimeState,validate_child_ticket
    pin={'path':a.runtime_spec,'sha256':a.runtime_spec_sha};context,spec=bootstrap(ROOT,pin,a.owner_sha)
    if spec['stage']!=a.stage:raise ValueError('CLI stage differs from independently pinned runtime spec')
    if a.stage not in IMPLEMENTED:raise ValueError('requested continuation caller is not yet implemented')
    if a.child:
        validate_child_ticket(context,pin,{'path':a.guardian_intent,'sha256':a.guardian_intent_sha},a.guardian_lock_fd,a.stage)
        result=dispatch(context,spec,pin);print(json.dumps({'stage':a.stage,'output':result,'stage_returned_success':True,'runtime_authorization_sha256':context.owner_sha}),flush=True);return 0
    from echo_routing.temporal.selected_risk_resume_guardian import execute
    result=execute(RuntimeState(context,spec,pin),a.stage);print(json.dumps(result,sort_keys=True),flush=True);return result['stage_exit_status']


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execute',action='store_true');p.add_argument('--stage',required=True,choices=sorted(stage_graph()));p.add_argument('--runtime-spec');p.add_argument('--runtime-spec-sha');p.add_argument('--owner-sha');p.add_argument('--child',action='store_true');p.add_argument('--guardian-intent');p.add_argument('--guardian-intent-sha');p.add_argument('--guardian-lock-fd',type=int)
    a=p.parse_args(argv)
    if not a.execute:
        print(json.dumps({'preparation_only':True,'execution_authorized':False,'requested_stage':a.stage,'implemented_stage_callers':sorted(IMPLEMENTED),'missing_stage_callers':sorted(set(stage_graph())-IMPLEMENTED),'all_stage_dispatchers_implemented':set(IMPLEMENTED)==set(stage_graph()),'full_chain_runtime_ready':False,'new_fixed_window_authorized':False,'new_runtime_directories_created':False}));return 0
    if not all([a.runtime_spec,a.runtime_spec_sha,a.owner_sha]):p.error('--execute requires independently supplied runtime-spec/raw SHA and new owner raw SHA; old owner is insufficient')
    if a.child and (not a.guardian_intent or not a.guardian_intent_sha or a.guardian_lock_fd is None):p.error('child requires inherited live guardian intent/raw SHA/original flock descriptor')
    os.environ['CUDA_VISIBLE_DEVICES']='0' if a.stage in ['source_FIT','source_CONFIRM'] else ''
    for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
    os.environ['PYTHONDONTWRITEBYTECODE']='1';os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    return launch(a)


if __name__=='__main__':raise SystemExit(main())
