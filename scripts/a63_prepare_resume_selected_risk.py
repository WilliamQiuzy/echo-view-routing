#!/usr/bin/env python3
"""Local freeze/final publication preparation; inspection never writes/launches.

A separately authorized raw new-window owner is mandatory for mutations. This
script cannot train, forward, deploy, acquire a queue or change an old artifact.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from echo_routing.temporal.selected_risk_resume_contract import RUN,PUB,stage_graph


def collect_receipts_only(root,spec_pin,owner_sha):
    """Administrative evidence only, including after expiry; never dispatch.

    Validate historical raw authority at its originally approved start, without
    conferring present work authority. Require an existing launched namespace;
    refuse after its publication manifest has sealed. No queue/child/model exists
    on this branch. Unresolved totals remain null and completion remains false.
    """
    import time,uuid,resource
    from echo_routing.temporal.selected_risk_resume_runtime import read_pin
    from echo_routing.temporal.selected_risk_resume_context import Context
    from echo_routing.temporal.selected_risk_resume_finish import Finish
    spec=read_pin(root,spec_pin);owner=read_pin(root,spec['owner'])
    if spec['owner']['sha256']!=owner_sha or not isinstance(owner.get('T0_epoch'),int) or time.time()<owner['T0_epoch']:raise ValueError('only historically activated, externally pinned new-window receipts may be collected')
    ctx=Context(root,spec['owner'],spec['proposal'],spec['baseline'],now=owner['T0_epoch']+1);ctx.bind_runtime_spec(spec_pin,spec)
    ctx.output('receipt_only_namespace_guard');ctx.output('receipt_only_namespace_guard',public=True)
    if not (ctx.root/RUN/'caller_intents').is_dir() or (ctx.root/RUN/'caller_intents').is_symlink():raise ValueError('no launched namespace: preparation cannot fabricate runtime closure')
    sealed=ctx.root/PUB/'public_source_manifest.json'
    if sealed.exists() or sealed.is_symlink():raise ValueError('sealed public manifest cannot receive new receipt-only files')
    finish=Finish(ctx,None,None,{},{});resources=finish._resources();aux,formal,instances,pending=finish._physical_prefix();intents=[]
    for folder in ['caller_intents','launches']:
        base=ctx.root/RUN/folder
        for path in base.glob('*.json') if base.exists() else []:
            if path.is_symlink():raise ValueError('intent symlink forbidden')
            raw=path.read_bytes();pin={'path':str(path.relative_to(ctx.root)),'sha256':hashlib.sha256(raw).hexdigest()};ctx.raw(pin);intents.append(pin)
    usage=resource.getrusage(resource.RUSAGE_SELF)
    value={'schema':'A63_resume_administrative_incomplete_snapshot_v1','status':'incomplete_receipt_collection','resources':resources,'aux_optimizer_updates_lower_bound':aux,'formal_optimizer_updates_lower_bound':formal,'C0_C1_instances_lower_bound':instances,'pending_private_evidence_paths':pending,'preserved_current_intent_pins':intents,'source_runtime_spec':spec_pin,'runtime_authorization_sha256':owner_sha,'historical_identity_check_does_not_reopen_expired_work_authority':True,'execution_authority_reopened':False,'new_child_or_queue_or_model_or_optimizer':False,'SELF_partial_CPU_seconds_through_snapshot':usage.ru_utime+usage.ru_stime,'SELF_actual_OS_exit_code':None,'snapshot_CPU_is_nested_if_an_external_parent_meter_charges_whole_process':True,'unknown_write_exit_tail_retained':True,'current_orphan_liveness_not_reconciled_or_signalled':True,'overall_complete':False}
    return ctx.write_json('incomplete_receipt_only_'+uuid.uuid4().hex+'.json',value,public=True)


def finalize(ctx,spec,terminal_pin,parent_pin,negative_reviews=None):
    from echo_routing.temporal.selected_risk_resume_runtime import RuntimeState
    terminal=ctx.read(terminal_pin);parent=ctx.read(parent_pin)
    if terminal.get('overall_complete') is not False or parent.get('stage') not in ['close_full','close_negative'] or parent.get('stage_exit_status')!=0 or parent.get('observed_worker_exit_code')!=0 or parent.get('inner_model_observed_OS_exit_code')!=0 or parent.get('actual_child_directly_waited') is not True or parent.get('whole_child_accounting_complete') is not True or parent.get('runtime_authorization_sha256')!=ctx.owner_sha or parent.get('runtime_spec_sha256')!=ctx._bound_runtime_spec_pin['sha256']:raise ValueError('actual same-owner whole-caller direct OS0 must follow the terminal snapshot before final publication')
    state=RuntimeState(ctx,spec,ctx._bound_runtime_spec_pin);spent=state.spent();storage=state.storage()
    if parent['stage']=='close_negative':
        if not isinstance(negative_reviews,dict) or set(negative_reviews)!={'scientific_review','resource_review','visual_QA'}:raise ValueError('negative snapshot still requires independent actual-science/resource/visual disposition before publication')
        for key in ['scientific_review','resource_review']:
            review=ctx.read(negative_reviews[key])
            if review.get('approved') is not True or review.get('independent') is not True or review.get('terminal_snapshot_sha256')!=terminal_pin['sha256'] or review.get('parent_OS_receipt_sha256')!=parent_pin['sha256'] or review.get('runtime_authorization_sha256')!=ctx.owner_sha:raise ValueError('independent negative review must bind actual snapshot/parent OS/current owner')
        qa=ctx.read(negative_reviews['visual_QA'])
        if terminal['status']=='coverage_negative':
            receipt_path=PUB+'/stageS_figures/render_receipt.json';raw=(ctx.root/receipt_path).read_bytes();receipt_sha=hashlib.sha256(raw).hexdigest();ctx.raw({'path':receipt_path,'sha256':receipt_sha});receipt=json.loads(raw)
            expected={PUB+'/'+r['path']:r['sha256'] for r in receipt['files']}
            if qa.get('passed') is not True or qa.get('actual_visual_QA') is not True or qa.get('figure_receipt_sha256')!=receipt_sha or qa.get('file_pins')!=expected:raise ValueError('coverage-negative actual PNG/PDF/CSV QA remains mandatory')
            for path,h in expected.items():ctx.raw({'path':path,'sha256':h})
        elif qa.get('not_applicable') is not True or qa.get('coverage_evaluated') is not terminal['coverage_evaluated'] or qa.get('no_complete_actual_rendering') is not True:raise ValueError('incomplete/unassessed rendering disposition must preserve actual coverage status without an invented QA pass')
    # This final cost cutoff includes the closed collector's entire parent CPU;
    # this administrative finalizer and root publication remain separate tails.
    costs=ctx.write_json('postcaller_cost_cutoff.json',{'canonical_costs':spent,'storage_A63_upper_bytes':storage,'parent_OS_receipt':parent_pin,'terminal_snapshot':terminal_pin,'negative_independent_reviews':negative_reviews,'this_administrative_finalizer_and_root_publication_not_in_cutoff':True,'overall_complete':False},public=True)
    paths=[];folder=ctx.root/PUB
    for p in folder.rglob('*'):
        if p.is_symlink():raise ValueError('public snapshot symlink forbidden')
        if not p.is_file():continue
        rel=str(p.relative_to(ctx.root))
        if rel==PUB+'/public_source_manifest.json':raise FileExistsError('public manifest already frozen; no overwrite')
        raw=p.read_bytes();h=hashlib.sha256(raw).hexdigest();ctx.raw({'path':rel,'sha256':h});paths.append({'path':rel,'sha256':h,'bytes':len(raw)})
    value={'schema':'A63_resume_publication_snapshot_v1','files':sorted(paths,key=lambda r:r['path']),'file_count':len(paths),'self_excluded':PUB+'/public_source_manifest.json','runtime_authorization_sha256':ctx.owner_sha,'terminal_snapshot':terminal_pin,'parent_OS_receipt':parent_pin,'postcaller_cost_cutoff':costs,'source_and_main_actual_Git_copy_and_user_index_final_verification_pending':True,'finalizer_and_root_private_tail_receipts_required':True,'overall_complete':False}
    return ctx.write_json('public_source_manifest.json',value,public=True)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execute',action='store_true');p.add_argument('--make-freeze',action='store_true');p.add_argument('--finalize-publication',action='store_true');p.add_argument('--collect-receipts-only',action='store_true');p.add_argument('--runtime-spec');p.add_argument('--runtime-spec-sha');p.add_argument('--owner-sha');p.add_argument('--git-commit');p.add_argument('--terminal');p.add_argument('--terminal-sha');p.add_argument('--parent-receipt');p.add_argument('--parent-receipt-sha');p.add_argument('--scientific-review');p.add_argument('--scientific-review-sha');p.add_argument('--resource-review');p.add_argument('--resource-review-sha');p.add_argument('--visual-QA');p.add_argument('--visual-QA-sha');a=p.parse_args(argv)
    if not a.execute:
        print(json.dumps({'preparation_only':True,'execution_authorized':False,'registered_stages':list(stage_graph()),'mutations_or_models_or_queue':False}));return 0
    if sum([a.make_freeze,a.finalize_publication,a.collect_receipts_only])!=1 or not all([a.runtime_spec,a.runtime_spec_sha,a.owner_sha]):p.error('select one explicit operation and independently supply new owner/raw spec pins')
    if a.collect_receipts_only:
        print(json.dumps(collect_receipts_only(ROOT,{'path':a.runtime_spec,'sha256':a.runtime_spec_sha},a.owner_sha)));return 0
    from echo_routing.temporal.selected_risk_resume_runtime import bootstrap
    pin={'path':a.runtime_spec,'sha256':a.runtime_spec_sha};ctx,spec=bootstrap(ROOT,pin,a.owner_sha)
    if a.make_freeze:
        if not a.git_commit:p.error('actual locally committed Git HEAD required; deployment has no Git builder')
        from echo_routing.temporal.selected_risk_resume_freeze import make
        contract=make(ctx,spec['stage'],pin,spec,a.git_commit)
        final={**spec,'freeze':contract};out=ctx.write_json('runtime_specs/'+spec['stage']+'.'+contract['sha256'][:12]+'.json',final);ctx.bind_runtime_spec(out,final);ctx.verify_stage(spec['stage'],contract,required=set());print(json.dumps({'runtime_spec':out,'freeze':contract,'actual_stage_executed':False}));return 0
    if not all([a.terminal,a.terminal_sha,a.parent_receipt,a.parent_receipt_sha]):p.error('both terminal and actual directly waited parent receipt external pins required')
    reviews=None
    if a.scientific_review or a.resource_review or a.visual_QA:
        if not all([a.scientific_review,a.scientific_review_sha,a.resource_review,a.resource_review_sha,a.visual_QA,a.visual_QA_sha]):p.error('all negative independent review/QA pins required together')
        reviews={'scientific_review':{'path':a.scientific_review,'sha256':a.scientific_review_sha},'resource_review':{'path':a.resource_review,'sha256':a.resource_review_sha},'visual_QA':{'path':a.visual_QA,'sha256':a.visual_QA_sha}}
    print(json.dumps(finalize(ctx,spec,{'path':a.terminal,'sha256':a.terminal_sha},{'path':a.parent_receipt,'sha256':a.parent_receipt_sha},reviews)));return 0


if __name__=='__main__':raise SystemExit(main())
