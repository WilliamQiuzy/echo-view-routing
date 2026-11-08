"""Concrete review adoption and honest full/negative continuation closure.

No model/source forward, threshold, sampler or optimizer is available here.
Closing snapshots are cutoffs: current parent wait4 and publication are later
mandatory evidence. They never pre-certify their own OS or main integration.
"""
import hashlib
import json
import math
import os
import time
from pathlib import Path
from echo_routing.temporal.selected_risk_resume_contract import RUN,PUB,stage_graph,storage_cap

SOURCE='echo_routing/temporal/selected_risk_resume_finish.py'


def validate_review(value,scope,owner_sha,package_sha,summary_sha):
    if value.get('approved') is not True or value.get('independent') is not True or (value.get('scope'),value.get('runtime_authorization_sha256'),value.get('package_sha256'),value.get('summary_sha256'))!=(scope,owner_sha,package_sha,summary_sha) or not isinstance(value.get('reviewer_identity'),str) or not value['reviewer_identity'].strip():raise ValueError('actual independent review of exact current snapshot required; no automatic approval')
    return True


def negative_state(mode,coverage,aux_updates,formal_updates,instances):
    if mode=='incomplete':
        return {'passed':True,'status':'incomplete','coverage_description':('unassessed' if coverage is None else 'passed the fixed supply gates' if coverage.get('passed') is True else 'failed the fixed supply gates'),'coverage_evaluated':coverage is not None,'coverage_passed':None if coverage is None else coverage.get('passed'),'failed_coverage_checks':None,'aux_optimizer_updates':aux_updates,'formal_optimizer_updates':formal_updates,'C0_C1_instances':instances,'pass_branch':{k:'partial_or_completed_prefix_preserved' for k in ['formal','calibration','outer','paired_CI','supports','native_regression']},'scientific_goal_evaluated':False,'research_goal_succeeded':None,'overall_complete':False}
    if mode not in ['resource_stopped','coverage_negative'] or formal_updates!=0 or instances!=0:raise ValueError('negative preformal closure cannot conceal C0/C1 construction or updates')
    if mode=='coverage_negative':
        from echo_routing.temporal.selected_risk_terminal import _coverage
        failed=_coverage(coverage)
        if aux_updates!=48000:raise ValueError('negative coverage requires actual complete48000 auxiliary updates')
    else:
        if coverage is not None:raise ValueError('resource-stopped unassessed coverage stays null, never coverage failure')
        failed=[]
    return {'status':mode,'coverage_description':('unassessed' if coverage is None else 'passed the fixed supply gates' if coverage.get('passed') is True else 'failed the fixed supply gates'),'coverage_evaluated':coverage is not None,'coverage_passed':None if coverage is None else False,'failed_coverage_checks':failed,'aux_optimizer_updates':aux_updates,'formal_optimizer_updates':0,'C0_C1_instances':0,'passed':True,'pass_branch':{k:'not_run' for k in ['formal','calibration','outer','paired_CI','supports','native_regression']},'scientific_goal_evaluated':False,'research_goal_succeeded':None,'overall_complete':False}


def validate_full(package,actual_caps_fit):
    expected=dict(outer=238,comparisons=140,inner=34,support=56,native=34,regression=32,curves=24000)
    if package.get('passed') is not True or package.get('counts')!=expected or package.get('actual_visual_QA_passed') is not True or actual_caps_fit is not True:raise ValueError('complete actual matrix, external visual QA and known actual-cap compliance required')
    return True


def write_text(ctx,name,value):
    ctx.revalidate();path=PUB+'/'+name;fd,leaf=ctx._parent(path,create=True)
    try:
        child=os.open(leaf,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600,dir_fd=fd)
        with os.fdopen(child,'wb') as f:f.write(value.encode());f.flush();os.fsync(f.fileno())
        os.fsync(fd)
    finally:os.close(fd)
    return {'path':path,'sha256':hashlib.sha256(value.encode()).hexdigest()}


class Finish:
    def __init__(self,context,inputs,freeze_pin,descriptors,arguments):self.ctx=context;self.inputs=inputs;self.freeze_pin=freeze_pin;self.descriptors=descriptors;self.args=arguments
    def _enter(self,stage):return self.ctx.enter(stage,self.freeze_pin,self.descriptors,required={SOURCE})
    def _state(self):
        from echo_routing.temporal.selected_risk_resume_runtime import RuntimeState
        return RuntimeState(self.ctx,self.ctx.read(self.ctx._bound_runtime_spec_pin),self.ctx._bound_runtime_spec_pin)
    def _package(self,e):
        pin=self.descriptors['package']['payload_pin'];p=e['package']['payload']
        if p.get('passed') is not True:raise ValueError('actual completed package required')
        for item in p['file_pins']:self.ctx.raw(item)
        return pin,p
    def _review(self,stage,scope):
        freeze,e=self._enter(stage);pin,p=self._package(e);review_pin=self.args['independent_review'];review=self.ctx.read(review_pin);summary_sha=self.descriptors['aggregate']['payload_pin']['sha256'];validate_review(review,scope,self.ctx.owner_sha,pin['sha256'],summary_sha)
        if review.get('figure_receipt_sha256')!=p['figure_receipt']['sha256'] or review.get('model_audit_sha256')!=self.descriptors['audit_formal']['payload_pin']['sha256']:raise ValueError('review must bind actual figures and independent final model audit')
        if scope=='resource_history_OS':
            state=self._state();workers=state._worker_rows();ledger=state.ctx.root/RUN/'processes.jsonl';raw=ledger.read_bytes()
            if review.get('canonical_process_ledger_sha256')!=hashlib.sha256(raw).hexdigest() or review.get('closed_receipt_ids')!=[r['receipt_id'] for r in workers]:raise ValueError('independent resource review must cover the exact current direct-waited worker ledger')
            if any(r.get('whole_child_accounting_complete') is not True or r.get('unresolved_worker_uncertainty') is True for r in workers):raise ValueError('unresolved worker cannot receive complete resource review')
            state.spent()
        return self.ctx.write_json(stage+'_adoption.json',{'passed':True,'independent_review':review_pin,'package_sha256':pin['sha256'],'summary_sha256':summary_sha,'reviewer_identity':review['reviewer_identity'],'scope':scope,'approval_is_accuracy_not_positive_research_result':True},public=True)
    def scientific_review(self):return self._review('scientific_review','scientific_accuracy')
    def resource_review(self):return self._review('resource_review','resource_history_OS')
    def _resources(self):
        state=self._state();workers=state._worker_rows();error=None
        try:spent=state.spent()
        except ValueError as exc:
            # Preserve exact unknown raw workers. Never replace unknown total by
            # zero, partial SELF, a plan count, or a prospective cap.
            spent=None;error=str(exc)
        storage=None
        try:storage=state.storage()
        except (ValueError,OSError) as exc:error=(error or '')+'; storage '+str(exc)
        now=time.time();caps=None if spent is None or storage is None else (spent['new_GPU_wall_seconds']<=5400 and spent['new_CPU_accounting_seconds']<=28800 and spent['cumulative_GPU_hours']+2<=24 and spent['cumulative_CPU_accounting_hours']<=72 and storage<=storage_cap(self.ctx.owner) and 38650409858+storage<=40000000000 and now<=self.ctx.owner['deadline_epoch'])
        return {'canonical_closed_costs':spent,'actual_worker_receipts':workers,'tracked_storage_upper_bytes':storage,'storage_cap':storage_cap(self.ctx.owner),'storage_amendment_sha256':self.ctx.owner.get('storage_amendment_sha256'),'storage_approval_receipt_sha256':self.ctx.owner.get('storage_approval_receipt',{}).get('sha256'),'storage_scope':'declared complete local bound plus current old/new RUN/PUB logical bytes; not physical blocks/RAM/global host peak','actual_caps_fit':caps,'unknown_error':error,'cutoff_epoch':now,'current_collector_parent_wait4_not_yet_in_snapshot':True,'tail_and_main_publication_pending':True,'overall_complete':False}
    def close_full(self):
        freeze,e=self._enter('close_full');pin,p=self._package(e);qa=self.ctx.read(self.args['visual_QA'])
        if qa.get('passed') is not True or qa.get('actual_visual_QA') is not True or qa.get('figure_receipt_sha256')!=p['figure_receipt']['sha256']:raise ValueError('independent actual PNG/PDF/CSV QA must bind exact figure receipt')
        receipt=self.ctx.read(p['figure_receipt'])
        expected={PUB+'/'+r['path']:r['sha256'] for r in receipt['files']}
        if qa.get('file_pins')!=expected:raise ValueError('actual visual QA must bind every rendered asset exact byte SHA')
        for path,h in expected.items():self.ctx.raw({'path':path,'sha256':h})
        resources=self._resources();validate_full({'passed':p['passed'],'counts':receipt['counts'],'actual_visual_QA_passed':qa['passed']},resources['actual_caps_fit'])
        summary=e['aggregate']['payload'];acceptance=self.ctx.read({'path':PUB+'/acceptance.json','sha256':summary['acceptance_sha256']})
        from echo_routing.temporal.selected_risk_metrics import acceptance as validate_acceptance
        if acceptance!=validate_acceptance(summary['rows'],summary['paired_comparisons']):raise ValueError('scientific verdict differs from exact sealed full point/CI matrix')
        rp=self.ctx.write_json('resources.json',resources,public=True)
        return self.ctx.write_json('terminal_snapshot.json',{'passed':True,'scientific_matrix_complete':True,'research_verdict':acceptance,'resources':rp,'actual_caps_fit':resources['actual_caps_fit'],'independent_reviews':{k:self.descriptors[k]['payload_pin'] for k in ['scientific_review','resource_review']},'visual_QA':self.args['visual_QA'],'actual_parent_OS_receipt_pending':True,'root_main_publication_pending':True,'overall_complete':False},public=True)
    def _physical_prefix(self):
        from echo_routing.temporal.selected_risk_storage_engineering import load_trace
        run=self.ctx.root/RUN;aux=0;pending=[];models=run/'models';instances=len(list(models.glob('seed*/C[01]'))) if models.exists() else 0;formal=0
        for family in ['aux','models']:
            folder=run/family
            heads=sorted({p.parent for p in folder.rglob('physical_updates.jsonl*')}) if folder.exists() else []
            for head in heads:
                raw=head/'physical_updates.jsonl';gz=head/'physical_updates.jsonl.gz';ar=head/'physical_updates.jsonl.archive.json'
                for p in [raw,gz,ar]:
                    if p.is_symlink():raise ValueError('physical trace symlink forbidden')
                if gz.exists() and ar.exists():
                    archive=json.loads(ar.read_text());rows=load_trace(gz,archive)
                    if raw.exists() and hashlib.sha256(raw.read_bytes()).hexdigest()!=archive['uncompressed_sha256']:raise ValueError('coexisting raw/archive prefixes disagree')
                elif raw.exists():rows=[json.loads(x) for x in raw.read_bytes().splitlines()]
                else:
                    pending.append(str(head.relative_to(self.ctx.root)));continue
                if any(r['u']!=i for i,r in enumerate(rows)):raise ValueError('ordered physical committed update prefix required')
                if family=='aux':aux+=len(rows)
                else:formal+=len(rows)
            pending.extend(str(p.relative_to(self.ctx.root)) for p in folder.rglob('*pending*') if p.is_file()) if folder.exists() else None
        return aux,formal,instances,pending
    def close_negative(self):
        freeze,e=self._enter('close_negative');aux,formal,instances,pending=self._physical_prefix();coverage=e.get('coverage',{}).get('payload');mode=self.args['negative_mode'];workers=self._state()._worker_rows();formal_attempted=any(r['stage'] in ['train_formal','audit_formal','fit_risk_scores','calibrate','outer','aggregate','supports','forensics','package'] for r in workers)
        if formal_attempted and mode!='incomplete':raise ValueError('attempted postcoverage formal/readout work requires honest incomplete mode, never preformal not_run')
        state=negative_state(mode,coverage,aux,formal,instances)
        if state['status']=='coverage_negative':
            from echo_routing.temporal.selected_risk_resume_contract import _successful
            _successful('audit_aux',e['audit_aux']['payload']);_successful('source_FIT',e['source_FIT']['payload'])
            from echo_routing.temporal.selected_risk_resume_present import render_stageS
            from echo_routing.temporal.selected_risk_storage_engineering import load_trace
            curves=[]
            for model in e['audit_aux']['payload']['models']:
                folder=(self.ctx.root/model['path']).parent;archive=json.loads((folder/'physical_updates.jsonl.archive.json').read_text());rows=load_trace(folder/'physical_updates.jsonl.gz',archive)
                curves.extend({'seed':model['seed'],'held_fold':model['held_fold'],'role':model['role'],'update':r['u']+1,'loss':r['loss']} for r in rows)
            render_stageS(self.ctx.output('coverage',public=True).parent,coverage,curves,synthetic=False)
        resources=self._resources();rp=self.ctx.write_json('resources_negative.json',resources,public=True)
        state.update(aux_optimizer_updates_lower_bound=aux,aux_optimizer_updates=None if pending else aux,formal_optimizer_updates_lower_bound=formal,formal_optimizer_updates=None if pending else formal,C0_C1_instances_lower_bound=instances,C0_C1_instances=None if formal_attempted and instances==0 else instances,resources=rp,pending_private_evidence_paths=pending,source_manifest_pins={k:self.descriptors[k]['payload_pin'] for k in ['source_FIT','source_CONFIRM'] if k in self.descriptors},teacher_forward_units=None,teacher_units_not_inferred_from_planned_payloads=True,old_resource_stop_history_unchanged=True,actual_parent_OS_receipt_pending=True,root_main_publication_pending=True)
        saved={}
        for name,key in [('outer_metrics.json','row_count'),('summary.json','comparison_count'),('support_summary.json','record_count'),('forensics_summary.json','native_count')]:
            path=self.ctx.root/PUB/name
            saved[name]=None if not path.exists() else json.loads(path.read_bytes()).get(key)
        state['persisted_endpoint_counts']=saved;state['private_paired_bundle_files']=len(list((self.ctx.root/RUN/'paired').rglob('*.private.npz'))) if (self.ctx.root/RUN/'paired').exists() else 0;state['absent_public_output_is_not_evidence_no_calculation_ran']=True
        write_text(self.ctx,'NEGATIVE_FINDINGS.md','# A63 continuation stopped\n\nStatus: '+state['status']+'. Coverage '+state['coverage_description']+'. '+('Partial formal/readout work and unknown tails are preserved; missing public endpoints are not described as not_run.' if mode=='incomplete' else 'Formal models, calibration and all outer endpoints were not run.')+' This is not evidence against the scientific hypothesis. Costs and unknown/pending evidence are retained in resources_negative.json. The directly waiting caller receipt and root publication remain separate.\n')
        return self.ctx.write_json('negative_terminal_snapshot.json',state,public=True)
