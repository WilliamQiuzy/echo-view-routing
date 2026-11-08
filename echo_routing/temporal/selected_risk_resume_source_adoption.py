"""Receipt-only adoption of complete FIT caches after a CPU-only stage overrun.

The failed source receipt and physical spend remain immutable. This contains
no model constructor, predictor, optimizer or cache writer. Geometry requires
this new stage's own actual OS0 and Git-sealed output, not just the cost proof.
"""
import math
from echo_routing.temporal.selected_risk_resume_contract import RUN,digest,storage_cap,_successful

SOURCE='echo_routing/temporal/selected_risk_resume_source_adoption.py'


def validate_original_attempt(parent,spec,spec_sha,owner):
    if parent.get('stage')!='source_FIT' or spec.get('stage')!='source_FIT' or any(type(parent.get(k)) is not int or parent[k]!=v for k,v in [('stage_exit_status',1),('observed_worker_exit_code',1),('inner_model_observed_OS_exit_code',0)]) or parent.get('guardian_termination_reason')!='actual_final_cap_exceeded' or parent.get('guardian_error') is not None:raise ValueError('only complete FIT innerOS0 / preserved CPU-finalcap outerOS1 can be adopted')
    if parent.get('whole_child_accounting_complete') is not True or parent.get('actual_child_directly_waited') is not True or parent.get('unresolved_worker_uncertainty') is not False or parent.get('current_remote_caps_unknown') is not False or parent.get('runtime_authorization_sha256')!=spec['owner']['sha256'] or parent.get('runtime_spec_sha256')!=spec_sha or parent.get('GPU_job') is not True or parent.get('CUDA_context_occupied') is not True:raise ValueError('exact known whole-attempt identity and accounting required')
    cpu=parent.get('measured_CPU_core_seconds');wall=parent.get('full_process_wall_seconds');space=parent.get('sampled_tracked_storage_upper_bytes');end=parent.get('finished_epoch');limits=spec['arguments']
    if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<0 for v in [cpu,wall,space,end]) or not limits['stage_CPU_upper_seconds']<cpu<=6000 or wall>limits['stage_wall_upper_seconds'] or space>storage_cap(owner) or end>owner['deadline_epoch']:raise ValueError('known CPU-only engineering stage underestimate within original source6000 and other hard limits required')
    return True


def verify_existing_FIT(context,inputs,producer,recovery,workers):
    """Historical actual-Git proof plus current complete readonly6000 payloads."""
    from echo_routing.temporal.safety_git import verify
    from echo_routing.temporal.selected_risk_resume_source import source_identity
    from echo_routing.temporal.anchor_cache import sha
    from echo_routing.temporal.anchor_inputs import workload
    from scripts.selected_risk_common import CFG,read
    import numpy as np
    if not isinstance(recovery,dict) or set(recovery)!={'original_parent_receipt','original_runtime_spec','cache_verification'}:raise ValueError('three exact external readonly recovery pins required')
    parent=context.read(recovery['original_parent_receipt']);spec=context.read(recovery['original_runtime_spec']);proof=context.read(recovery['cache_verification'])
    validate_original_attempt(parent,spec,recovery['original_runtime_spec']['sha256'],context.owner);_successful('source_FIT',producer)
    actual=[r for r in workers if r.get('stage')=='source_FIT' and r.get('model_pid') is not None]
    if len(actual)!=1 or actual[0]!=parent or parent['runtime_authorization_sha256']!=context.owner_sha:raise ValueError('one exact closed original teacher attempt required; no unknown/unclosed claim bypass')
    producer_pin=inputs.producer_FIT_pin if hasattr(inputs,'producer_FIT_pin') else context.read(context._bound_runtime_spec_pin)['FIT_producer']
    if context.read(producer_pin)!=producer or producer.get('runtime_authorization_sha256')!=context.owner_sha or proof.get('schema')!='A63_resume_closed_teacher_readonly_verification_v1' or proof.get('passed') is not True or proof.get('complete_cache_verified') is not True or proof.get('cache_only_validation_not_stage_resource_success') is not True or proof.get('original_source_stage_exit_status')!=1 or proof.get('actual_parent_OS_receipt')!=recovery['original_parent_receipt'] or proof.get('producer_manifest')!=producer_pin or proof.get('runtime_spec')!=recovery['original_runtime_spec'] or proof.get('created_own_teacher_attempts')!=1 or proof.get('committed_payloads')!=6000 or (proof.get('model_pid'),proof.get('model_start_ticks'),proof.get('boot_id'))!=(parent['model_pid'],parent['model_start_ticks'],parent['boot_id']) or parent['boot_id']!=spec['expected_boot_id']:raise ValueError('independent complete-cache audit must preserve exact original failure/payload/spec')
    contract=context.read(spec['freeze'])
    if contract['runtime_authorization_sha256']!=context.owner_sha or contract['stage']!='source_FIT' or contract['runtime_spec_semantics_sha256']!=digest({k:v for k,v in spec.items() if k!='freeze'}) or producer['source_freeze_git']!=contract['git_commit'] or producer['input_manifest_sha256']!=inputs.metadata_pin['sha256']:raise ValueError('original source freeze/input/complete semantics differ')
    objects=dict(contract['objects']);history=context._historical_expected()
    if set(contract['historical_frozen_imports'])!=set(history):raise ValueError('all literal historical source proofs required')
    for path,v in contract['historical_frozen_imports'].items():
        if {k:v[k] for k in ['git_commit','blob_oid','sha256']}!=history[path] or objects.pop(path,None)!=v['sha256']:raise ValueError('original historical source identity differs')
        verify(context.contract_proof(contract,v['proof']),v['git_commit'],{path:v['sha256']})
    verify(context.contract_proof(contract,contract['actual_Git_object_proof']),contract['git_commit'],objects)
    cfg=read(CFG);records=producer['records'];reported=proof['records'];units={'a08':0,'jepa':0}
    if len(records)!=4 or len(reported)!=4 or {(r['source'],r['role']) for r in records}!={(s,r) for s in units for r in ['F0_mixed','F1_mixed']}:raise ValueError('all four unique complete FIT teacher caches required')
    for role in ['F0_mixed','F1_mixed']:
        qs=inputs.sequences(role);ps=inputs._teachers(role,qs);volume=workload(qs)
        if len(qs)!=1500:raise ValueError('exact original1500 full axes perfold required')
        for name,values in ps.items():
            r=next(r for r in records if (r['source'],r['role'])==(name,role));a=next(r for r in reported if (r['source'],r['role'])==(name,role));identity=source_identity(name,role,inputs.manifest['roles'][role]['sha256'],context.owner_sha,contract['objects'],cfg['source'][name],'FIT');path=RUN+'/producer_FIT/'+name+'/'+role
            if r['path']!=path or r['identity']!=identity or r['input_sha256']!=inputs.manifest['roles'][role]['sha256'] or r['committed_sequences']!=1500 or r['resumed_committed_prefix_sequences']!=0 or any(a.get(k)!=r.get(k) for k in ['path','identity','input_sha256','journal_sha256']) or a.get('committed_payloads')!=1500 or sha(context.root/path/'committed.jsonl')!=r['journal_sha256'] or len(values)!=1500 or any(np.asarray(v).dtype!=np.float32 or np.asarray(v).shape!=(len(q['refs']),5) for q,v in zip(qs,values)):raise ValueError('every original complete FP32 teacher payload/journal/input/axis must still match')
            u=volume['a08_stride5_windows' if name=='a08' else 'jepa_chunks'];units[name]+=u
            if r['actual_new_neural_units_this_invocation']!=u or r['committed_logical_neural_units']!=u:raise ValueError('actual source units differ from original complete successful callback records')
    return {'original_parent':parent,'original_source_contract':spec['freeze'],'producer_manifest':producer_pin,'cache_verification':recovery['cache_verification'],'committed_neural_units_by_producer':units,'actual_created_source_attempts':1}


class SourceAdoption:
    def __init__(self,context,inputs,freeze,descriptors,arguments):self.ctx=context;self.inputs=inputs;self.freeze=freeze;self.descriptors=descriptors;self.args=arguments
    def source_FIT_adoption(self):
        freeze,e=self.ctx.enter('source_FIT_adoption',self.freeze,self.descriptors,required={SOURCE,'echo_routing/temporal/anchor_cache.py'})
        from echo_routing.temporal.selected_risk_resume_runtime import RuntimeState
        state=RuntimeState(self.ctx,self.ctx.read(self.ctx._bound_runtime_spec_pin),self.ctx._bound_runtime_spec_pin)
        checked=verify_existing_FIT(self.ctx,self.inputs,e['source_FIT']['payload'],self.args['source_CPU_final_cap_recovery'],state._worker_rows())
        return self.ctx.write_json('source_FIT_adoption.'+self.ctx.runtime_spec_semantics_sha256[:12]+'.json',{'passed':True,'complete':True,'committed_payloads':6000,'producer_manifest':checked['producer_manifest'],'original_source_contract':checked['original_source_contract'],'original_source_parent_receipt':self.args['source_CPU_final_cap_recovery']['original_parent_receipt'],'cache_verification':checked['cache_verification'],'original_source_stage_exit_status':1,'original_teacher_inner_OS_exit_code':0,'original_source_stage_cap_failure_preserved':True,'original_source_CPU_seconds':checked['original_parent']['measured_CPU_core_seconds'],'actual_created_source_attempts':1,'committed_neural_units_by_producer':checked['committed_neural_units_by_producer'],'new_teacher_forwards':0,'new_optimizer_updates':0,'cache_writes':0,'aux_optimizer_updates':0,'formal_optimizer_updates':0,'own_actual_parent_OS_pending':True},public=True)
