"""Actual offline-Git auxiliary/selectors/final-head barriers and source adoption.

Proofs are built locally from actual committed public files, then verified here
without a server .git. This module performs no forward, fitting or optimizer.
The new outer reader verifies the barrier/adoption itself; a caller boolean
cannot unlock the fresh confirmation bank.
"""
from echo_routing.temporal.selected_risk_resume_contract import PUB,_successful
from echo_routing.temporal.selected_risk_resume_inputs import ReadOnlyInputs
from echo_routing.temporal.selected_risk_metrics import readout_plan

SEEDS=(2026110461,2026110462)
SOURCE='echo_routing/temporal/selected_risk_resume_seals.py'
RISK_REPORT_PATHS={'FIT_risk_scores_manifest':PUB+'/FIT_risk_scores_manifest.json','FIT_risk_diagnostics':PUB+'/FIT_risk_diagnostics.json','risk_diagnostics':PUB+'/inner_risk_diagnostics.json'}
HEAD_CORE_PATHS={'models_manifest':PUB+'/models_manifest.json','calibration':PUB+'/calibration.json','readout_plan':PUB+'/readout_plan.json',**RISK_REPORT_PATHS}


def validate_aux_core(models,audit,training_sha):
    rows=models.get('models',[])
    if len(rows)!=8 or {(r['seed'],r['held_fold'],r['role']) for r in rows}!={(s,h,r) for s in SEEDS for h in [0,1] for r in ['utility','risk']} or any(r['step']!=6000 for r in rows):raise ValueError('exact eight final seed/fold/aux role records required')
    if audit.get('passed') is not True or (audit.get('aux_optimizer_updates'),audit.get('physical_models'),audit.get('fullstate_count'))!=(48000,8,32) or audit['models']!=rows or models['aux_training_inputs_sha256']!=training_sha or audit['aux_training_inputs_sha256']!=training_sha:raise ValueError('independent48000/eight/32 audit and training inputs must match actual models')
    return True


def validate_head_core(models,calibration,plan):
    if plan!=readout_plan():raise ValueError('exact complete fixed A63 endpoint plan required')
    _successful('seal_heads',models)
    rows=models.get('models',[]);states=models.get('fullstates',[])
    if len(rows)!=4 or {(r['seed'],r['arm'],r['step']) for r in rows}!={(s,a,6000) for s in SEEDS for a in ['C0','C1']} or len(states)!=16 or {(r['seed'],r['arm'],r['step'],r['kind']) for r in states}!={(s,a,k,'milestone') for s in SEEDS for a in ['C0','C1'] for k in [2000,4000,6000]}|{(s,a,6000,'latest') for s in SEEDS for a in ['C0','C1']}:raise ValueError('four actual final models and every16 fullstate endpoint required')
    if calibration.get('passed') is not True or len(calibration.get('thresholds',[]))!=4 or {(r['seed'],r['policy']) for r in calibration['thresholds']}!={(s,a) for s in SEEDS for a in ['C0','C1']} or len(calibration.get('inner_records',[]))!=34:raise ValueError('exact four CAL thresholds and34 inner endpoints required')
    key=lambda r:(r['population'],r['seed'],r['policy'],r['system'])
    if {key(r) for r in calibration['inner_records']}!={key(r) for r in plan['inner_rows']}:raise ValueError('every unique prespecified inner endpoint must be present')
    if any(r.get('parameters')!=6625 for r in rows):raise ValueError('four unchanged6625-parameter final heads required')
    for r in rows:
        final=next(v for v in states if (v['seed'],v['arm'],v['step'],v['kind'])==(r['seed'],r['arm'],6000,'milestone'))
        if (final['path'],final['sha256'])!=(r['path'],r['sha256']):raise ValueError('final head must equal actual audited step6000 fullstate pin')
    for key,path in RISK_REPORT_PATHS.items():
        pin=calibration.get(key)
        if not isinstance(pin,dict) or set(pin)!={'path','sha256'} or pin['path']!=path or not isinstance(pin['sha256'],str) or len(pin['sha256'])!=64 or any(c not in '0123456789abcdef' for c in pin['sha256']):raise ValueError('exact required risk diagnostic public pin absent: '+key)
    for key in ['thresholds','inner_records','legacy_thresholds','fixed_thresholds','model_audit_sha256',*RISK_REPORT_PATHS]:
        if models.get(key)!=calibration.get(key):raise ValueError('models/calibration crossbinding differs: '+key)
    if len(models.get('fixed_reference_models',[]))!=8 or {(r['seed'],r['arm']) for r in models['fixed_reference_models']}!={(s,a) for s in SEEDS for a in ['S-L','risk','R1','R2']}:raise ValueError('all eight original S-L/risk/R1/R2 reference model pins required')
    if len(calibration.get('legacy_thresholds',[]))!=2 or len(calibration.get('fixed_thresholds',[]))!=6:raise ValueError('original two L0 and six M0/R1/R2 thresholds must remain explicit')
    return True


def _core(context,core,expected_paths):
    from echo_routing.temporal.safety_git import verify
    if set(core)!=set(expected_paths)|{'git_commit','actual_Git_object_proof'}:raise ValueError('exact declared public core/proof inputs required')
    objects={}
    values={}
    for key,path in expected_paths.items():
        pin=core[key]
        if pin['path']!=path:raise ValueError('public core must stay in exact new namespace')
        values[key]=context.read(pin);objects[path]=pin['sha256']
    verify(context.read(core['actual_Git_object_proof']),core['git_commit'],objects)
    return values,objects


def _head_core(context,core):
    values,objects=_core(context,core,HEAD_CORE_PATHS)
    models,cal=values['models_manifest'],values['calibration']
    validate_head_core(models,cal,values['readout_plan'])
    for key in RISK_REPORT_PATHS:
        if models[key]!=core[key] or cal[key]!=core[key]:raise ValueError('actual-Git risk core must equal both model and calibration report pins')
    _successful('fit_risk_scores',values['FIT_risk_scores_manifest'])
    report=values['FIT_risk_diagnostics'];inner=values['risk_diagnostics']
    if report.get('passed') is not True or report.get('complete') is not True or report.get('FIT_q_scores_manifest')!=core['FIT_risk_scores_manifest'] or report.get('seeds')!=list(SEEDS) or len(report.get('records',[]))!=2 or {r.get('seed') for r in report['records']}!=set(SEEDS) or report.get('FIT_report_generated_after_CAL') is not True or report.get('FIT_scores_used_to_select_model_or_tau') is not False or report.get('new_MLP_calls_during_reporting')!=0:raise ValueError('complete fixed-score FIT report must bind its pre-CAL actual-Git q manifest')
    if set(inner.get('records',{}))!={p+'/seed'+str(seed) for p in ['CAL','AUDIT'] for seed in SEEDS} or inner.get('FIT_risk_diagnostics')!=core['FIT_risk_diagnostics']:raise ValueError('single four-cell CAL/AUDIT report must reference the same FIT diagnostic')
    return values,objects


class VerifiedOuterInputs(ReadOnlyInputs):
    def __init__(self,*args,**kwargs):
        self._confirmation_pin=kwargs.get('CONFIRM_producer_pin');self._outer_checked=False
        super().__init__(*args,**kwargs)
    @property
    def outer_adoption_verified(self):return getattr(self,'_outer_checked',False)

    def authorize_outer(self,head_descriptor,adoption_descriptor):
        h=self.context.evidence(head_descriptor);a=self.context.evidence(adoption_descriptor);head=h['payload'];adopt=a['payload'];_successful('seal_heads',head);_successful('adopt_confirmation',adopt)
        core={k:head[k] for k in [*HEAD_CORE_PATHS,'git_commit','actual_Git_object_proof']}
        v,objects=_head_core(self.context,core)
        finals=[{k:r[k] for k in ['path','sha256']} for r in v['models_manifest']['models']+v['models_manifest']['fixed_reference_models']]
        if head['all_final_checkpoints']!=finals:raise ValueError('head seal must bind all12 new and original final weight pins')
        for pin in finals:self.context.raw(pin)
        if self._confirmation_pin is None or adopt.get('head_seal_sha256')!=head_descriptor['payload_pin']['sha256'] or adopt.get('producer_manifest_sha256')!=self._confirmation_pin['sha256'] or (adopt.get('aux_updates_at_binding'),adopt.get('formal_updates_at_binding'),adopt.get('new_source_forwards'))!=(48000,24000,0):raise ValueError('actual48000/24000 head-sealed readonly adoption required before fresh outer reader')
        producer=self.context.read(self._confirmation_pin)
        if adopt['records']!=producer['records']:raise ValueError('adoption cannot rename/change original production48000/formal0 rows')
        for role in ['a63_confirmation_selection','a63_confirmation_calibration']:self._teachers(role,self.sequences(role))
        self._outer_checked=True;return v


class Seals:
    def __init__(self,context,inputs,freeze_pin,descriptors,arguments):
        self.ctx=context;self.inputs=inputs;self.freeze_pin=freeze_pin;self.descriptors=descriptors;self.args=arguments
    def _enter(self,stage):return self.ctx.enter(stage,self.freeze_pin,self.descriptors,required={SOURCE,'echo_routing/temporal/selected_risk_resume_context.py'})

    def seal_aux(self):
        freeze,e=self._enter('seal_aux');core=self.args['core'];v,objects=_core(self.ctx,core,{'models_manifest':PUB+'/aux_models_manifest.json','audit':PUB+'/aux_model_audit.json','training_inputs':PUB+'/aux_training_inputs.json'})
        validate_aux_core(v['models_manifest'],v['audit'],core['training_inputs']['sha256'])
        if core['audit']['sha256']!=self.descriptors['audit_aux']['payload_pin']['sha256'] or core['training_inputs']['sha256']!=self.descriptors['prepare_aux']['payload_pin']['sha256']:raise ValueError('auxiliary core must equal independently verified stage predecessors')
        finals=[{k:r[k] for k in ['path','sha256']} for r in v['models_manifest']['models']]
        for p in finals:self.ctx.raw(p)
        return self.ctx.write_json('aux_identity.json',{'passed':True,**core,'objects':objects,'all_final_checkpoints':finals,'aux_optimizer_updates':48000,'formal_optimizer_updates':0,'physical_models':8,'fullstate_count':32},public=True)

    def seal_selectors(self):
        freeze,e=self._enter('seal_selectors');core=self.args['core'];v,objects=_core(self.ctx,core,{'cache_manifest':PUB+'/held_selector_cache_manifest.json','models_manifest':PUB+'/aux_models_manifest.json','audit':PUB+'/aux_model_audit.json','training_inputs':PUB+'/aux_training_inputs.json'})
        manifest=v['cache_manifest'];records=manifest['records']
        if core['cache_manifest']['sha256']!=self.descriptors['held_score']['payload_pin']['sha256'] or manifest.get('complete') is not True or len(records)!=8 or {(r['seed'],r['role']) for r in records}!={(s,r) for s in SEEDS for r in ['F0_native','F0_mixed','F1_native','F1_mixed']} or manifest.get('effect_lookup_before_seal') is not False:raise ValueError('complete all-axes label-free two-seed selector cache must precede effect lookup')
        validate_aux_core(v['models_manifest'],v['audit'],core['training_inputs']['sha256'])
        from echo_routing.temporal.selected_risk_compact_cache import CompactSupplyCache
        from echo_routing.temporal.anchor_cache import sha
        refs={}
        for seed in SEEDS:
            for role in ['F0_native','F0_mixed','F1_native','F1_mixed']:
                qs,aa,jj,bb=self.inputs.bank(role,purpose='selector');r=next(r for r in records if (r['seed'],r['role'])==(seed,role));c=CompactSupplyCache(self.ctx.root/r['path'],r['identity'],bb,[p.argmax(1) for p in aa],[p.argmax(1) for p in jj],readonly=True,feature_reference=refs.get(role))
                if c.cursor!=len(qs) or sha(c.journal)!=r['journal_sha256'] or r['input_sha256']!=self.inputs.manifest['roles'][role]['sha256']:raise ValueError('complete eager immutable selector source/journal/input binding required')
                if seed==SEEDS[0]:refs[role]=c
        allowance=self.ctx.evidence(self.descriptors['admit_aux'])['payload']['selected_table_atomic_bytes_upper']
        return self.ctx.write_json('selector_identity.json',{'passed':True,'complete':True,**core,'objects':objects,'cache_manifest_sha256':core['cache_manifest']['sha256'],'effect_lookup_before_seal':False,'aux_optimizer_updates':48000,'formal_optimizer_updates':0,'selected_table_atomic_bytes_upper':allowance},public=True)

    def seal_heads(self):
        freeze,e=self._enter('seal_heads');core=self.args['core'];v,objects=_head_core(self.ctx,core)
        if core['calibration']['sha256']!=self.descriptors['calibrate']['payload_pin']['sha256'] or core['FIT_risk_scores_manifest']!=self.descriptors['fit_risk_scores']['payload_pin'] or v['models_manifest']['model_audit_sha256']!=self.descriptors['audit_formal']['payload_pin']['sha256']:raise ValueError('actual formal audit/FIT score/calibration stage identities must match final core')
        finals=[{k:r[k] for k in ['path','sha256']} for r in v['models_manifest']['models']+v['models_manifest']['fixed_reference_models']]
        for p in finals:self.ctx.raw(p)
        return self.ctx.write_json('outer_identity.json',{'passed':True,**core,'objects':objects,'all_final_checkpoints':finals,'formal_optimizer_updates':24000,'physical_models':4,'fullstate_count':16,'aux_optimizer_updates':48000,'no_outer_outcome_for_model_or_tau_choice':True},public=True)

    def adopt_confirmation(self):
        freeze,e=self._enter('adopt_confirmation');producer=e['source_CONFIRM']['payload'];head=e['seal_heads']['payload']
        # Head core and all final weights are verified before reader adoption.
        v,objects=_head_core(self.ctx,{k:head[k] for k in [*HEAD_CORE_PATHS,'git_commit','actual_Git_object_proof']})
        for p in head['all_final_checkpoints']:self.ctx.raw(p)
        total=0
        for role in ['a63_confirmation_selection','a63_confirmation_calibration']:
            qs=self.inputs.sequences(role);self.inputs._teachers(role,qs);total+=2*len(qs)
        if total!=480:raise ValueError('all480 fresh teacher payloads must be eagerly readonly adopted')
        return self.ctx.write_json('source_adoption.json',{'passed':True,'committed_payloads':480,'records':producer['records'],'producer_manifest_sha256':self.descriptors['source_CONFIRM']['payload_pin']['sha256'],'head_seal_sha256':self.descriptors['seal_heads']['payload_pin']['sha256'],'aux_updates_at_binding':48000,'formal_updates_at_binding':24000,'new_source_forwards':0,'source_aux_updates_at_production':48000,'source_formal_updates_at_production':0},public=True)
