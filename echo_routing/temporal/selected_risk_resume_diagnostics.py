"""Real cache-only56 supports/34 native cubes/32 anonymous regressions.

Original48 support records and16 case records remain byte-bound references.
Only new eight supports are solved; all units are original frozen Ntop/noop.
Private case locators, choices and RLE are never exported to public aggregates.
"""
import hashlib
import json
import numpy as np
from echo_routing.temporal.selected_risk_resume_contract import PUB,storage_cap
from echo_routing.temporal.selected_risk_metrics import readout_plan,SEEDS,POLICIES
from echo_routing.temporal.selected_risk_resume_readout import Readout,Inference,decoded,F_FIT,plain

SOURCE='echo_routing/temporal/selected_risk_resume_diagnostics.py'
OLD='docs/results/2026-11-04_anchor_context_risk/'
READONLY_INPUTS={OLD+'action_space_diagnostic.json':'74f90cc3fb4f2a519fb22aa33f94d1e0fd2994d818f4a108f0efd952dbe120f8',OLD+'regression_summary.json':'70cfb57246001e055b50c25c4b7f9ad7b47372e089690311632a384a8742c5ee',OLD+'readonly_context/case_evidence.anonymous.json':'a441f5384101ee547f294cbe0ce8f0ca29be17b13a5299a9ad8c21532fcafe29'}


def fit_inference_costs(context,calibration,descriptor):
    """Read actual FIT inference descriptors; whole-caller CPU stays canonical.

    The last successful consumer's calls are not an all-attempt counter. Keep
    unknown historical inference calls null until an independent attempt audit.
    No reporting path scores a model or adds nested SELF/axis CPU to the ledger.
    """
    import math
    from echo_routing.temporal.selected_risk_resume_seals import RISK_REPORT_PATHS
    from echo_routing.temporal.selected_risk_resume_contract import _successful
    pins={k:calibration[k] for k in RISK_REPORT_PATHS}
    if any(not isinstance(p,dict) or set(p)!={'path','sha256'} or p['path']!=RISK_REPORT_PATHS[k] for k,p in pins.items()) or pins['FIT_risk_scores_manifest']!=descriptor['payload_pin']:raise ValueError('FIT inference cost descriptors must use the actual-Git scoring predecessor and shared report pins')
    values={k:context.read(p) for k,p in pins.items()};scores=values['FIT_risk_scores_manifest'];_successful('fit_risk_scores',scores)
    report=values['FIT_risk_diagnostics'];inner=values['risk_diagnostics']
    if report.get('passed') is not True or report.get('complete') is not True or report.get('FIT_q_scores_manifest')!=pins['FIT_risk_scores_manifest'] or inner.get('FIT_risk_diagnostics')!=pins['FIT_risk_diagnostics']:raise ValueError('FIT/inner cost report references differ from the sealed q production')
    nested=scores.get('informational_consumer_CPU_seconds_to_snapshot')
    if isinstance(nested,bool) or not isinstance(nested,(int,float)) or not math.isfinite(nested) or nested<0:raise ValueError('actual consumer nested CPU descriptor required')
    attempts=scores.get('actual_all_attempt_MLP_calls')
    if attempts is not None and (type(attempts) is not int or attempts<0):raise ValueError('all-attempt inference count is nonnegative or explicitly unknown')
    return {'shared_reports':pins,'actual_new_MLP_calls_this_process':scores['actual_new_MLP_calls_this_process'],'actual_new_candidate_outputs_this_process':scores['actual_new_candidate_outputs_this_process'],'actual_all_attempt_MLP_calls':attempts,'all_attempt_counts_require_independent_closed_caller_recount':attempts is None,'nested_informational_CPU_seconds_to_snapshot':nested,'extra_canonical_CPU_charge_seconds':0,'CPU_charge_scope':'whole closed caller only; consumer and axis CPU are nested descriptors','new_teacher_or_S_L_or_aux_forward_calls':0,'optimizer_updates':0}


def old_support(value):
    rows=value['fixed_N_top_support'];keys=lambda r:tuple(r[k] for k in ['table','half','seed','safety'])
    specs=[s for s in readout_plan()['support_cells'] if s['table']!='a63_confirmation']
    if value['support_count']!=48 or len(rows)!=48 or len({keys(r) for r in rows})!=48 or {keys(r) for r in rows}!={keys(r) for r in specs}:raise ValueError('all48 literal original support endpoints required without resolving')
    return [{**{k:r[k] for k in ['table','half','seed','safety']},'readonly_A62_reference':True,'result':r} for r in rows]


def native_result(qs,bases,values,policy):
    from echo_routing.temporal.event_risk_forensics import TOTALS,_result
    from echo_routing.temporal.proposal_native_attribution import joint_counts
    from echo_routing.temporal.proposal_actions import apply
    from echo_routing.temporal.anchor_edits import majority
    if not qs or len(qs)!=len(bases) or values is not None and len(values)!=len(qs):raise ValueError('complete aligned native axes required')
    cube=np.zeros((5,5,5),np.int64);totals={k:0 for k in TOTALS};changed=accepted=0
    for i,(q,b) in enumerate(zip(qs,bases)):
        if q['kind']!='native' or not np.all(q['labels']==q['labels'][0]):raise ValueError('constant propagated native references required')
        v=None if values is None else values[i];row=None if v is None else v[policy];p=b if row is None else row['prediction'];index=None if row is None else row['accepted_index']
        if index is None:
            if not np.array_equal(p,b):raise ValueError('no-top native must be exact Base')
        elif isinstance(index,(bool,np.bool_)) or not isinstance(index,(int,np.integer)) or not 0<=index<len(v['actions']) or not np.array_equal(p,apply(b,v['actions'][index])):raise ValueError('actual single accepted action/prediction mismatch')
        if majority(p)!=majority(b):raise ValueError('originalG native majority protection changed')
        costs=joint_counts(q['labels'],b,p);cube+=np.asarray(costs['true_Base_prediction_joint5cube'])
        for k in TOTALS:totals[k]+=costs[k]
        changed+=costs['changed_positions']>0;accepted+=index is not None
    if changed!=accepted:raise ValueError('accepted action must change nonempty positions')
    return _result(cube,totals,qs,int(changed),accepted,[],policy)


def case_records(axis,q,base,a08,jepa,value,seed):
    from echo_routing.temporal.safety_targets import effects
    from echo_routing.temporal.event_risk_features import context_features
    from echo_routing.temporal.event_risk_metrics import _effect
    top=value['E']['top_index'];cost=None;context=None;key=None;span=None
    if top is not None:
        action=value['actions'][top];cost,*_=_effect(base,action,q['labels'],F_FIT);cx=context_features(a08,jepa,base,action);context={'minimum_overlap_over_full_run':float(cx[91]),'minimum_full_Base_run_A08_margin':float(cx[93])};key=hashlib.sha256(json.dumps(action.key,separators=(',',':')).encode()).hexdigest();span={'start':action.start,'end':action.end,'old':action.old,'jepa':action.jepa}
    rows=[]
    for policy in POLICIES:
        r=value[policy];accepted=r['accepted_index'] is not None
        out={'axis':axis,'seed':seed,'policy':policy,'top_exists':top is not None,'N_admitted':value['N']['accepted_index'] is not None,'accepted':accepted,'candidate_key_sha256':key,'candidate_action':span,'context_summary':context,'gate_score':r.get('gate_score') if policy not in ['E','N'] else None,'tau':r.get('tau') if policy not in ['E','N'] else None,'development_regression_not_independent_holdout':True}
        for k,ck in [('H','H'),('R','R'),('dFP16','dFP'),('short16_lost','short16_lost')]:out[k]=None if cost is None else int(cost[ck]);out['actual_'+k]=int(cost[ck]) if accepted else 0
        rows.append(out)
    return rows


class Diagnostics(Readout):
    def _setup(self,stage):
        freeze,e=self.ctx.enter(stage,self.freeze_pin,self.descriptors,required={SOURCE,*READONLY_INPUTS})
        self.inputs.authorize_outer(self.descriptors['seal_heads'],self.descriptors['adopt_confirmation']);refs=self._reference();core=e['seal_heads']['payload'];models=self.ctx.read(core['models_manifest']);auditpin={'path':PUB+'/formal_model_audit.json','sha256':models['model_audit_sha256']};cov=e['coverage']['payload'];infer=Inference(self.ctx,self.inputs,refs,auditpin,cov,freeze['objects']);cal=self.ctx.read(core['calibration']);return e,refs,infer,cal
    def _decode(self,bank,scored,seed,cal,refs):
        tau={a:next(r['tau'] for r in cal['thresholds'] if (r['seed'],r['policy'])==(seed,a)) for a in ['C0','C1']}
        return [decoded(b,v,v['q'],refs.thresholds()[seed],tau) for b,v in zip(bank[3],scored)]
    def supports(self):
        e,refs,infer,cal=self._setup('supports');pin={'path':OLD+'action_space_diagnostic.json','sha256':READONLY_INPUTS[OLD+'action_space_diagnostic.json']};rows=old_support(self.ctx.read(pin));private=[]
        from echo_routing.temporal.event_risk_kind import support
        for half in ['selection','calibration']:
            bank,scored,_=self._outer_bank('a63_confirmation',half,refs,infer,readonly=True)
            for seed in SEEDS:
                values=self._decode(bank,scored[seed],seed,cal,refs)
                for safety in ['event-safe','acceptance-safe']:
                    result=support(bank[0],bank[3],values,safety,F_FIT,private_details=True);choices=result.pop('private_details');private.append({'half':half,'seed':seed,'safety':safety,'choices':choices});rows.append({'table':'a63_confirmation','half':half,'seed':seed,'safety':safety,'readonly_A62_reference':False,'result':result})
        if len(rows)!=56:raise ValueError('all56 support records required')
        choice=self.ctx.save_value('support_choices.private.npz',private)
        return self.ctx.write_json('support_summary.json',{'passed':True,'record_count':56,'records':plain(rows),'old48_literal_pin':pin,'new_solved_records':8,'private_choices':{k:choice[k] for k in ['path','sha256']},'summary_sha256':self.descriptors['aggregate']['payload_pin']['sha256'],'new_NN_forwards':0,'new_optimizer_updates':0,'hindsight_not_single_tau_guarantee':True},public=True)
    def forensics(self):
        e,refs,infer,cal=self._setup('forensics');rows=[];regression=[];private=[];outer=e['outer']['payload'];oldpin={'path':OLD+'regression_summary.json','sha256':READONLY_INPUTS[OLD+'regression_summary.json']};literal=self.ctx.read(oldpin)['records']
        for half in ['selection','calibration']:
            bank,scored,_=self._outer_bank('main',half,refs,infer,readonly=True);ix=[i for i,q in enumerate(bank[0]) if q['kind']=='native'];qs=[bank[0][i] for i in ix];bb=[bank[3][i] for i in ix]
            for seed,policy in [(None,'Base')]+[(s,p) for s in SEEDS for p in POLICIES]:
                values=None if seed is None else self._decode(bank,scored[seed],seed,cal,refs);result=native_result(qs,bb,None if values is None else [values[i] for i in ix],policy);point=next(v for v in outer['rows'] if (v['table'],v['half'],v['seed'],v['policy'])==('main',half,seed,policy))
                if (result['Base_correct_to_wrong'],result['Base_wrong_to_correct'],result['true_prediction_frame_confusion5'])!=(point['H']['native'],point['R']['native'],self.ctx.load_value(next(r['pin'] for r in outer['private_bundles'] if (r['table'],r['half'],r['system'])==('main',half,point['system'])))['safety_summary']['position_confusion']['native']):raise ValueError('native joint/gross frame confusion differs from sealed actual point')
                rows.append({'population':half,'seed':seed,'policy':policy,'result':result})
        evidence=self.ctx.read({'path':OLD+'readonly_context/case_evidence.anonymous.json','sha256':READONLY_INPUTS[OLD+'readonly_context/case_evidence.anonymous.json']})
        # Original anonymous case is rebound by the pinned original C6 payload
        # hash and role metadata. No incoming raw identifier is trusted/exported.
        for case in [c for c in evidence['cases'] if c['seed']==SEEDS[0]]:
            role=case['source_role'];bank=refs.bank(role);original=refs.scores(SEEDS[0],role,bank);needle=case['evidence']['primary_policy_payload_sha256'];indices=[i for i,v in enumerate(original) if v['old_reference'].get('payload_sha256')==needle]
            if len(indices)!=1 or self.inputs.manifest['roles'][role]['sha256']!=case['evidence']['role_metadata_sha256']:raise ValueError('original anonymous case not uniquely bound to physical C6 payload/metadata')
            i=indices[0];axis=case['case'].replace(' ','_')
            for seed in SEEDS:
                scored=infer.score(seed,role,bank,readonly=True);v=self._decode(bank,scored,seed,cal,refs)[i];computed=case_records(axis,bank[0][i],bank[3][i],bank[1][i],bank[2][i],v,seed)
                for r in computed:
                    if r['policy'] in ['L0','M0','R1','R2']:
                        old=next(x for x in literal if (x['axis'],x['seed'],x['policy'])==(axis,seed,r['policy']))
                        if any(r[k]!=old[k] for k in ['H','R','dFP16','short16_lost','actual_H','actual_R','actual_dFP16','actual_short16_lost','candidate_key_sha256','accepted','gate_score','tau']):raise ValueError('original16 literal regression values changed')
                        r['readonly_A62_record']=old
                    regression.append(r)
                from echo_routing.temporal.segments import labels_to_segments
                rle=lambda p:[{'start':s.start,'end':s.end,'label':int(s.label)} for s in labels_to_segments(p)]
                private.append({'axis':axis,'seed':seed,'original_index':i,'original_id':bank[0][i]['id'],'truth':rle(bank[0][i]['labels']),'Base':rle(bank[3][i]),'predictions':{p:rle(v[p]['prediction']) for p in POLICIES}})
        if len(rows)!=34 or len(regression)!=32:raise ValueError('complete34 native and32 regression records required')
        rp=self.ctx.save_value('regression_RLE.private.npz',private);nativepin=self.ctx.write_json('native_attribution.json',{'records':plain(rows),'record_count':34,'old_native_reused_across_tables_not_independent':True},public=True);regpin=self.ctx.write_json('regression_summary.json',{'records':plain(regression),'record_count':32,'old16_literal_pin':oldpin,'private_RLE':{k:rp[k] for k in ['path','sha256']},'development_not_independent_holdout':True},public=True)
        return self.ctx.write_json('forensics_summary.json',{'passed':True,'native_count':34,'regression_count':32,'native':nativepin,'regression':regpin,'new_NN_forwards':0,'new_optimizer_updates':0},public=True)

    def package(self):
        """Cache-only full figures/reports; a negative verdict is a valid result."""
        from echo_routing.temporal.selected_risk_metrics import acceptance
        from echo_routing.temporal.selected_risk_resume_context import strict_json
        from echo_routing.temporal.selected_risk_resume_freeze import _raw
        from echo_routing.temporal.selected_risk_resume_present import render_full,_write
        from echo_routing.temporal.selected_risk_resume_runtime import RuntimeState
        from echo_routing.temporal.selected_risk_storage_engineering import load_trace
        from echo_routing.temporal.selected_risk_resume_contract import RUN
        from pathlib import PurePosixPath
        import math
        freeze,e=self.ctx.enter('package',self.freeze_pin,self.descriptors,required={SOURCE,'echo_routing/temporal/selected_risk_resume_present.py','echo_routing/temporal/selected_risk_resume_runtime.py','echo_routing/temporal/selected_risk_storage_engineering.py'})
        summary=e['aggregate']['payload'];support=e['supports']['payload'];forensics=e['forensics']['payload'];cal=e['calibrate']['payload'];audit=e['audit_formal']['payload'];aux=e['audit_aux']['payload'];coverage=e['coverage']['payload'];physical=e['train_formal']['payload']
        fit_costs=fit_inference_costs(self.ctx,cal,self.descriptors['fit_risk_scores'])
        if any(p.get('passed') is not True for p in [summary,support,forensics,cal,audit,aux,coverage,physical]) or (audit.get('formal_optimizer_updates'),audit.get('physical_models'),audit.get('fullstate_count'))!=(24000,4,16) or (aux.get('aux_optimizer_updates'),aux.get('physical_models'),aux.get('fullstate_count'))!=(48000,8,32):raise ValueError('complete independently audited formal/auxiliary and diagnostic inputs required')
        rows=summary['rows'];comparisons=summary['paired_comparisons'];verdict=acceptance(rows,comparisons)
        acceptance_pin={'path':PUB+'/acceptance.json','sha256':summary['acceptance_sha256']}
        if self.ctx.read(acceptance_pin)!=verdict:raise ValueError('report verdict must equal actual fixed complete-matrix acceptance')
        native=self.ctx.read(forensics['native'])['records'];regression=self.ctx.read(forensics['regression'])['records']
        if (len(rows),len(comparisons),len(cal['inner_records']),len(support['records']),len(native),len(regression))!=(238,140,34,56,34,32):raise ValueError('complete fixed package input matrix required')
        models=audit['models']
        if len(models)!=4 or {(r['seed'],r['arm'],r['step']) for r in models}!={(s,a,6000) for s in SEEDS for a in ['C0','C1']}:raise ValueError('all four actually audited final formal heads required')
        physical_records=physical.get('records',[])
        if (physical.get('formal_optimizer_updates'),physical.get('physical_models'),physical.get('fullstate_count'))!=(24000,4,16) or len(physical_records)!=4 or {(r['seed'],r['arm']) for r in physical_records}!={(s,a) for s in SEEDS for a in ['C0','C1']}:raise ValueError('all four actual-Git physical completion records required')
        curves=[];archives=[]
        for row in models:
            folder=str(PurePosixPath(row['path']).parent)
            if folder!=RUN+f'/models/seed{row["seed"]}/{row["arm"]}':raise ValueError('formal curves must come from their exact audited head directory')
            self.ctx.raw({k:row[k] for k in ['path','sha256']})
            receipt_path=folder+'/physical_updates.jsonl.archive.json';raw,h=_raw(self.ctx,receipt_path);archive=strict_json(raw)
            completion_path=folder+'/completion.json';original=next(r for r in physical_records if (r['seed'],r['arm'])==(row['seed'],row['arm']));completion_pin=original.get('completion')
            if not isinstance(completion_pin,dict) or completion_pin.get('path')!=completion_path:raise ValueError('original physical completion pin required for exact audited head')
            completion=self.ctx.read(completion_pin);completion_sha=completion_pin['sha256']
            if archive['identity']!=row['identity'] or completion.get('passed') is not True or completion.get('identity')!=row['identity'] or (completion.get('physical_updates'),completion.get('fullstate_count'))!=(6000,4) or completion.get('raw_trace_sha256')!=archive['uncompressed_sha256']:raise ValueError('formal archive must reproduce independently audited completed head trace')
            path=folder+'/physical_updates.jsonl.gz';_,gzsha=_raw(self.ctx,path)
            if gzsha!=original.get('archive_sha256'):raise ValueError('actual archive bytes differ from original physical completion record')
            trace=load_trace(self.ctx.root/path,archive)
            if len(trace)!=6000:raise ValueError('all6000 actual loss points per head required')
            for u,value in enumerate(trace):
                loss=value.get('loss')
                if type(value.get('u')) is not int or value['u']!=u or value.get('identity')!=row['identity'] or isinstance(loss,bool) or not isinstance(loss,(int,float)) or not math.isfinite(loss) or loss<0:raise ValueError('ordered finite actual audited loss trace required')
                curves.append({'seed':row['seed'],'arm':row['arm'],'update':u+1,'loss':loss})
            archives.append({'seed':row['seed'],'arm':row['arm'],'archive':{'path':path,'sha256':gzsha},'archive_receipt':{'path':receipt_path,'sha256':h},'uncompressed_sha256':archive['uncompressed_sha256'],'audited_completion':{'path':completion_path,'sha256':completion_sha}})
        spec_pin=self.ctx._bound_runtime_spec_pin;state=RuntimeState(self.ctx,self.ctx.read(spec_pin),spec_pin);spent=state.spent();storage=state.storage()
        resources={'known_CPU_seconds':spent['new_measured_CPU_seconds'],'CPU_accounting_seconds':spent['new_CPU_accounting_seconds'],'incremental_GPU_seconds':spent['new_GPU_wall_seconds'],'incremental_tracked_storage_upper_bytes':storage,'actual_caps_fit':None,'CPU_cap':28800,'GPU_cap':5400,'storage_cap':storage_cap(self.ctx.owner),'storage_amendment_sha256':self.ctx.owner.get('storage_amendment_sha256'),'storage_approval_receipt_sha256':self.ctx.owner.get('storage_approval_receipt',{}).get('sha256'),'cumulative_GPU_hours':spent['cumulative_GPU_hours'],'cumulative_CPU_accounting_hours':spent['cumulative_CPU_accounting_hours'],'scope':'closed-receipt cutoff before package rendering; current package/caller, later QA/reviews/closure are not yet closed or charged here; tracked bytes are not a physical peak'}
        pub=self.ctx.output('README.md',public=True).parent
        rendered=render_full(pub,rows,comparisons,cal['inner_records'],support['records'],native,regression,curves,resources)
        render_path=PUB+'/full_figures/render_receipt.json';render_raw,render_sha=_raw(self.ctx,render_path)
        if strict_json(render_raw)!=rendered or rendered.get('synthetic') is not False or rendered.get('actual_visual_QA_passed') is not False or rendered.get('overall_complete') is not False or rendered.get('counts')!={'outer':238,'comparisons':140,'inner':34,'support':56,'native':34,'regression':32,'curves':24000}:raise ValueError('actual complete rendering receipt required before package publication')
        findings='# A63 fixed-bank findings\n\n'
        findings+='The following are separate prespecified judgments, not a forced success label.\n\n'
        findings+='```json\n'+json.dumps(verdict,ensure_ascii=False,sort_keys=True,indent=2)+'\n```\n\n'
        findings+='hard_protection_pass requires C1 to preserve native votes/confusion and all gross-H, per-kind short-FP and true-short-loss constraints across all 28 fixed cells. joint_objective_pass additionally requires A61 recovery minima of 6/5 (selection seeds 61/62) and 4/3 (calibration seeds 61/62), at least one fresh JEPA-exclusive short16 recovery in each half/seed cell, and positive observed fresh-bank benefit versus Base with a positive 99.375% paired interval lower bound. H63_claim_supported additionally requires a positive observed C1-versus-C0 sampling contrast and interval lower bound in all four fresh half/seed cells, using the same prespecified eight-comparison primary family.\n\n'
        findings+='These are conditional fixed-bank results with shared sources; sessions are not confirmed patients. Case A/B are exposed development regressions. No independent clinical or whole-vision safety claim follows. Visual QA and overall closure remain pending.\n'
        readme='# A63 selected-action-risk results\n\nThe package contains 238 whole-task endpoints, 140 fixed paired comparisons using the same 5000 draws, 34 CAL/AUDIT endpoints, 56 supports, 34 native cubes, 32 regression cells and all 4×6000 actual formal loss points. Native/multi/pair domains remain explicit.\n\nSee [FINDINGS](FINDINGS.md), [figures and anonymous CSV](full_figures/README.md), [render receipt](full_figures/render_receipt.json), [shared FIT diagnostics](FIT_risk_diagnostics.json) and [shared CAL/AUDIT diagnostics](inner_risk_diagnostics.json). FIT final-risk scoring uses separately recorded MLP calls/outputs; its nested CPU is included once by the whole-caller receipt. An unknown all-attempt inference total remains null pending independent attempt reconciliation. The renderer does not certify model provenance, capacity, visual QA or overall closure.\n\nResources are a closed-receipt cutoff before this package; current package/caller and later administrative costs must be accounted separately. actual_caps_fit remains unknown until actual resource closure.\n'
        artifacts={'render_receipt':{'path':render_path,'sha256':render_sha}}
        for name,text in [('README',readme),('FINDINGS',findings)]:
            path=self.ctx.output(name+'.md',public=True);raw=text.encode();_write(path,raw);artifacts[name]={'path':str(path.relative_to(self.ctx.root)),'sha256':hashlib.sha256(raw).hexdigest()}
        file_pins=list(artifacts.values())
        for rendered_file in rendered['files']:
            path=PUB+'/'+rendered_file['path'];raw,h=_raw(self.ctx,path)
            if h!=rendered_file['sha256'] or len(raw)!=rendered_file['bytes']:raise ValueError('actual rendered public file differs from rendering receipt')
            file_pins.append({'path':path,'sha256':h})
        if len({p['path'] for p in file_pins})!=len(file_pins):raise ValueError('package output pins must be unique')
        return self.ctx.write_json('package_receipt.json',{'passed':True,'figure_receipt':artifacts['render_receipt'],'file_pins':file_pins,'acceptance':verdict,'acceptance_pin':acceptance_pin,'artifacts':artifacts,'rendered_files':rendered['files'],'counts':rendered['counts'],'formal_curve_archives':archives,'resources':resources,'FIT_risk_inference_costs':fit_costs,'risk_diagnostic_reports':fit_costs['shared_reports'],'input_stage_payloads':{k:self.descriptors[k]['payload_pin'] for k in ['aggregate','supports','forensics','calibrate','audit_formal','audit_aux','coverage','train_formal','fit_risk_scores']},'actual_visual_QA_passed':False,'overall_complete':False,'new_NN_forwards':0,'new_optimizer_updates':0},public=True)
