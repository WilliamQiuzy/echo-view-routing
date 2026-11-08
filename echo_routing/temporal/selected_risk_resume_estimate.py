"""Complete continuation forecast from frozen empirical profiles, never spend.

No new GT/head-outcome argument is used for cost selection. The original wide
estimates/failures remain valid history. Whole completed phases may be removed
only after the caller independently verifies their actual-Git/current payloads;
partial work receives no guessed refund. Actual geometry/predraw/layout gates
remain mandatory before persistent feature allocation or any optimizer.
"""
import copy
from echo_routing.temporal.selected_risk_resume_contract import _successful

OLDPUB='docs/results/2026-11-05_selected_action_risk/'


FIT_SCORE_HEADER_UPPER=640
FIT_SCORE_RECEIPT_UPPER=512
FIT_PUBLIC_AGGREGATE_UPPER=2000000


def fit_diagnostic_cost(candidate_count,nonempty_axes,total_axes):
    """Two-seed/four-head immutable per-nonempty-axis scores and actual exposure.

    Runtime consumer must enforce these codec/receipt/report bounds. Model
    calls are CPU-only;120 core-s/wall-s is a prospective engineering allowance,
    not a measured duration or universal hardware guarantee. No old term is
    reduced. No candidate values, GT or q outcomes select these coefficients.
    """
    if any(type(x) is not int or x<0 for x in [candidate_count,nonempty_axes,total_axes]) or nonempty_axes>min(candidate_count,total_axes):raise ValueError('canonical candidate and complete axis volumes required')
    c=candidate_count;a=nonempty_axes;n=total_axes
    artifacts=2*(32*c+(FIT_SCORE_HEADER_UPPER+FIT_SCORE_RECEIPT_UPPER)*a)
    global_one=18*c+8*(n+1)+4096
    parts={'all_score_artifacts_and_receipts':artifacts,'global_exposure_membership_offsets_and_headers':2*global_one,'one_atomic_extra_copy':max(FIT_SCORE_HEADER_UPPER+32*c,global_one,FIT_PUBLIC_AGGREGATE_UPPER),'pending_room':1024,'public_FIT_aggregate_server_and_local_copy':2*FIT_PUBLIC_AGGREGATE_UPPER}
    return {'CPU_seconds':120.,'remaining_wall_seconds':120.,'GPU_seconds':0.,'storage_upper_bytes':sum(parts.values()),'storage_components':parts,'candidate_upper_per_seed':c,'nonempty_axis_upper_per_seed':a,'complete_axes_per_seed':n,'both_seeds_and_all_four_final_risk_heads':True,'existing_C6_or_q_readonly_reuse_does_not_reduce_this_forecast':True,'runtime_codec_and_output_size_checks_required':True}


# These actual source bytes enforce the future producer's fixed FP32 storage.
TEACHER_FP32_LAYOUT_SOURCE_PINS={'echo_routing/temporal/anchor_cache.py': 'f98718880fb8195a3571b9a0f3fa6bccb7b4265d8ecc7ed8416ed957cdb3e96a', 'echo_routing/temporal/selected_risk_resume_source.py': '6d6ea8a634d75a812c51b1524dc8a94f67da028d895230fea2fc9195db06d5d2'}


def teacher_future_fp32_bound(roles):
    """Known persistent dtype, not a teacher numerical/precision change.

    The historical40T bound remains valid and retained. New ProbabilityCache
    casts to FP32 before np.savez, so only20T bytes per probability array are
    written. Keep4096/header+journal for every payload, two producer atomics and
    the entire1MiB metadata/manifests allowance. No compression is assumed.
    """
    import math
    values=[]
    for row in roles:
        t=row['positions'];n=row['sequences'];mt=row.get('max_length',t)
        if any(type(x) is not int or x<0 for x in [t,n,mt]) or mt>t or t<n:raise ValueError('fixed nonnegative complete role dimensions required')
        values.append((t,n,mt))
    t=sum(v[0] for v in values);n=sum(v[1] for v in values);mt=max((v[2] for v in values),default=0)
    return 2*(20*t+4096*n)+2*20*mt+1048576


def remaining_storage_components(work,materialized):
    """Remove completed FUTURE allocations only; current bytes never decrease.

    ``materialized`` is produced only by verify_materialized_storage below.
    Traces, fullstates, selected tables, FIT reports, bundles and every reserve
    remain conservatively forecast even when some of their files already exist.
    """
    out=copy.deepcopy(work);done=set(materialized)
    if ('seal_selectors' in done or 'prepare_formal' in done) and 'prepare_aux' not in done:raise ValueError('materialized later layout requires complete auxiliary inputs')
    if 'prepare_formal' in done and 'seal_selectors' not in done:raise ValueError('formal inputs require immutable complete selector cache')
    if 'prepare_aux' not in done:return out
    c=out['storage_components'];layout=out['feature_layout_components'];old=c['minimal_aux_plus_compact_shared_cache_phase_peak']
    remaining=(0 if 'seal_selectors' in done else layout['complete_two_seed_compact_cache'])+(0 if 'prepare_formal' in done else layout['formal_targets_only_no_persistent205'])
    if not 0<=remaining<=old:raise ValueError('invalid remaining complete feature allocation')
    c['minimal_aux_plus_compact_shared_cache_phase_peak']=remaining;out['storage_upper_bytes']-=old-remaining
    # Keep the whole original39.936MB draw allowance until formal preparation
    # actually finishes: each formal file atomically holds TWO draw arrays plus
    # its target/domain metadata. A lone one-array temporary is insufficient
    # before those writes. Afterwards all12 arrays are current retained bytes;
    # preserve the original extra one-array room conservatively.
    key='12_draw_arrays_plus_atomic_temp';draw=6000*64*8;count=12 if 'prepare_formal' in done else 0
    if c[key]!=13*draw:raise ValueError('original thirteen-array future draw layout changed')
    c[key]-=count*draw;out['storage_upper_bytes']-=count*draw
    out['verified_future_storage_obligations_completed']=sorted(done)
    return out


def _closed_storage_stage(context,spec,verified,workers,stage):
    """Actual parent wait + pinned child stdout must name the real payload."""
    import json
    if stage not in verified:return None
    _successful(stage,verified[stage]);pin=spec['predecessors'][stage]['payload_pin'];matches=[]
    def contains(value):
        return value==pin if isinstance(value,dict) and set(value)=={'path','sha256'} else any(contains(v) for v in value.values()) if isinstance(value,dict) else any(contains(v) for v in value) if isinstance(value,list) else False
    for row in workers:
        if row.get('stage')!=stage or row.get('model_pid') is None:continue
        if any(type(row.get(k)) is not int or row[k]!=0 for k in ['stage_exit_status','observed_worker_exit_code','inner_model_observed_OS_exit_code']) or row.get('whole_child_accounting_complete') is not True or row.get('actual_child_directly_waited') is not True or row.get('unresolved_worker_uncertainty') is True or row.get('runtime_authorization_sha256')!=context.owner_sha:continue
        for log in row.get('log_pins',[]):
            if not log['path'].endswith('.stdout'):continue
            for line in context.raw(log).splitlines():
                try:value=json.loads(line)
                except ValueError:continue
                if isinstance(value,dict) and value.get('stage')==stage and value.get('stage_returned_success') is True and value.get('runtime_authorization_sha256')==context.owner_sha and contains(value.get('output')):matches.append(row)
    if len(matches)!=1:raise ValueError('storage credit needs one actual closed parent and externally pinned successful child output: '+stage)
    return matches[0]


def verify_materialized_storage(context,inputs,verified,workers):
    """Read-only current payload/codec/journal proofs, never an asserted refund.

    No model, forward, optimizer, deletion or ledger mutation is possible here.
    Completion changes future obligations; all existing files remain in du.
    """
    relevant={'source_FIT','source_CONFIRM','prepare_aux','seal_selectors','prepare_formal'}&set(verified)
    if not relevant:return set()
    import numpy as np
    from echo_routing.temporal.selected_risk_resume_contract import RUN
    from echo_routing.temporal.selected_risk_resume_storage import storage_bound
    from echo_routing.temporal.selected_risk_resume_S import array_sha
    spec=context.read(context._bound_runtime_spec_pin);done=set()
    for stage in ['source_FIT','source_CONFIRM']:
        proofpin=spec['arguments'].get('completed_teacher_cache_verifications',{}).get(stage)
        if stage not in relevant or proofpin is None:continue
        recovery=spec['arguments'].get('source_CPU_final_cap_recovery')
        if stage=='source_FIT' and recovery is not None:
            if proofpin!=recovery['cache_verification']:raise ValueError('materialized failed-source recovery must use the exact independent cache proof')
            from echo_routing.temporal.selected_risk_resume_source_adoption import verify_existing_FIT
            verify_existing_FIT(context,inputs,verified[stage],recovery,workers);done.add(stage);continue
        row=_closed_storage_stage(context,spec,verified,workers,stage);proof=context.read(proofpin);phase=stage.split('_')[1];producer=verified[stage];pin=spec['FIT_producer' if phase=='FIT' else 'CONFIRM_producer']
        if proof.get('schema')!='A63_resume_closed_teacher_readonly_verification_v1' or proof.get('passed') is not True or proof.get('complete_cache_verified') is not True or proof.get('phase')!=phase or proof.get('producer_manifest')!=pin or pin!=spec['predecessors'][stage]['payload_pin'] or context.read(proof['actual_parent_OS_receipt'])!=row or (proof.get('model_pid'),proof.get('model_start_ticks'),proof.get('boot_id'))!=(row['model_pid'],row['model_start_ticks'],row['boot_id']):raise ValueError('independent complete teacher cache/current actual OS proof mismatch')
        records=producer['records'];reported=proof.get('records',[])
        if len(reported)!=4 or {(r['source'],r['role']) for r in reported}!={(r['source'],r['role']) for r in records}:raise ValueError('all four independently verified teacher caches required')
        for role in sorted({r['role'] for r in records}):
            qs=inputs.sequences(role);ps=inputs._teachers(role,qs)
            for source,values in ps.items():
                original=next(r for r in records if (r['source'],r['role'])==(source,role));audit=next(r for r in reported if (r['source'],r['role'])==(source,role))
                if any(audit.get(k)!=original.get(k) for k in ['path','identity','input_sha256','journal_sha256']) or audit.get('committed_payloads')!=len(qs) or audit.get('dtype')!='float32' or audit.get('all_payload_raw_SHA_verified') is not True or any(np.asarray(v).dtype!=np.float32 or np.asarray(v).shape!=(len(q['refs']),5) for v,q in zip(values,qs)):raise ValueError('current complete source cache differs from independent recorded layout')
        done.add(stage)
    if 'prepare_aux' in relevant:
        _closed_storage_stage(context,spec,verified,workers,'prepare_aux');prepared=verified['prepare_aux'];draws=0
        from echo_routing.temporal.selected_risk_training import aux_plans
        from echo_routing.temporal.safety_data import exact
        for fold in prepared['folds']:
            pin=fold['data']
            if pin['path']!=RUN+f'/aux/held{fold["held_fold"]}/training_inputs.private.npz':raise ValueError('exact immutable auxiliary input path required')
            raw=context.raw(pin);value=context.load_value(pin)
            if set(value)!={'data','plans'} or storage_bound(value)['artifact_bytes']!=len(raw):raise ValueError('complete lossless typed auxiliary artifact required')
            data=value['data'];plans=value['plans']
            if len(data['targets'])!=fold['candidate_count'] or len(data['position_labels'])!=fold['changed_positions'] or data['F_fit']!=fold['F_fit'] or data['scale']!=fold['scale'] or set(plans)!={'2026110461','2026110462'}:raise ValueError('full current auxiliary allocation differs from its pinned public dimensions')
            for seed in [2026110461,2026110462]:
                expected=aux_plans(data,seed)
                if not exact(plans[str(seed)],expected):raise ValueError('independent complete auxiliary draw reconstruction differs')
                for plan in expected.values():
                    if plan['indices'].shape!=(6000,64) or plan['indices'].dtype!=np.int64:raise ValueError('complete fixed int64 auxiliary draw arrays required')
                    draws+=1
        if draws!=8:raise ValueError('all eight actual auxiliary draw arrays required')
        done.add('prepare_aux')
    if 'seal_selectors' in relevant:
        _closed_storage_stage(context,spec,verified,workers,'seal_selectors');selector=verified['seal_selectors'];manifest=context.read(selector['cache_manifest']);records=manifest['records'];refs={}
        from echo_routing.temporal.selected_risk_compact_cache import CompactSupplyCache
        from echo_routing.temporal.anchor_cache import sha
        if manifest.get('complete') is not True or len(records)!=8 or {(r['seed'],r['role']) for r in records}!={(s,r) for s in [2026110461,2026110462] for r in ['F0_native','F0_mixed','F1_native','F1_mixed']}:raise ValueError('all eight complete selector cache records required')
        for seed in [2026110461,2026110462]:
            for role in ['F0_native','F0_mixed','F1_native','F1_mixed']:
                qs,a,j,b=inputs.bank(role,purpose='selector');r=next(r for r in records if (r['seed'],r['role'])==(seed,role));c=CompactSupplyCache(context.root/r['path'],r['identity'],b,[v.argmax(1) for v in a],[v.argmax(1) for v in j],readonly=True,feature_reference=refs.get(role))
                if c.cursor!=len(qs) or sha(c.journal)!=r['journal_sha256'] or r['input_sha256']!=inputs.manifest['roles'][role]['sha256']:raise ValueError('complete current immutable selector payloads/journal/input required')
                if seed==2026110461:refs[role]=c
        done.add('seal_selectors')
    if 'prepare_formal' in relevant:
        _closed_storage_stage(context,spec,verified,workers,'prepare_formal');prepared=verified['prepare_formal']
        from echo_routing.temporal.selected_risk_resume_formal import validate_matched_draws
        if len(prepared.get('records',[]))!=2 or {r['seed'] for r in prepared['records']}!={2026110461,2026110462}:raise ValueError('both actually prepared formal domains required')
        for row in prepared['records']:
            pin=row['data'];raw=context.raw(pin);data=context.load_value(pin)
            if pin['path']!=RUN+f'/formal/seed{row["seed"]}/training_inputs.private.npz' or storage_bound(data)['artifact_bytes']!=len(raw) or set(data)!={'candidate_digests','unsafe','kind_indices','draws'} or len(data['unsafe'])!=row['candidate_count']:raise ValueError('complete exact typed formal allocation required')
            validate_matched_draws(data['draws'],data['kind_indices'],len(data['unsafe']))
            if any(data['draws'][a].dtype!=np.int64 or array_sha(data['draws'][a])!=row['draw_SHA'][a] for a in ['C0','C1']):raise ValueError('current actual formal global draw arrays changed')
        done.add('prepare_formal')
    return done


def formal_trace_projection(context,geometry):
    from echo_routing.temporal.selected_risk_resume_trace_bound import formal_trace_raw_bound
    roles=geometry['roles']
    if set(roles)!={'F0_native','F1_native','F0_mixed','F1_mixed'} or any(type(r['geometry']['C']) is not int or r['geometry']['C']<0 for r in roles.values()):raise ValueError('complete four-role observed global candidate volume required')
    bound=formal_trace_raw_bound(sum(r['geometry']['C'] for r in roles.values()))
    for pin in bound['source_pins']:context.raw(pin)
    return bound


def remove_complete(work,verified):
    out=copy.deepcopy(work);p=out['phases'];removed=[]
    for phase,stage in [('FIT','source_FIT'),('CONFIRM','source_CONFIRM')]:
        if stage in verified:
            _successful(stage,verified[stage]);value=out['source_'+phase+'_GPU_forecast']
            if value<0 or value>p['teacher_CPU_including_entry_load_IO']:raise ValueError('invalid completed teacher forecast component')
            out['GPU_seconds']-=value;p['teacher_CPU_including_entry_load_IO']-=value;removed.append({'stage':stage,'GPU_seconds':value,'CPU_seconds':value})
    phases={'audit_aux':['aux48000_updates'],'audit_formal':['formal24000_updates'],'seal_selectors':['D_X_held_score_all_features_2seeds_and_cache_IO'],'coverage':['selected_effects_signatures_coverage_recount'],'aggregate':['140paired_5000_same_draws_and_independent_recount'],'supports':['56_support_48readonly_8new_joint_solves'],'forensics':['238outer_34inner_descriptors_native_regression_support']}
    for stage,keys in phases.items():
        if stage in verified:
            _successful(stage,verified[stage]);amount=sum(p[k] for k in keys)
            for k in keys:p[k]=0.
            removed.append({'stage':stage,'CPU_seconds':amount,'GPU_seconds':0.})
    cpu=sum(r['CPU_seconds'] for r in removed);out['CPU_seconds']-=cpu;out['remaining_wall_seconds']-=cpu
    if min(out['CPU_seconds'],out['GPU_seconds'])<0 or p['resource_OS_independent_audit_publication']<3600:raise ValueError('completed-phase subtraction cannot erase original closing or produce negative work')
    out['verified_whole_phase_forecast_components_removed']=removed;return out


def estimate(context,inputs,current_upper,verified):
    from scripts.selected_risk_common import prior_path
    from echo_routing.temporal.selected_risk_cost_units import remaining
    from echo_routing.temporal.selected_risk_forecast import teacher_bytes,verify_cost_inputs,COST_INPUTS
    import json
    # Original public profiles are part of the immutable old840 publication.
    # Runtime verifier uses the already reviewed offline Git proof, never git.
    verify_cost_inputs(context.root)
    def old(name):
        from echo_routing.temporal.selected_risk_resume_context import CLOSED_MANIFEST_ALIAS,OLD_MANIFEST
        path=OLDPUB+name;row=next(r for r in context.read({'path':CLOSED_MANIFEST_ALIAS,'sha256':OLD_MANIFEST})['files'] if r['path']==path)
        row={**row,'path':context.closed_publication_path(path)}
        return context.read({k:row[k] for k in ['path','sha256']})
    profile=old('cost_profile.json');shape=old('cost_shape_profile.json')
    if shape.get('passed') is not True or profile.get('passed') is not True or shape['input_manifest_sha256']!=inputs.metadata_pin['sha256'] or profile['input_manifest_sha256']!=inputs.metadata_pin['sha256']:raise ValueError('actual frozen engineering profile and original input bindings required')
    producer=context.read({'path':'docs/results/2026-11-04_anchor_context_risk/producer_manifest.json','sha256':COST_INPUTS['docs/results/2026-11-04_anchor_context_risk/producer_manifest.json']})
    wall=context.read({'path':'docs/results/2026-11-04_anchor_context_risk/process_producer_1.json','sha256':COST_INPUTS['docs/results/2026-11-04_anchor_context_risk/process_producer_1.json']})['full_process_wall_seconds']
    units={name:sum(r['committed_logical_neural_units'] if 'committed_logical_neural_units' in r else r['actual_neural_forward_units'] for r in producer['records'] if r['source']==name) for name in ['a08','jepa']}
    sessions=sorted(sum(inputs.partition()['sessions'],[]))
    work=remaining(profile,shape,inputs.manifest['roles'],old_source_wall=wall,old_a08_units=units['a08'],old_jepa_units=units['jepa'],current_upper=current_upper,sessions=sessions)
    from echo_routing.temporal.selected_risk_resume_runtime import RuntimeState
    specpin=context._bound_runtime_spec_pin;spec=context.read(specpin)
    materialized=verify_materialized_storage(context,inputs,verified,RuntimeState(context,spec,specpin)._worker_rows())
    # Prospective exact producer-layout refinement only. Old executed modules
    # and original40T forecast remain byte-identical; old caches stay in du.
    for path,h in TEACHER_FP32_LAYOUT_SOURCE_PINS.items():context.raw({'path':path,'sha256':h})
    old_teacher=work['storage_components']['teacher_payload_journal_atomic_upper']
    future_roles=[]
    for stage,names in [('source_FIT',['F0_mixed','F1_mixed']),('source_CONFIRM',['a63_confirmation_selection','a63_confirmation_calibration'])]:
        if stage not in materialized:future_roles.extend(names)
    new_teacher=teacher_future_fp32_bound([inputs.manifest['roles'][r] for r in future_roles])
    if new_teacher>old_teacher:raise ValueError('FP32 producer format cannot enlarge or negate original teacher projection')
    work['legacy_float64_teacher_future_upper_bytes']=old_teacher
    work['teacher_FP32_producer_layout_source_pins']=dict(TEACHER_FP32_LAYOUT_SOURCE_PINS)
    work['teacher_future_layout_refinement_does_not_change_producer_values']=True
    if materialized:work['read_only_materialized_storage_phases']=sorted(materialized)
    work['storage_components']['teacher_payload_journal_atomic_upper']=new_teacher
    work['storage_upper_bytes']-=old_teacher-new_teacher
    # Additional independently bound rich/summary/bundle artifacts and the new
    # protocol are prospective costs, not a reduction of any original floor.
    work['phases']['continuation_provenance_codec_and_protocol_CPU']=360.
    work['CPU_seconds']+=360.;work['remaining_wall_seconds']+=360.
    work['storage_components']['continuation_bound_rich_private_bundles_allowance']=48000000
    work['storage_upper_bytes']+=48000000
    # Keep the broad projection if actual observables exceed it; the measured
    # shape gate rejects an unsupported training domain. No smaller GT-driven
    # estimate is selected. Full predraw bytes are an additional explicit bound.
    if 'observe_geometry' in verified:
        g=verified['observe_geometry'];_successful('observe_geometry',g)
        from echo_routing.temporal.selected_risk_aux_pipeline import validate_measured_shapes
        validate_measured_shapes(g['shape_maxima'])
        if g['input_manifest_sha256']!=inputs.metadata_pin['sha256'] or len(g['aux_full_predraw'])!=8:raise ValueError('complete current observable geometry/all eight6000 draws required')
        projected=work['projected_geometry']
        for role in ['F0_mixed','F1_mixed']:
            actual=g['roles'][role]['geometry']
            if any(actual.get('all_'+k,actual[k])>projected[role].get('all_'+k,projected[role][k])+1e-9 for k in ['N','T','T2','C','C2','TC','K','KT','CR']):raise ValueError('actual teacher geometry exceeds frozen empirical density2 projection; fresh engineering review required')
        full=sum(r['trace_raw_byte_upper'] for r in g['aux_full_predraw'])
        if full>work['original_empirical_trace_bytes']:raise ValueError('actual full auxiliary6000-index raw bytes exceed admitted complete trace forecast')
        work['actual_aux_full_predraw_raw_bytes']=full
        # Deflate stored-block overhead safely covers incompressible data. Do
        # not use the empirical .5 ratio as an actual hard-storage guarantee.
        formal_trace=formal_trace_projection(context,g);formal_per_head=formal_trace['per_head_raw_bytes']
        work['formal_trace_structural_bound']=formal_trace
        work['legacy_formal_trace_8192_per_head_raw_bytes']=6000*8192
        perhead=[r['trace_raw_byte_upper'] for r in g['aux_full_predraw']]+[formal_per_head]*4
        gzip_worst=sum(n+5*((n+16382)//16383)+64 for n in perhead)
        complete_and_active=gzip_worst+max(perhead)
        key='retained_archive_plus_one_active_raw_and_atomic_peak'
        extra=max(0,complete_and_active-work['storage_components'][key]);work['storage_components'][key]+=extra;work['storage_upper_bytes']+=extra
        work['actual_predraw_all_trace_gzip_worst_plus_active_bytes']=complete_and_active
    # Full original FIT D/X/all risk diagnostics include genuine final-risk
    # inference, exact global-map exposure, cache IO and both publication copies.
    # Native G265 is fixed by the approved original bank; mixed geometry uses
    # the same frozen density2 forecast, never new labels/head outcomes.
    roles=work['projected_geometry'];c=265+sum(roles[r]['C'] for r in ['F0_mixed','F1_mixed'])
    axes=2736+sum(roles[r]['N'] for r in ['F0_mixed','F1_mixed']);nonempty=265+sum(min(roles[r]['C'],roles[r]['N']) for r in ['F0_mixed','F1_mixed'])
    fit=fit_diagnostic_cost(c,nonempty,axes)
    work['phases']['FIT_final_R1_R2_C0_C1_q_and_actual_global_exposure_CPU']=fit['CPU_seconds']
    work['CPU_seconds']+=fit['CPU_seconds'];work['remaining_wall_seconds']+=fit['remaining_wall_seconds']
    work['storage_components']['FIT_final_risk_q_exposure_and_report_complete_bound']=fit['storage_upper_bytes'];work['storage_upper_bytes']+=fit['storage_upper_bytes'];work['FIT_diagnostic_complete_bound']=fit
    work=remove_complete(work,verified)
    work=remaining_storage_components(work,materialized)
    if spec['arguments'].get('source_CPU_final_cap_recovery') is not None:
        workers=RuntimeState(context,spec,specpin)._worker_rows()
        teacher_spent=sum(r['measured_CPU_core_seconds'] for r in workers if r['stage'] in ['source_FIT','source_CONFIRM'])
        remaining_teacher=work['phases']['teacher_CPU_including_entry_load_IO']
        work['teacher_CPU_allocation']={'actual_closed_source_CPU_seconds':teacher_spent,'remaining_teacher_CPU_forecast':remaining_teacher,'original_card_source_CPU_limit':6000.,'fits':teacher_spent+remaining_teacher<=6000}
        if not work['teacher_CPU_allocation']['fits']:raise ValueError('actual source plus all remaining teacher CPU exceeds original source6000 allocation')
        if 'source_FIT_adoption' not in verified:
            work['phases']['readonly_failed_source_recovery_CPU']=120.;work['CPU_seconds']+=120.;work['remaining_wall_seconds']+=120.
    work['prepared_data_atomic_bytes_upper']=work['feature_layout_components']['minimal_aux_atomic_peak']
    work['formal_data_atomic_bytes_upper']=39936000
    work['selected_table_atomic_bytes_upper']=work['selected_record_bound']['atomic_coexistence_upper_bytes']
    work['source_stage_payload_byte_upper']={phase:teacher_future_fp32_bound([inputs.manifest['roles'][r] for r in roles]) for phase,roles in [('FIT',['F0_mixed','F1_mixed']),('CONFIRM',['a63_confirmation_selection','a63_confirmation_calibration'])]}
    work['scope']='estimated full72000/238/34/140x5000/56/34/32 through actual closure/main; CPU/GPU cumulative budgets not reset; storage follows separately raw-approved lifecycle cap; not a physical guarantee'
    return work
