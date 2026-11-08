"""A63 prospective empirical full-pass forecast; no new GT/head-output inputs.

Old native bank is identical. Constructed per-position work densities are
multiplied by TWO before any new teacher geometry exists. This is a forecast,
not a grammar guarantee. Every volume and the grammar worst case remain public.
"""
import math
TERMS=('N','T','T2','C','C2','TC','K','KT','CR')
DRAW_PLAN_BYTES=13*6000*64*8 # eight aux + four formal + one atomic temp
SCOPE={'aux_updates':48000,'formal_updates':24000,'outer_rows':238,'inner_rows':34,'comparisons':140,'replicates':5000,'support':56,'native':34,'regression':32,'fullstates':48}

def nonnegative(x):
    if isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) or x<0:raise ValueError('finite nonnegative measured or projected value required')
    return float(x)

def project(old,role,density=2):
    if density!=2:raise ValueError('prospective constructed density factor fixed at TWO')
    t=nonnegative(role['positions']);n=nonnegative(role['sequences']);mt=nonnegative(role['max_length']);oldT=nonnegative(old['T'])
    if not oldT or not mt or not n:raise ValueError('complete nonempty role and old observable source required')
    out={k:math.ceil(nonnegative(v)*t/oldT*density) for k,v in old.items() if k in {*TERMS,*('all_'+x for x in TERMS)}}
    out.update(N=int(n),T=int(t),T2=int(t*mt),max_T=int(mt));return out

def teacher_bytes(roles):
    # Two float64 T-by-5 probability payloads, <=4096B per payload/journal
    # entry, one atomic payload per producer plus 1MiB manifests/filesystem room.
    t=sum(nonnegative(r['positions']) for r in roles);n=sum(nonnegative(r['sequences']) for r in roles)
    mt=max((r.get('max_length',r['positions']) for r in roles),default=0)
    return math.ceil(2*(40*t+4096*n)+2*40*mt+1048576)

def teacher_forecast(a08,jepa,old_a08,old_jepa,old_wall,startup):
    if min(old_a08,old_jepa,old_wall)<=0:raise ValueError('separate physical source units and occupied wall required')
    # A08 windows and JEPA chunks are NOT interchangeable units.
    return 1.25*max(nonnegative(a08)/old_a08,nonnegative(jepa)/old_jepa)*old_wall+nonnegative(startup)

def add(gs):
    return {k:sum(float(g.get(k,0)) for g in gs) for k in set().union(*(set(g) for g in gs)) if k!='max_T'}

def backend(g,coef):
    return 2*sum(nonnegative(g.get('all_'+k,g.get(k,0)))*nonnegative(coef[k]) for k in TERMS)

def remaining(profile,roles,*,old_source_wall,old_source_units=None,old_a08_units=None,old_jepa_units=None,current_upper):
    # old_source_units remains only a fabricated-fixture compatibility argument.
    if old_a08_units is None:old_a08_units=old_source_units
    if old_jepa_units is None:old_jepa_units=old_source_units
    fitnames=['F0_mixed','F1_mixed'];cnames=['a63_confirmation_selection','a63_confirmation_calibration']
    new={r:project(profile['geometry']['mixed'],roles[r]) for r in fitnames+cnames}
    fit=add([profile['geometry']['native'],*[new[r] for r in fitnames]])
    confirm=add([new[r] for r in cnames]);train=profile['training'];coef=profile['backend_coefficients']
    a=lambda names,key:sum(roles[r][key] for r in names)
    fitgpu=teacher_forecast(a(fitnames,'a08_stride5_windows'),a(fitnames,'jepa_chunks'),old_a08_units,old_jepa_units,old_source_wall,100)
    cgpu=teacher_forecast(a(cnames,'a08_stride5_windows'),a(cnames,'jepa_chunks'),old_a08_units,old_jepa_units,old_source_wall,50)
    phases={
      'aux48000_updates':2*24000*(train['utility_update_seconds']+train['risk_update_seconds']),
      'formal24000_updates':2*24000*train['event_update_seconds'],
      'fullstate768_write_read':2*768*train['fullstate_write_read_seconds'],
      'complement_prepare_and_independent_recount':4*backend(fit,coef),
      'D_X_held_score_all_features_2seeds_and_cache_IO':8*backend(fit,coef),
      'selected_effects_signatures_coverage_recount':4*backend(fit,coef),
      'fixed_new_confirmation_and_new_q':12*backend(confirm,coef),
      # Complete historical rows/CI/readout are retained even though frozen
      # old policy metrics can be reused. This floor is independent of C/K.
      '238outer_34inner_descriptors_native_regression_support':2400+8*backend(confirm,coef),
      '140paired_5000_same_draws_and_independent_recount':2*140*profile['CI5000_seconds'],
      '56_support_48readonly_8new_joint_solves':600,
      'independent_models_states_calibration_Git_audits':900,
      'figures_PDF_visual_QA_and_report':900,
      'resource_OS_independent_audit_publication':3600,
      'teacher_CPU_including_entry_load_IO':fitgpu+cgpu}
    C=fit['C'];K=fit['K'];N=fit['N'];teacher=teacher_bytes([roles[r] for r in fitnames+cnames])
    # Complement datasets jointly one full new FIT; two simultaneously retained
    # representations/copies, raw joint D/X perseed, and future event205 data.
    feature=math.ceil(2*(8*109*C+8*47*K+8*96*C+8*4*C+128*N)+2*(8*217*C+4096*N)+4*205*C)
    states=48*250000+2*250000 # deliberately above actual small-head state size
    native_mean=profile['geometry']['native']['C']/max(1,profile['geometry']['native'].get('nonempty_query_axes',profile['geometry']['native']['N']))
    mixed_mean=2*profile['geometry']['mixed']['C']/max(1,profile['geometry']['mixed'].get('nonempty_query_axes',profile['geometry']['mixed']['N']))
    # Empirical JSON trace size includes FULL index arrays and identity/grad fields.
    # Actual pre-draw byte bounds are recomputed after teachers, before training.
    traces=math.ceil(24000*(3*2000+2*64*7+64*7+32*(native_mean+mixed_mean)*7))
    selected_records=4*(sum(roles[r]['sequences'] for r in fitnames)+profile['geometry']['native']['C'])*4096 # per-seed D/X, append-only bounded digest rows; no expanded tuples
    upper=max(nonnegative(current_upper),300000000)+teacher+feature+states+traces+DRAW_PLAN_BYTES+selected_records+128000000
    cpu=sum(phases.values());return {'estimator':'old_observable_density2_full_scope_v1','scope':dict(SCOPE),'projected_geometry':new,'phases':phases,'CPU_seconds':cpu,'GPU_seconds':fitgpu+cgpu,'source_FIT_GPU_forecast':fitgpu,'source_CONFIRM_GPU_forecast':cgpu,'remaining_wall_seconds':cpu+7200,'storage_upper_bytes':math.ceil(upper),'storage_components':{'current_namespace_plus_admin_upper':max(current_upper,300000000),'teacher_payload_journal_atomic_upper':teacher,'features_joint_cache_and_event_copies':feature,'retained_fullstates_atomic_upper':states,'12_draw_arrays_plus_atomic_temp':DRAW_PLAN_BYTES,'selected_digest_records_4096_bytes_each':selected_records,'72000_trace_rows_empirical_index_aware_forecast':traces,'closing_reserve':128000000},'old_geometry_is_stratified':True,'new_observed_GT_or_head_outputs_used':False,'empirical_estimate_not_grammar_guarantee':True,'separate_A08_J_ratios_maxed_not_summed':True}


COST_INPUTS={'docs/results/2026-11-04_anchor_context_risk/cost_profile.json': 'bff4ea23faf14f66d88dd415972a69b5c22ecaac5704ff48473337eec13c14d6', 'docs/results/2026-11-04_anchor_context_risk/process_producer_1.json': '86992a9f25db3769147de5c4792fc611544298c6f835275057c8eee1756f5b69', 'docs/results/2026-11-04_anchor_context_risk/producer_manifest.json': '2be60e89a34c3e329845318c01ea304ee79439573aba4d946aaee386c473727b'}
COST_INPUT_GIT='db0d04c6543813e2bc3d3456d3a6ecdfd09f327e'
def verify_cost_inputs(root):
    from echo_routing.temporal.anchor_cache import sha
    from echo_routing.temporal.safety_git import verify
    import json
    from pathlib import Path
    if any(sha(Path(root)/p)!=h for p,h in COST_INPUTS.items()):raise ValueError('prespecified closed engineering cost input bytes changed')
    fp=Path(root)/COST_INPUT_PROOF
    if sha(fp)!=COST_INPUT_PROOF_SHA:raise ValueError('literal prebuilt actual-Git cost proof changed')
    proof=json.loads(fp.read_text());verify(proof,COST_INPUT_GIT,COST_INPUTS);return proof

COST_INPUT_PROOF='docs/results/2026-11-05_selected_action_risk/cost_input_actual_Git_proof.json'
COST_INPUT_PROOF_SHA='4fb8ac23766e403d036b3dcfda725f4c1a8811e9f31241616b9282f38a9108ca'
