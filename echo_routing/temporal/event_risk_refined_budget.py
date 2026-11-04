"""Prospective measured factorization; no outcome/GT/model-selection interface.

Empty-axis timing supplies N/T/T²; long-envelope timing supplies C/C²/TC;
long-K supplies K/KT; legal high-RLE supplies CR. Each term is nonnegative.
We verify every synthetic measured profile is dominated before factor-two
uncertainty. A dense matcher runtime is never assigned to a per-T coefficient.
All original complete-workload counts, remaining phase floors and caps stay.
"""
import math
from echo_routing.temporal.event_risk_measured_budget import remaining as original_remaining
from echo_routing.temporal.event_risk_data import ROLES

KEYS=('N','T','T2','C','C2','TC','K','KT','CR')


def volumes(g):
    # Pre-G proposal/partition work is retained even if no eligible action.
    return {k:float(g.get('all_'+k,g[k])) if k in ['C','C2','TC','K','KT','CR'] else float(g[k]) for k in KEYS}


def estimate(g,coef):
    values=volumes(g)
    if any(not math.isfinite(x) or x<0 for x in [*values.values(),*coef.values()]):raise ValueError('finite nonnegative measured work/cost required')
    return 2*sum(values[k]*coef[k] for k in KEYS)


def coefficients(profile,field):
    rows={p['name']:p for p in profile['profiles']}
    if len(rows)!=len(profile['profiles']) or set(rows)!={'empty','envelope','legal_RLE','long_K'}:raise ValueError('complete independently frozen four-profile measurement required')
    source={'N':'empty','T':'empty','T2':'empty','C':'envelope','C2':'envelope','TC':'envelope','K':'long_K','KT':'long_K','CR':'legal_RLE'}
    c={k:rows[n][field]/max(1.,volumes(rows[n]['geometry'])[k]) for k,n in source.items()}
    if any(not math.isfinite(v) or v<0 for v in c.values()):raise ValueError('real nonnegative measured timings required')
    for p in rows.values():
        if estimate(p['geometry'],c)/2+1e-12<p[field]:raise ValueError('factorized terms do not dominate measured profile')
    return c


def validate_geometry(original,refined):
    if set(original['geometry'])!=set(refined['geometry']):raise ValueError('all original observable roles retained')
    for role,old in original['geometry'].items():
        new=refined['geometry'][role]
        if any(old[k]!=new[k] for k in ['N','T','C','TC','K','input_sha256']):raise ValueError('original role geometry/input changed')
    return True


def remaining(original,refined,manifest,fit_checks=None,sources_done=False,formal_done=False,prior_producer=None):
    validate_geometry(original,refined)
    if set(refined['future_structural_bounds'])!=set(ROLES):raise ValueError('all540 future-source bounds required')
    old=[g for r,g in refined['geometry'].items() if r not in ['fit','inner_cal']]
    new=[refined['future_structural_bounds'][r] for r in ROLES];inner=refined['geometry']['inner_cal'];groups=old+new+[inner]
    for role in ROLES:
        g=refined['future_structural_bounds'][role];r=manifest['roles'][role]
        if g['N']!=r['sequences'] or g['T']!=r['positions'] or g['max_T']!=r['max_length']:raise ValueError('future frozen-role length volumes differ')
    fields=['event_inference_seconds','fixed_inference_seconds','six_gate_summary_seconds','three_oracle_options_coverage_solve_seconds','raw_cache_init_write_read_seconds'];coef={f:coefficients(refined,f) for f in fields}
    def cost(g,field):return estimate(g,coef[field])
    solver=refined['integer_solver']
    if solver['streams']<150 or solver['options_each_excluding_noop']<max([g.get('max_C',g.get('C',0)) for g in refined['geometry'].values()]+[(3*g['max_T']+1)//2 for g in new]):raise ValueError('synthetic pooled solver does not dominate per-stream option and stream counts')
    value=original_remaining(original,manifest,fit_checks,sources_done,formal_done,prior_producer);phases=value['phases']
    phases['new_and_cached_q_inference']=2*sum(cost(g,'event_inference_seconds') for g in groups)
    phases['fixed_old_heads_new540']=6*sum(cost(g,'fixed_inference_seconds') for g in new)
    phases['action_descriptors_recounts']=8*sum(cost(g,'six_gate_summary_seconds') for g in groups)
    # Three safety domains are already inside each measured oracle backend.
    # Retain the prior conservative logical-pass multiplicities plus all54
    # pooled integer solves (6 new full ceilings and48 N-top support records).
    phases['oracles_support_native_regression']=12*sum(cost(refined['future_structural_bounds'][r],'three_oracle_options_coverage_solve_seconds') for r in ROLES[2:])+4*sum(cost(g,'six_gate_summary_seconds') for g in groups)+2*54*solver['seconds']
    phases['raw_joint_cache_IO']=4*sum(cost(g,'raw_cache_init_write_read_seconds') for g in groups)
    cpu=sum(v for k,v in phases.items() if k!='teacher_GPU_wall');value.update(CPU_seconds=cpu,remaining_wall_seconds=cpu+value['GPU_seconds']+300,phases=phases,factorized_coefficients=coef,measured_profile_domination=True,original_volume_terms_retained=list(KEYS),N1_alias_not_doublecharged=True,estimator_version='legal_source_geometry_v2',actual_spend_is_separate_from_forecast=True)
    if not all(math.isfinite(x) and x>=0 for x in [cpu,value['remaining_wall_seconds']]):raise ValueError('invalid complete revised forecast')
    return value
