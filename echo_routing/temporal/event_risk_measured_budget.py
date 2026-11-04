"""Prospective complete A62 workload from measured CPU backends, never outcomes.

Nonnegative observable N/T/C/TC/K terms retain empty-axis and worst geometry.
Synthetic backend timings have factor two, training factor1.5; actual spend is
charged independently. None of these forecasts is a hardware lower bound.
"""
import math
from echo_routing.temporal.event_risk_data import ROLES
SCOPE={'formal_updates':24000,'engineering_updates':72,'outer_rows':156,'inner_rows':26,'comparisons':240,'replicates':5000,'new_full_oracles':6,'old_readonly_oracles':30,'support':48,'native':24,'base_native':2,'regression':16,'fullstates':16}


def upper(row):
    n=int(row['sequences']);t=int(row['positions']);maxT=int(row['max_length'])
    # At most T disjoint raw differences plus ceil(T/2) nontrivial J runs.
    return {'N':n,'T':t,'C':2*t,'TC':2*t*maxT,'K':2*t}


def remaining(profile,manifest,fit_checks=None,sources_done=False,formal_done=False,prior_producer=None):
    profiles=profile['profiles']
    if not profiles or any(p['T']<1 or p['C']<0 for p in profiles):raise ValueError('actual synthetic geometry profiles required')
    def cost(g,field):
        zero=max(p[field] for p in profiles if p['C']==0)
        non=[p for p in profiles if p['C']>0]
        coefficients={'T':max(p[field]/p['T'] for p in non),'C':max(p[field]/p['C'] for p in non),'TC':max(p[field]/p['T_times_C'] for p in non),'K':max(p[field]/max(1,p['K_total']) for p in non)}
        return 2*(g['N']*zero+sum(g[k]*v for k,v in coefficients.items()))
    old=[g for r,g in profile['geometry'].items() if r not in ['fit','inner_cal']]
    new=[upper(manifest['roles'][r]) for r in ROLES]
    # CAL/AUDIT unchanged native subsets together are bounded by complete628
    # native inner geometry; its old300 mixed were excluded by the cost scanner.
    inner=profile['geometry']['inner_cal'];groups=old+new+[inner]
    qcost=2*sum(cost(g,'full_q_inference_seconds') for g in groups)
    # Fixed S-L+position+FP modules only new540 axes, not old score-cache roles.
    fixed=6*sum(cost(g,'full_q_inference_seconds') for g in new)
    # Source and complete-task descriptors: two read/recount passes per seed,
    # plus CAL and AUDIT. Native/pair reuse is charged as composition work.
    descriptions=8*sum(cost(g,'six_gate_summary_seconds') for g in groups)
    # Six new full candidate oracles (3 safety filters x2 halves),48 N-top
    # supports, native24,regression16,independent descriptors retain high-RLE work.
    oracle=12*sum(cost(upper(manifest['roles'][r]),'six_gate_summary_seconds') for r in ROLES[2:])+4*sum(cost(g,'six_gate_summary_seconds') for g in groups)
    cache=4*sum(cost(g,'raw_cache_init_write_read_seconds') for g in groups)
    unit=profile['synthetic_CI_unit_upper'];maxnative=max(manifest['roles'].get(r,{}).get('native',unit['native_sessions']*3) for r in ['selection','calibration'])
    if unit!={'native_sessions':100,'multi_streams':150,'pair_streams':300} or maxnative>300:raise ValueError('synthetic complete CI units do not dominate actual native role; separately qualify larger units')
    evaluate=2*(156*2+26)*profile['synthetic_fulltask_metric_seconds']
    ci=2*240*profile['synthetic_CI5000_seconds']
    if fit_checks is None:update=max(.01,40*profile['max_64candidate_forward_seconds']);disk=.02
    else:update=fit_checks['max_complete_update_seconds'];disk=fit_checks['max_disk_save_load_seconds']
    training=0 if formal_done else 1.5*24000*update+256*max(.001,disk)
    engineering=0 if fit_checks is not None else 72*max(update,.02)+300
    # Original checkpoint formats are only66KB; intentionally allow1MiB/atomic.
    weights=4*4*1048576+4*2*1048576
    empty_bytes=max(p['payload_bytes'] for p in profiles if p['C']==0);bytes_per_C=max(p['payload_bytes']/p['C'] for p in profiles if p['C']>0)
    payload=4*sum(g['N']*empty_bytes+g['C']*bytes_per_C for g in groups)
    teacher_bytes=2*sum(manifest['roles'][r]['positions']*5*8+manifest['roles'][r]['sequences']*20000 for r in ROLES)
    private=max(100000000,payload+teacher_bytes+weights+60000000)
    gpu=0.
    if not sources_done:
        if prior_producer:
            records=prior_producer['records'];rates={s:max(r['wall_seconds']/r['committed_total_neural_units'] for r in records if r['source']==s) for s in ['a08','jepa']}
            gpu=120+2*sum(manifest['roles'][r]['a08_stride5_windows']*rates['a08']+manifest['roles'][r]['jepa_chunks']*rates['jepa'] for r in ROLES)
        else:gpu=1200.  # Tiny pure fixture only; actual CLI always supplies pinned prior.
    phases={'teacher_GPU_wall':gpu,'teacher_CPU_upper':gpu,'FIT_context_prepare_and_recount':600 if fit_checks is None else 0,'FIT72_engineering':engineering,'formal24000_and_IO':training,'new_and_cached_q_inference':qcost,'fixed_old_heads_new540':fixed,'full_task_metrics_independent_recount':evaluate,'action_descriptors_recounts':descriptions,'raw_joint_cache_IO':cache,'paired240x5000':ci,'oracles_support_native_regression':oracle,'model_and_calibration_independent_audits':600,'six_figures_PDF_QA_report':600,'resource_history_exit_collector_and_handoff':3600}
    cpu=sum(v for k,v in phases.items() if k!='teacher_GPU_wall')
    # Reserve3600 CPU seconds for actual closure; this is a reserve, not spend.
    cpu=max(cpu,3600.)
    if not all(math.isfinite(x) and x>=0 for x in [cpu,gpu,private]):raise ValueError('nonfinite complete prospective forecast')
    return {'GPU_seconds':gpu,'CPU_seconds':cpu,'remaining_wall_seconds':cpu+gpu+300,'storage_upper_bytes':math.ceil(private),'phases':phases,'scope':{**SCOPE,'formal_updates':0 if formal_done else 24000,'engineering_updates':0 if fit_checks is not None else 72},'factors':{'synthetic_backends':2,'formal_update':1.5},'original_caps_and_deadline_unchanged':True,'actual_spend_is_separate_from_forecast':True}
