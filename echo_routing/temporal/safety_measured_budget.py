"""Measured component forecast; scientific work/caps and old estimate unchanged."""
import math
from echo_routing.temporal.safety_budget import complete,estimated_inference

HALVES=['selection','calibration']
OUTER_ROLES=[h for h in HALVES]+[p+h for p in ['old_stress_','confirmation_','a60_confirmation_','a61_confirmation_'] for h in HALVES]
EXPECTED_ROLES=['fit','inner_cal',*OUTER_ROLES]


def add(a,b):return {k:a[k]+b[k] for k in a}


def future(pin):
    n=int(pin['sequences']);T=int(pin['max_length']);P=int(pin['positions']);C=1.5*P+n;maxC=T+math.ceil(T/2)
    return {'N':n,'T':P,'C':C,'C2':C*maxC,'TC':1.5*P*T+P,'all_C':C,'all_TC':1.5*P*T+P,'K':2*P,'native_N':0,'native_T':0,'native_T2':0}


def complete_measured(roles,checks,costs,measurement,retained_bytes,formal_pending=True):
    if set(roles)!=set(EXPECTED_ROLES) or not measurement['passed'] or measurement['new_optimizer_updates']!=0 or measurement['new_neural_forwards']!=0 or measurement['A61_outer_head_outcomes_used'] or measurement['outer_label_values_used']:raise ValueError('complete cost-only measurement with unchanged full role matrix required')
    t=measurement['timings'];required=['task_16','task_701','dense_16','dense_701','long_701','native_16','native_701','empty_descriptor','probability_IO','integer_solver','trace_utility','trace_risk','state40','six_figures']
    if not set(required)<=set(t):raise ValueError('all fixed evaluator/cache/descriptor/oracle/audit/plot component measurements required')
    for name in required:
        for key,value in t[name].items():
            if key.endswith('seconds') and (not isinstance(value,(float,int)) or not math.isfinite(value) or value<0):raise ValueError('finite nonnegative actual component time required')
    g={r:measurement['old_observable_geometry'][r] if not r.startswith('a61_confirmation_') else future(roles[r]) for r in OUTER_ROLES}
    for h in HALVES:
        bench=t['task_701'];pairs=g[h]['N']-g[h]['native_N']-120
        if g[h]['native_N']>bench['native_sessions'] or pairs>bench['pairs'] or 120>bench['multi'] or pairs<0:raise ValueError('synthetic5000 bootstrap geometry fails full task unit bound')
    max_candidates=max([g[r]['C'] for r in OUTER_ROLES if not r.startswith('a61_confirmation_')]+[roles[r]['max_length']+math.ceil(roles[r]['max_length']/2) for r in OUTER_ROLES if r.startswith('a61_confirmation_')])
    if t['integer_solver']['streams']<120 or t['integer_solver']['options_each']<1+max_candidates:raise ValueError('integer whole-bank solver benchmark fails declared candidate bound')
    base=complete(roles,checks,costs,retained_bytes,formal_pending);dense=t['dense_701'];long=t['long_701'];short=t['task_16'];large=t['task_701']
    def evaluation(v):return 2*(short['evaluation_prepare_max_seconds']*v['N']/short['N']+large['evaluation_prepare_max_seconds']*v['T']/(large['N']*large['T']))
    def descriptor(v):return 2*(t['empty_descriptor']['seconds']*v['N']+dense['descriptor_max_seconds']*(v['C']/dense['C']+v['all_C']/dense['all_C']+v['C2']/dense['C2']+v['TC']/dense['TC']+v['all_TC']/dense['all_TC'])+long['descriptor_max_seconds']*(v['K']/long['K']+v['T']/long['T']))
    def IO(v):
        ds=dense['cache_write_seconds']+dense['cache_read_seconds'];ls=long['cache_write_seconds']+long['cache_read_seconds'];small=t['dense_16']['cache_write_seconds']+t['dense_16']['cache_read_seconds']
        return 2*(small*v['N']+ds*(v['C']/dense['C']+v['TC']/dense['TC'])+ls*(v['T']/long['T']+v['K']/long['K']))
    def inference(v):return estimated_inference({'sequences':v['N'],'T_sum':v['T'],'C_sum':v['C'],'TC_sum':v['TC'],'K_sum':v['K']},costs)
    def oracle(v):return 2*(t['dense_16']['oracle_max_seconds']*v['N']+dense['oracle_max_seconds']*(v['C']/dense['C']+v['TC']/dense['TC'])+long['oracle_max_seconds']*(v['K']/long['K']+v['T']/long['T']))
    contexts=[]
    for r in OUTER_ROLES:
        half='selection' if r.endswith('selection') else 'calibration';contexts.append((r,g[r] if r in HALVES else add(g[half],g[r])))
    metrics=3*sum((33 if r in HALVES else 31)*evaluation(v) for r,v in contexts)
    descriptors=4*28*sum(descriptor(v) for r,v in contexts)
    heads=8*sum(inference(v) for v in g.values())+2*sum(inference(g[r]) for r in OUTER_ROLES if r.startswith('a61_confirmation_'))
    caches=(4*8+2*2)*sum(IO(v) for v in g.values())
    probability=t['probability_IO'];p_rate=(probability['write_seconds']+probability['read_seconds'])/(probability['N']*probability['T']);source_read=2*p_rate*(8*sum(v['T'] for v in g.values())+6*sum(g[h]['T'] for h in HALVES))
    inner=costs['role_geometry']['inner_cal'];ig={'N':inner['sequences'],'T':inner['T_sum'],'C':inner['C_sum'],'TC':inner['TC_sum'],'K':inner['K_sum']}
    # Full inference upper also covers cached calibration/redecode without NN.
    inner_work=(64*inference(ig)+112*evaluation(ig)+24*IO(ig)) if formal_pending else 0.
    native=0.
    for h in HALVES:
        v=g[h];native+=28*2*(t['native_16']['seconds']*v['native_N']+t['native_701']['seconds']*(v['native_T']/701+v['native_T2']/701**2))
    bootstrap=240*2*max(t['task_16']['paired5000_seconds'],t['task_701']['paired5000_seconds'])+60
    oracles=6*sum(oracle(v) for v in g.values())+60*2*t['integer_solver']['seconds']
    audits=3*(8*t['trace_utility']['generation_IO_and_audit_seconds']+2*t['trace_risk']['generation_IO_and_audit_seconds'])+2*t['state40']['seconds']
    package=3*t['six_figures']['seconds']+300
    phases={'risk_and_utility_training_including_durable_IO':base['phase_CPU_forecasts_seconds']['risk_and_utility_training_including_durable_IO'],'all24inner_8calibration_and_full_inner_audit':inner_work,'all314_metrics_and_preparation':metrics,'all280_policy_historical_fullaxis_descriptors_and_recount':descriptors,'all80joint_20historical_cache_IO_source_IO_and_inference':heads+caches+source_read,'all240paired5000':bootstrap,'all60oracle_16reproduction':oracles,'all56native_cubes':native,'all10model_fullstate_and60000trace_audits_curve_IO':audits,'six_figures_tables_final_review_CPU_room':package,'fresh240_dualproducer_CPU':base['phase_CPU_forecasts_seconds']['fresh240_dualproducer_CPU']}
    base.update(remaining_CPU_seconds=sum(phases.values()),remaining_serial_wall_seconds=sum(phases.values())+base['remaining_GPU_seconds']+1200,phase_CPU_forecasts_seconds=phases,whole_workflow_closure_integration_wall_room_seconds=1200,superseded_original_phase_forecast_not_deleted=True,all_original_fresh_candidate_geometry_bounds_preserved=True,basis='actual new synthetic full-component timings;nonnegative N/T/C/C2/TC/K volume scaling with >=2 factors;fixed training×1.5/source×2 unchanged;all endpoints intact;forecast not a proven runtime bound and not spent charges')
    return base
