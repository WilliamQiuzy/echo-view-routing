"""Prospective RLE-factorized cost model: measured geometry, no outcomes."""
import math
from echo_routing.temporal.safety_measured_budget import OUTER_ROLES


def observed(roles, report):
    if not report['passed'] or report['new_optimizer_updates'] or report['new_neural_forwards_in_this_cost_stage'] or report['A61_outer_head_outcomes_used'] or report['outer_GT_values_used']:
        raise ValueError('complete cost-only geometry required')
    g=report['observable_geometry_all10roles']
    if set(g)!=set(OUTER_ROLES):raise ValueError('all10roles required')
    fields=['N','T','C','C2','TC','all_C','all_TC','K','CR','max_C','max_T','native_N','native_T','native_T2']
    for r,v in g.items():
        if any(not isinstance(v[k],int) or v[k]<0 for k in fields):raise ValueError('integer nonnegative observable volumes required')
        pin=roles[r]
        if v['N']!=pin['sequences'] or v['T']!=pin['positions'] or v['input_sha256']!=pin['sha256'] or v['max_T']>pin['max_length']:raise ValueError('observable frozen-axis binding changed')
        N,T,C,M=v['N'],v['T'],v['C'],v['max_C']
        if not (C<=v['all_C']<=1.5*T+N and M<=v['max_T']+math.ceil(v['max_T']/2) and v['C2']<=C*M and v['TC']<=T*M and v['all_TC']<=1.5*T*v['max_T']+T and v['K']<=2*T and 2*C<=v['CR']<=2*v['TC'] and v['native_N']<=N and v['native_T']<=T and v['native_T2']<=v['native_T']*v['max_T']):raise ValueError('original grammar/position/run bounds violated')
    return g


def densities(t, report):
    low,high,kernel=report['low_RLE_dense'],report['retained_high_RLE_dense'],report['all_unequal_pair_kernel'];long=t['long_701']
    for row in [low,high,kernel,long,t['empty_descriptor'],t['dense_16']]:
        for k,v in row.items():
            if isinstance(v,(int,float)) and (not math.isfinite(v) or v<0):raise ValueError('finite nonnegative measured costs required')
    if min(low[k] for k in ['C','all_C','TC','all_TC'])<=0 or high['CR']<=0 or kernel['C2']<=0 or min(long['K'],long['T'])<=0:raise ValueError('nonzero cost-kernel denominators required')
    def descriptor(v):
        return 2*(t['empty_descriptor']['seconds']*v['N']+low['descriptor_max_seconds']*sum(v[k]/low[k] for k in ['C','all_C','TC','all_TC'])+kernel['seconds']*v['C2']/kernel['C2']+high['descriptor_max_seconds']*v['CR']/high['CR']+long['descriptor_max_seconds']*(v['K']/long['K']+v['T']/long['T']))
    def oracle(v):
        return 2*(t['dense_16']['oracle_max_seconds']*v['N']+low['oracle_max_seconds']*(v['C']/low['C']+v['TC']/low['TC'])+high['oracle_max_seconds']*v['CR']/high['CR']+long['oracle_max_seconds']*(v['K']/long['K']+v['T']/long['T']))
    return descriptor,oracle


def role_costs(roles,report):
    expected={*OUTER_ROLES,'inner_cal'}
    if not report['passed'] or report['new_optimizer_updates'] or report['new_NN_forwards'] or report['A61_outer_head_outcomes_used'] or report['actual_GT_values_used'] or set(report['roles'])!=expected:raise ValueError('complete synthetic role timing matrix required')
    for r,v in report['roles'].items():
        if v['N']!=roles[r]['sequences'] or v['T']!=roles[r]['positions'] or v['input_sha256']!=roles[r]['sha256'] or v['synthetic_truth_max_runs']!=8 or not v['synthetic_score_matrix_no_heads']:raise ValueError('whole-role frozen geometry/source binding changed')
        for k in ['descriptor_max_seconds','cache_write_seconds','cache_read_seconds','oracle_max_seconds','geometry_and_dummy_policy_seconds']:
            if not math.isfinite(v[k]) or v[k]<0:raise ValueError('finite nonnegative actual role timings required')
        for label in ['descriptor','oracle']:
            a=v['all6_'+label+'_seconds']
            if len(a)!=6 or any(not math.isfinite(x) or x<0 for x in a) or max(a)!=v[label+'_max_seconds']:raise ValueError('all6 actual paths and conservative max required')
    return report['roles']
