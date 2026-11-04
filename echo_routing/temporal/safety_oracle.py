"""A61 post-freeze single-action G ceilings with true net/gross safety constraints."""
from echo_routing.temporal.proposal_oracle import actions
from echo_routing.temporal.proposal_actions import apply
from echo_routing.temporal.anchor_edits import counts
from echo_routing.temporal.anchor_oracle import fractional_oracle
from echo_routing.temporal.safety_targets import effects
from echo_routing.temporal.segments import labels_to_segments
from echo_routing.evaluate.failure_units import match_segments

SAFETIES=['unrestricted','net-safe','gross-safe']


def allowed(base,raw_a,raw_j,truth,domain,safety):
    if safety not in SAFETIES:raise ValueError('prespecified oracle safety domain required')
    result=[]
    for a in actions(base,raw_a,raw_j,domain,True):
        if safety=='unrestricted':result.append(a);continue
        value=effects(base,a,truth,0.)
        if value['dFP']<=0 and (value['dE']<=0 if safety=='net-safe' else value['H']==0):result.append(a)
    return result


def options(base,raw_a,raw_j,truth,domain,safety):
    items=[(None,base)]+[(a.key,apply(base,a)) for a in allowed(base,raw_a,raw_j,truth,domain,safety)]
    return [{'M':m,'P':p,'T':t,'span':key} for key,pred in items for m,p,t in [counts(pred,truth)]]


def solve(pools,domain,safety):
    if domain not in ['old','expanded'] or safety not in SAFETIES:raise ValueError('fixed oracle domain/safety required')
    return {**fractional_oracle(pools),'domain':domain,'guard':'G','safety':safety,'scope':'exact pooled multi IoU50 F1 maxONEaction+noop;true net-safe=dE<=0/dFP<=0,gross-safe=H0/dFP<=0;label-assisted notdeployable/notuniversalmodelceiling'}


def coverage(base,raw_a,raw_j,truth,domain,safety):
    true=labels_to_segments(truth);matched=set(match_segments(labels_to_segments(base),true));recovered=set()
    pool=allowed(base,raw_a,raw_j,truth,domain,safety)
    for a in pool:recovered|=set(match_segments(labels_to_segments(apply(base,a)),true))-matched
    return {'baseline_missed_true_units':len(true)-len(matched),'candidate_union_recovered_true_units':len(recovered),'candidate_union_recovered_short10_true_units':sum(true[k].length<=10 for k in recovered),'candidate_union_recovered_short16_true_units':sum(true[k].length<=16 for k in recovered),'baseline_missed_short10_true_units':sum(s.length<=10 and k not in matched for k,s in enumerate(true)),'baseline_missed_short16_true_units':sum(s.length<=16 and k not in matched for k,s in enumerate(true)),'eligible_actions':len(pool),'union_is_jointly_reachable_claim':False}
