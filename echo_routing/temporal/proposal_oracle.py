"""Post-readout multi-only exact one-action domain diagnosis, never inference."""
from echo_routing.temporal.proposal_actions import propose,partition,apply
from echo_routing.temporal.anchor_edits import counts
from echo_routing.temporal.anchor_oracle import fractional_oracle
from echo_routing.temporal.segments import labels_to_segments
from echo_routing.evaluate.failure_units import match_segments


def actions(base,raw_a,raw_j,domain,guarded):
    if domain not in ['old','expanded']:raise ValueError('declared old/expanded oracle domain required')
    pool=[a for a in propose(base,raw_a,raw_j) if domain=='expanded' or a.old]
    return partition(base,pool)[0] if guarded else pool


def action_options(base,raw_a,raw_j,truth,domain,guarded=False):
    options=[(None,base)]+[(a.key,apply(base,a)) for a in actions(base,raw_a,raw_j,domain,guarded)]
    return [{'M':m,'P':p,'T':t,'span':key} for key,y in options for m,p,t in [counts(y,truth)]]


def solve(options,domain,guard):
    if domain not in ['old','expanded'] or guard not in ['U','G']:raise ValueError('prospective domain and guard required')
    return {**fractional_oracle(options),'domain':domain,'guard':guard,'scope':f'exact pooled multi IoU50 F1 over {domain} canonical complete-label one-action + no-op domain;label assisted;not deployable;not whole-family upper bound'}


def missed_coverage(base,raw_a,raw_j,truth,domain,guarded=False):
    true=labels_to_segments(truth);matched=set(match_segments(labels_to_segments(base),true));union=set()
    for action in actions(base,raw_a,raw_j,domain,guarded):union|=set(match_segments(labels_to_segments(apply(base,action)),true))-matched
    return {'baseline_missed_true_units':len(true)-len(matched),'candidate_union_recovered_true_units':len(union),'candidate_union_recovered_short10_true_units':sum(true[k].length<=10 for k in union),'candidate_union_recovered_short16_true_units':sum(true[k].length<=16 for k in union),'baseline_missed_short10_true_units':sum(s.length<=10 and k not in matched for k,s in enumerate(true)),'baseline_missed_short16_true_units':sum(s.length<=16 and k not in matched for k,s in enumerate(true)),'union_is_jointly_reachable_claim':False}
