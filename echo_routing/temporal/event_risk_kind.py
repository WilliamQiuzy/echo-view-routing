"""Explicit A59/A61 'pairs' schema adapter; not a decoder or metadata rewrite."""
from echo_routing.temporal.event_risk_metrics import support as _support


def canonical_kind(kind):
    if kind not in ['native','multi','pair','pairs']:raise ValueError('unqualified sequence kind; only pinned legacy pairs alias accepted')
    return 'pair' if kind=='pairs' else kind


def support(sequences,baselines,values,safety,fit_f1,private_details=False):
    # The frozen evaluator already groups this exact legacy type as pair.
    # Keep original IDs/labels/axis/objects and all inference inputs untouched.
    canonical=[q if canonical_kind(q['kind'])==q['kind'] else {**q,'kind':'pair'} for q in sequences]
    result=_support(canonical,baselines,values,safety,fit_f1,private_details)
    result['legacy_kind_alias']={'pairs':'pair'};result['original_metadata_kind_not_modified']=True
    return result
