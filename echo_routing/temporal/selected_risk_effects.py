"""A63 private selected-event extraction after immutable label-free score selection.

Uses the qualified original full-axis matcher and original membership F_fit.
The returned source IDs, canonical IDs and signatures must remain server-only.
"""
import numpy as np
from echo_routing.temporal.safety_targets import effects
from echo_routing.temporal.proposal_actions import apply
from echo_routing.temporal.anchor_edits import majority
from echo_routing.temporal.segments import labels_to_segments
from echo_routing.evaluate.failure_units import match_segments
from echo_routing.temporal.selected_risk_supply import digest,draw_signature,event_signature
F_FIT=.8917446116307443


def unmatched_short(prediction,truth):
    pred=labels_to_segments(prediction);true=labels_to_segments(truth);matched=set(match_segments(pred,true,.5).values())
    return {(s.start,s.end,s.label) for i,s in enumerate(pred) if s.length<=16 and i not in matched}

def record(sequence,base,action,context,*,D,X):
    if not isinstance(D,(bool,np.bool_)) or not isinstance(X,(bool,np.bool_)):raise ValueError('immutable selector flags required')
    ctx=np.asarray(context,np.float64)
    if ctx.shape!=(96,) or not np.isfinite(ctx).all() or not 0<=ctx[91]<=1:raise ValueError('actual frozen label-free fullrun context96 required')
    truth=np.asarray(sequence['labels']);edited=apply(base,action);cost=effects(base,action,truth,F_FIT);unsafe=cost['H']>0 or cost['dFP']>0
    oldshort=unmatched_short(base,truth);newshort=unmatched_short(edited,truth)
    if len(newshort)-len(oldshort)!=cost['dFP']:raise ValueError('whole-axis shortFP effect mismatch')
    newfp=sorted(newshort-oldshort) if cost['dFP']>0 else []
    event=event_signature(sequence['refs'],base,edited,truth,new_fp_segments=newfp) if unsafe else None
    true=labels_to_segments(truth);before=set(match_segments(labels_to_segments(base),true));after=set(match_segments(labels_to_segments(edited),true));recovered=after-before;lost=before-after
    shortrecovered=sum(true[i].length<=16 for i in recovered);shortlost=sum(true[i].length<=16 for i in lost)
    k=int(np.count_nonzero(edited!=np.asarray(base)))
    if event is not None and event['H']!=cost['H']:raise ValueError('event H source tuples do not match gross effect')
    return {'canonical_id':digest([sequence['id'],action.key]),'kind':sequence['kind'],'D':bool(D),'X':bool(X),**cost,'unsafe':unsafe,'changed_K':k,'neutral':k-cost['H']-cost['R'],'short16_lost':int(shortlost),'short16_recovered':int(shortrecovered),'Jexclusive_short16_recovered':int(shortrecovered if action.jepa and not action.old else 0),'event_signature':None if event is None else event['signature'],'draw_signature':draw_signature(sequence['refs'],base,edited),'responsible_sessions':[] if event is None else event['responsible_sessions'],'harm_transitions':[] if event is None else event['harm_transitions'],'new_FP16_segments':len(newfp),'min_overlap':float(ctx[91]),'replacement_class':majority(np.asarray(action.replacement)),'old':action.old,'jepa':action.jepa}
