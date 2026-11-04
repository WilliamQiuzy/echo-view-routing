"""Fixed A61 top/N; matched CAL-only event gates without fallback."""
import math
import numpy as np
from echo_routing.temporal.safety_policy import decode as old_decode
from echo_routing.temporal.proposal_actions import apply

POLICIES=('E','N','L0','M0','R1','R2')


def decode(base,actions,old_scores,q1,q2,taus):
    scores=np.asarray(old_scores,np.float64);q1=np.asarray(q1,np.float64);q2=np.asarray(q2,np.float64);n=len(actions)
    if scores.shape!=(n,6) or q1.shape!=(n,) or q2.shape!=(n,) or not all(np.isfinite(v).all() for v in [scores,q1,q2]) or any((v<0).any() or (v>1).any() for v in [q1,q2]):raise ValueError('aligned finite originalC6 and new two unit-range event vectors required')
    if set(taus)!=set(POLICIES[2:]) or any(not math.isfinite(x) or not 0<=x<=1 for x in taus.values()):raise ValueError('four exact frozen C thresholds required')
    risks=[dict(zip(['predicted_H','predicted_R','predicted_neutral','pFP','r'],r[1:])) for r in scores]
    out={p:old_decode(base,actions,scores[:,0],risks,p) for p in ['E','N']};top=out['E']['top_index'];admitted=out['N']['accepted_index'] is not None
    for p,value in [('L0',scores[:,5]),('M0',scores[:,5]),('R1',q1),('R2',q2)]:
        accepted=top if admitted and value[top]<taus[p] else None
        out[p]={**out['N'],'prediction':np.asarray(base).copy() if accepted is None else apply(base,actions[accepted]),'accepted':None if accepted is None else actions[accepted],'accepted_index':accepted,'top_vetoed':top is not None and accepted is None,'gate_score':None if top is None else float(value[top]),'tau':float(taus[p])}
    return out


def calibrate(rows,population='CAL'):
    if population!='CAL':raise ValueError('AUDIT/outer cannot fit thresholds')
    for r in rows:
        if not isinstance(r['N'],(bool,np.bool_)) or not isinstance(r['unsafe'],(bool,np.bool_)) or any(not math.isfinite(r[k]) or not 0<=r[k]<=1 for k in ['r','q1','q2']):raise ValueError('complete finite CAL top events required')
    group=[r for r in rows if r['N']];bad=[r for r in group if r['unsafe']];out={'total_axes':len(rows),'N_admitted':len(group),'unsafe_N_top':len(bad),'calibration_limited':len(group)<20 or len(bad)<5,'population':'same frozen original N-top set for all three gates','independent_safety_guarantee':False}
    for name,key in [('M0','r'),('R1','q1'),('R2','q2')]:
        tau=min(r[key] for r in bad) if bad else (1. if group else 0.);accepted=[r for r in group if r[key]<tau]
        if any(r['unsafe'] for r in accepted):raise ValueError('strict first-unsafe CAL property failed')
        out[name]={'tau':float(tau),'accepted':len(accepted),'unsafe_accepted':0,'threshold_unsafe_ties':sum(r[key]==tau for r in bad),'all_noop':not accepted}
    return out
