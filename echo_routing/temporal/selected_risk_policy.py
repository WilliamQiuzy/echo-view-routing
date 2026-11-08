"""A63 fixed deployment top/old-N then strict C0/C1 veto; no fallback."""
import math
import numpy as np
from echo_routing.temporal.selected_risk_supply import top
from echo_routing.temporal.proposal_actions import partition,apply
ARMS=('C0','C1')

def decode(base,actions,c6,q,taus):
    eligible,rejected=partition(base,actions);c6=np.asarray(c6,np.float64);q=np.asarray(q,np.float64)
    if rejected or len(eligible)!=len(actions) or c6.shape!=(len(actions),6) or q.shape!=(len(actions),2) or not np.isfinite(c6).all() or not np.isfinite(q).all() or np.any((q<0)|(q>1)) or np.any(c6[:,1:4]<0) or np.any((c6[:,4:]<0)|(c6[:,4:]>1)):raise ValueError('complete canonical G/C6/unit-q original-axis rows required')
    if set(taus)!=set(ARMS) or any(isinstance(t,bool) or not isinstance(t,(float,int)) or not math.isfinite(t) or not 0<=t<=1 for t in taus.values()):raise ValueError('both frozen finite strict event thresholds required')
    selected=top([a.key for a in actions],c6[:,0],c6[:,1],c6[:,2],c6[:,4]);i=selected['top_index'];out={}
    for col,arm in enumerate(ARMS):
        score=None if i is None else float(q[i,col]);accepted=i if selected['N_admitted'] and score<taus[arm] else None
        out[arm]={**selected,'accepted_index':accepted,'accepted':None if accepted is None else actions[accepted],'prediction':np.asarray(base).copy() if accepted is None else apply(base,actions[accepted]),'gate_score':score,'tau':taus[arm]}
    return out

def calibrate(rows):
    admitted=[]
    for r in rows:
        q=r['q']
        if q is not None and (not isinstance(q,(float,int)) or isinstance(q,bool) or not math.isfinite(q) or not 0<=q<=1):raise ValueError('non-null q finite in unit range required')
        if r['N_admitted']:
            if q is None or r['H']<0:raise ValueError('N-admitted true top requires q and full-axis harm')
            admitted.append(r)
    bad=[r for r in admitted if r['H']>0 or r['dFP']>0];tau=min(r['q'] for r in bad) if bad else (1. if admitted else 0.)
    accepted=[r for r in admitted if r['q']<tau]
    if any(r['H']>0 or r['dFP']>0 for r in accepted):raise ValueError('first unsafe strict empirical calibration mismatch')
    return {'tau':float(tau),'total_axes':len(rows),'N_admitted':len(admitted),'unsafe_N_top':len(bad),'unsafe_tied_at_threshold':sum(r['q']==tau for r in bad),'accepted':len(accepted),'accepted_unsafe':0,'scope':'fixed exposed CAL original deployment N-top only; no independent or conformal safety claim'}

def production_phase(phase,coverage=None,aux_audit=None):
    if phase=='FIT':return {'roles':['F0_mixed','F1_mixed'],'payloads':6000,'aux_updates_at_production':0,'formal_updates_at_production':0}
    if phase!='CONFIRM' or coverage is None or aux_audit is None or coverage.get('passed') is not True or set(coverage.get('seeds',{}))!={'2026110461','2026110462'} or any(coverage['seeds'][s].get('passed') is not True for s in coverage['seeds']) or aux_audit.get('passed') is not True or (aux_audit.get('aux_optimizer_updates'),aux_audit.get('physical_models'),aux_audit.get('fullstate_count'))!=(48000,8,32):raise ValueError('both exact seed coverage plus complete48000/8/32 actual auxiliary audit precedes confirmation teacher forward')
    return {'roles':['a63_confirmation_selection','a63_confirmation_calibration'],'payloads':480,'aux_updates_at_production':48000,'formal_updates_at_production':0}
