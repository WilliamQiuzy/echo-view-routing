"""A61 common top-first action, empirical veto and calibration; no fallback."""
import math
import numpy as np
from echo_routing.temporal.anchor_edits import labels
from echo_routing.temporal.proposal_actions import apply,partition


def checked_risk(row):
    names=['predicted_H','predicted_R','pFP','r']
    if any(k not in row or not math.isfinite(row[k]) for k in names) or min(row['predicted_H'],row['predicted_R'])<0 or not 0<=row['pFP']<=1 or not 0<=row['r']<=1:raise ValueError('finite nonnegative gross risks and unit-range evidence required')
    return row['predicted_H']-row['predicted_R']<=0 and row['pFP']<.5


def decode(base,actions,utility,risks,policy,tau=None):
    base=labels(base);eligible,rejected=partition(base,actions);utility=np.asarray(utility,np.float64)
    if policy not in ['E','N','C'] or rejected or len(eligible)!=len(actions) or utility.shape!=(len(actions),) or not np.isfinite(utility).all() or len(risks)!=len(actions):raise ValueError('complete common G domain/finite signed utility/risk rows required')
    if policy=='C' and (tau is None or not math.isfinite(tau) or not 0<=tau<=1):raise ValueError('frozen finite unit-range calibration threshold required')
    n_flags=[checked_risk(row) for row in risks]
    positive=[i for i,v in enumerate(utility) if v>0]
    index=min(positive,key=lambda i:(-utility[i],actions[i].key)) if positive else None
    n_admitted=index is not None and n_flags[index]
    accepted=index if index is not None and (policy=='E' or n_admitted and (policy=='N' or risks[index]['r']<tau)) else None
    return {'prediction':base.copy() if accepted is None else apply(base,actions[accepted]),'accepted':None if accepted is None else actions[accepted],
            'accepted_index':accepted,'top_index':index,'N_admitted':bool(n_admitted),'r':None if index is None else float(risks[index]['r']),
            'positive_utility_count':len(positive),'top_vetoed':index is not None and accepted is None}


def calibrate(rows):
    admitted=[r for r in rows if r['N_admitted']]
    if any(not math.isfinite(r['r']) or not 0<=r['r']<=1 or r['H']<0 for r in admitted):raise ValueError('finite full-axis inner top effects/risk required')
    unsafe=[r for r in admitted if r['H']>0 or r['dFP']>0]
    tau=min(r['r'] for r in unsafe) if unsafe else (1. if admitted else 0.)
    accepts=[r for r in admitted if r['r']<tau]
    bad=sum(r['H']>0 or r['dFP']>0 for r in accepts)
    if bad:raise ValueError('strict first-unsafe calibration failed')
    return {'tau':float(tau),'inner_sequences':len(rows),'N_admitted':len(admitted),'unsafe_N_tops':len(unsafe),'unsafe_tied_at_threshold':sum(r['r']==tau for r in unsafe),
            'C_accepted':len(accepts),'C_unsafe':int(bad),'all_noop_C':not accepts,'scope':'all928 inner diagnostic/calibration;not independent/exchangeable/conformal safety evidence'}
