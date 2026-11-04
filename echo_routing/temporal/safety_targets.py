"""A61 full-axis gross effects; supervision never enters proposal/inference."""
import numpy as np
from echo_routing.temporal.anchor_edits import aligned,counts
from echo_routing.temporal.proposal_actions import apply
from echo_routing.temporal.proposal_features import targets as original_targets


def position_labels(base,action,truth):
    base,truth=aligned(base,truth);edited=apply(base,action);changed=edited!=base
    return np.where(base[changed]==truth[changed],0,np.where(edited[changed]==truth[changed],1,2)).astype(np.int64)


def effects(base,action,truth,fit_f1):
    base,truth=aligned(base,truth);edited=apply(base,action)
    u,de,dfp=original_targets(base,action,truth,fit_f1);before=counts(base,truth);after=counts(edited,truth)
    h=int(np.count_nonzero((base==truth)&(edited!=truth)))
    repair=int(np.count_nonzero((base!=truth)&(edited==truth)))
    if h-repair!=de:raise ValueError('gross/net complete-axis cost mismatch')
    safe=h==0 and dfp<=0
    return {'u':float(u),'H':h,'R':repair,'dE':h-repair,'dFP':int(dfp),
            'safe':bool(safe),'v':float(u if safe else -1-abs(u)),
            'delta_M':after[0]-before[0],'delta_P':after[1]-before[1]}
