"""Observable proposal computational volumes; no reference labels or scores."""
import time,numpy as np
from echo_routing.temporal.proposal_actions import propose,partition,apply


def runs(y):return 1+int(np.count_nonzero(y[1:]!=y[:-1]))


def geometry(base,raw_a,raw_j):
    all_actions=propose(base,raw_a,raw_j);actions,_=partition(base,all_actions);T=len(base);C=len(actions);edited=[apply(base,x) for x in actions]
    return {'N':1,'T':T,'C':C,'C2':C*C,'TC':T*C,'all_C':len(all_actions),'all_TC':T*len(all_actions),'K':sum(int(np.count_nonzero(v!=base)) for v in edited),'CR':sum(runs(base)+runs(v) for v in edited),'max_C':C,'max_T':T}


def pair_kernel(C):
    began=time.perf_counter();actual=np.r_[np.arange(C,dtype=np.float64)+1,0.];pred=np.r_[np.sin(np.arange(C)),0.];i,j=np.triu_indices(C+1,1);keep=actual[i]!=actual[j];i,j=i[keep],j[keep];np.count_nonzero(np.sign(actual[i]-actual[j])==np.sign(pred[i]-pred[j]));np.count_nonzero(np.sign(pred[i]-pred[j])==0)
    return {'C':C,'C2':C*C,'informative_pair_operations':len(i),'seconds':time.perf_counter()-began,'all_unequal_targets_synthetic':True}
