"""Explicit raw-class grouping for original-native baseline endpoints."""
import numpy as np


def grouped_probabilities(raw, mapping):
    p=np.asarray(raw,float)
    if p.ndim!=2 or not np.isfinite(p).all(): raise ValueError('finite raw probability matrix required')
    out=np.zeros((len(p),6),float)
    for j in range(p.shape[1]): out[:,mapping.get(j,5)]+=p[:,j]
    return out


def stfm_probabilities(logits, prior_once=False):
    x=np.asarray(logits,float)
    evidence=np.where(x>20,x,np.logaddexp(0,x))
    alpha=evidence.sum(0)+(.8 if prior_once else .8*len(x))
    return alpha/alpha.sum()
