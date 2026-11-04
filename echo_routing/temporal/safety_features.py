"""A61 observable changed-position47; exact qualified J109 candidate features."""
import numpy as np
from echo_routing.temporal.anchor_features import probabilities
from echo_routing.temporal.proposal_actions import apply
from echo_routing.temporal.proposal_features import candidate_features


def utility_features(prob_a,prob_j,base,action):return candidate_features(prob_a,prob_j,base,action,'J')


def position_features(prob_a,prob_j,base,action):
    a,j=probabilities(prob_a),probabilities(prob_j);edited=apply(base,action);base=np.asarray(base)
    if a.shape!=j.shape or len(a)!=len(base):raise ValueError('same complete probability/base axes required')
    changed=np.flatnonzero(edited!=base);T=len(base);length=action.end-action.start;K=len(changed);eye=np.eye(5)
    result=[]
    for t in changed:
        left,right=max(0,t-16),min(T,t+17)
        row=[*a[t],*j[t],*eye[base[t]],*eye[edited[t]]]
        for prob in [a,j]:
            row.extend(prob[left:t].mean(0) if t>left else np.zeros(5))
            row.extend(prob[t+1:right].mean(0) if right>t+1 else np.zeros(5))
        row.extend([int(t>left),int(right>t+1),*np.log1p([T,length,K]),(t-action.start+.5)/length,(t+.5)/T])
        result.append(row)
    return matrix(result)


def matrix(value):
    value=np.asarray(value,np.float64)
    if value.ndim!=2 or value.shape[1]!=47 or not len(value) or not np.isfinite(value).all():raise ValueError('nonempty finite N47 changed-position features required')
    return value


def fit_position_normalizer(value):
    x=matrix(value);return {'mean':x.mean(0),'std':np.maximum(x.std(0),1e-6)}


def normalize_positions(value,normalizer):
    x=matrix(value);m,s=np.asarray(normalizer['mean']),np.asarray(normalizer['std'])
    if m.shape!=(47,) or s.shape!=(47,) or not np.isfinite(m).all() or not np.isfinite(s).all() or (s<1e-6).any():raise ValueError('complete FIT47 normalizer required')
    return ((x-m)/s).astype(np.float32)
