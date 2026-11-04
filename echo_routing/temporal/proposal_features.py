"""A60 29 common+40+40 observables and separate full-axis supervised targets."""
import numpy as np
from echo_routing.temporal.anchor_features import probabilities,regions,source_features
from echo_routing.temporal.anchor_edits import aligned,counts,predicted_count
from echo_routing.temporal.proposal_actions import apply
from echo_routing.temporal.segments import labels_to_segments
from echo_routing.evaluate.failure_units import match_segments


def evidence(prob,base,action):
    prob=probabilities(prob);edited=apply(base,action);changed=np.flatnonzero(edited!=base)
    if len(prob)!=len(base) or not len(changed):raise ValueError('matched nonempty complete changed-position evidence required')
    proposed=prob[changed,edited[changed]];original=prob[changed,np.asarray(base)[changed]]
    diff=proposed-original
    logratio=np.log(np.maximum(proposed,1e-8))-np.log(np.maximum(original,1e-8))
    return np.r_[source_features(prob,(action.start,action.end)),diff.mean(),diff.min(),diff.max(),logratio.mean()]


def candidate_features(prob_a,prob_j,base,action,arm):
    a=probabilities(prob_a);j=probabilities(prob_j)
    if a.shape!=j.shape or arm not in ['A','J'] or len(a)!=len(base):raise ValueError('matched original axes and A/J arm required')
    edited=apply(base,action);base=np.asarray(base);s,e=action.start,action.end;onehot=np.eye(5)
    k=int(np.count_nonzero(edited!=base));bounds=regions(len(base),(s,e))
    common=[*np.log1p([right-left for left,right in bounds]),*onehot[base[s:e]].mean(0),
        *onehot[edited[s:e]].mean(0),*(onehot[base[s-1]] if s else np.zeros(5)),
        *(onehot[base[e]] if e<len(base) else np.zeros(5)),int(s>0),int(e<len(base)),
        predicted_count(edited)-predicted_count(base),np.log1p(k),k/(e-s),k/len(base)]
    block=evidence(a,base,action)
    result=np.r_[common,block,block if arm=='A' else evidence(j,base,action)].astype(np.float64)
    if result.shape!=(109,) or not np.isfinite(result).all():raise ValueError('fixed109 finite observables required')
    return result


def short_fp(prediction,truth):
    prediction,truth=aligned(prediction,truth);pred=labels_to_segments(prediction);true=labels_to_segments(truth)
    used=set(match_segments(pred,true,.5).values())
    return sum(segment.length<=16 and index not in used for index,segment in enumerate(pred))


def targets(base,action,truth,fit_f1):
    base,truth=aligned(base,truth)
    if not np.isfinite(fit_f1) or not 0<=fit_f1<=1:raise ValueError('frozen FIT pooled operating point required')
    edited=apply(base,action);before=counts(base,truth);after=counts(edited,truth)
    return np.asarray([2*(after[0]-before[0])-fit_f1*(after[1]-before[1]),
       int(np.count_nonzero(edited!=truth))-int(np.count_nonzero(base!=truth)),
       short_fp(edited,truth)-short_fp(base,truth)],np.float64)


def feature_matrix(value):
    value=np.asarray(value,dtype=np.float64)
    if value.ndim!=2 or value.shape[1]!=109 or not len(value) or not np.isfinite(value).all():raise ValueError('nonempty finite FIT N109 matrix required')
    return value


def fit_normalizer(fit_features):
    value=feature_matrix(fit_features)
    return {'mean':value.mean(0),'std':np.maximum(value.std(0),1e-6)}


def normalize(features,normalizer):
    value=feature_matrix(features);mean=np.asarray(normalizer['mean']);std=np.asarray(normalizer['std'])
    if mean.shape!=(109,) or std.shape!=(109,) or not np.isfinite(mean).all() or not np.isfinite(std).all() or (std<1e-6).any():raise ValueError('frozen finite FIT109 normalizer required')
    return ((value-mean)/std).astype(np.float32)


def target_matrix(value):
    value=np.asarray(value,dtype=np.float64)
    if value.ndim!=2 or value.shape[1]!=3 or not len(value) or not np.isfinite(value).all():raise ValueError('nonempty finite complete FIT N3 targets required')
    return value


def fit_target_scales(fit_targets):return np.maximum(np.sqrt(np.mean(target_matrix(fit_targets)**2,axis=0)),1.)


def checked_scales(scales):
    scales=np.asarray(scales,dtype=np.float64)
    if scales.shape!=(3,) or not np.isfinite(scales).all() or (scales<1).any():raise ValueError('three shared uncentered FIT RMS scales floor1 required')
    return scales


def scale_targets(values,scales):return (target_matrix(values)/checked_scales(scales)).astype(np.float32)


def physical_outputs(values,scales):return target_matrix(values)*checked_scales(scales)
