"""Descriptive, additive cached-error bins; no model or threshold selection."""
import numpy as np

def class_stats(confusion):
    c=np.asarray(confusion);out=[]
    for k in range(5):
        true,pred,tp=int(c[k].sum()),int(c[:,k].sum()),int(c[k,k]);out.append({'class':k,'true':true,'predicted':pred,'correct':tp,'precision':tp/pred if pred else 0.,'recall':tp/true if true else 0.,'f1':2*tp/(true+pred) if true+pred else 0.})
    return out

def correctness_cells(candidate,A08):
    a,b=np.asarray(candidate,bool),np.asarray(A08,bool)
    if a.shape!=b.shape:raise ValueError('paired correctness arrays must share the complete axis')
    return {'both_correct':int((a&b).sum()),'candidate_only_correct':int((a&~b).sum()),'A08_only_correct':int((~a&b).sum()),'both_wrong':int((~a&~b).sum()),'denominator':int(a.size)}

def position_diagnostics(labels,predictions,probabilities):
    g,p,z=np.asarray(labels,int),np.asarray(predictions,int),np.asarray(probabilities,float)
    if g.ndim!=1 or p.shape!=g.shape or z.shape!=(len(g),5) or not np.isfinite(z).all() or (z<0).any() or not np.allclose(z.sum(1),1,atol=1e-5):raise ValueError('complete finite normalized original-position cache required')
    change=np.flatnonzero(np.diff(g))+1;near=(np.min(abs(np.arange(len(g))[:,None]-change),axis=1)<=5) if len(change) else np.zeros(len(g),bool);truth=z[np.arange(len(g)),g];other=z.copy();other[np.arange(len(g)),g]=-np.inf;margin=truth-other.max(1);top=np.sort(z,axis=1);confidence=top[:,-1]-top[:,-2];out={}
    for name,mask in [('near_true_change_5samples',near),('stable_interior',~near)]:
        cm=np.zeros((5,5),np.int64);np.add.at(cm,(g[mask],p[mask]),1);out[name]={'positions':int(mask.sum()),'errors':int((g[mask]!=p[mask]).sum()),'raw_true_class_probability_sum':float(truth[mask].sum()),'raw_true_minus_other_margin_sum':float(margin[mask].sum()),'raw_top2_confidence_sum':float(confidence[mask].sum()),'confusion':cm.tolist()}
    cuts=np.flatnonzero(np.diff(p))+1;out.update({'true_changes':len(change),'predicted_cuts':len(cuts),'predicted_cuts_near_true_change':int(near[cuts].sum()),'predicted_cuts_far_from_true_change':int((~near[cuts]).sum()),'predicted_cuts_on_exact_equal_label_adjacency':int((g[cuts]==g[cuts-1]).sum())});return out
