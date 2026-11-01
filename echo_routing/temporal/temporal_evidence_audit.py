"""Post-A52 cache-only description and new dense-supervision qualification."""
import numpy as np


def singleton_view(records,ordinal,pts):
    found=[]
    for axis in (ordinal,pts):
        for origin in (0,1):
            for inclusive in (0,1):
                labels={r['true_label'] for r in records if r['start']-origin<=axis<r['end']-origin+inclusive}
                if len(labels)!=1:return None
                found.append(next(iter(labels)))
    if len(set(found))!=1 or found[0] not in ('A4C','A5C'):return None
    return int(found[0]=='A5C')


def informative(labels,physical_refs,marker):
    if len(labels)!=8 or len(physical_refs)!=8 or marker not in (0,1):raise ValueError('fixed eight slots and binary marker required')
    known=[x for x in labels if x is not None]
    distinct={y:{ref for label,ref in zip(labels,physical_refs) if label==y} for y in (0,1)}
    return len(known)>=4 and sum(x!=marker for x in known)>=2 and all(len(distinct[y])>=2 for y in (0,1))


def auc_partition(y,scores,groups):
    y=np.asarray(y,bool);s=np.asarray(scores,float);g=np.asarray(groups)
    if y.ndim!=1 or s.shape!=y.shape or g.shape!=y.shape or not np.isfinite(s).all():raise ValueError('aligned finite vectors required')
    p,n=np.flatnonzero(y),np.flatnonzero(~y);wins=(s[p,None]>s[None,n])+.5*(s[p,None]==s[None,n]);same=g[p,None]==g[None,n]
    total=int(wins.size);wn=int(same.sum());bn=total-wn;w=float(wins[same].sum());b=float(wins[~same].sum())
    per=[];unipolar=0
    for k in set(g.tolist()):
        pos=s[(g==k)&y];neg=s[(g==k)&~y]
        if not len(pos) or not len(neg):unipolar+=1;continue
        per.append(float(((pos[:,None]>neg[None,:])+.5*(pos[:,None]==neg[None,:])).mean()))
    return {'global_AUC':float(wins.mean()) if total else None,'total_pairs':total,
      'within_AUC':w/wn if wn else None,'within_pairs':wn,'between_AUC':b/bn if bn else None,'between_pairs':bn,
      'reconstructed_AUC':(w+b)/total if total else None,'equal_group_within_AUC':float(np.mean(per)) if per else None,
      'within_eligible_groups':len(per),'single_class_groups':unipolar,'groups_total':len(set(g.tolist()))}


def dense_gate(counts):
    minima={'fit':{'requests':80,'patients':8,'videos':8,'marker_classes':2},'diagnostic':{'requests':20,'patients':2,'videos':2,'marker_classes':2}}
    checks={role:{key:counts[role][key]>=value for key,value in m.items()} for role,m in minima.items()}
    return {'passed':all(ok for d in checks.values() for ok in d.values()),'checks':checks,'minimums':minima}
