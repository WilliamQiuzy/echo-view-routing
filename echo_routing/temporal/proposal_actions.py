"""A60 complete-axis old/raw and JEPA-run actions; inference has no truth inputs."""
from dataclasses import dataclass
import numpy as np
from echo_routing.temporal.anchor_edits import labels,aligned,components,majority


@dataclass(frozen=True)
class Action:
    start:int
    end:int
    replacement:tuple
    old:bool=False
    jepa:bool=False

    @property
    def key(self):return (self.start,self.end,self.replacement)


def canonical(base,start,end,replacement,old=False,jepa=False):
    base=labels(base);replacement=labels(np.asarray(replacement))
    if isinstance(start,bool) or isinstance(end,bool) or int(start)!=start or int(end)!=end or not 0<=start<end<=len(base) or len(replacement)!=end-start:
        raise ValueError('complete aligned nonempty action interval required')
    changed=np.flatnonzero(base[start:end]!=replacement)
    if not len(changed):return None
    left,right=int(changed[0]),int(changed[-1])+1
    return Action(int(start)+left,int(start)+right,tuple(map(int,replacement[left:right])),bool(old),bool(jepa))


def apply(base,action):
    base=labels(base)
    if not isinstance(action,Action):raise ValueError('canonical immutable action required')
    normalized=canonical(base,action.start,action.end,action.replacement,action.old,action.jepa)
    if normalized is None or normalized.key!=action.key:raise ValueError('action must be nonempty and trimmed at unchanged ends')
    out=base.copy();out[action.start:action.end]=action.replacement
    return out


def propose(base,raw_a,raw_j):
    base,raw_a=aligned(base,raw_a);base,raw_j=aligned(base,raw_j);pool={}
    def add(action):
        if action is None:return
        prior=pool.get(action.key)
        pool[action.key]=action if prior is None else Action(action.start,action.end,action.replacement,prior.old or action.old,prior.jepa or action.jepa)
    for s,e in components(base,raw_a):add(canonical(base,s,e,raw_a[s:e],old=True))
    bounds=np.r_[0,np.flatnonzero(raw_j[1:]!=raw_j[:-1])+1,len(raw_j)]
    for s,e in zip(bounds[:-1],bounds[1:]):add(canonical(base,int(s),int(e),raw_j[s:e],jepa=True))
    return [pool[key] for key in sorted(pool)]


def partition(base,actions):
    current=majority(base);eligible,rejected=[],[]
    if [a.key for a in actions]!=sorted({a.key for a in actions}):raise ValueError('unique canonical action order required')
    for action in actions:
        (eligible if majority(apply(base,action))==current else rejected).append(action)
    return eligible,rejected


def decode(base,actions,outputs,policy):
    if policy not in ['O','E','R']:raise ValueError('declared same-weight O/E/R policy required')
    base=labels(base);eligible,rejected=partition(base,actions)
    if rejected or len(eligible)!=len(actions):raise ValueError('all scored actions must pass observable full-axis G first')
    outputs=np.asarray(outputs,dtype=np.float64)
    if outputs.shape!=(len(actions),3) or not np.isfinite(outputs).all():raise ValueError('one finite signed physical three-output row per action required')
    admitted=[i for i,a in enumerate(actions) if outputs[i,0]>0 and (policy!='O' or a.old)]
    vetoed=[i for i in admitted if policy=='R' and (outputs[i,1]>0 or outputs[i,2]>0)]
    admitted=[i for i in admitted if i not in vetoed]
    index=min(admitted,key=lambda i:(-outputs[i,0],actions[i].key)) if admitted else None
    action=None if index is None else actions[index]
    return {'prediction':base.copy() if action is None else apply(base,action),'accepted':action,
            'accepted_index':index,'risk_vetoed_indices':vetoed,'positive_utility_count':int(np.count_nonzero(outputs[:,0]>0)),
            'admitted_positive_count':len(admitted)}
