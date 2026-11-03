"""A58 exact conditional age-state scores; the last age bucket is unbounded overflow."""
from __future__ import annotations
import numpy as np
import torch

def _validate(unary,transition,duration):
    if unary.ndim!=2 or min(unary.shape)<1:raise ValueError('nonempty time-by-class unary required')
    t,k=unary.shape
    if transition.shape!=(t-1,k,k) or duration.ndim!=2 or duration.shape[0]!=k or duration.shape[1]<2:raise ValueError('complete adjacent transitions and at least two age states required')
    if any(not x.is_floating_point() or x.dtype!=unary.dtype or x.device!=unary.device or not bool(torch.isfinite(x).all()) for x in [unary,transition,duration]):raise ValueError('same-device/dtype finite floating scores required')
    return t,k,duration.shape[1]

def safe_logsumexp(x,dim):
    """All-unreachable rows stay -inf with zero gradient, without -inf/-inf backward."""
    reachable=torch.isfinite(x).any(dim=dim)
    safe=torch.where(reachable.unsqueeze(dim),x,torch.zeros_like(x))
    value=torch.logsumexp(safe,dim=dim)
    return torch.where(reachable,value,torch.full_like(value,-torch.inf))

def path_score(unary,transition,duration,labels):
    t,k,ages=_validate(unary,transition,duration)
    if labels.shape!=(t,) or labels.dtype!=torch.long or bool(((labels<0)|(labels>=k)).any()):raise ValueError('one valid integer class per original position required')
    lab=labels.tolist();ends=[];lengths=[];age=0
    for i,c in enumerate(lab):
        age=age+1 if i and lab[i-1]==c else 1
        if i==t-1 or lab[i+1]!=c:ends.append(c);lengths.append(min(age,ages)-1)
    score=unary[torch.arange(t,device=unary.device),labels].sum()+duration[ends,lengths].sum()
    if t>1:score=score+transition[torch.arange(t-1,device=unary.device),labels[:-1],labels[1:]].sum()
    return score

def log_partition(unary,transition,duration):
    """One unique age path per label path. End potentials apply only at changes/terminal."""
    t,k,ages=_validate(unary,transition,duration)
    if k==1:return path_score(unary,transition,duration,torch.zeros(t,dtype=torch.long,device=unary.device))
    alpha=unary[0,:,None];diagonal=torch.eye(k,dtype=torch.bool,device=unary.device)
    for i in range(1,t):
        width=alpha.shape[1]
        ended=torch.logsumexp(alpha+duration[:,:width],dim=1)
        switch=safe_logsumexp((ended[:,None]+transition[i-1]).masked_fill(diagonal,-torch.inf),0)
        stay=alpha+transition[i-1].diagonal()[:,None]
        if width==ages:
            # Only previous explicit-last and overflow states merge. No end charge on either stay.
            overflow=torch.logsumexp(stay[:,-2:],1,keepdim=True)
            alpha=torch.cat([switch[:,None],stay[:,:-2],overflow],1)
        else:alpha=torch.cat([switch[:,None],stay],1)
        alpha=alpha+unary[i,:,None]
    return torch.logsumexp((alpha+duration[:,:alpha.shape[1]]).flatten(),0)

@torch.no_grad()
def viterbi(unary,transition,duration):
    """Exact MAP with full-prefix lexicographic ties; FP64 accumulation of fixed scores."""
    t,k,ages=_validate(unary,transition,duration)
    if k==1:
        path=torch.zeros(t,dtype=torch.long,device=unary.device);return path,float(path_score(unary,transition,duration,path))
    u,v,g=[x.detach().cpu().double().numpy() for x in [unary,transition,duration]]
    scores=u[0,:,None].copy();ranks=np.arange(k,dtype=np.int64)[:,None];backs=[]
    for i in range(1,t):
        old_width=scores.shape[1];width=min(old_width+1,ages);new=np.empty((k,width));back=np.empty((k,width),np.int64)
        ended=scores+g[:,:old_width]
        for c in range(k):
            candidates=ended+v[i-1,:,c,None];candidates=candidates.copy();candidates[c]=-np.inf;best=candidates.max();ties=np.flatnonzero(candidates.ravel()==best);chosen=ties[np.argmin(ranks.ravel()[ties])];new[c,0]=best;back[c,0]=chosen
            if old_width<ages:
                new[c,1:]=scores[c]+v[i-1,c,c];back[c,1:]=c*old_width+np.arange(old_width)
            else:
                new[c,1:-1]=scores[c,:-2]+v[i-1,c,c];back[c,1:-1]=c*old_width+np.arange(old_width-2)
                candidate=scores[c,-2:];best=candidate.max();ties=np.flatnonzero(candidate==best)+(old_width-2);chosen=ties[np.argmin(ranks[c,ties])];new[c,-1]=best+v[i-1,c,c];back[c,-1]=c*old_width+chosen
        new+=u[i,:,None];prefix=ranks.ravel()[back.ravel()];labels=np.repeat(np.arange(k),width);order=np.lexsort((labels,prefix));new_ranks=np.empty(k*width,np.int64);new_ranks[order]=np.arange(k*width);ranks=new_ranks.reshape(k,width);scores=new;backs.append(back)
    terminal=scores+g[:,:scores.shape[1]];best=terminal.max();ties=np.flatnonzero(terminal.ravel()==best);idx=int(ties[np.argmin(ranks.ravel()[ties])]);path=np.empty(t,np.int64)
    for i in range(t-1,-1,-1):
        width=min(i+1,ages);c,a=divmod(idx,width);path[i]=c
        if i:idx=int(backs[i-1][c,a])
    return torch.from_numpy(path).to(unary.device),float(best)

def _posterior(p,classes):
    if p.ndim!=2 or p.shape[1]!=classes or not len(p) or not p.is_floating_point() or not bool(torch.isfinite(p).all()) or bool((p<0).any()) or not torch.allclose(p.sum(1),torch.ones(len(p),device=p.device,dtype=p.dtype),atol=1e-5,rtol=0):raise ValueError('complete finite normalized original-axis posterior required')

def _prefix(mask,length):
    if mask is None:return length
    if mask.shape!=(length,) or mask.dtype!=torch.bool:raise ValueError('boolean original-position prefix mask required')
    n=int(mask.sum())
    if not n or not bool(mask[:n].all()) or bool(mask[n:].any()):raise ValueError('nonempty contiguous valid prefix required')
    return n

def trajectory_features(p):
    previous=torch.cat([p[:1],p[:-1]],0);delta=p-previous
    return torch.cat([p,previous,delta,delta.abs()],1)

class StructuredEvidence(torch.nn.Module):
    """Frozen posterior trajectories to shared transition scores and capped-end potentials."""
    def __init__(self,arm,seed,classes=5,ages=129):
        super().__init__()
        if arm not in ['D0','D1','D2','D3'] or classes<1 or ages<2:raise ValueError('declared arm/classes/age states required')
        self.arm,self.seed,self.classes,self.ages=arm,int(seed),classes,ages
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed);self.transition=torch.nn.Sequential(torch.nn.Linear(8*classes,64),torch.nn.GELU(),torch.nn.Linear(64,classes*classes));torch.nn.init.zeros_(self.transition[-1].weight);torch.nn.init.zeros_(self.transition[-1].bias)
        self.duration=torch.nn.Parameter(torch.zeros(classes,ages),requires_grad=arm!='D0')
    def scores(self,a08,jepa,mask=None):
        _posterior(a08,self.classes);_posterior(jepa,self.classes)
        if a08.shape!=jepa.shape or a08.device!=jepa.device:raise ValueError('identical original axes and device required')
        n=_prefix(mask,len(a08));dtype=self.duration.dtype;a=a08[:n].detach().to(dtype);b=jepa[:n].detach().to(dtype)
        if self.arm in ['D0','D1']:first,second,unary=a,a,a
        elif self.arm=='D2':first,second,unary=a,b,a
        else:first,second,unary=b,b,b
        features=torch.cat([trajectory_features(first),trajectory_features(second)],1)[1:];v=self.transition(features).reshape(n-1,self.classes,self.classes)
        return unary.clamp_min(1e-6).log(),v,self.duration
    def nll(self,a08,jepa,labels,mask=None):
        n=_prefix(mask,len(a08))
        if labels.shape!=(len(a08),):raise ValueError('label axis differs from original posterior axis')
        u,v,g=self.scores(a08,jepa,mask);return (log_partition(u,v,g)-path_score(u,v,g,labels[:n]))/n
    @torch.no_grad()
    def decode(self,a08,jepa,mask=None):return viterbi(*self.scores(a08,jepa,mask))[0]
