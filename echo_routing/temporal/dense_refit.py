"""Fixed full-data CE versus label-protected temporal stability on dense logits."""
import torch
from torch.nn import functional as F
from echo_routing.temporal.dense_sequence import dense_ce

def eligible_edges(labels,mask):
    if labels.shape!=mask.shape or labels.ndim!=2 or labels.shape[1]!=64:raise ValueError('original64-position labels/mask required')
    pair=labels.reshape(-1,32,2);valid=mask.bool().reshape(-1,32,2).all(-1);pure=pair[:,:,0]==pair[:,:,1]
    return valid[:,:-1]&valid[:,1:]&pure[:,:-1]&pure[:,1:]&(pair[:,:-1,0]==pair[:,1:,0])

def stability(stages,labels,mask):
    edge=eligible_edges(labels,mask);denominator=(5*edge.sum(1).clamp(min=1)).float();values=[]
    for z in stages:
        if z.shape!=(len(labels),5,32):raise ValueError('five-class32-tubelet stage required')
        ell=F.log_softmax(z.float(),dim=1);delta=ell[:,:,1:]-ell[:,:,:-1].detach();values.append(((delta.square().clamp(max=16)*edge[:,None]).sum((1,2))/denominator).mean())
    if not values:raise ValueError('atleast one stage required')
    return torch.stack(values).mean()

def objective(stages,labels,mask,coefficient):
    ce=dense_ce(stages,labels,mask);s=stability(stages,labels,mask);loss=ce if coefficient==0 else ce+coefficient*s
    return loss,{'CE':ce,'S':s,'weighted_S':coefficient*s}

def diagnostics(stages,labels,mask,coefficient=.15):
    edge=eligible_edges(labels,mask);_,parts=objective(stages,labels,mask,coefficient);a=torch.autograd.grad(parts['CE'],stages,retain_graph=True);b=torch.autograd.grad(parts['weighted_S'],stages,retain_graph=True);aa=sum(x.float().square().sum() for x in a);bb=sum(x.float().square().sum() for x in b);dot=sum((x.float()*y.float()).sum() for x,y in zip(a,b));clamped=[]
    for z in stages:
        ell=F.log_softmax(z.float(),1);clamped.append(int((((ell[:,:,1:]-ell[:,:,:-1].detach()).square()>16)&edge[:,None]).sum()))
    return {k:float(v.detach()) for k,v in parts.items()}|{'CE_logit_gradient_norm':float(aa.sqrt()),'weighted_S_logit_gradient_norm':float(bb.sqrt()),'logit_gradient_cosine':float(dot/(aa*bb).sqrt()) if aa>0 and bb>0 else None,'eligible_edges_per_chunk':edge.sum(1).tolist(),'clamped_class_edges_per_stage':clamped}
