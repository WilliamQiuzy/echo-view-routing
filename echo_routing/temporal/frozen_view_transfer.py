"""A53 explicitly frozen feature supplement and real linear CPU readouts."""
import numpy as np
import torch


def pooled(features,groups):
    x=np.asarray(features,dtype=np.float64);out=[]
    for rows,weights in groups:
        rows=np.asarray(rows,np.int64);w=np.asarray(weights,np.float64)
        if not len(rows) or len(rows)!=len(w) or np.any(rows<0) or np.any(rows>=len(x)) or not np.isfinite(w).all() or np.any(w<0) or abs(w.sum()-1)>1e-10:
            raise ValueError('complete normalized duration/multiplicity membership required')
        out.append(w@x[rows])
    return np.asarray(out)


def scaler(features,rows):
    x=np.asarray(features,dtype=np.float64)[np.unique(rows)];mu=x.mean(0);sd=x.std(0);sd[sd<1e-6]=1
    return mu,sd


def design(g,s,gscale,sscale):
    a=(np.asarray(g,np.float64)-gscale[0])/gscale[1]
    b=np.zeros_like(a) if s is None else (np.asarray(s,np.float64)-sscale[0])/sscale[1]
    return np.concatenate([a,b],axis=1)


def new_head():
    h=torch.nn.Linear(1024,1,dtype=torch.float64)
    with torch.no_grad():h.weight.zero_();h.bias.fill_(-2)
    return h


def objective(z):
    return torch.nn.functional.softplus(-z[0])+torch.nn.functional.softplus(z[1:]).mean()


def step(h,opt,x,u):
    from echo_routing.temporal.dev_train import lr_factor
    opt.param_groups[0]['lr']=1e-3*lr_factor(u,60,600)
    opt.zero_grad(set_to_none=True);loss=objective(h(torch.from_numpy(x)).flatten());loss.backward()
    norm=float(torch.nn.utils.clip_grad_norm_(h.parameters(),1.));assert np.isfinite(norm)
    opt.step();return {'u':u,'loss':float(loss.detach()),'grad_norm':norm,'lr':opt.param_groups[0]['lr'],'pooled_run_logits':4,'loss_backwards':1,'CNN_backwards':0}
