"""A45 frozen-input denoising reconstruction support models and conservative veto."""
import numpy as np
import torch
from torch import nn

def partition_mask(footprints, sessions):
    allowed=set(sessions)
    return np.asarray([bool(fp) and set(fp)<=allowed for fp in footprints],bool)

def run_embeddings(pk,emb,off):
    out=[]
    for i in range(len(pk['offsets'])-1):
        a,b=pk['offsets'][i:i+2];rows=pk['rows'][a:b]-off;v=pk['weights'][a:b]
        if abs(float(v.sum())-1)>1e-10:raise ValueError('membership weights do not sum to one')
        out.append(np.sum(np.asarray(emb[rows],np.float64)*v[:,None],axis=0))
    return np.asarray(out,np.float64)

def normalizer(x):
    x=np.asarray(x,np.float64)
    return x.mean(axis=0),np.maximum(x.std(axis=0),1e-3)

def calibrate(cal):
    cal=np.asarray(cal,np.float64)
    if not np.isfinite(cal).all():raise ValueError('nonfinite calibration score')
    k=(99*(len(cal)+1)+99)//100
    return float(np.sort(cal)[k-1]) if k<=len(cal) else float('inf')

def outlier(error,threshold):
    return np.asarray(error)>threshold

def pvalues(cal,error):
    cal=np.asarray(cal,np.float64)
    return np.asarray([(1+int((cal>=e).sum()))/(len(cal)+1) for e in np.asarray(error)])

def deploy(base,proposal,error,threshold):
    b=np.array(base,copy=True); change=(b==2)&(np.asarray(proposal)>0)&outlier(error,threshold)
    b[change]=3
    return b

class SupportAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(512,128),nn.GELU(),nn.Linear(128,32),nn.GELU(),nn.Linear(32,128),nn.GELU(),nn.Linear(128,512))

    def forward(self,x):
        return self.net(x)

def plans(neg,pos,fold,updates=600):
    a,u=[],[]
    for step in range(updates):
        rng=np.random.default_rng([20261028,45,int(fold),step,2])
        common=rng.choice(neg,16,replace=True);second=rng.choice(neg,16,replace=True);positive=rng.choice(pos,16,replace=True)
        a.append(np.r_[common,second]);u.append(np.r_[common,positive])
    return np.asarray(a,np.int64),np.asarray(u,np.int64)

@torch.no_grad()
def errors(net,x,mu,sd):
    z=torch.from_numpy(np.asarray((np.asarray(x)-mu)/sd,np.float32))
    return torch.mean((net(z)-z)**2,dim=1).double().numpy()
