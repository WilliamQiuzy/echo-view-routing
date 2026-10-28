"""A46: explicit displacement-field representation, not a proposal veto.

RAFT/appearance are frozen input transforms; every MotionRepair parameter is
trained. S changes spatial organization of the identical motion tensor only.
"""
import hashlib

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def valid_pair(a, b):
    """The original10Hz references advance3 source frames at30fps; no splice motion."""
    return a[0] == b[0] and int(b[1])-int(a[1]) == 3 and float(a[2]) == float(b[2])


def tile_permutation(fold, row):
    h=hashlib.sha256(f'A46|{int(fold)}|{int(row)}'.encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8],'big')).permutation(49)


def shuffle_tiles(z, perms):
    """Same49-tile permutation for every time/vector component of each window."""
    n,c,t,h,w=z.shape
    if (h,w)!=(56,56) or perms.shape!=(n,49):
        raise ValueError('expected56x56 maps and one49-tile permutation/window')
    tiles=z.reshape(n,c,t,7,8,7,8).permute(0,3,5,1,2,4,6).reshape(n,49,c,t,8,8)
    y=tiles[torch.arange(n,device=z.device)[:,None],perms]
    return y.reshape(n,7,7,c,t,8,8).permute(0,3,4,1,5,2,6).reshape(n,c,t,56,56)


def motion_inputs(bank, pair_index, markers, perms=None):
    """Pair bank(N,2,56,56) -> window fields(B,3,15,56,56), invalid pairs zero.

    A pair id of-1 is never used to index the bank's last pair. Marker is added
    after spatial shuffling and denotes the original target time, not anatomy.
    """
    ix=torch.as_tensor(pair_index,dtype=torch.long,device=bank.device)
    marker=torch.as_tensor(markers,dtype=torch.long,device=bank.device).clamp(max=14)
    z=bank[ix.clamp(min=0)].float()*(ix>=0)[:,:,None,None,None]
    z=z.permute(0,2,1,3,4)
    if perms is not None:
        z=shuffle_tiles(z,torch.as_tensor(perms,dtype=torch.long,device=bank.device))
    mark=torch.zeros(len(ix),1,15,56,56,device=bank.device,dtype=z.dtype)
    mark[torch.arange(len(ix),device=bank.device),0,marker]=1
    return torch.cat([z,mark],dim=1)


class DeterministicPool(nn.Module):
    """Exact adaptive(1,4,4) bins via slice means: deterministic CUDA backward.

    PyTorch's adaptive_avg_pool3d CUDA backward is nondeterministic. These
    floor/ceil bins implement the same mathematical operator, including overlap.
    """
    def forward(self,x):
        h,w=x.shape[-2:]
        out=[]
        for i in range(4):
            for j in range(4):
                out.append(x[:,:,:,i*h//4:((i+1)*h+3)//4,j*w//4:((j+1)*w+3)//4].mean((2,3,4)))
        return torch.stack(out,dim=-1).reshape(x.shape[0],x.shape[1],1,4,4)


class MotionRepair(nn.Module):
    """Learn local spatiotemporal displacement, fuse with frozen RGB appearance."""
    def __init__(self):
        super().__init__()
        self.motion=nn.Sequential(
            nn.Conv3d(3,16,(3,5,5),(1,2,2),(1,2,2)),nn.GroupNorm(4,16),nn.SiLU(),
            nn.Conv3d(16,32,3,(1,2,2),1),nn.GroupNorm(8,32),nn.SiLU(),
            nn.Conv3d(32,64,3,(1,2,2),1),nn.GroupNorm(8,64),nn.SiLU(),
            DeterministicPool(),nn.Flatten(),nn.Linear(1024,128),nn.GELU())
        self.appearance=nn.LayerNorm(512)
        self.fusion=nn.Sequential(nn.Linear(640,128),nn.GELU(),nn.Linear(128,1))
        nn.init.zeros_(self.fusion[-1].weight)
        nn.init.constant_(self.fusion[-1].bias,-2.)

    def forward(self,x,embedding):
        return self.fusion(torch.cat([self.appearance(embedding),self.motion(x)],dim=1)).flatten()


def loss_from_scores(scores, weights, slots):
    """Full multiplicity-weighted means; slot0 positive, slots1..3 protected negatives."""
    r=scores.new_zeros(4).index_add(0,slots,scores*weights)
    return F.softplus(-r[0])+F.softplus(r[1:]).mean(),r


def deploy(base, scores):
    p=np.asarray(base).copy()
    p[(p==2)&(np.asarray(scores)>0)]=3
    return p
