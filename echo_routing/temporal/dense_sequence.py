"""A57: dense video tubelets to complete five-class sequences, without Base cuts."""
import contextlib
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

LENGTH=64
STRIDE=32
CLIP=16
PATCHES=196
CLASSES=5
DILATIONS=(1,2,4,8,16,32)


def chunk_starts(n):
    if int(n)!=n or n<1:raise ValueError('positive original sequence length required')
    return list(range(0,int(n),STRIDE))


def chunk_positions(n,start):
    if start not in chunk_starts(n):raise ValueError('start outside fixed stride32 inference grid')
    positions=np.arange(start,start+LENGTH,dtype=np.int64)
    return np.minimum(positions,n-1),positions<n


def spatial_time_features(tokens,batch):
    if tokens.ndim!=3 or tokens.shape[0]!=batch*4 or tokens.shape[1]!=8*PATCHES:
        raise ValueError('four clips of eight-by196 dense tokens required')
    # tubelet j spans source positions[2j,2j+1], center2j+.5.
    return tokens.float().reshape(batch,4,8,PATCHES,-1).mean(3).reshape(batch,32,-1).transpose(1,2).contiguous()


def token_mask(sample_mask):
    if sample_mask.ndim!=2 or sample_mask.shape[1]!=64:raise ValueError('original64-position mask required')
    return sample_mask.bool().reshape(-1,32,2).any(-1)


def expand_logits(logits):
    if logits.ndim!=3 or logits.shape[1:]!=(5,32):raise ValueError('five logits for each of32 tubelets required')
    return logits.repeat_interleave(2,dim=-1)


def dense_ce(stages,labels,sample_mask):
    if not stages or labels.shape!=sample_mask.shape or labels.ndim!=2 or labels.shape[1]!=64 or not sample_mask.any():raise ValueError('valid original labels/masks and stages required')
    valid=sample_mask.bool()
    if ((labels[valid]<0)|(labels[valid]>=CLASSES)).any():raise ValueError('only public five-family labels0..4 allowed')
    targets=labels.long().masked_fill(~valid,-100)
    return torch.stack([F.cross_entropy(expand_logits(z).float(),targets,ignore_index=-100,reduction='sum')/valid.sum() for z in stages]).mean()


def stitch_probabilities(chunks,n,starts=None):
    expected=chunk_starts(n);starts=expected if starts is None else list(starts)
    values=np.asarray(chunks,dtype=np.float64)
    if starts!=expected or values.shape!=(len(starts),64,CLASSES):raise ValueError('complete fixed-grid64x5 chunks required')
    if not np.isfinite(values).all() or (values<0).any() or not np.allclose(values.sum(-1),1.,atol=1e-4):raise ValueError('finite normalized five-class probabilities required')
    total=np.zeros((n,CLASSES));denominator=np.zeros(n);weights=np.minimum(np.arange(64)+1,64-np.arange(64))
    for row,start in enumerate(starts):
        length=min(64,n-start);w=weights[:length];total[start:start+length]+=values[row,:length]*w[:,None];denominator[start:start+length]+=w
    if not (denominator>0).all():raise ValueError('every original position needs positive coverage')
    return total/denominator[:,None]


class ResidualTemporalLayer(nn.Module):
    def __init__(self,channels,dilation):
        super().__init__();self.dilated=nn.Conv1d(channels,channels,3,padding=dilation,dilation=dilation);self.pointwise=nn.Conv1d(channels,channels,1);self.dropout=nn.Dropout(.1)
    def forward(self,x,mask):
        x=x*mask;hidden=F.relu(self.dilated(x));hidden=self.dropout(self.pointwise(hidden));return (x+hidden)*mask


class TemporalStage(nn.Module):
    def __init__(self,in_dim,channels):
        super().__init__();self.project=nn.Conv1d(in_dim,channels,1);self.layers=nn.ModuleList([ResidualTemporalLayer(channels,d) for d in DILATIONS]);self.classifier=nn.Conv1d(channels,CLASSES,1)
    def forward(self,x,mask):
        x=self.project(x*mask)*mask
        for layer in self.layers:x=layer(x,mask)
        return self.classifier(x)*mask


class TwoStageTCN(nn.Module):
    def __init__(self,in_dim=768,channels=128):
        super().__init__();self.first=TemporalStage(in_dim,channels);self.second=TemporalStage(CLASSES,channels)
    def forward(self,x,valid):
        mask=valid[:,None].to(x.dtype);first=self.first(x,mask);second=self.second(torch.softmax(first,dim=1)*mask,mask);return [first,second]


class DenseSequence(nn.Module):
    """Encoder receives genuine RGB clips. Only J trains it; all heads emit full timelines."""
    def __init__(self,encoder,arm,dim=768,channels=128,bf16=True,seed=2026110257):
        super().__init__()
        if arm not in ('L','F','J','J-low','J-high'):raise ValueError('fixed local/frozen/joint arms required')
        self.encoder=encoder;self.arm=arm;self.joint=arm.startswith('J');self.bf16=bool(bf16)
        for p in encoder.parameters():p.requires_grad_(self.joint)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed);self.head=nn.Conv1d(dim,CLASSES,1) if arm=='L' else TwoStageTCN(dim,channels)
        self.head.to(next(encoder.parameters()).device)
    def train(self,mode=True):
        super().train(mode);self.encoder.eval()  # backbone dropout/drop-path are declared0 in all arms.
        return self
    def forward(self,u8,sample_mask):
        if u8.ndim!=5 or u8.shape[1]!=64 or u8.shape[-1]!=3 or u8.dtype!=torch.uint8:raise ValueError('RGB uint8[B,64,H,W,3] required')
        valid=token_mask(sample_mask)
        if len(valid)!=len(u8):raise ValueError('chunk mask batch mismatch')
        batch=len(u8);clips=u8.reshape(batch*4,16,*u8.shape[2:]);x=clips.float()/255.;mean=x.new_tensor([.485,.456,.406]);std=x.new_tensor([.229,.224,.225]);x=((x-mean)/std).permute(0,4,1,2,3).contiguous()
        grad=contextlib.nullcontext() if self.joint else torch.no_grad()
        with grad,torch.autocast(x.device.type,dtype=torch.bfloat16,enabled=self.bf16 and x.device.type=='cuda'):
            tokens=self.encoder(x)
        features=spatial_time_features(tokens,batch)
        if self.arm=='L':return [self.head(features)*valid[:,None].to(features.dtype)]
        return self.head(features,valid)


def probabilities(stages):
    return torch.softmax(expand_logits(stages[-1]).float(),dim=1).transpose(1,2)
