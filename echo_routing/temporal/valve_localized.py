"""A48 named transfer from box-supervised valve localization, no detector gate.

Weights: CardiovascularImagingVHIR/AoVdetector, CC BY-NC-SA4.0. Only licensed
torchvision code is used; upstream annotation/training scripts are not executed.
"""
import re
import torch
from torch import nn
from torch.nn import functional as F
from .clinical_pretraining import loss_from_scores


def canonical_backbone(raw,expected):
    state={};renamed={}
    for original,value in raw.items():
        if not original.startswith('backbone.'):continue
        key=original[len('backbone.'):]
        # Exact documented torchvision FPN._load_from_state_dict v1->v2 migration.
        match=re.fullmatch(r'fpn\.(inner_blocks|layer_blocks)\.(\d+)\.(weight|bias)',key)
        new=f'fpn.{match[1]}.{match[2]}.0.{match[3]}' if match else key
        if new!=key:renamed[key]=new
        if new in state:raise ValueError('conflicting FPN state aliases')
        state[new]=value
    if set(state)!=set(expected):raise ValueError('complete exact backbone keys required')
    for k,v in state.items():
        if not isinstance(v,torch.Tensor) or v.shape!=expected[k].shape or v.dtype!=expected[k].dtype or not torch.isfinite(v).all():
            raise ValueError('invalid source tensor '+k)
    return state,renamed


def bin_pool(x):
    """Same floor/ceil adaptive-average bins, deterministic GEMM backward.

    CUDA adaptive_avg_pool2d backward is atomic/nondeterministic in the pinned
    runtime. Separable normalized indicator matrices implement identical bins.
    """
    mats=[]
    for length in x.shape[-2:]:
        bins=torch.arange(14,device=x.device)
        start=torch.div(bins*length,14,rounding_mode='floor')
        end=torch.div((bins+1)*length+13,14,rounding_mode='floor')
        at=torch.arange(length,device=x.device)
        mat=((at[None,:]>=start[:,None])&(at[None,:]<end[:,None])).to(x.dtype)
        mats.append(mat/(end-start).to(x.dtype)[:,None])
    return torch.matmul(torch.matmul(mats[0],x),mats[1].T)


def pyramid_maps(outputs):
    if not all(str(i) in outputs for i in range(4)):
        raise ValueError('all four spatial pyramid scales required')
    parts=[]
    for i in range(4):
        x=outputs[str(i)]
        if x.ndim!=4 or x.shape[1]!=256:
            raise ValueError('expected 256-channel spatial pyramid')
        parts.append(bin_pool(x))
    return torch.cat(parts,dim=1)


def set_trainable(backbone):
    for name,p in backbone.named_parameters():
        p.requires_grad_(name.startswith('body.layer4.') or name.startswith('fpn.'))
    assert any(p.requires_grad for p in backbone.parameters())
    return backbone


class ValveHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokens=nn.LayerNorm(1024)
        self.spatial=nn.Sequential(nn.Conv2d(1024,8,1),nn.GELU(),nn.Flatten(),nn.Linear(1568,128),nn.GELU())
        self.appearance=nn.LayerNorm(512)
        self.fusion=nn.Sequential(nn.Linear(640,128),nn.GELU(),nn.Linear(128,1))
        nn.init.zeros_(self.fusion[2].weight);nn.init.constant_(self.fusion[2].bias,-2.)

    def forward(self,maps,embedding):
        z=self.tokens(maps.permute(0,2,3,1)).permute(0,3,1,2)
        return self.fusion(torch.cat([self.spatial(z),self.appearance(embedding)],dim=1)).flatten()


def make_backbone(state=None):
    from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
    from torchvision.ops.misc import FrozenBatchNorm2d
    b=resnet_fpn_backbone(backbone_name='resnet50',weights=None,norm_layer=FrozenBatchNorm2d,trainable_layers=3)
    for mod in b.modules():
        if isinstance(mod,FrozenBatchNorm2d):mod.eps=0.
    if state is not None:b.load_state_dict(state,strict=True)
    return set_trainable(b).eval()


class ValveRepair(nn.Module):
    def __init__(self,backbone):
        super().__init__()
        from torchvision.models.detection.transform import GeneralizedRCNNTransform
        self.backbone=backbone
        self.transform=GeneralizedRCNNTransform(800,1333,[.485,.456,.406],[.229,.224,.225],size_divisible=32)
        self.head=ValveHead()
        self.eval()

    def forward(self,frames,embedding):
        if frames.dtype!=torch.uint8 or frames.ndim!=4 or frames.shape[1:]!=(224,224,3):
            raise ValueError('original clean 224 RGB uint8 marker required')
        images=frames.permute(0,3,1,2).to(embedding.device).float()/255.
        batch,_=self.transform(list(images),None)
        return self.head(pyramid_maps(self.backbone(batch.tensors)),embedding)


def two_pass_backward(model,x,e,weights,slots,microbatch=4):
    n=len(x)
    if n!=len(e) or n!=len(weights) or n!=len(slots) or n==0:raise ValueError('complete aligned occurrences required')
    with torch.no_grad():
        scores=torch.cat([model(x[j:j+microbatch],e[j:j+microbatch]) for j in range(0,n,microbatch)])
    z=scores.detach().requires_grad_();loss,r=loss_from_scores(z,weights,slots)
    coef=torch.autograd.grad(loss,z)[0]
    replay=[]
    for j in range(0,n,microbatch):
        s=model(x[j:j+microbatch],e[j:j+microbatch])
        s.backward(coef[j:j+microbatch]);replay.append(s.detach())
    err=float((scores-torch.cat(replay)).abs().max())
    if not torch.isfinite(loss) or not torch.isfinite(coef).all() or err>1e-4:raise ValueError('invalid whole-run score replay')
    return {'loss':float(loss.detach()),'run_scores':r.detach().cpu().tolist(),'first_occurrences':n,'second_occurrences':n,'zero_coefficients':int((coef==0).sum()),'replay_max_abs':err}
