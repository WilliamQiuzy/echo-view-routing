"""A47 frozen clinical-pretraining comparison, whole-run binary repair.

PanEcho attribution: CarDS-Yale/PanEcho, commit836c46e, weights CC BY-NC-SA4.0.
This module does not use the clinical task outputs as frame anatomical labels.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def marker_refs(windows, markers):
    if len(windows) != len(markers):
        raise ValueError('one original marker per window is required')
    result=[]
    for row, marker in zip(windows, markers):
        k=int(marker)
        if k != marker or not 0 <= k < len(row):
            raise ValueError('marker must index its original window')
        result.append(tuple(row[k]))
    return result


def extract_image_state(source, expected, arm):
    if arm == 'P':
        prefix='encoder.encoder.model.'
        raw=source['weights']
        state={k[len(prefix):]:v for k,v in raw.items() if k.startswith(prefix)}
    elif arm == 'G':
        state=dict(source)
        for key, shape in [('classifier.2.weight',(1000,768)),('classifier.2.bias',(1000,))]:
            v=state.pop(key, None)
            if v is None or tuple(v.shape) != shape or not torch.isfinite(v).all():
                raise ValueError('generic source final classifier identity mismatch')
    else:
        raise ValueError('unknown pretrained state')
    if set(state) != set(expected):
        raise ValueError('image encoder must load every exact key; no filling or extra keys')
    for k, v in state.items():
        if v.shape != expected[k].shape or v.dtype != expected[k].dtype or not torch.isfinite(v).all():
            raise ValueError('invalid image state tensor: '+k)
    return state


def clinical_head_names(weights):
    names=sorted({k.split('.')[0] for k in weights if k.split('.')[0].endswith('_head')})
    # Official prose advertises39 tasks; the pinned released state actually has40.
    if len(names)!=40 or not {'LVOTDiam_head','AVStructure_head'}.issubset(names):
        raise ValueError('pinned release clinical-head identity mismatch')
    return names


class ClinicalRepair(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokens=nn.LayerNorm(768)
        self.spatial=nn.Sequential(nn.Conv2d(768,16,1), nn.GELU(), nn.Flatten(),
                                   nn.Linear(16*7*7,128), nn.GELU())
        self.appearance=nn.LayerNorm(512)
        self.fusion=nn.Sequential(nn.Linear(640,128), nn.GELU(), nn.Linear(128,1))
        nn.init.zeros_(self.fusion[2].weight)
        nn.init.constant_(self.fusion[2].bias,-2.)

    def forward(self, maps, appearance):
        x=self.tokens(maps.permute(0,2,3,1)).permute(0,3,1,2)
        z=self.spatial(x)
        return self.fusion(torch.cat([self.appearance(appearance),z],1)).flatten()


def loss_from_scores(scores, weights, slots):
    if scores.ndim != 1 or scores.shape != weights.shape or scores.shape != slots.shape:
        raise ValueError('one weight and run slot per actual window occurrence')
    if not torch.isfinite(scores).all() or not torch.isfinite(weights).all() or (weights<0).any():
        raise ValueError('invalid score or duration weight')
    if (slots<0).any() or (slots>3).any():
        raise ValueError('one positive and three negative whole runs required')
    totals=weights.new_zeros(4).index_add(0,slots,weights)
    if not torch.allclose(totals,torch.ones_like(totals),atol=1e-6,rtol=1e-6):
        raise ValueError('complete membership weights must sum to one per run')
    r=scores.new_zeros(4).index_add(0,slots,scores*weights)
    return F.softplus(-r[0])+F.softplus(r[1:]).mean(),r


def deploy(base, scores):
    base=np.asarray(base);scores=np.asarray(scores)
    if base.shape != scores.shape or not np.isfinite(scores[base==2]).all():
        raise ValueError('finite full run scores required for every original Base A4C')
    pred=base.copy()
    pred[(base==2)&(scores>0)]=3
    return pred
