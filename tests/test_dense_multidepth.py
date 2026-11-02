"""Preparation-only structural tests: no real pretrained model forward."""
import copy,importlib
from types import SimpleNamespace
import pytest
import torch
from torch import nn

def api():return importlib.import_module('echo_routing.temporal.dense_multidepth')
def stub():return SimpleNamespace(hierarchical_layers=[2,5,8,11],out_layers_distillation=[2,5,8,11],out_layers=None,norms_block=[0,1,2,3],return_hierarchical=False)
def test_official_depth_selection_is_explicit_feature_concat_not_a_guessed_slice():
 a=api();old=stub();new=copy.deepcopy(old);a.enable_official_hierarchy(new);assert new.return_hierarchical and not old.return_hierarchical
 for key,value in [('hierarchical_layers',[1,4,7,10]),('out_layers_distillation',[11]),('out_layers',[2,5,8,11]),('norms_block',[0,1,2])]:
  bad=stub();setattr(bad,key,value)
  with pytest.raises(ValueError):a.enable_official_hierarchy(bad)

class MetadataEncoder(nn.Module):
 def __init__(self):
  super().__init__();self.img_mod_embed=nn.Parameter(torch.zeros(1));self.video_mod_embed=nn.Parameter(torch.zeros(1));self.patch_embed_img=nn.Module();self.patch_embed_img.proj=nn.Linear(1,1);self.norms_block=nn.ModuleList([nn.LayerNorm(3) for _ in range(4)]);self.hierarchical_layers=[2,5,8,11];self.out_layers_distillation=[2,5,8,11];self.out_layers=None;self.return_hierarchical=False
 def forward(self,*args):raise AssertionError('structural preparation test must not perform neural forward')

def test_exact_image_roles_and_all_six_new_norm_trainables_are_retained():
 a=api();m=MetadataEncoder();inactive=a.image_only_inactive_names(dict(m.named_parameters()));assert inactive==['img_mod_embed','patch_embed_img.proj.weight','patch_embed_img.proj.bias']
 j=a.wrap_encoder(copy.deepcopy(m),'J-MD',seed=57);f=a.wrap_encoder(copy.deepcopy(m),'F-MD',seed=57)
 assert all(not p.requires_grad for p in f.encoder.parameters())
 assert all(p.requires_grad==(name not in inactive) for name,p in j.encoder.named_parameters())
 assert all(j.encoder.norms_block[i].weight.requires_grad and j.encoder.norms_block[i].bias.requires_grad for i in range(3))
 assert all(torch.equal(v,j.head.state_dict()[k]) for k,v in f.head.state_dict().items())

def test_same_tcn_width_but_projection_capacity_difference_is_explicit():
 m=api().wrap_encoder(MetadataEncoder(),'F-MD',seed=57)
 assert tuple(m.head.first.project.weight.shape)==(128,3072,1)
 assert len(m.head.first.layers)==len(m.head.second.layers)==6
 assert sum(p.numel() for p in m.head.parameters())==1184906
 assert m.feature_contract['normalized_blocks_zero_based']==[2,5,8,11] and m.feature_contract['feature_dim']==3072
