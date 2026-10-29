"""A48 strict transfer-source contract; synthetic states only."""
import importlib
import importlib.util
import pytest
import torch
import copy
from torch import nn


def core():
    name='echo_routing.temporal.valve_localized'
    assert importlib.util.find_spec(name) is not None,'A48 valve-source module missing'
    torch.set_num_threads(2)
    return importlib.import_module(name)


def test_only_documented_fpn_v1_names_migrate_without_filling():
    m=core();expected={'body.layer4.0.conv1.weight':torch.zeros(2,2),
                      'fpn.inner_blocks.0.0.weight':torch.zeros(2,2)}
    raw={'backbone.body.layer4.0.conv1.weight':torch.ones(2,2),
         'backbone.fpn.inner_blocks.0.weight':torch.ones(2,2),'roi_heads.unused':torch.zeros(1)}
    state,renamed=m.canonical_backbone(raw,expected)
    assert set(state)==set(expected) and renamed=={'fpn.inner_blocks.0.weight':'fpn.inner_blocks.0.0.weight'}
    assert state['fpn.inner_blocks.0.0.weight'] is raw['backbone.fpn.inner_blocks.0.weight']
    for bad in [dict(raw,**{'backbone.surprise':torch.zeros(1)}),
                dict(raw,**{'backbone.fpn.inner_blocks.0.0.weight':torch.zeros(2,2)}),
                {'backbone.fpn.inner_blocks.0.weight':torch.zeros(2,2)},
                dict(raw,**{'backbone.body.layer4.0.conv1.weight':torch.zeros(3,2)})]:
        with pytest.raises(ValueError):m.canonical_backbone(bad,expected)


def test_four_pyramid_scales_are_all_used_in_fixed_order():
    m=core();assert hasattr(m,'pyramid_maps'),'four-scale representation missing'
    maps={str(i):torch.full((2,256,32//(2**i),32//(2**i)),float(i),requires_grad=True) for i in range(4)}
    out=m.pyramid_maps(dict(reversed(list(maps.items()))));assert out.shape==(2,1024,14,14)
    for i in range(4):assert torch.equal(out[:,i*256:(i+1)*256],torch.full((2,256,14,14),float(i)))
    out.sum().backward();assert all(z.grad.abs().sum()>0 for z in maps.values())
    with pytest.raises(ValueError):m.pyramid_maps({'0':maps['0']})


def test_only_layer4_and_fpn_are_trainable():
    m=core();assert hasattr(m,'set_trainable'),'backbone binding missing'
    b=nn.Module();b.body=nn.ModuleDict({'layer3':nn.Linear(2,2),'layer4':nn.Linear(2,2)});b.fpn=nn.ModuleDict({'inner':nn.Linear(2,2),'out':nn.Linear(2,2)})
    m.set_trainable(b)
    assert all(p.requires_grad==(n.startswith('body.layer4.') or n.startswith('fpn.')) for n,p in b.named_parameters())


def test_spatial_head_count_init_and_independent_windows():
    m=core();assert hasattr(m,'ValveHead'),'live spatial head missing'
    torch.manual_seed(48);h=m.ValveHead().double().eval();x=torch.randn(3,1024,14,14,dtype=torch.float64);e=torch.randn(3,512,dtype=torch.float64)
    assert sum(p.numel() for p in h.parameters())==294281
    assert torch.equal(h(x,e),torch.full((3,),-2.,dtype=torch.float64))
    with torch.no_grad():h.fusion[2].weight.fill_(.01)
    assert torch.allclose(h(x,e),torch.cat([h(x[i:i+1],e[i:i+1]) for i in range(3)]),atol=1e-12,rtol=1e-12)


def test_two_real_passes_match_retained_graph_including_zero_occurrence():
    m=core();assert hasattr(m,'two_pass_backward'),'whole-run two-pass backward missing'
    class Toy(nn.Module):
        def __init__(self):super().__init__();self.net=nn.Linear(3,1).double();self.seen=[]
        def forward(self,x,e):self.seen.append(len(x));return self.net(x+e).flatten()
    torch.manual_seed(48);h=Toy();x=torch.randn(8,3,dtype=torch.float64);e=torch.randn(8,3,dtype=torch.float64);w=torch.tensor([.2,.8,0.,1.,.5,.5,.3,.7],dtype=torch.float64);slots=torch.tensor([0,0,1,1,2,2,3,3]);s=h(x,e);r=torch.stack([(s[slots==i]*w[slots==i]).sum() for i in range(4)]);loss=torch.logaddexp(r.new_zeros(()),-r[0])+torch.logaddexp(r.new_zeros(3),r[1:]).mean();loss.backward();expected=torch.cat([p.grad.flatten() for p in h.parameters()]);h.zero_grad(set_to_none=True);h.seen=[]
    result=m.two_pass_backward(h,x,e,w,slots,3);actual=torch.cat([p.grad.flatten() for p in h.parameters()]);assert torch.allclose(expected,actual,atol=1e-12,rtol=1e-12);assert h.seen==[3,3,2,3,3,2] and result['first_occurrences']==8 and result['second_occurrences']==8 and result['zero_coefficients']==1
    assert abs(float(loss)-result['loss'])<1e-12


def test_repeat_and_optimizer_resume_on_same_source_path():
    m=core();assert hasattr(m,'ValveHead') and hasattr(m,'two_pass_backward'),'model/gradient contract missing'
    torch.manual_seed(948);x=torch.randn(4,1024,14,14);e=torch.randn(4,512);w=torch.ones(4);slots=torch.arange(4)
    def fresh():
        torch.manual_seed(948);h=m.ValveHead().eval();return h,torch.optim.AdamW(h.parameters(),lr=.001,weight_decay=.0001)
    def step(h,o,n):
        for _ in range(n):o.zero_grad(set_to_none=True);m.two_pass_backward(h,x,e,w,slots,2);torch.nn.utils.clip_grad_norm_(h.parameters(),1);o.step()
    h,o=fresh();step(h,o,6);goal=copy.deepcopy(h.state_dict())
    for resume in [False,True]:
        hh,oo=fresh()
        if resume:
            step(hh,oo,3);st=copy.deepcopy((hh.state_dict(),oo.state_dict()));hh,oo=fresh();hh.load_state_dict(st[0]);oo.load_state_dict(st[1]);step(hh,oo,3)
        else:step(hh,oo,6)
        assert all(torch.equal(v,hh.state_dict()[k]) for k,v in goal.items())
