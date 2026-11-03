import importlib
import torch
import pytest

def module():return importlib.import_module('echo_routing.temporal.dense_refit')

def test_stability_zero_identical_outputs_and_only_later_endpoint_gradient():
    a=module();labels=torch.zeros(1,64,dtype=torch.long);mask=torch.zeros(1,64,dtype=torch.bool);mask[:,:4]=True;z=torch.zeros(1,5,32,requires_grad=True)
    assert a.stability([z,z],labels,mask).item()==0
    with torch.no_grad():z[0,0,1]=1
    s=a.stability([z],labels,mask);g=torch.autograd.grad(s,z)[0]
    assert s>0 and g[:,:,0].abs().sum()==0 and g[:,:,1].abs().sum()>0 and g[:,:,2:].abs().sum()==0

def test_all_four_labels_and_valid_positions_gate_each_tubelet_edge():
    a=module();labels=torch.zeros(1,64,dtype=torch.long);mask=torch.zeros(1,64,dtype=torch.bool);mask[:,:4]=True
    assert a.eligible_edges(labels,mask).sum()==1
    for i in range(4):
        mixed=labels.clone();mixed[0,i]=1;assert a.eligible_edges(mixed,mask).sum()==0
        pad=mask.clone();pad[0,i]=False;assert a.eligible_edges(labels,pad).sum()==0

def test_stage_chunk_normalization_independent_formula_and_coefficient_zero():
    a=module();torch.manual_seed(7);z=[torch.randn(2,5,32,requires_grad=True) for _ in range(2)];labels=torch.zeros(2,64,dtype=torch.long);mask=torch.zeros(2,64,dtype=torch.bool);mask[0,:6]=True;mask[1,:2]=True
    manual=[]
    for stage in z:
        ell=torch.log_softmax(stage.float(),1);manual.append(((ell[0,:,1:3]-ell[0,:,:2].detach()).square().clamp(max=16)).sum()/(2*5)/2)
    assert torch.allclose(a.stability(z,labels,mask),torch.stack(manual).mean())
    from echo_routing.temporal.dense_sequence import dense_ce
    ce=dense_ce(z,labels,mask);loss,parts=a.objective(z,labels,mask,0.)
    assert torch.equal(loss,ce) and all(torch.equal(x,y) for x,y in zip(torch.autograd.grad(loss,z,retain_graph=True),torch.autograd.grad(ce,z,retain_graph=True)))
    zero=a.stability(z,labels,torch.zeros_like(mask));assert torch.isfinite(zero) and zero.item()==0
