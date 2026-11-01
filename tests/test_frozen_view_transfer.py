import numpy as np
import torch
from echo_routing.temporal.frozen_view_transfer import pooled, scaler, design, new_head, objective


def test_complete_multiplicity_mean_and_scaling_commute_with_linear_score():
    x=np.array([[1.,3.],[4.,7.],[9.,2.]])
    groups=[(np.array([0,1,1]),np.array([.2,.3,.5])),(np.array([2]),np.array([1.]))]
    mean=pooled(x,groups);assert np.allclose(mean[0],.2*x[0]+.8*x[1],rtol=0,atol=1e-14)
    mu,sd=scaler(x,[0,1]);w=np.array([.4,-.7]);b=-.3
    direct=sum(v*(((x[i]-mu)/sd)@w+b) for i,v in zip(*groups[0]))
    assert abs(((mean[0]-mu)/sd)@w+b-direct)<1e-12


def test_control_supplement_is_structurally_zero_and_init_is_minus_two():
    g=np.ones((4,512));mu=np.zeros(512);sd=np.ones(512)
    x=design(g,None,(mu,sd),None);assert not x[:,512:].any()
    h=new_head();assert np.array_equal(h(torch.from_numpy(x)).detach().numpy(),np.full((4,1),-2.))


def test_linear_loss_gradient_matches_independent_sigmoid_formula():
    x=torch.arange(4*1024,dtype=torch.float64).reshape(4,1024)/4000;h=new_head();z=h(x).flatten();loss=objective(z);loss.backward()
    s=1/(1+np.exp(2));coeff=np.array([s-1,s/3,s/3,s/3]);ref=x.numpy().T@coeff
    assert np.max(abs(h.weight.grad.numpy()[0]-ref))<1e-12
    assert abs(h.bias.grad.item()-coeff.sum())<1e-12
