"""Independent enumerable label-path oracles for A58; no images or old-model calls."""
import importlib,itertools
import numpy as np
import pytest
import torch

def module():return importlib.import_module('echo_routing.temporal.structured_evidence')

def oracle_score(unary,transition,duration,path):
    total=sum(unary[t,c] for t,c in enumerate(path));age=0
    for t,c in enumerate(path):
        age=age+1 if t and path[t-1]==c else 1
        if t:total=total+transition[t-1,path[t-1],c]
        if t==len(path)-1 or c!=path[t+1]:total=total+duration[c,min(age,duration.shape[1])-1]
    return total

@pytest.mark.parametrize('classes,length,ages',[(2,1,3),(2,6,3),(3,5,3)])
def test_exact_partition_MAP_and_parameter_gradients_against_all_label_paths(classes,length,ages):
    m=module();rng=torch.Generator().manual_seed(58+length)
    u=torch.randn(length,classes,generator=rng,dtype=torch.float64,requires_grad=True)
    v=torch.randn(length-1,classes,classes,generator=rng,dtype=torch.float64,requires_grad=True)
    g=torch.randn(classes,ages,generator=rng,dtype=torch.float64,requires_grad=True)
    paths=list(itertools.product(range(classes),repeat=length));scores=torch.stack([oracle_score(u,v,g,p) for p in paths]);want=torch.logsumexp(scores,0);got=m.log_partition(u,v,g)
    torch.testing.assert_close(got,want,atol=1e-10,rtol=0)
    gold=paths[len(paths)//3];torch.testing.assert_close(m.path_score(u,v,g,torch.tensor(gold)),oracle_score(u,v,g,gold),atol=1e-10,rtol=0)
    actual_grad=torch.autograd.grad(got,(u,v,g),retain_graph=True,allow_unused=True);expected_grad=torch.autograd.grad(want,(u,v,g),allow_unused=True)
    for actual,expected in zip(actual_grad,expected_grad):
        if actual is None or expected is None:assert actual is expected
        else:torch.testing.assert_close(actual,expected,atol=1e-10,rtol=0)
    path,score=m.viterbi(u,v,g);assert tuple(path.tolist())==paths[int(scores.argmax())];assert abs(score-float(scores.max().detach()))<1e-10
    assert float((got-m.path_score(u,v,g,torch.tensor(gold))).detach())>=-1e-10

def test_global_lexicographic_ties_use_full_prefix_and_zero_score_is_argmax():
    m=module();u=torch.zeros(6,3,dtype=torch.float64);v=torch.zeros(5,3,3,dtype=torch.float64);g=torch.zeros(3,3,dtype=torch.float64)
    # Rank order of the final age state is not the lexicographic path order.
    path,score=m.viterbi(u,v,g);assert path.tolist()==[0]*6 and score==0
    u=torch.tensor([[0.,1.,1.],[1.,1.,0.],[0.,0.,2.]],dtype=torch.float64)
    path,_=m.viterbi(u,v[:2],g);assert torch.equal(path,u.argmax(-1))

@pytest.mark.parametrize('length',[1,128,129,130,257])
def test_one_class_unbounded_path_and_overflow_end_score_charged_once(length):
    m=module();u=torch.zeros(length,1,dtype=torch.float64,requires_grad=True);v=torch.zeros(length-1,1,1,dtype=torch.float64,requires_grad=True);g=torch.arange(129,dtype=torch.float64).reshape(1,129).requires_grad_();want=float(min(length,129)-1)
    z=m.log_partition(u,v,g);assert float(z.detach())==want;path,score=m.viterbi(u,v,g);assert path.tolist()==[0]*length and score==want
    loss=(z-m.path_score(u,v,g,path))/length;loss.backward();assert float(loss.detach())==0 and torch.isfinite(u.grad).all() and torch.isfinite(g.grad).all()
    if length>1:assert torch.isfinite(v.grad).all()

def test_all_unreachable_reduction_has_zero_finite_gradient():
    m=module();x=torch.tensor([[-torch.inf,-torch.inf],[0.,-torch.inf]],requires_grad=True);y=m.safe_logsumexp(x,1);assert y[0]==-torch.inf and y[1]==0
    torch.where(torch.isfinite(y),y,torch.zeros_like(y)).sum().backward();assert torch.isfinite(x.grad).all();assert torch.equal(x.grad[0],torch.zeros(2))

@pytest.mark.parametrize('length',[1,128,129,130])
def test_reachable_age_frontier_overflow_has_finite_gradients_and_zero_potential_partition(length):
    m=module();u=(torch.randn(length,2,dtype=torch.float64)*.03).requires_grad_();v=torch.zeros(length-1,2,2,dtype=torch.float64,requires_grad=True);g=torch.zeros(2,129,dtype=torch.float64,requires_grad=True)
    z=m.log_partition(u,v,g);torch.testing.assert_close(z,torch.logsumexp(u,1).sum(),atol=1e-10,rtol=0);loss=(z-m.path_score(u,v,g,torch.arange(length)%2))/length;loss.backward()
    assert torch.isfinite(u.grad).all() and torch.isfinite(g.grad).all()
    if length>1:assert torch.isfinite(v.grad).all()

def test_integer_score_ties_match_first_globally_lexicographic_enumerated_path():
    m=module();rng=torch.Generator().manual_seed(5801);paths=list(itertools.product(range(3),repeat=4))
    for _ in range(12):
        u=torch.randint(-1,2,(4,3),generator=rng).double();v=torch.randint(-1,2,(3,3,3),generator=rng).double();g=torch.randint(-1,2,(3,2),generator=rng).double();scores=torch.stack([oracle_score(u,v,g,p) for p in paths]);got,value=m.viterbi(u,v,g);assert tuple(got.tolist())==paths[int(scores.argmax())] and value==float(scores.max())

def test_four_arm_head_initialization_features_liveness_and_frozen_inputs():
    m=module();a=torch.softmax(torch.randn(11,5),-1).requires_grad_();b=torch.softmax(torch.randn(11,5),-1).requires_grad_();models={arm:m.StructuredEvidence(arm,2026110358) for arm in ['D0','D1','D2','D3']}
    assert sum(p.numel() for p in models['D1'].parameters())==4894
    assert sum(p.numel() for p in models['D0'].parameters() if p.requires_grad)==4249
    assert all(torch.equal(models['D1'].state_dict()[k],models['D2'].state_dict()[k]) for k in models['D1'].state_dict())
    for arm in ['D0','D1','D2']:
        path=models[arm].decode(a,b);assert torch.equal(path,a.argmax(-1))
    model=models['D2'];opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.001,weight_decay=.01);y=torch.arange(11)%5
    loss=model.nll(a,b,y);loss.backward();assert torch.equal(model.transition[0].weight.grad,torch.zeros_like(model.transition[0].weight));assert a.grad is None and b.grad is None;opt.step();opt.zero_grad(set_to_none=True);model.nll(a,b,y).backward();assert model.transition[0].weight.grad.abs().max()>0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())

def test_masked_loss_is_original_valid_prefix_mean_and_invalid_inputs_fail():
    m=module();model=m.StructuredEvidence('D2',58);a=torch.softmax(torch.randn(9,5),-1);b=torch.softmax(torch.randn(9,5),-1);y=torch.arange(9)%5;mask=torch.arange(9)<4
    torch.testing.assert_close(model.nll(a,b,y,mask),model.nll(a[:4],b[:4],y[:4]));assert len(model.decode(a,b,mask))==4
    for invalid in [torch.zeros(9,dtype=torch.bool),torch.tensor([True,False,True,False,False,False,False,False,False])]:
        with pytest.raises(ValueError):model.nll(a,b,y,invalid)
    with pytest.raises(ValueError):model.decode(a[:0],b[:0])
    bad=a.clone();bad[0,0]=float('nan')
    with pytest.raises(ValueError):model.decode(bad,b)
    with pytest.raises(ValueError):model.decode(a*2,b)
