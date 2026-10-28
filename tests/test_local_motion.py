"""Critical invariants of A46: correspondence, marker, chain rule and protected deployment."""
import copy

import numpy as np
import torch

from echo_routing.temporal import local_motion as m

torch.set_num_threads(1)


def test_explicit_pool_equals_adaptive_pool_values_and_gradients():
    x=torch.randn(2,3,5,7,7,dtype=torch.float64,requires_grad=True)
    y=m.DeterministicPool()(x)
    ref=torch.nn.functional.adaptive_avg_pool3d(x,(1,4,4))
    v=torch.randn_like(y)
    a=torch.autograd.grad((v*y).sum(),x,retain_graph=True)[0]
    b=torch.autograd.grad((v*ref).sum(),x)[0]
    assert torch.allclose(y,ref,atol=1e-14,rtol=0)
    assert torch.allclose(a,b,atol=1e-14,rtol=0)


def test_pair_identity_rejects_splices_repeats_and_noncontiguous_frames():
    a = ('recording', 3, 1.)
    assert m.valid_pair(a, ('recording', 6, 1.))
    assert not m.valid_pair(a, a)
    assert not m.valid_pair(a, ('other', 6, 1.))
    assert not m.valid_pair(a, ('recording', 6, 1.1))
    assert not m.valid_pair(a, ('recording', 9, 1.))


def test_sham_preserves_vector_time_trajectory_and_value_multisets():
    z = torch.arange(2*2*15*56*56).reshape(2,2,15,56,56).float()
    p = np.stack([m.tile_permutation(0, 10), m.tile_permutation(0, 11)])
    y = m.shuffle_tiles(z, torch.tensor(p))
    assert not torch.equal(y, z)
    assert torch.equal(y.flatten(3).sort(-1).values, z.flatten(3).sort(-1).values)
    # Each source tile keeps all vector/time binding and local coordinates.
    for n in range(2):
        for dest, src in enumerate(p[n]):
            dy,dx=divmod(dest,7);sy,sx=divmod(int(src),7)
            assert torch.equal(y[n,:,:,dy*8:dy*8+8,dx*8:dx*8+8],z[n,:,:,sy*8:sy*8+8,sx*8:sx*8+8])
    assert np.array_equal(p[0],m.tile_permutation(0,10))
    assert not np.array_equal(p[0],p[1])


def test_pair_bank_invalid_entries_are_exact_zero_and_marker_is_original_time():
    bank=torch.ones(2,2,56,56)
    ix=np.full((2,15),-1);ix[0,3]=1
    x=m.motion_inputs(bank,ix,np.array([3,15]))
    assert x.shape==(2,3,15,56,56)
    assert torch.equal(x[1,:2],torch.zeros_like(x[1,:2]))
    assert x[0,:2].sum()==2*56*56
    assert x[0,2,3].sum()==56*56 and x[1,2,14].sum()==56*56
    assert x[:,2].sum()==2*56*56


def test_weighted_bce_gradient_matches_explicit_repeated_samples():
    s=torch.tensor([.2,-.4,.8,-.3,-.5],dtype=torch.float64,requires_grad=True)
    v=torch.tensor([.25,.75,1.,1.,1.],dtype=torch.float64)
    seg=torch.tensor([0,0,1,2,3])
    loss,r=m.loss_from_scores(s,v,seg)
    # positive has1 and3 repeated samples; negative1/2/3 are separate runs.
    rp=(s[0]+3*s[1])/4
    ref=torch.logaddexp(torch.zeros_like(rp),-rp)+(torch.logaddexp(torch.zeros_like(s[2]),s[2])+torch.logaddexp(torch.zeros_like(s[3]),s[3])+torch.logaddexp(torch.zeros_like(s[4]),s[4]))/3
    assert torch.allclose(loss,ref,atol=1e-14,rtol=0)
    a=torch.autograd.grad(loss,s,retain_graph=True)[0];b=torch.autograd.grad(ref,s)[0]
    assert torch.allclose(a,b,atol=1e-14,rtol=0) and r.shape==(4,)


def test_zero_head_is_base_and_learned_motion_has_real_gradients():
    torch.manual_seed(17);net=m.MotionRepair().eval()
    x=torch.randn(4,3,15,56,56);e=torch.randn(4,512)
    opt=torch.optim.AdamW(net.parameters(),lr=.001)
    init=copy.deepcopy(net.state_dict())
    assert torch.equal(net(x,e),torch.full((4,),-2.))
    for _ in range(4):
        opt.zero_grad();s=net(x,e);loss=torch.nn.functional.softplus(-s[0])+torch.nn.functional.softplus(s[1:]).mean();loss.backward();opt.step()
    assert all(not torch.equal(v,init[k]) for k,v in net.state_dict().items())
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
    assert float(sum(p.grad.square().sum() for p in net.motion.parameters()).detach())>0


def test_one_way_deployment_retains_zero_negative_and_other_labels():
    b=np.array([0,1,2,2,2,3,4]);r=np.array([8,8,0,1,-1,8,8])
    assert np.array_equal(m.deploy(b,r),[0,1,2,3,2,3,4])


def test_promotion_can_delete_a_correct_boundary():
    b=np.array([2,3]);p=m.deploy(b,np.array([1.,-2.]))
    assert np.count_nonzero(b[1:]!=b[:-1])==1
    assert np.count_nonzero(p[1:]!=p[:-1])==0
