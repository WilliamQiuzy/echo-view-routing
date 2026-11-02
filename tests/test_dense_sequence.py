import copy,importlib
import numpy as np
import torch
import pytest

torch.set_num_threads(1)
def api():return importlib.import_module('echo_routing.temporal.dense_sequence')


def test_tubelet_grid_pair_expansion_and_masked_original_position_ce():
    a=api();x=torch.arange(4*8*196*3,dtype=torch.float32).reshape(4,1568,3)
    out=a.spatial_time_features(x,1)
    assert out.shape==(1,3,32)
    assert torch.equal(out[0,:,8],x.reshape(4,8,196,3)[1,0].mean(0))
    z=torch.zeros(1,5,32,requires_grad=True);z.data[0,0,0]=2
    labels=torch.full((1,64),99,dtype=torch.long);labels[0,:2]=torch.tensor([0,1]);mask=torch.arange(64)[None]<2
    loss=a.dense_ce([z,z],labels,mask)
    expected=.5*(torch.nn.functional.cross_entropy(z[:,:,:1].transpose(1,2).reshape(1,5),torch.tensor([0]))+torch.nn.functional.cross_entropy(z[:,:,:1].transpose(1,2).reshape(1,5),torch.tensor([1])))
    assert torch.allclose(loss,expected)
    loss.backward();assert z.grad[:,:,1:].abs().sum()==0
    assert torch.equal(a.expand_logits(z)[:,:,0],a.expand_logits(z)[:,:,1])


def test_chunk_footprints_tail_padding_and_probability_stitch():
    a=api();assert a.chunk_starts(65)==[0,32,64]
    idx,valid=a.chunk_positions(65,64);assert valid.sum()==1 and np.all(idx==64)
    probs=np.zeros((3,64,5));probs[0,:,0]=1;probs[1,:,1]=1;probs[2,:,2]=1
    p=a.stitch_probabilities(probs,65)
    assert p.shape==(65,5) and np.allclose(p.sum(1),1)
    assert np.allclose(p[32],[32/33,1/33,0,0,0])
    assert np.allclose(p[64],[0,32/33,1/33,0,0])
    with pytest.raises(ValueError):a.stitch_probabilities(probs,65,starts=[0,31,64])


def test_training_chunk_can_center_a_true_change_off_the_inference_grid():
    idx,valid=api().chunk_positions(100,3)
    assert idx[0]==3 and idx[-1]==66 and valid.all()


def test_tcn_mask_blocks_arbitrary_padded_FEATURE_values_not_raw_pixels():
    a=api();head=a.TwoStageTCN(3,channels=4).eval();x=torch.randn(2,3,32);mask=torch.arange(32)[None].repeat(2,1)<torch.tensor([7,19])[:,None];changed=x.clone();changed.masked_fill_(~mask[:,None],1e6)
    left,right=head(x,mask),head(changed,mask)
    assert len(left)==2 and all(v.shape==(2,5,32) for v in left)
    assert all(torch.equal(l,r) and torch.count_nonzero(l.masked_select(~mask[:,None].expand_as(l)))==0 for l,r in zip(left,right))


class ToyDenseEncoder(torch.nn.Module):
    def __init__(self):super().__init__();self.fc=torch.nn.Linear(3,3)
    def forward(self,x):
        # Only a shape/gradient fixture; no medical image representation claim.
        t=x.mean((3,4)).transpose(1,2).reshape(len(x),8,2,3).mean(2)
        return self.fc(t)[:,:,None].expand(-1,-1,196,-1).reshape(len(x),1568,3)


def test_frozen_joint_same_initial_predictions_and_real_backbone_gradient():
    a=api();encoder=ToyDenseEncoder();f=a.DenseSequence(copy.deepcopy(encoder),'F',dim=3,channels=4,bf16=False,seed=57).eval();j=a.DenseSequence(copy.deepcopy(encoder),'J',dim=3,channels=4,bf16=False,seed=57).eval();u8=torch.randint(0,256,(2,64,2,2,3),dtype=torch.uint8);mask=torch.ones(2,64,dtype=torch.bool);labels=torch.arange(128).reshape(2,64)%5
    assert all(torch.equal(l,r) for l,r in zip(f(u8,mask),j(u8,mask)))
    a.dense_ce(f(u8,mask),labels,mask).backward();a.dense_ce(j(u8,mask),labels,mask).backward()
    assert all(p.grad is None for p in f.encoder.parameters())
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for p in j.encoder.parameters())
    assert f(u8,mask)[-1].shape==(2,5,32)


def test_joint_dropout_optimizer_rng_resume_matches_uninterrupted():
    a=api();torch.manual_seed(123);enc=ToyDenseEncoder();u=torch.randint(0,256,(2,64,2,2,3),dtype=torch.uint8);mask=torch.ones(2,64,dtype=torch.bool);y=torch.arange(128).reshape(2,64)%5
    def build():
        m=a.DenseSequence(copy.deepcopy(enc),'J',dim=3,channels=4,bf16=False,seed=57).train();return m,torch.optim.AdamW(m.parameters(),lr=.001)
    def step(m,o):o.zero_grad();a.dense_ce(m(u,mask),y,mask).backward();o.step()
    m,o=build();rng=torch.get_rng_state();
    for _ in range(4):step(m,o)
    expected=copy.deepcopy(m.state_dict());n,p=build();torch.set_rng_state(rng)
    for _ in range(2):step(n,p)
    st=copy.deepcopy(n.state_dict());opt=copy.deepcopy(p.state_dict());state=torch.get_rng_state();n,p=build();n.load_state_dict(st);p.load_state_dict(opt);torch.set_rng_state(state)
    for _ in range(2):step(n,p)
    assert all(torch.equal(v,n.state_dict()[k]) for k,v in expected.items())
