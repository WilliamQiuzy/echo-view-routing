from pathlib import Path
import importlib.util
import numpy as np
import pytest
import torch


def api():
    p=Path(__file__).parents[1]/'echo_routing/temporal/pw_transfer.py'
    assert p.exists(), 'clinical-purpose transfer implementation missing'
    s=importlib.util.spec_from_file_location('pw_transfer',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def test_coefficients_match_independent_class_balanced_reference():
    m=api();s=torch.tensor([-.4,.7,-2.,1.,-.3],dtype=torch.float64)
    v=torch.tensor([.25,.75,1.,1.,1.],dtype=torch.float64);seg=torch.tensor([0,0,1,2,3])
    loss,r,g=m.leaf_coefficients(s,v,seg)
    rr=np.array([.25*(-.4)+.75*.7,-2.,1.,-.3]);dr=1/(1+np.exp(-rr));dr[0]-=1;dr[1:]/=3
    assert np.allclose(r.numpy(),rr,atol=1e-12)
    assert np.allclose(g.numpy(),v.numpy()*dr[seg.numpy()],atol=1e-12)
    assert float(loss)==pytest.approx(np.logaddexp(0,-rr[0])+np.logaddexp(0,rr[1:]).mean())


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__();self.backbone=torch.nn.Conv2d(3,2,1);self.head=torch.nn.Linear(2,1)
        self.register_buffer('frozen_reference',torch.tensor([4.]))
    def forward(self,x):return self.head(self.backbone(x).mean((2,3))).flatten()


def test_two_real_passes_equal_retained_graph_parameter_gradients():
    m=api();torch.manual_seed(41);a=Toy();b=Toy();b.load_state_dict(a.state_dict())
    frames=np.random.default_rng(4).integers(0,256,(5,3,3,3),dtype=np.uint8)
    rows=np.arange(5);v=torch.tensor([.3,.7,1.,1.,1.]);seg=torch.tensor([0,0,1,2,3])
    m.two_pass(a,frames,rows,v,seg,torch.device('cpu'))
    scores=torch.cat([b(m.normalise(torch.from_numpy(frames[i:i+1]))) for i in rows])
    r=torch.stack([(scores[seg==k]*v[seg==k]).sum() for k in range(4)])
    (torch.nn.functional.softplus(-r[0])+torch.nn.functional.softplus(r[1:]).mean()).backward()
    for (_,pa),(_,pb) in zip(a.named_parameters(),b.named_parameters()):assert torch.allclose(pa.grad,pb.grad,atol=1e-6,rtol=1e-5)


def test_bundle_contains_all_mutable_tensors_and_complete_state_identity():
    m=api();torch.manual_seed(7);base=Toy();target=Toy();target.load_state_dict(base.state_dict())
    with torch.no_grad():target.backbone.weight.add_(.2);target.head.bias.sub_(2)
    bundle=m.make_bundle(target,'base-sha',{'step':600})
    assert set(bundle['trainable'])=={n for n,p in target.named_parameters() if p.requires_grad}
    restored=m.restore_bundle(base,bundle,'base-sha')
    assert m.state_sha(restored.state_dict())==m.state_sha(target.state_dict())
    with pytest.raises(ValueError):m.restore_bundle(Toy(),bundle,'wrong-sha')
    corrupt=Toy();corrupt.load_state_dict(restored.state_dict());corrupt.frozen_reference.add_(1)
    with pytest.raises(ValueError):m.restore_bundle(corrupt,bundle,'base-sha')
