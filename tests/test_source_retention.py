import copy
import importlib
import numpy as np
import pytest
import torch
from echo_routing.temporal import direct_view_training as durable


def api():
    return importlib.import_module('echo_routing.temporal.source_retention')


def sequences():
    return [{'id':f'fit{c}_{j}', 'kind':'native','labels':np.full(20,c),
             'refs':[(f'fit{c}_{j}',i,1.) for i in range(20)]}
            for c in range(5) for j in range(3)]


def test_allowlist_rejects_protected_and_mislabeled_references():
    a=api(); seq=sequences(); allowed={q['id']:int(q['labels'][0]) for q in seq}
    assert len(a.native_allowlist(seq,allowed))==15
    bad=copy.deepcopy(seq); bad[0]['refs'][2]=('protected_test',2,1.)
    with pytest.raises(ValueError,match='allowlist'): a.native_allowlist(bad,allowed)
    bad=copy.deepcopy(seq); bad[0]['labels'][3]=4
    with pytest.raises(ValueError,match='label'): a.native_allowlist(bad,allowed)
    with pytest.raises(ValueError,match='allowlist'): a.native_allowlist(seq,{})


def test_plans_cover_source_and_replay_with_exact_class_exposure():
    a=api(); seq=sequences(); source=np.array([0]*11+[1]*9)
    sp,rp=a.exposure_plans(source,seq,updates=20)
    assert sp.shape==rp.shape==(20,8)
    assert set(sp.ravel())==set(range(20))
    assert np.all(source[sp].sum(1)==4)
    labels=np.array([q['labels'][0] for q in seq]); assert np.bincount(labels[rp].ravel(),minlength=5).tolist()==[32]*5
    assert set(rp.ravel())==set(range(15))
    sp2,rp2=a.exposure_plans(source,seq,updates=20)
    assert np.array_equal(sp,sp2) and np.array_equal(rp,rp2)


def test_zero_replay_backward_and_separate_loss_normalization():
    a=api(); torch.manual_seed(3); x=torch.randn(16,4); y=torch.arange(16)%5
    m=torch.nn.Linear(4,5); initial=copy.deepcopy(m.state_dict()); calls=[]
    def forward(i):
        z=m(x[i:i+1]);z.register_hook(lambda g:calls.append(g.detach().clone()));return z
    rec=a.accumulate(forward,y[:8],y[8:],0.)
    g=[p.grad.clone() for p in m.parameters()]
    assert rec['physical_forwards']==rec['physical_backwards']==16 and len(calls)==16
    assert all(torch.count_nonzero(v)==0 for v in calls[8:])
    ref=torch.nn.Linear(4,5);ref.load_state_dict(initial)
    loss=torch.nn.functional.cross_entropy(ref(x[:8]),y[:8]);loss.backward()
    assert all(torch.allclose(p.grad,q,atol=2e-7) for p,q in zip(ref.parameters(),g))
    assert rec['source_loss']==pytest.approx(float(loss.detach()),abs=2e-7)
    m.zero_grad();calls.clear();a.accumulate(forward,y[:8],y[8:],1.)
    ref.zero_grad();both=torch.nn.functional.cross_entropy(ref(x[:8]),y[:8])+torch.nn.functional.cross_entropy(ref(x[8:]),y[8:]);both.backward()
    assert all(torch.allclose(p.grad,q.grad,atol=2e-7) for p,q in zip(m.parameters(),ref.parameters()))
    assert all(torch.count_nonzero(v)>0 for v in calls)


def test_ambiguous_optimizer_resume_is_rejected(tmp_path):
    durable.write_json(tmp_path/'update_guard.json',{'pending':1,'last_committed':0})
    with pytest.raises(RuntimeError,match='possibly executed'): durable.validate_recovery(tmp_path,1,1)
