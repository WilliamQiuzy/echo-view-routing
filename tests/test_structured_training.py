import importlib,copy
import pytest
import torch
from echo_routing.temporal.structured_evidence import StructuredEvidence
from echo_routing.temporal.dense_durable import atomic_torch_save

def equality(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equality(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return len(a)==len(b) and all(equality(x,y) for x,y in zip(a,b))
    return a==b

def test_actual_optimizer_binding_repeat_and_disk_resume_keep_exact_model_and_moments(tmp_path):
    m=importlib.import_module('echo_routing.temporal.structured_training');torch.manual_seed(58);batch=[(torch.softmax(torch.randn(n,5),1),torch.softmax(torch.randn(n,5),1),torch.arange(n)%5) for n in [9,15]]
    def run(stop,z=None):
        model=StructuredEvidence('D2',2026110358);opt=m.optimizer(model);start=0 if z is None else m.restore(model,opt,z,'fixed-input')
        for u in range(start,stop):m.step(model,opt,batch,u)
        return m.checkpoint(model,opt,stop,'fixed-input')
    expected=run(4);repeat=run(4);assert equality(expected['model'],repeat['model']) and equality(expected['optimizer'],repeat['optimizer'])
    path=tmp_path/'checkpoint.pt';atomic_torch_save(run(2),path);resumed=run(4,torch.load(path,weights_only=True));assert equality(expected['model'],resumed['model']) and equality(expected['optimizer'],resumed['optimizer'])
    model=StructuredEvidence('D2',2026110358);opt=m.optimizer(model)
    with pytest.raises(ValueError):m.restore(model,opt,expected,'changed-input')
    assert m.optimizer_binding(model,opt)
    model=StructuredEvidence('D0',58);opt=m.optimizer(model);assert all(p is not model.duration for g in opt.param_groups for p in g['params']);before=model.duration.detach().clone();m.step(model,opt,batch,0);assert torch.equal(before,model.duration)

def test_changed_optimizer_recipe_fails_binding():
    m=importlib.import_module('echo_routing.temporal.structured_training')
    model=StructuredEvidence('D2',58);opt=m.optimizer(model)
    opt.param_groups[0]['weight_decay']=.05
    assert not m.optimizer_binding(model,opt)

@pytest.mark.parametrize('fault',['cursor','recipe'])
def test_checkpoint_counter_and_optimizer_recipe_fail_closed(fault):
    m=importlib.import_module('echo_routing.temporal.structured_training')
    model=StructuredEvidence('D2',58);opt=m.optimizer(model)
    batch=[(torch.softmax(torch.randn(n,5),1),torch.softmax(torch.randn(n,5),1),torch.arange(n)%5) for n in [7,11]]
    m.step(model,opt,batch,0);payload=m.checkpoint(model,opt,1,'fixed')
    wrong=copy.deepcopy(payload)
    if fault=='cursor':
        wrong['update']=2
        # Even a matching two-update warmup LR cannot disguise one actual Adam step.
        wrong['optimizer']['param_groups'][0]['lr']=2e-5
    else:wrong['optimizer']['param_groups'][0]['betas']=(.8,.9)
    fresh=StructuredEvidence('D2',58);freshopt=m.optimizer(fresh)
    with pytest.raises(ValueError,match='cursor|counter|optimizer'):
        m.restore(fresh,freshopt,wrong,'fixed')
