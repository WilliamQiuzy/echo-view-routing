"""Independent optimizer identity checks with small tensor fixtures only."""
import importlib
import pytest
import torch

def fixture():
    names=['head.weight','encoder.video.weight','encoder.norms_block.0.weight','encoder.img_mod_embed']
    initial={k:torch.ones(2) for k in names}
    state={k:v.clone() for k,v in initial.items()}
    for k in names[:-1]:state[k].add_(.1)
    groups=[{'params':[0],'name':'head','base_lr':.001,'lr':.0005,'betas':(.9,.999),'eps':1e-8,'weight_decay':.05},{'params':[1,2],'name':'active_video','base_lr':3e-5,'lr':1.5e-5,'betas':(.9,.999),'eps':1e-8,'weight_decay':.05}]
    moments={i:{'step':torch.tensor(4.),'exp_avg':torch.ones(2),'exp_avg_sq':torch.ones(2)} for i in range(3)}
    return state,initial,{'param_groups':groups,'state':moments}

def test_optimizer_identity_includes_new_norms_excludes_image_and_rejects_corruption():
    a=importlib.import_module('echo_routing.temporal.multidepth_audit');s,b,o=fixture()
    result=a.verify_optimizer(s,b,o,4,True,['img_mod_embed'],.5)
    assert result['active_video_optimizer_tensors']==2 and result['changed_video_tensors']==2
    o['state'][2]['exp_avg_sq'][0]=-1
    with pytest.raises(AssertionError):a.verify_optimizer(s,b,o,4,True,['img_mod_embed'],.5)

def test_frozen_encoder_and_parameter_groups_are_checked_by_exact_tensor_binding():
    a=importlib.import_module('echo_routing.temporal.multidepth_audit');s,b,o=fixture();o['param_groups'][1]['params']=[2,1];o['state'][1]['exp_avg_sq']=torch.ones(3)
    with pytest.raises(AssertionError):a.verify_optimizer(s,b,o,4,True,['img_mod_embed'],.5)
    s,b,o=fixture();o['param_groups']=o['param_groups'][:1];o['state']={0:o['state'][0]}
    with pytest.raises(AssertionError):a.verify_optimizer(s,b,o,4,False,['img_mod_embed'],.5)
