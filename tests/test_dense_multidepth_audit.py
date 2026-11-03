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
    return state,{k[8:]:v for k,v in initial.items() if k.startswith('encoder.')},{'param_groups':groups,'state':moments}

def test_optimizer_identity_includes_new_norms_excludes_image_and_rejects_corruption():
    a=importlib.import_module('echo_routing.temporal.multidepth_audit');s,b,o=fixture()
    result=a.verify_optimizer(s,b,o,4,True,['img_mod_embed'],.5,{'head.weight':(2,)})
    assert result['active_video_optimizer_tensors']==2 and result['changed_video_tensors']==2
    o['state'][2]['exp_avg_sq'][0]=-1
    with pytest.raises(AssertionError):a.verify_optimizer(s,b,o,4,True,['img_mod_embed'],.5,{'head.weight':(2,)})

def test_frozen_encoder_and_parameter_groups_are_checked_by_exact_tensor_binding():
    a=importlib.import_module('echo_routing.temporal.multidepth_audit');s,b,o=fixture();o['param_groups'][1]['params']=[2,1];o['state'][1]['exp_avg_sq']=torch.ones(3)
    with pytest.raises(AssertionError):a.verify_optimizer(s,b,o,4,True,['img_mod_embed'],.5,{'head.weight':(2,)})
    s,b,o=fixture();o['param_groups']=o['param_groups'][:1];o['state']={0:o['state'][0]}
    with pytest.raises(AssertionError):a.verify_optimizer(s,b,o,4,False,['img_mod_embed'],.5,{'head.weight':(2,)})

def test_missing_frozen_source_key_and_extra_or_reshaped_head_are_rejected():
    a=importlib.import_module('echo_routing.temporal.multidepth_audit')
    for kind in ['missing_image','extra_head','wrong_head_shape']:
        s,b,o=fixture()
        if kind=='missing_image':del s['encoder.img_mod_embed']
        elif kind=='extra_head':s['head.extra']=torch.ones(2)
        else:s['head.weight']=torch.ones(3)
        with pytest.raises(AssertionError):a.verify_optimizer(s,b,o,4,True,['img_mod_embed'],.5,{'head.weight':(2,)})

def test_MD_decoder_contract_binds_manifest_feature_kind_and_exact_checkpoint_coverage():
    import copy
    a=importlib.import_module('echo_routing.temporal.multidepth_audit');feature={'kind':'fourdepth'};models={'feature_contract':feature,'outer_rows':[{'sha256':'one','feature_contract':feature},{'sha256':'two','feature_contract':feature}]};choices={'feature_contract':feature,'model_manifest_sha256':'manifest','choices_by_checkpoint_sha256':{'one':{'params':{}},'two':{'params':{}}}}
    a.verify_decoder_binding(models,choices,'manifest',feature)
    for kind in ['manifest','feature','missing','extra']:
        bad=copy.deepcopy(choices)
        if kind=='manifest':bad['model_manifest_sha256']='swapped'
        elif kind=='feature':bad['feature_contract']={'kind':'finalonly'}
        elif kind=='missing':del bad['choices_by_checkpoint_sha256']['two']
        else:bad['choices_by_checkpoint_sha256']['extra']={'params':{}}
        with pytest.raises(AssertionError):a.verify_decoder_binding(models,bad,'manifest',feature)
