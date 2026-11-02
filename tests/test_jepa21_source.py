import importlib
import pytest
import torch

def api():return importlib.import_module('echo_routing.temporal.jepa21_source')

def test_strict_all_key_import_with_no_silent_missing_or_extra():
    a=api();m=torch.nn.Linear(3,2);state={'module.backbone.'+k:v.clone() for k,v in m.state_dict().items()};got=a.strict_target_state({'target_encoder':state},m);assert set(got)==set(m.state_dict())
    bad=dict(state);bad.pop('module.backbone.bias')
    with pytest.raises(ValueError):a.strict_target_state({'target_encoder':bad},m)
    bad=dict(state);bad['module.backbone.extra']=torch.ones(1)
    with pytest.raises(ValueError):a.strict_target_state({'target_encoder':bad},m)

def test_full_video_trainables_explicitly_exclude_image_and_unused_norms():
    a=api();names=['img_mod_embed','video_mod_embed','patch_embed_img.proj.weight','patch_embed.proj.weight','norms_block.0.weight','norms_block.2.bias','norms_block.3.weight','blocks.0.attn.qkv.weight']
    assert a.inactive_video_names(names)==['img_mod_embed','patch_embed_img.proj.weight','norms_block.0.weight','norms_block.2.bias']
