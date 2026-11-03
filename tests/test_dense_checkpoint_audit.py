import importlib
import pytest
import torch

def test_final_joint_checkpoint_binds_all_latest_tensors_not_only_its_own_filehash():
    a=importlib.import_module('echo_routing.temporal.dense_checkpoint_audit');state={'encoder.v':torch.ones(2),'head.w':torch.ones(3)};frozen={'model':{k:v.clone() for k,v in state.items()},'feature_contract':{'kind':'dense'}}
    assert a.compare_final_checkpoint(frozen,state,True,{'kind':'dense'})=={'tensors':2,'parameters':5,'all_weights_equal_latest_optimizer_model':True}
    frozen['model']['head.w'][0]+=1
    with pytest.raises(AssertionError):a.compare_final_checkpoint(frozen,state,True,{'kind':'dense'})

def test_frozen_head_checkpoint_requires_exact_names_values_and_feature_identity():
    a=importlib.import_module('echo_routing.temporal.dense_checkpoint_audit');state={'encoder.v':torch.ones(2),'head.w':torch.ones(3),'head.b':torch.zeros(1)};frozen={'head':{'w':torch.ones(3),'b':torch.zeros(1)},'feature_contract':{'kind':'fourdepth'}}
    assert a.compare_final_checkpoint(frozen,state,False,{'kind':'fourdepth'})['tensors']==2
    with pytest.raises(AssertionError):a.compare_final_checkpoint(frozen,state,False,{'kind':'finalonly'})
    del frozen['head']['b']
    with pytest.raises(AssertionError):a.compare_final_checkpoint(frozen,state,False,{'kind':'fourdepth'})
