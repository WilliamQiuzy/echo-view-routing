"""Bind evaluated milestone tensor bytes to the audited final optimizer model."""
import torch

def compare_final_checkpoint(checkpoint,latest_state,joint,feature_contract=None):
    if feature_contract is not None:assert checkpoint.get('feature_contract')==feature_contract
    expected=latest_state if joint else {k[5:]:v for k,v in latest_state.items() if k.startswith('head.')}
    actual=checkpoint['model'] if joint else checkpoint['head'];assert set(actual)==set(expected)
    assert all(actual[k].shape==v.shape and actual[k].dtype==v.dtype and torch.equal(actual[k],v) for k,v in expected.items())
    return {'tensors':len(expected),'parameters':sum(v.numel() for v in expected.values()),'all_weights_equal_latest_optimizer_model':True}
