import importlib
import pytest

def api():return importlib.import_module('echo_routing.temporal.dense_durable')
def test_checkpoint_cursor_rejects_gaps_and_preserves_replay_accounting():
    a=api();rows=[{'u':u,'attempt':1} for u in range(5)]
    out=a.resume_accounting(rows,3)
    assert out=={'committed_physical_updates':5,'checkpoint_next_update':3,'invalidated_after_checkpoint':2}
    with pytest.raises(ValueError):a.resume_accounting(rows,6)
    with pytest.raises(ValueError):a.resume_accounting([{'u':0,'attempt':1},{'u':2,'attempt':1}],0)
def test_atomic_checkpoint_writes_do_not_replace_good_state_on_failure(tmp_path):
    a=api();p=tmp_path/'checkpoint.pt';a.atomic_torch_save({'step':3},p)
    import torch
    assert torch.load(p,weights_only=True)['step']==3
    class Bad:
        def __reduce__(self):raise RuntimeError('intentional save failure')
    with pytest.raises(RuntimeError):a.atomic_torch_save({'bad':Bad()},p)
    assert torch.load(p,weights_only=True)['step']==3
