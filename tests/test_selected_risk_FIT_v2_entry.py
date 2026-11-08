import ast
import copy
import importlib
from pathlib import Path
import sys
import pytest

ROOT=Path(__file__).resolve().parents[1]

def _producer(path,new):
    tree=ast.parse((ROOT/path).read_text());f=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='produce')
    if new:
        assert isinstance(f.body[0],ast.If) and ast.unparse(f.body[0].test)=="phase != 'FIT'"
        f.body.pop(0)
    for node in ast.walk(f):
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='freeze' for t in node.targets):node.value=ast.Constant(value='checked-stage-freeze')
        if isinstance(node,ast.Dict):
            for i,key in enumerate(node.keys):
                if isinstance(key,ast.Constant) and key.value=='source_production_stage':node.values[i]=ast.Constant(value='literal-actual-stage')
    return ast.dump(f,include_attributes=False)

def test_FIT_producer_body_preserves_every_original_neural_cache_and_counter_operation():
    assert _producer('scripts/cache_selected_risk.py',False)==_producer('scripts/a63_cache_selected_risk_FIT_v2.py',True)

def test_guardian_uses_original_owned_engine_with_GPU_true_only(monkeypatch):
    m=importlib.import_module('scripts.a63_supervise_selected_risk_FIT_v2');seen=[]
    monkeypatch.setattr(m,'authorize',lambda a:True);monkeypatch.setattr(m,'verify_shared_freeze',lambda stage:{})
    monkeypatch.setattr(m,'actual_receipts',lambda:[]);monkeypatch.setattr(m,'stage_exit_status',lambda r:0)
    monkeypatch.setattr(m,'execute',lambda *a:seen.append(a) or {'test_only':True})
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0')
    monkeypatch.setattr(sys,'argv',['guard','--execute','--seconds','4500',m.STAGE,'--','python',m.ENTRY,'--execute','--phase','FIT'])
    assert m.main()==0 and len(seen)==1 and seen[0][-1] is True and seen[0][-2]==4500
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    with pytest.raises(ValueError):m.main()
    assert len(seen)==1

def test_guardian_rejects_over4500_before_engine(monkeypatch):
    m=importlib.import_module('scripts.a63_supervise_selected_risk_FIT_v2');monkeypatch.setattr(m,'authorize',lambda a:True)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0');monkeypatch.setattr(sys,'argv',['guard','--execute','--seconds','4501',m.STAGE,'--','python',m.ENTRY,'--execute','--phase','FIT'])
    monkeypatch.setattr(m,'execute',lambda *a:pytest.fail('forbidden child launched'))
    with pytest.raises(ValueError):m.main()

def test_freezer_checks_actual_admission_before_any_proof_or_contract_write(monkeypatch,tmp_path):
    m=importlib.import_module('scripts.a63_freeze_selected_risk_FIT_v2');monkeypatch.setattr(m,'RUN',tmp_path)
    def bad():raise ValueError('actual v3 not mirrored')
    monkeypatch.setattr(m,'verify_admission',bad)
    monkeypatch.setattr(m,'verify_freeze',lambda stage:pytest.fail('proof read/build reached before admission guard'))
    with pytest.raises(ValueError,match='actual v3 not mirrored'):m.make(m.STAGE)
    assert not list(tmp_path.rglob('*'))
