import importlib.util
from pathlib import Path
import pytest


def module():
    p=Path(__file__).resolve().parents[1]/'scripts/a63_resume_selected_risk.py'
    s=importlib.util.spec_from_file_location('resume_entry_under_test',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def test_default_inspection_never_constructs_context_or_model(monkeypatch,capsys):
    m=module()
    monkeypatch.setattr(m,'launch',lambda *a:(_ for _ in ()).throw(AssertionError('preparation cannot launch')))
    assert m.main(['--stage','observe_geometry'])==0
    assert 'preparation_only' in capsys.readouterr().out


def test_execute_without_new_raw_owner_pin_is_a_parser_refusal():
    m=module()
    with pytest.raises(SystemExit) as e:m.main(['--execute','--stage','source_FIT'])
    assert e.value.code==2


def test_child_without_inherited_guardian_ticket_cannot_dispatch(monkeypatch):
    m=module();monkeypatch.setattr(m,'dispatch',lambda *a:(_ for _ in ()).throw(AssertionError('no guardian')))
    with pytest.raises(SystemExit) as e:m.main(['--execute','--child','--stage','train_aux','--runtime-spec','spec.json','--runtime-spec-sha','a'*64,'--owner-sha','b'*64])
    assert e.value.code==2


def test_all33_stage_names_dispatch_to_concrete_qualified_drivers(monkeypatch):
    m=module()
    from echo_routing.temporal.selected_risk_resume_contract import stage_graph
    assert m.IMPLEMENTED==set(stage_graph()) and len(m.IMPLEMENTED)==33
    import ast
    tree=ast.parse(Path(m.__file__).read_text())
    imports={n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)}
    assert {'echo_routing.temporal.selected_risk_resume_readout','echo_routing.temporal.selected_risk_resume_diagnostics','echo_routing.temporal.selected_risk_resume_finish'}<=imports


def test_admission_result_versions_preserve_previous_success_and_failed_caps(tmp_path,monkeypatch):
    import scripts.a63_resume_selected_risk as entry
    import echo_routing.temporal.selected_risk_resume_runtime as runtime
    import json
    class State:
        def __init__(self,*a):pass
        def admission(self):return {'all_caps_fit':False,'remaining_complete_workload':{'prepared_data_atomic_bytes_upper':1,'formal_data_atomic_bytes_upper':2,'selected_table_atomic_bytes_upper':3}}
    monkeypatch.setattr(runtime,'RuntimeState',State)
    class C:
        runtime_spec_semantics_sha256='a'*64;owner_sha='b'*64
        def write_json(self,name,value,public=False):
            p=tmp_path/name
            with p.open('x') as f:json.dump(value,f)
            return {'path':name,'sha256':'c'*64}
    ctx=C();old=tmp_path/'admit_FIT.json';old.write_bytes(b'original successful raw admission')
    spec={'stage':'admit_FIT','metadata':{'sha256':'d'*64},'predecessors':{}}
    for mark in ['a','e']:
        ctx.runtime_spec_semantics_sha256=mark*64
        with pytest.raises(ValueError,match='full remaining'):entry._admit(ctx,spec,{'path':'spec','sha256':'f'*64})
        value=json.loads((tmp_path/('admit_FIT.'+mark*12+'.json')).read_bytes())
        assert value['passed'] is False and value['all_caps_fit'] is False
        assert old.read_bytes()==b'original successful raw admission'


def test_fit_score_phase_is_mandatory_before_calibration_and_concrete(monkeypatch,capsys):
    from echo_routing.temporal.selected_risk_resume_contract import stage_graph,_successful
    g=stage_graph()
    assert len(g)==33 and set(g['fit_risk_scores']['requires'])=={'audit_formal','train_formal','prepare_formal','coverage','seal_selectors'}
    assert {'fit_risk_scores','prepare_formal','seal_selectors'}<=set(g['calibrate']['requires'])
    with pytest.raises(ValueError):_successful('fit_risk_scores',{'passed':True,'complete':False})
    m=module();monkeypatch.setattr(m,'launch',lambda *a:(_ for _ in ()).throw(AssertionError('inspection cannot launch')))
    assert m.main(['--stage','fit_risk_scores'])==0
    assert 'execution_authorized' in capsys.readouterr().out
    import ast
    tree=ast.parse(Path(m.__file__).read_text())
    assert any(isinstance(n,ast.Constant) and n.value=='fit_risk_scores' for n in ast.walk(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='dispatch')))
