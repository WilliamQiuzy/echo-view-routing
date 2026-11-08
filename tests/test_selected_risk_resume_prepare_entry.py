import importlib.util
from pathlib import Path
import pytest


def module():
    path=Path(__file__).resolve().parents[1]/'scripts/a63_prepare_resume_selected_risk.py';sp=importlib.util.spec_from_file_location('prepentry',path);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m


def test_default_inspection_and_missing_raw_owner_never_call_a_mutation(capsys):
    m=module();assert m.main([])==0
    assert '"execution_authorized": false' in capsys.readouterr().out
    with pytest.raises(SystemExit) as ex:m.main(['--execute','--make-freeze'])
    assert ex.value.code==2


def test_receipt_only_collects_expired_unknown_prefix_without_dispatch_or_claim(tmp_path,monkeypatch):
    import importlib.util,json
    fixture=Path(__file__).with_name('test_selected_risk_resume_context.py');sp=importlib.util.spec_from_file_location('fixture',fixture);h=importlib.util.module_from_spec(sp);sp.loader.exec_module(h);ctx,put=h.setup(tmp_path)
    m=module();spec=ctx.read(ctx._bound_runtime_spec_pin);spec.update(prep_ledger=None,local_storage=None);pin=put('adminspec.json',spec)
    from echo_routing.temporal.selected_risk_resume_contract import RUN,PUB
    (tmp_path/RUN/'caller_intents').mkdir(parents=True)
    worker=put(PUB+'/failed_unknown.json',{'receipt_id':'unknown','stage':'train_formal','unresolved_worker_uncertainty':True,'measured_CPU_core_seconds':None,'model_pid':None})
    (tmp_path/RUN/'processes.jsonl').write_text(json.dumps({'receipt_id':'unknown','receipt':worker})+'\n')
    trace=tmp_path/RUN/'models/seed2026110461/C0/physical_updates.jsonl';trace.parent.mkdir(parents=True);trace.write_text('{"u":0}\n')
    monkeypatch.setattr('time.time',lambda:ctx.owner['deadline_epoch']+1)
    monkeypatch.setattr('subprocess.Popen',lambda *a,**k:(_ for _ in ()).throw(AssertionError('no child allowed')))
    result=m.collect_receipts_only(tmp_path,pin,ctx.owner_sha);v=ctx.read(result)
    assert v['formal_optimizer_updates_lower_bound']==1 and v['resources']['actual_caps_fit'] is None
    assert v['resources']['actual_worker_receipts'][0]['measured_CPU_core_seconds'] is None
    assert v['execution_authority_reopened'] is False and v['overall_complete'] is False
