import hashlib
import json
import os
import pytest


def test_bootstrap_does_not_accept_preparation_as_authority(tmp_path):
    from echo_routing.temporal.selected_risk_resume_runtime import bootstrap
    p=tmp_path/'spec.json';p.write_text(json.dumps({'preparation_only':True}))
    with pytest.raises(ValueError):bootstrap(tmp_path,{'path':'spec.json','sha256':hashlib.sha256(p.read_bytes()).hexdigest()},'a'*64)
    assert sorted(x.name for x in tmp_path.iterdir())==['spec.json']


def test_final_prep_cutoff_is_owner_bound_before_budget_rows_are_compiled(tmp_path):
    from echo_routing.temporal.selected_risk_resume_runtime import RuntimeState
    class C:
        baseline_pin={'path':'baseline','sha256':'a'*64}
        owner={}
        def read(self,p):return {'preparation_ledger_pin':{'path':'real-final-ledger','sha256':'b'*64}}
    c=C();s=RuntimeState(c,{'stage':'admit_FIT','prep_ledger':{'path':'empty-ledger','sha256':'c'*64}},None)
    with pytest.raises(ValueError):s.spent()


def test_receipt_after_deadline_preserves_failed_actual_os_without_reopening_work(tmp_path,monkeypatch):
    import importlib.util
    from pathlib import Path
    path=Path(__file__).with_name('test_selected_risk_resume_context.py');sp=importlib.util.spec_from_file_location('context_fixture_helper',path);helper=importlib.util.module_from_spec(sp);sp.loader.exec_module(helper)
    from echo_routing.temporal.selected_risk_resume_runtime import RuntimeState
    ctx,_=helper.setup(tmp_path);s=RuntimeState(ctx,{'stage':'observe_geometry'},{'path':'spec','sha256':'a'*64})
    intent=ctx.write_json('launches/observe_geometry.nonce.intent.json',{'nonce':'nonce','guardian_pid':os.getpid(),'runtime_authorization_sha256':ctx.owner_sha})
    ctx._clock=lambda:ctx.owner['deadline_epoch']+1
    monkeypatch.setattr('echo_routing.temporal.selected_risk_resume_runtime.time.time',lambda:ctx.owner['deadline_epoch']+1)
    with pytest.raises(ValueError):ctx.write_json('no-more-work.json',{})
    row={'receipt_id':'resume:observe_geometry:nonce','supervisor_pid':os.getpid(),'stage':'observe_geometry','nonce':'nonce','model_pid':999,'actual_child_directly_waited':True,'observed_worker_exit_code':7,'stage_exit_status':1}
    s.caller_ticket=lambda:({'path':'caller','sha256':'e'*64},{})
    result=s.record(row,intent)
    assert result['observed_worker_exit_code']==7 and result['stage_exit_status']==1
    assert result['window_expired_at_receipt'] is True
    assert ctx.read(result['public_receipt'])['source_intent_sha256']==intent['sha256']
    assert not (tmp_path/'runs/a63_selected_action_risk_resume_v1/no-more-work.json').exists()


def test_malformed_receipt_is_rejected_before_any_publication(tmp_path):
    import importlib.util
    from pathlib import Path
    path=Path(__file__).with_name('test_selected_risk_resume_context.py');sp=importlib.util.spec_from_file_location('context_fixture_helper',path);helper=importlib.util.module_from_spec(sp);sp.loader.exec_module(helper)
    from echo_routing.temporal.selected_risk_resume_runtime import RuntimeState
    ctx,_=helper.setup(tmp_path);s=RuntimeState(ctx,{'stage':'observe_geometry'},{'path':'spec','sha256':'a'*64})
    row={'supervisor_pid':os.getpid(),'stage':'observe_geometry','nonce':'nonce','model_pid':None,'observed_worker_exit_code':None,'stage_exit_status':1}
    with pytest.raises(ValueError):s.record(row,None)
    assert not (tmp_path/'docs/results/2026-11-05_a63_resume_execution').exists()


def test_whole_caller_is_only_charge_and_exit_allowance_is_separate(monkeypatch):
    from echo_routing.temporal.selected_risk_resume_runtime import RuntimeState
    pin={'path':'prep','sha256':'b'*64}
    class C:
        owner={};baseline_pin={'path':'baseline','sha256':'a'*64}
        def read(self,p):return {'preparation_ledger_pin':pin} if p==self.baseline_pin else {'file_pins':{},'receipts':[]}
    s=RuntimeState(C(),{'stage':'admit_FIT','prep_ledger':pin},None)
    row={'receipt_id':'whole','finished_epoch':5.,'caller_exit_CPU_allowance_seconds':1.,'measured_CPU_core_seconds':2.,'covered_charge_groups':['guardian']}
    s._worker_rows=lambda:[row]
    monkeypatch.setattr('echo_routing.temporal.selected_risk_resume_budget.totals',lambda base,prep,workers:(prep,workers))
    prep,workers=s.spent()
    assert workers==[row] and len(prep)==1
    assert prep[0]['measured_CPU_core_seconds'] is None and prep[0]['CPU_accounting_allowance_seconds']==1.
    row['unresolved_worker_uncertainty']=True
    with pytest.raises(ValueError):s.spent()


def test_guardian_requires_actual_whole_caller_parent_ticket(monkeypatch):
    from echo_routing.temporal.selected_risk_resume_runtime import RuntimeState
    from echo_routing.temporal.selected_risk_resume_contract import RUN
    spec={'stage':'admit_FIT','expected_boot_id':'boot'};sp={'path':'s','sha256':'c'*64}
    class C:
        owner={};owner_sha='a'*64
        def read(self,p):return {'caller_pid':os.getppid(),'caller_start_ticks':42,'runtime_authorization_sha256':self.owner_sha,'runtime_spec':sp,'stage':'admit_FIT','boot_id':'boot'}
    s=RuntimeState(C(),spec,sp)
    monkeypatch.delenv('A63_RESUME_CALLER_INTENT_PATH',raising=False)
    with pytest.raises(ValueError):s.caller_ticket()
    monkeypatch.setenv('A63_RESUME_CALLER_INTENT_PATH',RUN+'/caller_intents/own.json');monkeypatch.setenv('A63_RESUME_CALLER_INTENT_SHA','d'*64)
    monkeypatch.setattr('echo_routing.temporal.selected_risk_resume_guardian.start_ticks',lambda pid:42)
    assert s.caller_ticket()[0]['sha256']=='d'*64


def test_broken_canonical_ledger_is_never_an_empty_account(tmp_path):
    from echo_routing.temporal.selected_risk_resume_runtime import RuntimeState
    from echo_routing.temporal.selected_risk_resume_contract import RUN
    class C:root=tmp_path;owner={}
    s=RuntimeState(C(),{'stage':'admit_FIT'},None);p=tmp_path/RUN/'processes.jsonl';p.parent.mkdir(parents=True);p.symlink_to(p.parent/'missing-target')
    with pytest.raises(ValueError):s._worker_rows()


def test_negative_closing_admission_consumes_only_reserved_closure_not_unrun_matrix(monkeypatch):
    import time
    from echo_routing.temporal.selected_risk_resume_runtime import RuntimeState
    now=time.time()
    class C:
        owner={'T0_epoch':int(now)-1,'deadline_epoch':int(now)-1+43200,'execution_authorized':True,'preparation_only':False,'new_budget_granted':False}
        def enter(self,*a,**k):return {},{}
    s=RuntimeState(C(),{'stage':'close_negative','freeze':{},'predecessors':{},'arguments':{'stage_CPU_upper_seconds':900,'stage_wall_upper_seconds':900}},None)
    s.spent=lambda:dict(new_GPU_wall_seconds=0.,new_CPU_accounting_seconds=1000.,cumulative_GPU_hours=18.657838855,cumulative_CPU_accounting_hours=20.5)
    s.storage=lambda:200000000
    monkeypatch.setattr('echo_routing.temporal.selected_risk_resume_estimate.estimate',lambda *a:(_ for _ in ()).throw(AssertionError('must not demand unrun formal matrix for negative closure')))
    assert s.admission()['all_caps_fit'] is True


def test_private_postcutoff_admin_pins_are_once_and_cannot_disappear(tmp_path,monkeypatch):
    import hashlib
    from echo_routing.temporal.selected_risk_resume_runtime import RuntimeState
    from echo_routing.temporal.selected_risk_resume_contract import RUN
    def put(path,value):
        p=tmp_path/path;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value));return {'path':path,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
    prep_pin=put('prep.json',{'file_pins':{},'receipts':[],'review_raw_components_covered_not_recharged':[]});base_pin=put('base.json',{'preparation_ledger_pin':prep_pin})
    raw=put(RUN+'/administrative_fees/raw.json',{'CPU':1.25})
    row={'receipt_id':'extra','measurement_kind':'administrative_partial_receipt','source_receipt':raw['path'],'source_sha256':raw['sha256'],'source_CPU_field':'CPU','measured_CPU_core_seconds':1.25,'charge_group':'raw:'+raw['sha256'],'covers_raw_sha256':[],'GPU_job':False,'CUDA_context_occupied':False}
    pin=put(RUN+'/administrative_fees/fee.json',row)
    class C:
        root=tmp_path;owner={};baseline_pin=base_pin
        def read(self,p):
            raw=(tmp_path/p['path']).read_bytes();assert hashlib.sha256(raw).hexdigest()==p['sha256'];return json.loads(raw)
        def raw(self,p):self.read(p);return b''
    s=RuntimeState(C(),{'stage':'admit_FIT','prep_ledger':prep_pin,'arguments':{'additional_administrative_fee_receipts':[pin]}},None);s._worker_rows=lambda:[]
    monkeypatch.setattr('echo_routing.temporal.selected_risk_resume_budget.totals',lambda b,p,w:p)
    assert s.spent()==[row]
    s.spec['arguments']['additional_administrative_fee_receipts']=[pin,pin]
    with pytest.raises(ValueError):s.spent()
    s.spec['arguments']['additional_administrative_fee_receipts']=[];s._worker_rows=lambda:[{'additional_administrative_fee_receipts':[pin]}]
    with pytest.raises(ValueError):s.spent()
