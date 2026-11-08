"""Synthetic caller/wait4 receipts only: no scientific child or server launch."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import pytest


def module():
    p=Path(__file__).resolve().parents[1]/'scripts/a63_wait_resume_selected_risk.py'
    s=importlib.util.spec_from_file_location('a63_resume_caller_fixture',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def snapshot():
    return {'receipt_id':'resume:train_formal:'+'a'*32,'stage':'train_formal','nonce':'a'*32,
        'supervisor_pid':20,'supervisor_start_ticks':100,'model_pid':30,'model_start_ticks':200,
        'source_intent_sha256':'c'*64,'runtime_spec_sha256':'d'*64,'runtime_authorization_sha256':'e'*64,
        'caller_intent_sha256':'f'*64,'started_epoch':11.,'finished_epoch':19.,
        'measured_CPU_core_seconds':999.,'GPU_job':False,'CUDA_context_occupied':False,
        'observed_worker_exit_code':0,'actual_child_directly_waited':True,'stage_exit_status':0,
        'termination_reason':None,'current_remote_caps_unknown':False,'boot_id':'toy'}


def test_whole_child_CPU_replaces_nested_guardian_cost_not_addition():
    m=module();s=snapshot();r=m.canonical_row('train_formal','b'*32,10.,20.,20,100,0,SimpleNamespace(ru_utime=3.,ru_stime=2.),.5,s,{'path':'docs/results/guardian.json','sha256':'1'*64},{'path':'runs/caller.json','sha256':'f'*64},{'path':'spec.json','sha256':'d'*64},'e'*64,None)
    assert r['measured_CPU_core_seconds']==5.5 and r['GPU_job'] is False
    assert r['nonce']==s['nonce'] and r['source_intent_sha256']==s['source_intent_sha256']
    assert r['covered_charge_groups']==[s['receipt_id']] and r['unresolved_worker_uncertainty'] is False
    assert r['caller_exit_CPU_allowance_seconds']==1 and r['SELF_OS_exit_code'] is None


def test_absent_guardian_receipt_does_not_fabricate_model_OS_or_GPU_zero():
    m=module();r=m.canonical_row('train_formal','b'*32,10.,20.,20,100,1,SimpleNamespace(ru_utime=3.,ru_stime=2.),.5,None,None,{'path':'runs/caller.json','sha256':'f'*64},{'path':'spec.json','sha256':'d'*64},'e'*64,None)
    assert r['unresolved_worker_uncertainty'] is True and r['GPU_job'] is True
    assert r['GPU_observed_occupied_seconds'] is None and r['GPU_conservative_charged_seconds']==10
    assert r['observed_worker_exit_code']==1 and r['inner_model_observed_OS_exit_code'] is None
    assert r['stage_exit_status']==1 and r['measured_CPU_core_seconds'] is None and r['whole_child_accounting_complete'] is False


def test_wait4_interrupt_never_replays_or_signals_foreign_PID():
    m=module();calls=[]
    def wait(pid,flags):
        calls.append(pid)
        if len(calls)==1:raise KeyboardInterrupt()
        return pid,0,SimpleNamespace(ru_utime=1.,ru_stime=0.)
    child=SimpleNamespace(pid=123,returncode=None)
    result=m.direct_wait(child,wait4=wait)
    assert calls==[123,123] and child.returncode==0 and result[2]=='KeyboardInterrupt'


def test_inspection_and_missing_owner_never_spawn(monkeypatch,capsys):
    m=module();monkeypatch.setattr(m,'execute',lambda *a,**k:(_ for _ in ()).throw(AssertionError('scientific child attempted')))
    assert m.main(['--stage','train_formal'])==0
    assert 'false' in capsys.readouterr().out
    with pytest.raises(SystemExit):m.main(['--execute','--stage','train_formal'])


def test_lost_wait4_usage_is_unknown_not_zero_child_CPU():
    m=module();s=snapshot();r=m.canonical_row('train_formal','b'*32,10.,20.,20,100,None,None,.5,s,{'path':'docs/results/guardian.json','sha256':'1'*64},{'path':'runs/caller.json','sha256':'f'*64},{'path':'spec.json','sha256':'d'*64},'e'*64,'ChildProcessError')
    assert r['measured_CPU_core_seconds'] is None and r['known_partial_CPU_core_seconds']==.5
    assert r['unresolved_worker_uncertainty'] is True and r['actual_child_directly_waited'] is False


def test_wait4_status_lost_retains_unknown_instead_of_fake_wait():
    m=module();child=SimpleNamespace(pid=123,returncode=None)
    def wait(*args):raise ChildProcessError('already reaped externally')
    code,usage,error=m.direct_wait(child,wait4=wait)
    assert code is None and usage is None and 'ChildProcessError' in error and child.returncode is None


def test_broken_ledger_symlink_is_not_an_absent_empty_ledger(tmp_path):
    m=module();from echo_routing.temporal.selected_risk_resume_context import Context
    ctx=object.__new__(Context);ctx.root=tmp_path
    p=tmp_path/m.RUN/'processes.jsonl';p.parent.mkdir(parents=True);p.symlink_to(tmp_path/'absent-foreign')
    with pytest.raises((ValueError,OSError)):m._rows(ctx,'processes')


def test_receipt_only_writer_rejects_noncanonical_nonce_before_any_IO():
    m=module()
    class NoIO:
        def raw(self,*args):raise AssertionError('malformed receipt reached IO')
    with pytest.raises(ValueError):m._publish(NoIO(),{'caller_nonce':'../foreign'}, {}, {})


def test_missing_inner_model_OS_never_becomes_success_from_guardian_OS_zero():
    m=module();s=snapshot();s['observed_worker_exit_code']=None
    r=m.canonical_row('train_formal','b'*32,10.,20.,20,100,0,SimpleNamespace(ru_utime=3.,ru_stime=2.),.5,s,{'path':'docs/results/guardian.json','sha256':'1'*64},{'path':'runs/caller.json','sha256':'f'*64},{'path':'spec.json','sha256':'d'*64},'e'*64,None)
    assert r['inner_worker_OS_unknown'] is True and r['unresolved_worker_uncertainty'] is True
    assert r['stage_exit_status']==1 and r['measured_CPU_core_seconds'] is None and r['known_partial_CPU_core_seconds']==5.5
    assert r['whole_child_accounting_complete'] is False


def test_postdeadline_receipt_requires_current_physical_caller_birth(monkeypatch):
    m=module();import os
    spec={'path':'spec.json','sha256':'d'*64};nonce='b'*32
    own={'caller_pid':os.getpid(),'caller_nonce':nonce,'runtime_spec':spec,'runtime_authorization_sha256':'e'*64,'stage':'train_formal','caller_start_ticks':100,'boot_id':'toy'}
    row={'caller_nonce':nonce,'receipt_id':'resume_caller:train_formal:'+nonce,'stage':'train_formal','caller_start_ticks':100,'caller_boot_id':'toy'}
    class Context:
        owner_pin=proposal_pin=baseline_pin={};owner_sha='e'*64
        def raw(self,pin):return b'{}'
        def read(self,pin):return own
        def _parent(self,*args,**kwargs):raise AssertionError('reused PID reached filesystem write')
    monkeypatch.setattr(m,'_identity',lambda:(101,'toy'),raising=False)
    with pytest.raises(ValueError):m._publish(Context(),row,{},spec)


@pytest.mark.parametrize('inner_OS,reason',[(7,None),(0,'actual_cap_reached')])
def test_known_inner_failure_or_cap_cannot_be_erased_by_snapshot_success(inner_OS,reason):
    m=module();s=snapshot();s['observed_worker_exit_code']=inner_OS;s['termination_reason']=reason
    r=m.canonical_row('train_formal','b'*32,10.,20.,20,100,0,SimpleNamespace(ru_utime=3.,ru_stime=2.),.5,s,{'path':'docs/results/guardian.json','sha256':'1'*64},{'path':'runs/caller.json','sha256':'f'*64},{'path':'spec.json','sha256':'d'*64},'e'*64,None)
    assert r['stage_exit_status']==1 and r['inner_model_observed_OS_exit_code']==inner_OS
    assert r['unresolved_worker_uncertainty'] is False and r['measured_CPU_core_seconds']==5.5
    assert r['guardian_termination_reason']==reason
