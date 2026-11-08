"""Fixed cumulative account and prospective-window arithmetic; no stage runs."""
import copy
import importlib

import pytest


def module():return importlib.import_module('echo_routing.temporal.selected_risk_resume_budget')


def baseline():
    return dict(sha256='c290ec86944cdaefc094a2fd75c45c33747f177114cae0ea9706dec6937ac722',
        closed_A63_known_CPU_seconds=306.464139,closed_A63_explicit_CPU_allowance_seconds=128.,
        closed_A63_CPU_accounting_seconds=434.464139,closed_cumulative_CPU_accounting_hours=20.35189033788889,
        closed_new_A63_GPU_seconds=0.,closed_cumulative_GPU_hours=18.657838855)


def row(key='new',cpu=.5,allow=0.,gpu=False):
    return dict(receipt_id=key,measurement_kind='administrative_partial_receipt',
        source_receipt='docs/results/2026-11-05_a63_resume_preparation/'+key+'.json',source_sha256='a'*64,
        started_epoch=100.,finished_epoch=110. if gpu else 100.,full_process_wall_seconds=10. if gpu else 0.,
        measured_CPU_core_seconds=cpu,CPU_accounting_allowance_seconds=allow,
        GPU_job=gpu,CUDA_context_occupied=False,observed_worker_exit_code=None,
        measurement_host=None,model_pid=None)


def owner():
    return dict(T0_epoch=1793897041,deadline_epoch=1793897041+43200,
        execution_authorized=True,preparation_only=False,new_budget_granted=False)


def test_closed_baseline_is_inherited_once_and_unknown_admin_is_not_zero():
    m=module();s=m.totals(baseline(),[row()],[row('unknown',None,3.)])
    assert s['new_measured_CPU_seconds']==pytest.approx(306.964139)
    assert s['new_CPU_accounting_seconds']==pytest.approx(437.964139)
    assert s['administrative_CPU_allowance_seconds']==131.
    assert s['cumulative_CPU_accounting_hours']==pytest.approx(20.35189033788889+3.5/3600)
    assert s['new_GPU_wall_seconds']==0 and s['postclosure_unknown_CPU_receipts']==1


@pytest.mark.parametrize('change',['duplicate','old_path','baseline_recharge','baseline_reset','unknown_worker'])
def test_duplicate_old_or_unknown_worker_fees_cannot_silently_reset_account(change):
    m=module();b=baseline();prep=[row()];workers=[]
    if change=='duplicate':workers=[row()]
    if change=='old_path':prep[0]['source_receipt']='docs/results/2026-11-05_selected_action_risk/old.json'
    if change=='baseline_recharge':prep[0]['source_sha256']=b['sha256']
    if change=='baseline_reset':b['closed_A63_CPU_accounting_seconds']=0.
    if change=='unknown_worker':prep[0].update(measured_CPU_core_seconds=None,measurement_kind='actual_worker',CPU_accounting_allowance_seconds=3.)
    with pytest.raises(ValueError):m.totals(b,prep,workers)


def test_fresh_forecast_keeps_original_caps_and_reserves_inside_remaining():
    m=module();s=m.totals(baseline(),[],[]);o=owner()
    f=m.forecast(s,4000,21000,800000000,o['T0_epoch']+10,24000,o)
    assert f['all_caps_fit'] is True
    assert f['forecast_incremental_CPU_seconds_including_closure']==pytest.approx(434.464139+21000)
    assert f['forecast_incremental_GPU_seconds']==4000
    assert f['GPU_reserve_seconds']==7200 and f['CPU_closure_seconds']==3600
    assert f['forecast_cumulative_tracked_storage_upper_bytes']==38650409858+800000000
    assert f['unused_forecast_reserves_are_actual_spend'] is False


@pytest.mark.parametrize('change',['CPU','GPU','storage','unknown_storage','deadline','missing_closure','not_approved'])
def test_every_full_run_limit_is_failclosed_and_failure_is_preserved(change):
    m=module();s=m.totals(baseline(),[],[]);o=owner();g,c,storage,wall=4000,21000,800000000,24000
    if change=='CPU':c=28800
    if change=='GPU':g=5401
    if change=='storage':storage=1000000001
    if change=='unknown_storage':storage=None
    if change=='deadline':wall=43200
    if change=='missing_closure':c=3599
    if change=='not_approved':o['execution_authorized']=False
    out=m.forecast(s,g,c,storage,o['T0_epoch']+10,wall,o)
    assert out['all_caps_fit'] is False and out['failed_checks']


def test_stage_limits_inherit_spend_and_do_not_grant_a_second_GPU_budget():
    m=module();s=m.totals(baseline(),[],[row('GPU',gpu=True)]);o=owner()
    out=m.remaining_limits(s,o['T0_epoch']+10,o,potential_GPU_context=True)
    assert out['remaining_GPU_seconds_before_reserve']==5390
    assert out['remaining_CPU_seconds']==pytest.approx(28800-434.964139-3600)
    assert out['stage_wall_seconds']==5390
