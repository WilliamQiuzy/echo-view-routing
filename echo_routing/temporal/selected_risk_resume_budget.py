"""Continuation arithmetic: inherit closed A63 once, never grant new caps.

Raw SHA/charge-group/aggregate-child coverage registration, live queue/GPU/du,
actual worker receipts and raw new-window authority are caller obligations.
Unknown values stay explicit. A numerical forecast is not an actual admission.
"""
import copy
import math

from echo_routing.temporal import selected_risk_budget as old
from echo_routing.temporal.selected_risk_resume_contract import storage_cap

CLOSED = dict(closed_A63_known_CPU_seconds=306.464139,
              closed_A63_explicit_CPU_allowance_seconds=128.,
              closed_A63_CPU_accounting_seconds=434.464139,
              closed_cumulative_CPU_accounting_hours=20.35189033788889,
              closed_new_A63_GPU_seconds=0., closed_cumulative_GPU_hours=18.657838855)
CLOSED_SHA='c290ec86944cdaefc094a2fd75c45c33747f177114cae0ea9706dec6937ac722'


def totals(baseline, prep_rows, new_worker_rows):
    """Input rows must already be an exact-once, externally pinned compilation."""
    if not isinstance(baseline,dict) or baseline.get('sha256')!=CLOSED_SHA:
        raise ValueError('independently pinned original closed A63 receipt required')
    for k,v in CLOSED.items():
        if k not in baseline or not math.isclose(old.checked(baseline[k]),v,rel_tol=0,abs_tol=1e-10):
            raise ValueError('closed baseline cannot be reset or reinterpreted')
    if not isinstance(prep_rows,list) or not isinstance(new_worker_rows,list):
        raise ValueError('disjoint compiled preparation and new worker lists required')
    rows=prep_rows+new_worker_rows
    for row in rows:
        if row.get('source_sha256')==CLOSED_SHA:
            raise ValueError('closed cumulative receipt cannot be recharged as a fresh fee')
        path=row.get('source_receipt')
        if path is not None and not any(path.startswith(prefix) for prefix in (
                'docs/results/2026-11-05_a63_resume_preparation/',
                'docs/results/2026-11-05_a63_resume_execution/',
                'runs/a63_selected_action_risk_resume_v1/')):
            raise ValueError('old/source-foreign fee path cannot be charged to continuation')
    new=old.totals(rows)
    return dict(new_GPU_wall_seconds=CLOSED['closed_new_A63_GPU_seconds']+new['new_GPU_wall_seconds'],
        new_measured_CPU_seconds=CLOSED['closed_A63_known_CPU_seconds']+new['new_measured_CPU_seconds'],
        new_CPU_accounting_seconds=CLOSED['closed_A63_CPU_accounting_seconds']+new['new_CPU_accounting_seconds'],
        administrative_CPU_allowance_seconds=CLOSED['closed_A63_explicit_CPU_allowance_seconds']+new['administrative_CPU_allowance_seconds'],
        cumulative_GPU_hours=CLOSED['closed_cumulative_GPU_hours']+new['new_GPU_wall_seconds']/3600,
        cumulative_CPU_accounting_hours=CLOSED['closed_cumulative_CPU_accounting_hours']+new['new_CPU_accounting_seconds']/3600,
        postclosure_unknown_CPU_receipts=new['unknown_CPU_receipts'],
        postclosure_failed_stages=new['failed_stages'],
        postclosure_unknown_observed_worker_OS_statuses=new['unknown_observed_worker_OS_statuses'],
        postclosure_closed_receipts=new['closed_receipts'],receipt_records=copy.deepcopy(rows),
        inherited_baseline_receipt_sha256=CLOSED_SHA,baseline_inherited_once=True,
        inherited_unmetered_gaps_and_negative_receipts_retained=True,
        original_A63_incremental_caps=dict(GPU_seconds=5400,CPU_seconds=28800,storage_bytes=1000000000))


def _window(owner,now):
    old.checked(now)
    if not isinstance(owner,dict):raise ValueError('separately approved fixed-window metadata required')
    a,b=owner.get('T0_epoch'),owner.get('deadline_epoch')
    if type(a) is not int or type(b) is not int or a<=old.T0 or b-a!=43200:
        raise ValueError('new separately recorded12h window required, old window unchanged')
    authorized=(owner.get('execution_authorized') is True and owner.get('preparation_only') is False
                and owner.get('new_budget_granted') is False)
    return a,b,authorized


def forecast(spent, remaining_GPU_seconds, remaining_CPU_seconds, storage_A63_upper,
             now, remaining_wall_seconds, owner, *, closing=False):
    """Remaining CPU/wall and full A63 storage upper ALREADY include reserves.

    storage_A63_upper includes old closed artifacts, current prep/new RUN, all
    temporary coexistence and future closing storage. It is not a new delta or
    an observed physical peak. Null storage cannot establish fit. During the
    actual closing phase reserved resources may be consumed, not reset.
    """
    if type(closing) is not bool:raise ValueError('explicit closing phase boolean required')
    g=old.checked(remaining_GPU_seconds);c=old.checked(remaining_CPU_seconds)
    wall=old.checked(remaining_wall_seconds);a,b,authorized=_window(owner,now)
    if storage_A63_upper is not None:old.checked(storage_A63_upper)
    used_g=old.checked(spent['new_GPU_wall_seconds']);used_c=old.checked(spent['new_CPU_accounting_seconds'])
    cg=old.checked(spent['cumulative_GPU_hours']);cc=old.checked(spent['cumulative_CPU_accounting_hours'])
    increment_gpu=used_g+g;increment_cpu=used_c+c
    cumulative_gpu=cg+(g+old.GPU_CLOSURE_RESERVE_SECONDS)/3600
    cumulative_cpu=cc+c/3600
    storage=None if storage_A63_upper is None else old.PRIOR_STORAGE_UPPER_BYTES+storage_A63_upper;cap=storage_cap(owner)
    cpu_reserve=0 if closing else old.CPU_CLOSURE_RESERVE_SECONDS
    wall_reserve=0 if closing else old.WALL_CLOSURE_RESERVE_SECONDS
    storage_reserve=0 if closing else old.STORAGE_RESERVE_BYTES
    checks=dict(new_execution_authorization=authorized,window_start=a<=now,
        deadline=now+wall<=b,remaining_CPU_includes_closure=c>=cpu_reserve,
        remaining_wall_includes_closure=wall>=wall_reserve,
        A63_GPU_cap=increment_gpu<=old.GPU_CAP_SECONDS,
        A63_CPU_cap=increment_cpu<=old.CPU_CAP_SECONDS,
        A63_storage_cap=storage_A63_upper is not None and storage_reserve<=storage_A63_upper<=cap,
        cumulative_GPU_with_inherited_reserve=cumulative_gpu<=24,
        cumulative_CPU=cumulative_cpu<=72,cumulative_storage=storage is not None and storage<=40000000000)
    return dict(all_caps_fit=all(checks.values()),checks=checks,approved_A63_storage_cap_bytes=cap,CPU_GPU_cumulative_caps_not_reset=True,
        failed_checks=[k for k,v in checks.items() if not v],
        forecast_incremental_GPU_seconds=increment_gpu,
        forecast_incremental_CPU_seconds_including_closure=increment_cpu,
        forecast_cumulative_GPU_hours_including_reserve=cumulative_gpu,
        forecast_cumulative_CPU_accounting_hours=cumulative_cpu,
        forecast_cumulative_tracked_storage_upper_bytes=storage,
        forecast_incremental_tracked_storage_upper_bytes=storage_A63_upper,
        forecast_end_epoch=now+wall,deadline_epoch=b,old_deadline_epoch=old.DEADLINE_EPOCH,
        GPU_reserve_seconds=old.GPU_CLOSURE_RESERVE_SECONDS,CPU_closure_seconds=cpu_reserve,
        storage_closure_reserve_bytes_inside_increment=storage_reserve,
        wall_closure_reserve_seconds_inside_window=wall_reserve,
        unused_forecast_reserves_are_actual_spend=False,
        scope='numerical full-pass forecast only; raw authority/live claims/du/ledger and actual admission remain caller duties')


def remaining_limits(spent,now,owner,*,potential_GPU_context=True,closing=False):
    if type(potential_GPU_context) is not bool or type(closing) is not bool:
        raise ValueError('explicit GPU-context and closing booleans required')
    a,b,authorized=_window(owner,now)
    cpu=min(old.CPU_CAP_SECONDS-old.checked(spent['new_CPU_accounting_seconds']),
            (72-old.checked(spent['cumulative_CPU_accounting_hours']))*3600)
    gpu=min(old.GPU_CAP_SECONDS-old.checked(spent['new_GPU_wall_seconds']),
            (24-old.checked(spent['cumulative_GPU_hours']))*3600-old.GPU_CLOSURE_RESERVE_SECONDS)
    cpu-=0 if closing else old.CPU_CLOSURE_RESERVE_SECONDS
    wall=b-now-(0 if closing else old.WALL_CLOSURE_RESERVE_SECONDS)
    active=authorized and a<=now<b
    limit=max(0.,min(cpu,wall,gpu if potential_GPU_context else float('inf'))) if active else 0.
    return dict(stage_wall_seconds=limit,remaining_CPU_seconds=max(0.,cpu),
        remaining_GPU_seconds_before_reserve=max(0.,gpu),deadline_epoch=b,closing=closing,
        execution_authorization_established=authorized,original_caps_not_reset=True,
        storage_remaining_bytes=None,current_storage_unverified_by_this_function=True)
