import importlib

import pytest


def api():
    return importlib.import_module('echo_routing.temporal.anchor_resources')


def receipt(stage='one', start=100., end=120., gpu=False, context=False, cpu=13.):
    return {'stage': stage, 'measurement_host': 'synthetic-host', 'model_pid': 123 if stage == 'one' else 124,
            'started_epoch': start, 'finished_epoch': end, 'full_process_wall_seconds': end - start,
            'GPU_job': gpu, 'CUDA_context_occupied': context, 'measured_CPU_core_seconds': cpu,
            'observed_worker_exit_code': 0, 'receipt_id': stage}


def test_context_occupation_failed_work_and_all_prior_charges_are_retained_once():
    m = api()
    rows = [receipt(gpu=False, context=True), receipt('two', 130., 150., gpu=True)]
    rows[1]['observed_worker_exit_code'] = 1
    result = m.totals(rows)
    assert result['new_GPU_wall_seconds'] == 40.
    assert result['new_measured_CPU_seconds'] == 26.
    assert result['failed_stages'] == 1
    assert result['cumulative_GPU_hours'] == pytest.approx(17.31542736033599 + 40 / 3600)
    assert result['cumulative_CPU_accounting_hours'] == pytest.approx(17.9637378675 + 26 / 3600)
    assert result['prior_CPU_allowance_hours'] == .75
    assert result['GPU_intervals_disjoint']
    legacy = receipt()
    legacy.pop('model_pid')
    legacy.pop('measurement_host')
    r = m.totals([legacy])
    assert not r['complete_host_PID_start_provenance']
    assert r['receipt_based_identity_count'] == 1


@pytest.mark.parametrize('fault', ['duplicate', 'overlap', 'nan_start', 'reverse', 'undercharge', 'cpu_nan'])
def test_duplicate_overlap_nonfinite_or_undercharged_receipts_fail_closed(fault):
    m = api()
    rows = [receipt(gpu=True)]
    if fault == 'duplicate':
        rows.append(dict(rows[0]))
    elif fault == 'overlap':
        rows.append(receipt('two', 110., 140., gpu=True))
    elif fault == 'nan_start':
        rows[0]['started_epoch'] = float('nan')
    elif fault == 'reverse':
        rows[0]['finished_epoch'] = 90.
    elif fault == 'undercharge':
        rows[0]['full_process_wall_seconds'] = 1.
    elif fault == 'cpu_nan':
        rows[0]['measured_CPU_core_seconds'] = float('nan')
    with pytest.raises(ValueError):
        m.totals(rows)


def test_forecast_enforces_original_caps_reserve_storage_and_deadline():
    m = api()
    spent = m.totals([])
    now = 1793739000.
    result = m.forecast(spent, GPU_seconds=3600, CPU_seconds=7200, incremental_peak_bytes=100000000,
                        now=now, remaining_wall_seconds=3600)
    assert result['all_caps_fit']
    assert result['closure_GPU_reserve_hours'] == 2
    for field, amount in [('GPU_seconds', 6 * 3600), ('CPU_seconds', 55 * 3600),
                           ('incremental_peak_bytes', 7000000000), ('remaining_wall_seconds', 1000000)]:
        kwargs = dict(GPU_seconds=3600, CPU_seconds=7200, incremental_peak_bytes=100000000,
                      now=now, remaining_wall_seconds=3600)
        kwargs[field] = amount
        assert not m.forecast(spent, **kwargs)['all_caps_fit']
    limits = m.remaining_limits(spent, now, potential_GPU_context=True)
    assert limits['stage_wall_seconds'] <= (24 - 17.31542736033599 - 2) * 3600
    with pytest.raises(ValueError):
        m.forecast(spent, -1, 0, 0, now, 0)


def test_same_nonblocking_exclusive_queue_lock_is_held_through_output(tmp_path):
    m = api()
    path = tmp_path / 'shared.lock'
    with m.ExclusiveQueue(path):
        with pytest.raises(BlockingIOError):
            with m.ExclusiveQueue(path):
                pytest.fail('a duplicate executor acquired the queue')
    with m.ExclusiveQueue(path):
        (tmp_path / 'closure').write_text('outputs stay inside the held lock')
