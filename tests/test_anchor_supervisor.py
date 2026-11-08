import importlib
import json
import os
import sys

import pytest


def run(tmp_path, code, limit=5., storage=None):
    module = importlib.import_module('scripts.supervise_anchor_edits')
    return module.execute('synthetic', [sys.executable, '-c', code], tmp_path / 'run', tmp_path / 'pub',
                          module.totals([]), limit, False,
                          storage_probe=storage or (lambda: 0), GPU_probe=lambda: [])


def test_real_child_failure_is_waited_charged_and_shared_queue_released(tmp_path, synthetic_resource_window):
    synthetic_resource_window("echo_routing.temporal.anchor_resources")
    module = importlib.import_module('scripts.supervise_anchor_edits')
    row = run(tmp_path, 'sum(i*i for i in range(10000)); raise SystemExit(3)')
    assert row['observed_worker_exit_code'] == 3 and module.stage_exit_status(row) == 1
    assert row['full_process_wall_seconds'] > 0 and row['measured_CPU_core_seconds'] > 0
    assert row['child_exit_code_directly_waited']
    assert not (tmp_path / 'run/active_process.json').exists()
    assert json.loads((tmp_path / 'run/process_synthetic_1.json').read_text()) == row
    with module.ExclusiveQueue(tmp_path / 'run/exclusive_supervisor.lock'):
        pass
    with pytest.raises(ProcessLookupError):
        os.kill(row['model_pid'], 0)


@pytest.mark.parametrize('fault', ['timeout', 'guard_exception'])
def test_guard_timeout_or_exception_never_orphans_child_or_reports_success(tmp_path, fault, synthetic_resource_window):
    synthetic_resource_window("echo_routing.temporal.anchor_resources")
    module = importlib.import_module('scripts.supervise_anchor_edits')
    def failed_storage():
        raise RuntimeError('synthetic storage probe failure')
    row = run(tmp_path, 'import time; time.sleep(10)', limit=.15 if fault == 'timeout' else 5.,
              storage=failed_storage if fault == 'guard_exception' else None)
    assert row['termination_reason'] is not None
    assert module.stage_exit_status(row) == 1
    with pytest.raises(ProcessLookupError):
        os.kill(row['model_pid'], 0)
    # Observed OS0 cannot mask an independently failed guardian stage.
    assert module.stage_exit_status({**row, 'observed_worker_exit_code': 0}) == 1
    assert not (tmp_path / 'run/active_process.json').exists()


def test_locked_external_local_CPU_receipts_are_included_even_when_caller_is_old(tmp_path, synthetic_resource_window):
    synthetic_resource_window("echo_routing.temporal.anchor_resources")
    module = importlib.import_module('scripts.supervise_anchor_edits')
    pub = tmp_path / 'pub'
    pub.mkdir()
    row = {'stage': 'local_checks', 'receipt_id': 'local-fixed', 'started_epoch': 100., 'finished_epoch': 101.,
           'full_process_wall_seconds': 1., 'measured_CPU_core_seconds': 11., 'GPU_job': False,
           'CUDA_context_occupied': False, 'observed_worker_exit_code': 0}
    (pub / 'local_cpu_receipts.json').write_text(json.dumps({'receipts': [row]}))
    result = run(tmp_path, 'pass')
    expected = module.remaining_limits(module.totals([row]), result['started_epoch'])
    assert result['limits_at_launch']['remaining_CPU_seconds'] <= expected['remaining_CPU_seconds']
