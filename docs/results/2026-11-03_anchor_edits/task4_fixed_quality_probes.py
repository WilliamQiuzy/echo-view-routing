"""A59 fixed-supervisor quality probes; synthetic temporary CPU children only."""
import copy
import importlib
import sys
import time
import pytest


def initial(m, tmp_path):
    return m.execute('first_quality_probe', [sys.executable, '-c', 'pass'], tmp_path / 'run', tmp_path / 'pub',
                     m.totals([]), 1., False, storage_probe=lambda: 0, GPU_probe=lambda: [])


def test_external_and_locked_same_actual_receipt_are_merged_once(tmp_path):
    m = importlib.import_module('scripts.supervise_anchor_edits')
    first = initial(m, tmp_path)
    expected = m.remaining_limits(m.totals([first]), time.time())
    second = m.execute('second_quality_probe', [sys.executable, '-c', 'pass'], tmp_path / 'run', tmp_path / 'pub',
                       m.totals([first]), 1., False, storage_probe=lambda: 0, GPU_probe=lambda: [])
    assert second['limits_at_launch']['remaining_CPU_seconds'] == expected['remaining_CPU_seconds']
    assert second['limits_at_launch']['remaining_GPU_seconds_before_reserve'] == expected['remaining_GPU_seconds_before_reserve']


def test_same_receipt_identity_with_conflicting_cost_rejected_before_launch(tmp_path, monkeypatch):
    m = importlib.import_module('scripts.supervise_anchor_edits')
    first = initial(m, tmp_path)
    changed = copy.deepcopy(first)
    changed['measured_CPU_core_seconds'] += 1.
    def never_launch(*args, **kwargs):
        pytest.fail('receipt conflict reached child launch')
    monkeypatch.setattr(m.subprocess, 'Popen', never_launch)
    with pytest.raises(ValueError, match='conflicting'):
        m.execute('conflict_quality_probe', [sys.executable, '-c', 'pass'], tmp_path / 'run', tmp_path / 'pub',
                  m.totals([changed]), 1., False, storage_probe=lambda: 0, GPU_probe=lambda: [])
    assert not (tmp_path / 'run' / 'active_process.json').exists()


def test_launch_intent_persistence_failure_does_not_start_a_worker(tmp_path, monkeypatch):
    m = importlib.import_module('scripts.supervise_anchor_edits')
    original = m.write_json
    def fail_launch_intent(path, value):
        if path.name == 'active_process.json':
            raise OSError('injected launch-intent persistence failure')
        return original(path, value)
    def never_launch(*args, **kwargs):
        pytest.fail('failed durable launch intent reached Popen')
    monkeypatch.setattr(m, 'write_json', fail_launch_intent)
    monkeypatch.setattr(m.subprocess, 'Popen', never_launch)
    row = m.execute('failed_intent_quality_probe', [sys.executable, '-c', 'pass'], tmp_path / 'run', tmp_path / 'pub',
                    m.totals([]), 1., False, storage_probe=lambda: 0, GPU_probe=lambda: [])
    assert row['model_pid'] is None and not row['child_exit_code_directly_waited']
    assert row['observed_worker_exit_code'] is None and row['guardian_exception']
    assert m.stage_exit_status(row) == 1
