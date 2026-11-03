"""Read-only A59 supervisor contract probes; temporary synthetic CPU children only."""
import importlib
import sys
import time


def test_active_intent_is_durable_before_child_launch(tmp_path, monkeypatch):
    m = importlib.import_module('scripts.supervise_anchor_edits')
    run = tmp_path / 'run'
    observed = []
    original = m.subprocess.Popen
    def observe_launch(*args, **kwargs):
        observed.append((run / 'active_process.json').exists())
        return original(*args, **kwargs)
    monkeypatch.setattr(m.subprocess, 'Popen', observe_launch)
    row = m.execute('prelaunch_probe', [sys.executable, '-c', 'pass'], run, tmp_path / 'pub',
                    m.totals([]), 1., False, storage_probe=lambda: 0, GPU_probe=lambda: [])
    assert row['observed_worker_exit_code'] == 0
    assert observed and all(observed), 'worker launches before any durable active intent exists'


def test_new_locked_stage_accounts_for_prior_ledger_cost_even_with_stale_caller_snapshot(tmp_path):
    m = importlib.import_module('scripts.supervise_anchor_edits')
    run, pub = tmp_path / 'run', tmp_path / 'pub'
    before = m.totals([])
    first = m.execute('first_probe', [sys.executable, '-c', 'sum(i*i for i in range(100000))'],
                      run, pub, before, 1., False, storage_probe=lambda: 0, GPU_probe=lambda: [])
    assert first['measured_CPU_core_seconds'] > 0
    second = m.execute('second_probe', [sys.executable, '-c', 'pass'], run, pub, before, 1., False,
                       storage_probe=lambda: 0, GPU_probe=lambda: [])
    expected = m.remaining_limits(m.totals([first]), time.time(), potential_GPU_context=True)
    assert second['limits_at_launch']['remaining_CPU_seconds'] <= expected['remaining_CPU_seconds'], \
        'locked stage ignores completed ledger costs when caller computed spent before acquiring lock'
