"""A59 reader/closure probes; synthetic IO, no server or real experiment data."""
import importlib
import json
from pathlib import Path


def test_committed_readout_entrypoints_import_without_timing_checker_imports():
    r = importlib.import_module('scripts.readout_anchor_edits')
    assert r.role_for('main', 'selection') == 'selection'
    assert r.role_for('old_stress', 'calibration') == 'old_stress_calibration'
    assert r.role_for('fresh_confirmation', 'selection') == 'confirmation_selection'
    for name in ['scripts.aggregate_anchor_edits', 'scripts.oracle_anchor_edits', 'scripts.close_anchor_resources']:
        assert callable(importlib.import_module(name).main)


def test_terminal_direct_wait_claim_follows_unknown_server_OS_receipt(tmp_path, monkeypatch):
    m = importlib.import_module('scripts.close_anchor_resources')
    run, pub = tmp_path / 'run', tmp_path / 'pub'
    run.mkdir(); pub.mkdir()
    monkeypatch.setattr(m, 'RUN', run)
    monkeypatch.setattr(m, 'PUB', pub)
    monkeypatch.setattr(m, 'ROOT', tmp_path)
    record = {'stage': 'unknown_synthetic_stage', 'attempt': 1, 'receipt_id': 'synthetic:1',
              'measurement_host': 'synthetic-host', 'model_pid': 12345, 'supervisor_pid': 54321,
              'started_epoch': 1000., 'finished_epoch': 1001., 'full_process_wall_seconds': 1.,
              'measured_CPU_core_seconds': .5, 'GPU_job': False, 'CUDA_context_occupied': False,
              'observed_worker_exit_code': None, 'child_exit_code_directly_waited': False}
    (run / 'processes.jsonl').write_text(json.dumps(record) + '\n')
    models = []
    for seed in [2026110359, 2026110360]:
        for arm in ['A', 'J']:
            folder = run / 'models' / f'seed{seed}' / arm
            folder.mkdir(parents=True)
            (folder / 'latest.pt').write_bytes(b'x')
            models.append({'seed': seed, 'arm': arm,
                           'milestones': [{'path': f'fixture_{seed}_{arm}_{step}', 'sha256': 'a' * 64}
                                          for step in [2000, 4000, 6000]]})
    fixtures = {'local_cpu_receipts.json': {'file_pins': {}, 'receipts': []},
                'summary.json': {'formal_models': 4, 'formal_optimizer_updates': 24000, 'metric_rows': 70, 'paired_comparison_count': 96},
                'action_space_diagnostic.json': {'domain_count': 12},
                'models_manifest.json': {'passed': True, 'models': models},
                'formal_admission.json': {'storage': {'incremental_tracked_path_peak_upper_bytes': 100}},
                'anchor_edits_v1.json': {'source': {'a08': {'path': 'fixture_a', 'sha256': 'a' * 64},
                                                 'jepa': {'path': 'fixture_j', 'sha256': 'b' * 64}}}}
    for name in ['summary.json', 'action_space_diagnostic.json', 'models_manifest.json', 'local_cpu_receipts.json', 'formal_admission.json']:
        (pub / name).write_text(json.dumps(fixtures[name]))
    original_read = m.read
    monkeypatch.setattr(m, 'read', lambda path: fixtures[Path(path).name] if Path(path).name in fixtures else original_read(path))
    monkeypatch.setattr(m, 'verify_freeze', lambda name: {'objects': {}})
    monkeypatch.setattr(m, 'verify_pin', lambda pin: Path(pin['path']))
    def fake_output(args, **kwargs):
        if args[0] == 'nvidia-smi': return ''
        if args[0] == 'du': return f'0 {run}\n0 {pub}\n'
        raise AssertionError('unexpected external command')
    monkeypatch.setattr(m.subprocess, 'check_output', fake_output)
    m.main()
    resources = json.loads((pub / 'resources.json').read_text())
    terminal = json.loads((pub / 'terminal_exit_receipt.json').read_text())
    assert resources['actual_totals']['unknown_observed_worker_OS_statuses'] == 1
    assert terminal['worker_OS_exit_statuses'][0]['observed_worker_exit_code'] is None
    assert resources['new_worker_OS_statuses_directly_waited_not_inferred_from_manifests'] is False
    assert terminal['all_new_worker_statuses_directly_waited'] is False
