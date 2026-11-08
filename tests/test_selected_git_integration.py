"""Only new A63 administrative allowlist/transaction qualification in toy Git."""
import hashlib
import importlib
import json
import subprocess

import pytest


def git(repo, *args):
    return subprocess.check_output(['git', '-C', str(repo), *args], text=True).strip()


@pytest.fixture
def toy(tmp_path):
    repo = tmp_path/'toy'; repo.mkdir()
    subprocess.run(['git', 'init', '-q', '-b', 'main', str(repo)], check=True)
    git(repo, 'config', 'user.name', 'Toy'); git(repo, 'config', 'user.email', 'toy@example.invalid')
    (repo/'user.txt').write_text('committed\n'); git(repo, 'add', 'user.txt'); git(repo, 'commit', '-qm', 'base')
    for i in range(8): (repo/f'theory_{i}.txt').write_text(f'staged theory {i}\n')
    git(repo, 'add', *[f'theory_{i}.txt' for i in range(8)])
    (repo/'user.txt').write_text('user unstaged\n')
    p = 'docs/results/2026-11-05_selected_action_risk/toy.json'
    f = repo/p; f.parent.mkdir(parents=True); f.write_text('{"toy":true}\n')
    return repo, [{'path': p, 'sha256': hashlib.sha256(f.read_bytes()).hexdigest()}], git(repo, 'rev-parse', 'HEAD')


def test_A63_allowlist_never_admits_old_research_or_main_Index():
    m = importlib.import_module('echo_routing.temporal.selected_git_integration')
    for p in ['echo_routing/temporal/selected_risk_audit.py', 'echo_routing/temporal/selected_git_integration.py',
              'scripts/a63_common.py', 'scripts/crossfit_action_selectors.py', 'scripts/train_selected_action_risk.py',
              'tests/test_selected_risk_scope.py', 'tests/test_selected_git_integration.py',
              'configs/experiments/selected_action_risk_v1.json',
              'docs/superpowers/plans/2026-11-05-a63-selected-action-risk.md',
              'docs/results/2026-11-05_selected_action_risk/owner_approval.json']:
        assert m.allowed(p), p
    for p in ['echo_routing/temporal/event_risk_model.py', 'echo_routing/temporal/safety_training.py',
              'echo_routing/temporal/dense_readout.py', 'scripts/train_event_risk.py', 'docs/EXPERIMENT_INDEX.md',
              'docs/superpowers/plans/2026-11-05-a64-other.md', 'docs/results/2026-11-04_anchor_context_risk/x.json',
              'web/index.html', 'docs/results/2026-11-05_selected_action_risk/../x']:
        assert not m.allowed(p), p


def test_explicit_executed_short_name_scripts_do_not_open_a_generic_script_family():
    m = importlib.import_module('echo_routing.temporal.selected_git_integration')
    names = ['selected_risk_shared_contracts', 'supervise_selected_risk_extension',
             'selected_risk_common', 'probe_selected_risk_shape', 'cache_selected_risk',
             'freeze_shared_selected_risk', 'freeze_selected_risk', 'supervise_selected_risk',
             'collect_selected_risk_costs', 'prepare_selected_risk', 'probe_selected_risk',
             'admit_selected_risk', 'admit_selected_risk_refined']
    assert all(m.allowed('scripts/'+name+'.py') for name in names)
    for name in ['selected_risk_unreviewed', 'prepare_selected_risk_v2', 'train_event_risk', 'cache_action_safety']:
        assert not m.allowed('scripts/'+name+'.py')
    for name in ['dense_analysis', 'dense_runtime', 'dense_sequence', 'structured_metrics',
                 'structured_uncertainty', 'dense_readout', 'structured_inputs', 'jepa21_source']:
        assert not m.allowed('echo_routing/temporal/'+name+'.py')


def test_short_name_script_publication_is_new_file_only_with_eight_stages_preserved(toy):
    m = importlib.import_module('echo_routing.temporal.selected_git_integration')
    repo, _, head = toy; path = 'scripts/prepare_selected_risk.py'
    p = repo/path; p.parent.mkdir(); p.write_text('# Synthetic new A63 script\n')
    files = [dict(path=path, sha256=hashlib.sha256(p.read_bytes()).hexdigest())]
    before = (repo/'.git/index').read_bytes(); entries = m.index_entries(before)
    dry = m.integrate(repo, head, 'refs/heads/main', files, 'A63 short-name toy')
    result = m.integrate(repo, head, 'refs/heads/main', files, 'A63 short-name toy',
                         apply=True, expected_index_sha256=dry['expected_index_sha256'])
    assert result['status'] == 'applied'
    after = m.index_entries((repo/'.git/index').read_bytes())
    assert all(after[k] == v for k, v in entries.items())
    assert git(repo, 'diff-tree', '--no-commit-id', '--name-only', '-r', head, result['new_commit']) == path
    assert git(repo, 'show', result['new_commit']+':'+path) == p.read_text().strip()
    assert (repo/'user.txt').read_text() == 'user unstaged\n'


def test_dryrun_and_apply_exact_new_A63_file_preserve_all_eight_staged_and_working_edits(toy):
    m = importlib.import_module('echo_routing.temporal.selected_git_integration'); repo, files, head = toy
    before = (repo/'.git/index').read_bytes(); entries = m.index_entries(before)
    dry = m.integrate(repo, head, 'refs/heads/main', files, 'A63 toy')
    assert dry['status'] == 'dry_run' and git(repo, 'rev-parse', 'HEAD') == head and (repo/'.git/index').read_bytes() == before
    result = m.integrate(repo, head, 'refs/heads/main', files, 'A63 toy', apply=True, expected_index_sha256=dry['expected_index_sha256'])
    assert result['status'] == 'applied'
    after = m.index_entries((repo/'.git/index').read_bytes())
    assert all(after[k] == value for k, value in entries.items())
    assert git(repo, 'diff-tree', '--no-commit-id', '--name-only', '-r', head, result['new_commit']) == files[0]['path']
    assert (repo/'user.txt').read_text() == 'user unstaged\n'
    for i in range(8):
        assert git(repo, 'show', f':theory_{i}.txt') == f'staged theory {i}'
        assert not git(repo, 'ls-tree', '--name-only', result['new_commit'], '--', f'theory_{i}.txt')


def test_new_hook_or_foreign_lock_after_dryrun_rejects_without_bypass_or_cleanup(toy):
    m = importlib.import_module('echo_routing.temporal.selected_git_integration'); repo, files, head = toy
    dry = m.integrate(repo, head, 'refs/heads/main', files, 'A63'); before = (repo/'.git/index').read_bytes()
    hook = repo/'.git/hooks/pre-commit'; hook.write_text('#!/bin/sh\nexit 0\n'); hook.chmod(0o755)
    with pytest.raises(m.IntegrationError):
        m.integrate(repo, head, 'refs/heads/main', files, 'A63', apply=True, expected_index_sha256=dry['expected_index_sha256'])
    assert hook.exists() and (repo/'.git/index').read_bytes() == before
    hook.unlink(); lock = repo/'.git/index.lock'; lock.write_text('foreign')
    with pytest.raises(m.IntegrationError):
        m.integrate(repo, head, 'refs/heads/main', files, 'A63', apply=True, expected_index_sha256=dry['expected_index_sha256'])
    assert lock.read_text() == 'foreign' and git(repo, 'rev-parse', 'HEAD') == head


def test_partial_publish_keeps_evidence_and_does_not_force_restore(toy, monkeypatch):
    m = importlib.import_module('echo_routing.temporal.selected_git_integration'); repo, files, head = toy
    dry = m.integrate(repo, head, 'refs/heads/main', files, 'A63')
    def fail(_): raise RuntimeError('injected after index before ref publication')
    monkeypatch.setattr(m, 'commit_transaction', fail)
    with pytest.raises(m.IntegrationError) as exc:
        m.integrate(repo, head, 'refs/heads/main', files, 'A63', apply=True, expected_index_sha256=dry['expected_index_sha256'])
    report = exc.value.report
    assert report['status'] == 'partial' and report['index_published'] and not report['ref_published']
    assert (repo/'.git/index.lock').exists() and git(repo, 'rev-parse', 'HEAD') == head
    intent = json.loads(__import__('pathlib').Path(report['journal_path']).read_text())
    assert intent['manual_reconciliation_required']


def test_exclude_IO_failure_is_preserved_and_not_repaired_by_helper(toy):
    m = importlib.import_module('echo_routing.temporal.selected_git_integration'); repo, files, head = toy
    exclude = repo/'.git/info/exclude'; exclude.unlink(); exclude.mkdir()
    dry = m.integrate(repo, head, 'refs/heads/main', files, 'A63')
    assert m.integrate(repo, head, 'refs/heads/main', files, 'A63', apply=True, expected_index_sha256=dry['expected_index_sha256'])['status'] == 'applied'
    assert exclude.is_dir()
