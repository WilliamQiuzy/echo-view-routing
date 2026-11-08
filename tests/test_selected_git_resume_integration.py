"""New resume publication scope and transaction only; isolated temporary Git."""
import hashlib
import importlib.util
from pathlib import Path
import subprocess


def helper():
    path = Path(__file__).resolve().parents[1]/'echo_routing/temporal/selected_git_resume_integration.py'
    spec = importlib.util.spec_from_file_location('private_selected_git_resume_integration', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def git(repo, *args):
    return subprocess.check_output(['git', '-C', str(repo), *args], text=True).strip()


def test_resume_scope_has_only_new_public_namespaces_and_original_code_patterns():
    m = helper()
    positive = [
        'docs/results/2026-11-05_a63_resume_preparation/review.json',
        'docs/results/2026-11-05_a63_resume_execution/subdir/result.json',
        'docs/superpowers/plans/2026-11-05-a63-storage-window-amendment.md',
        'echo_routing/temporal/selected_git_resume_integration.py',
        'tests/test_selected_git_resume_integration.py',
        'echo_routing/temporal/selected_risk_resume_analysis.py',
        'tests/test_selected_risk_resume_analysis.py',
        'scripts/selected_risk_shared_contracts.py',
        'scripts/supervise_selected_risk_extension.py',
        'scripts/selected_risk_common.py', 'scripts/probe_selected_risk_shape.py',
        'scripts/cache_selected_risk.py', 'scripts/freeze_shared_selected_risk.py',
        'scripts/freeze_selected_risk.py', 'scripts/supervise_selected_risk.py',
        'scripts/collect_selected_risk_costs.py', 'scripts/prepare_selected_risk.py',
        'scripts/probe_selected_risk.py', 'scripts/admit_selected_risk.py',
        'scripts/admit_selected_risk_refined.py',
        'scripts/probe_selected_action_risk.py', 'scripts/a63_resume_selected_risk.py',
        'scripts/crossfit_action_selectors.py',
    ]
    negative = [
        'docs/results/2026-11-05_selected_action_risk/result.json',
        'docs/results/2026-11-05_a63_resume_preparation_other/result.json',
        'docs/results/2026-11-05_a63_resume_execution_other/result.json',
        'docs/results/2026-11-05_a62_resume_execution/result.json',
        'docs/results/2026-11-05_a63_resume_preparation',
        'docs/superpowers/plans/2026-11-05-a63-selected-action-risk.md',
        'docs/superpowers/plans/2026-11-05-a63-card-revision.receipt.json',
        'docs/superpowers/plans/2026-11-05-a63-storage-window-amendment.md/extra',
        'configs/experiments/selected_action_risk_v1.json',
        'echo_routing/temporal/selected_git_integration.py',
        'tests/test_selected_git_integration.py',
        'scripts/arbitrary_selected_risk.py',
        'echo_routing/temporal/other.py', 'tests/test_other.py',
        'RUN/result.json', 'README.md', '/tmp/result.json', '../result.json',
        'docs/results/2026-11-05_a63_resume_execution/../old.json',
        'docs/results/2026-11-05_a63_resume_execution/./new.json',
        'docs/results/2026-11-05_a63_resume_execution//new.json',
        'docs/results/2026-11-05_a63_resume_execution/bad\\name.json',
        'docs/results/2026-11-05_a63_resume_execution/bad\x00name.json',
    ]
    assert all(m.allowed(p) for p in positive)
    assert not any(m.allowed(p) for p in negative)


def test_toy_dryrun_then_apply_preserves_staged_and_unstaged_user_bytes(tmp_path):
    m = helper(); repo = tmp_path/'toy'; repo.mkdir()
    subprocess.run(['git', 'init', '-q', '-b', 'main', str(repo)], check=True)
    git(repo, 'config', 'user.name', 'Toy')
    git(repo, 'config', 'user.email', 'toy@example.invalid')
    (repo/'user.txt').write_text('committed\n')
    git(repo, 'add', 'user.txt')
    git(repo, '-c', 'commit.gpgsign=false', 'commit', '-qm', 'base')
    for i in range(8): (repo/f'theory_{i}.txt').write_text(f'staged theory {i}\n')
    git(repo, 'add', *[f'theory_{i}.txt' for i in range(8)])
    (repo/'user.txt').write_text('user unstaged\n')
    (repo/'theory_0.txt').write_text('user unstaged over staged theory 0\n')
    path = 'docs/results/2026-11-05_a63_resume_execution/toy.json'
    payload = repo/path; payload.parent.mkdir(parents=True); payload.write_bytes(b'{"toy":true}\n')
    files = [{'path': path, 'sha256': hashlib.sha256(payload.read_bytes()).hexdigest()}]
    head = git(repo, 'rev-parse', 'HEAD'); before = (repo/'.git/index').read_bytes()
    entries = m.index_entries(before)
    dry = m.integrate(repo, head, 'refs/heads/main', files, 'A63 resume toy')
    assert dry['status'] == 'dry_run'
    assert git(repo, 'rev-parse', 'HEAD') == head and (repo/'.git/index').read_bytes() == before
    result = m.integrate(repo, head, 'refs/heads/main', files, 'A63 resume toy',
                         apply=True, expected_index_sha256=dry['expected_index_sha256'])
    assert result['status'] == 'applied'
    after = m.index_entries((repo/'.git/index').read_bytes())
    assert all(after[k] == value for k, value in entries.items())
    assert git(repo, 'diff-tree', '--no-commit-id', '--name-only', '-r', head, result['new_commit']) == path
    assert subprocess.check_output(['git', '-C', str(repo), 'show', f'{result["new_commit"]}:{path}']) == payload.read_bytes()
    assert (repo/'user.txt').read_bytes() == b'user unstaged\n'
    assert (repo/'theory_0.txt').read_bytes() == b'user unstaged over staged theory 0\n'
    for i in range(8):
        assert git(repo, 'show', f':theory_{i}.txt') == f'staged theory {i}'
        assert not git(repo, 'ls-tree', '--name-only', result['new_commit'], '--', f'theory_{i}.txt')


def test_exact_completed_supply_cost_probe_test_path_only():
    m = helper()
    assert m.allowed('tests/test_a63_completed_supply_cost_probe.py')
    for path in [
        'tests/test_a63_completed_supply_cost_probe_extra.py',
        'tests/test_a63_completed_supply_cost_probe.py/extra',
        'tests/test_a62_completed_supply_cost_probe.py',
        'tests/test_a63_other_cost_probe.py',
        'tests/test_a63_completed_supply_cost_probe.py.bak',
        'tests/../test_a63_completed_supply_cost_probe.py',
        'tests/./test_a63_completed_supply_cost_probe.py',
        'tests//test_a63_completed_supply_cost_probe.py',
        '/tests/test_a63_completed_supply_cost_probe.py',
        '../tests/test_a63_completed_supply_cost_probe.py',
    ]:
        assert not m.allowed(path), path
