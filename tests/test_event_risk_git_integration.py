import hashlib,importlib,json,subprocess
from pathlib import Path
import pytest


def git(repo,*args):return subprocess.check_output(['git','-C',str(repo),*args],text=True).strip()


def fixture(tmp_path):
    repo=tmp_path/'toy';repo.mkdir();subprocess.run(['git','init','-q','-b','main',str(repo)],check=True)
    git(repo,'config','user.name','Toy');git(repo,'config','user.email','toy@example.invalid')
    (repo/'user.txt').write_text('committed\n');(repo/'docs').mkdir();(repo/'docs/EXPERIMENT_INDEX.md').write_text('original index document\n');git(repo,'add','user.txt','docs/EXPERIMENT_INDEX.md');git(repo,'commit','-qm','base')
    (repo/'user.txt').write_text('original staged edit\n');git(repo,'add','user.txt');(repo/'user.txt').write_text('extra unstaged edit\n');(repo/'docs/EXPERIMENT_INDEX.md').write_text('existing user document edit\n')
    for i in range(8):(repo/f'docs/user_theory_{i}.md').write_text(f'user staged theory {i}\n')
    git(repo,'add',*[f'docs/user_theory_{i}.md' for i in range(8)]);git(repo,'update-index','--assume-unchanged','user.txt')
    path='docs/results/2026-11-04_anchor_context_risk/toy.json';f=repo/path;f.parent.mkdir(parents=True);f.write_text('{"approved":true}\n');files=[{'path':path,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()}]
    return repo,files,git(repo,'rev-parse','HEAD')


def test_dryrun_then_commit_only_new_A62_files_preserves_user_stage_and_worktree(tmp_path):
    m=importlib.import_module('echo_routing.temporal.event_risk_git_integration');repo,files,head=fixture(tmp_path);index=repo/'.git/index';before=index.read_bytes();objects=sorted(str(p.relative_to(repo/'.git/objects')) for p in (repo/'.git/objects').rglob('*') if p.is_file())
    plan=m.integrate(repo,head,'refs/heads/main',files,'A62 public integration')
    assert plan['status']=='dry_run' and git(repo,'rev-parse','HEAD')==head and index.read_bytes()==before
    assert objects==sorted(str(p.relative_to(repo/'.git/objects')) for p in (repo/'.git/objects').rglob('*') if p.is_file())
    result=m.integrate(repo,head,'refs/heads/main',files,'A62 public integration',apply=True,expected_index_sha256=plan['expected_index_sha256'])
    assert result['status']=='applied' and result['all_original_index_entries_preserved']
    assert git(repo,'diff-tree','--no-commit-id','--name-only','-r',head,result['new_commit'])==files[0]['path']
    assert git(repo,'show',result['new_commit']+':user.txt')=='committed' and git(repo,'show',':user.txt')=='original staged edit'
    for i in range(8):
        assert git(repo,'show',f':docs/user_theory_{i}.md')==f'user staged theory {i}'
        assert not git(repo,'ls-tree','--name-only',result['new_commit'],'--',f'docs/user_theory_{i}.md')
    assert (repo/'user.txt').read_text()=='extra unstaged edit\n' and (repo/'docs/EXPERIMENT_INDEX.md').read_text()=='existing user document edit\n'
    assert not (repo/'.git/index.lock').exists() and not (repo/'.git/HEAD.lock').exists()


def test_plumbing_operates_when_exclude_is_not_a_readable_file(tmp_path):
    m=importlib.import_module('echo_routing.temporal.event_risk_git_integration');repo,files,head=fixture(tmp_path);exclude=repo/'.git/info/exclude';exclude.unlink();exclude.mkdir()
    assert subprocess.run(['git','-C',str(repo),'status','--porcelain'],capture_output=True).returncode!=0
    plan=m.integrate(repo,head,'refs/heads/main',files,'A62 despite unrelated exclude I/O')
    assert m.integrate(repo,head,'refs/heads/main',files,'A62 despite unrelated exclude I/O',apply=True,expected_index_sha256=plan['expected_index_sha256'])['status']=='applied'
    assert exclude.is_dir()


def test_lock_changed_HEAD_or_changed_index_never_publishes(tmp_path):
    m=importlib.import_module('echo_routing.temporal.event_risk_git_integration');repo,files,head=fixture(tmp_path);plan=m.integrate(repo,head,'refs/heads/main',files,'A62');lock=repo/'.git/index.lock';lock.write_text('another owner')
    with pytest.raises(m.IntegrationError) as e:m.integrate(repo,head,'refs/heads/main',files,'A62',apply=True,expected_index_sha256=plan['expected_index_sha256'])
    assert e.value.report['status']=='rejected' and git(repo,'rev-parse','HEAD')==head and lock.read_text()=='another owner'
    lock.unlink();git(repo,'add','docs/EXPERIMENT_INDEX.md')
    with pytest.raises(m.IntegrationError):m.integrate(repo,head,'refs/heads/main',files,'A62',apply=True,expected_index_sha256=plan['expected_index_sha256'])
    assert git(repo,'rev-parse','HEAD')==head
    git(repo,'commit','-qm','independent user commit');new=git(repo,'rev-parse','HEAD')
    with pytest.raises(m.IntegrationError):m.integrate(repo,head,'refs/heads/main',files,'A62',apply=True,expected_index_sha256=plan['expected_index_sha256'])
    assert git(repo,'rev-parse','HEAD')==new


def test_failure_after_index_publish_retains_partial_evidence_and_locks(tmp_path,monkeypatch):
    m=importlib.import_module('echo_routing.temporal.event_risk_git_integration');repo,files,head=fixture(tmp_path);plan=m.integrate(repo,head,'refs/heads/main',files,'A62')
    def fail(*args):
        assert (repo/'.git/HEAD.lock').exists() and (repo/'.git/refs/heads/main.lock').exists()
        raise RuntimeError('injected before ref commit')
    monkeypatch.setattr(m,'commit_transaction',fail)
    with pytest.raises(m.IntegrationError) as e:m.integrate(repo,head,'refs/heads/main',files,'A62',apply=True,expected_index_sha256=plan['expected_index_sha256'])
    r=e.value.report;assert r['status']=='partial' and git(repo,'rev-parse','HEAD')==head and r['index_published'] and not r['ref_published']
    assert (repo/'.git/index.lock').exists();journal=json.loads(Path(r['journal_path']).read_text());assert journal['status']=='partial' and journal['manual_reconciliation_required']
    assert git(repo,'show',':user.txt')=='original staged edit'


def test_after_ref_commit_changed_HEAD_is_partial_without_hiding_target_branch_publish(tmp_path,monkeypatch):
    m=importlib.import_module('echo_routing.temporal.event_risk_git_integration');repo,files,head=fixture(tmp_path);git(repo,'update-ref','refs/heads/other',head);plan=m.integrate(repo,head,'refs/heads/main',files,'A62');commit=m.commit_transaction
    def concurrent_HEAD(tx):commit(tx);git(repo,'symbolic-ref','HEAD','refs/heads/other')
    monkeypatch.setattr(m,'commit_transaction',concurrent_HEAD)
    with pytest.raises(m.IntegrationError) as e:m.integrate(repo,head,'refs/heads/main',files,'A62',apply=True,expected_index_sha256=plan['expected_index_sha256'])
    r=e.value.report;assert r['status']=='partial' and r['ref_published'] and git(repo,'rev-parse','refs/heads/main')==r['new_commit']
    assert git(repo,'symbolic-ref','HEAD')=='refs/heads/other' and (repo/'.git/index.lock').exists()


def test_cli_unknown_post_entry_cleanup_failure_never_claims_apply_did_not_start(tmp_path,monkeypatch,capsys):
    import sys
    from scripts import commit_event_risk_paths as cli
    repo,files,head=fixture(tmp_path);manifest=tmp_path/'paths.json';manifest.write_text(json.dumps(files))
    def escaped(*args,**kwargs):raise RuntimeError('post-entry cleanup escaped classification')
    monkeypatch.setattr(cli,'integrate',escaped);monkeypatch.setattr(sys,'argv',['tool','--repo',str(repo),'--expected-head',head,'--expected-branch','refs/heads/main','--paths-json',str(manifest),'--message','A62','--apply','--expected-index-sha256','0'*64])
    assert cli.main()==2
    value=json.loads(capsys.readouterr().out);assert value['status']=='partial' and value['publication_state_unknown'] and value['manual_reconciliation_required'] and not value['no_integration_apply_entered']


def test_only_exact_authorized_engineering_amendment_plan_is_added(tmp_path):
    m=importlib.import_module('echo_routing.temporal.event_risk_git_integration')
    path='docs/superpowers/plans/2026-11-04-a62-execution-authorization.json'
    assert m.allowed(path)
    assert not m.allowed('docs/superpowers/plans/2026-11-04-a62-other.md')
    repo,files,head=fixture(tmp_path);p=repo/path;p.parent.mkdir(parents=True);p.write_text('approved engineering amendment\n');files.append({'path':path,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    plan=m.integrate(repo,head,'refs/heads/main',files,'literal approved amendment')
    assert plan['status']=='dry_run' and git(repo,'rev-parse','HEAD')==head


def test_fresh_hooks_gate_after_dryrun_is_not_bypassed(tmp_path):
    m=importlib.import_module('echo_routing.temporal.event_risk_git_integration');repo,files,head=fixture(tmp_path)
    dry=m.integrate(repo,head,'refs/heads/main',files,'A62')
    hook=repo/'.git/hooks/pre-commit';hook.write_text('#!/bin/sh\nexit 0\n');hook.chmod(0o755)
    before=(repo/'.git/index').read_bytes()
    with pytest.raises(m.IntegrationError) as e:m.integrate(repo,head,'refs/heads/main',files,'A62',apply=True,expected_index_sha256=dry['expected_index_sha256'])
    assert 'hook' in str(e.value).lower() and git(repo,'rev-parse','HEAD')==head and (repo/'.git/index').read_bytes()==before
    assert hook.exists() and not (repo/'.git/index.lock').exists()
    hook.unlink();git(repo,'config','core.hooksPath','/dev/null')
    with pytest.raises(m.IntegrationError):m.integrate(repo,head,'refs/heads/main',files,'A62')


def test_split_sparse_and_environment_overrides_fail_closed(tmp_path,monkeypatch):
    m=importlib.import_module('echo_routing.temporal.event_risk_git_integration');repo,files,head=fixture(tmp_path)
    before=(repo/'.git/index').read_bytes();git(repo,'config','core.sparseCheckout','true')
    with pytest.raises(m.IntegrationError):m.integrate(repo,head,'refs/heads/main',files,'A62')
    assert (repo/'.git/index').read_bytes()==before
    git(repo,'config','--unset','core.sparseCheckout');git(repo,'update-index','--skip-worktree','user.txt')
    with pytest.raises(m.IntegrationError):m.integrate(repo,head,'refs/heads/main',files,'A62')
    git(repo,'update-index','--no-skip-worktree','user.txt');monkeypatch.setenv('GIT_CONFIG_COUNT','0')
    with pytest.raises(m.IntegrationError):m.integrate(repo,head,'refs/heads/main',files,'A62')


def test_exact_A62_allowlist_excludes_other_rounds_main_index_and_web():
    m=importlib.import_module('echo_routing.temporal.event_risk_git_integration')
    for p in ['echo_routing/temporal/event_risk_model.py','scripts/train_event_risk.py','tests/test_event_risk_scope.py','configs/experiments/anchor_context_risk_v1.json','docs/superpowers/plans/2026-11-04-a62-anchor-context-risk.md','docs/superpowers/plans/2026-11-04-a62-anchor-context-risk.basis.json','docs/results/2026-11-04_anchor_context_risk/report.json']:assert m.allowed(p)
    for p in ['echo_routing/temporal/safety_model.py','scripts/train_action_safety.py','docs/EXPERIMENT_INDEX.md','docs/superpowers/plans/2026-11-04-a63-plan.md','docs/results/2026-11-04_action_safety_ranking/report.json','web/index.html','docs/results/2026-11-04_anchor_context_risk/../x']:assert not m.allowed(p)


def test_default_replace_refs_and_legacy_grafts_are_rejected_without_modifying_them(tmp_path):
    m=importlib.import_module('echo_routing.temporal.event_risk_git_integration');repo,files,head=fixture(tmp_path)
    replacement=git(repo,'commit-tree',git(repo,'rev-parse',head+'^{tree}'),'-m','replacement toy object')
    git(repo,'replace',head,replacement);before=(repo/'.git/index').read_bytes()
    with pytest.raises(m.IntegrationError) as e:m.integrate(repo,head,'refs/heads/main',files,'A62')
    assert 'replace' in str(e.value).lower() and (repo/'.git/index').read_bytes()==before
    assert git(repo,'show-ref','--verify','refs/replace/'+head).split()[0]==replacement
    git(repo,'replace','-d',head);grafts=repo/'.git/info/grafts';grafts.write_text(head+'\n')
    with pytest.raises(m.IntegrationError) as e:m.integrate(repo,head,'refs/heads/main',files,'A62')
    assert 'graft' in str(e.value).lower() and grafts.read_text()==head+'\n'
