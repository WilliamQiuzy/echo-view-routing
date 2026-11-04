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
    path='docs/results/2026-11-04_action_safety_ranking/toy.json';f=repo/path;f.parent.mkdir(parents=True);f.write_text('{"approved":true}\n');files=[{'path':path,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()}]
    return repo,files,git(repo,'rev-parse','HEAD')


def test_dryrun_then_commit_only_new_A61_files_preserves_user_stage_and_worktree(tmp_path):
    m=importlib.import_module('echo_routing.temporal.safety_git_integration');repo,files,head=fixture(tmp_path);index=repo/'.git/index';before=index.read_bytes();objects=sorted(str(p.relative_to(repo/'.git/objects')) for p in (repo/'.git/objects').rglob('*') if p.is_file())
    plan=m.integrate(repo,head,'refs/heads/main',files,'A61 public integration')
    assert plan['status']=='dry_run' and git(repo,'rev-parse','HEAD')==head and index.read_bytes()==before
    assert objects==sorted(str(p.relative_to(repo/'.git/objects')) for p in (repo/'.git/objects').rglob('*') if p.is_file())
    result=m.integrate(repo,head,'refs/heads/main',files,'A61 public integration',apply=True,expected_index_sha256=plan['expected_index_sha256'])
    assert result['status']=='applied' and result['all_original_index_entries_preserved']
    assert git(repo,'diff-tree','--no-commit-id','--name-only','-r',head,result['new_commit'])==files[0]['path']
    assert git(repo,'show',result['new_commit']+':user.txt')=='committed' and git(repo,'show',':user.txt')=='original staged edit'
    for i in range(8):
        assert git(repo,'show',f':docs/user_theory_{i}.md')==f'user staged theory {i}'
        assert not git(repo,'ls-tree','--name-only',result['new_commit'],'--',f'docs/user_theory_{i}.md')
    assert (repo/'user.txt').read_text()=='extra unstaged edit\n' and (repo/'docs/EXPERIMENT_INDEX.md').read_text()=='existing user document edit\n'
    assert not (repo/'.git/index.lock').exists() and not (repo/'.git/HEAD.lock').exists()


def test_plumbing_operates_when_exclude_is_not_a_readable_file(tmp_path):
    m=importlib.import_module('echo_routing.temporal.safety_git_integration');repo,files,head=fixture(tmp_path);exclude=repo/'.git/info/exclude';exclude.unlink();exclude.mkdir()
    assert subprocess.run(['git','-C',str(repo),'status','--porcelain'],capture_output=True).returncode!=0
    plan=m.integrate(repo,head,'refs/heads/main',files,'A61 despite unrelated exclude I/O')
    assert m.integrate(repo,head,'refs/heads/main',files,'A61 despite unrelated exclude I/O',apply=True,expected_index_sha256=plan['expected_index_sha256'])['status']=='applied'
    assert exclude.is_dir()


def test_lock_changed_HEAD_or_changed_index_never_publishes(tmp_path):
    m=importlib.import_module('echo_routing.temporal.safety_git_integration');repo,files,head=fixture(tmp_path);plan=m.integrate(repo,head,'refs/heads/main',files,'A61');lock=repo/'.git/index.lock';lock.write_text('another owner')
    with pytest.raises(m.IntegrationError) as e:m.integrate(repo,head,'refs/heads/main',files,'A61',apply=True,expected_index_sha256=plan['expected_index_sha256'])
    assert e.value.report['status']=='rejected' and git(repo,'rev-parse','HEAD')==head and lock.read_text()=='another owner'
    lock.unlink();git(repo,'add','docs/EXPERIMENT_INDEX.md')
    with pytest.raises(m.IntegrationError):m.integrate(repo,head,'refs/heads/main',files,'A61',apply=True,expected_index_sha256=plan['expected_index_sha256'])
    assert git(repo,'rev-parse','HEAD')==head
    git(repo,'commit','-qm','independent user commit');new=git(repo,'rev-parse','HEAD')
    with pytest.raises(m.IntegrationError):m.integrate(repo,head,'refs/heads/main',files,'A61',apply=True,expected_index_sha256=plan['expected_index_sha256'])
    assert git(repo,'rev-parse','HEAD')==new


def test_failure_after_index_publish_retains_partial_evidence_and_locks(tmp_path,monkeypatch):
    m=importlib.import_module('echo_routing.temporal.safety_git_integration');repo,files,head=fixture(tmp_path);plan=m.integrate(repo,head,'refs/heads/main',files,'A61')
    def fail(*args):
        assert (repo/'.git/HEAD.lock').exists() and (repo/'.git/refs/heads/main.lock').exists()
        raise RuntimeError('injected before ref commit')
    monkeypatch.setattr(m,'commit_transaction',fail)
    with pytest.raises(m.IntegrationError) as e:m.integrate(repo,head,'refs/heads/main',files,'A61',apply=True,expected_index_sha256=plan['expected_index_sha256'])
    r=e.value.report;assert r['status']=='partial' and git(repo,'rev-parse','HEAD')==head and r['index_published'] and not r['ref_published']
    assert (repo/'.git/index.lock').exists();journal=json.loads(Path(r['journal_path']).read_text());assert journal['status']=='partial' and journal['manual_reconciliation_required']
    assert git(repo,'show',':user.txt')=='original staged edit'


def test_after_ref_commit_changed_HEAD_is_partial_without_hiding_target_branch_publish(tmp_path,monkeypatch):
    m=importlib.import_module('echo_routing.temporal.safety_git_integration');repo,files,head=fixture(tmp_path);git(repo,'update-ref','refs/heads/other',head);plan=m.integrate(repo,head,'refs/heads/main',files,'A61');commit=m.commit_transaction
    def concurrent_HEAD(tx):commit(tx);git(repo,'symbolic-ref','HEAD','refs/heads/other')
    monkeypatch.setattr(m,'commit_transaction',concurrent_HEAD)
    with pytest.raises(m.IntegrationError) as e:m.integrate(repo,head,'refs/heads/main',files,'A61',apply=True,expected_index_sha256=plan['expected_index_sha256'])
    r=e.value.report;assert r['status']=='partial' and r['ref_published'] and git(repo,'rev-parse','refs/heads/main')==r['new_commit']
    assert git(repo,'symbolic-ref','HEAD')=='refs/heads/other' and (repo/'.git/index.lock').exists()


def test_cli_unknown_post_entry_cleanup_failure_never_claims_apply_did_not_start(tmp_path,monkeypatch,capsys):
    import sys
    from scripts import commit_action_safety_paths as cli
    repo,files,head=fixture(tmp_path);manifest=tmp_path/'paths.json';manifest.write_text(json.dumps(files))
    def escaped(*args,**kwargs):raise RuntimeError('post-entry cleanup escaped classification')
    monkeypatch.setattr(cli,'integrate',escaped);monkeypatch.setattr(sys,'argv',['tool','--repo',str(repo),'--expected-head',head,'--expected-branch','refs/heads/main','--paths-json',str(manifest),'--message','A61','--apply','--expected-index-sha256','0'*64])
    assert cli.main()==2
    value=json.loads(capsys.readouterr().out);assert value['status']=='partial' and value['publication_state_unknown'] and value['manual_reconciliation_required'] and not value['no_integration_apply_entered']
