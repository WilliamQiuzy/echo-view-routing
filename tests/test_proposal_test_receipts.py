import importlib
import json
import sys
import pytest


def test_existing_stage_cannot_execute_child_again_and_claim_precedes_Popen(tmp_path,monkeypatch):
    m=importlib.import_module('scripts.run_proposal_tests');root=tmp_path/'project';(root/'scripts').mkdir(parents=True)
    child=root/'test_probe.py';marker=root/'physical_child_marker'
    child.write_text(f"from pathlib import Path\ndef test_probe():\n p=Path({str(marker)!r});p.write_text(p.read_text()+'x' if p.exists() else 'x')\n")
    monkeypatch.setattr(m,'__file__',str(root/'scripts/run_proposal_tests.py'))
    monkeypatch.setattr(sys,'argv',['runner','same_stage',str(child)])
    assert m.main()==0 and marker.read_text()=='x'
    with pytest.raises((ValueError,FileExistsError)):m.main()
    assert marker.read_text()=='x'


def test_claim_is_directory_durable_before_spawn_and_launch_failure_is_recorded(tmp_path,monkeypatch):
    import os,stat
    m=importlib.import_module('scripts.run_proposal_tests');root=tmp_path/'project';(root/'scripts').mkdir(parents=True)
    monkeypatch.setattr(m,'__file__',str(root/'scripts/run_proposal_tests.py'));monkeypatch.setattr(sys,'argv',['runner','launch_failure','unused.py'])
    folder=root/'docs/results/2026-11-03_jepa_proposals';claim=folder/'launch_failure.claim.json';synced=[];real=os.fsync
    def fsync(fd):
        if stat.S_ISDIR(os.fstat(fd).st_mode) and claim.exists():synced.append(True)
        return real(fd)
    monkeypatch.setattr(m.os,'fsync',fsync)
    def refuse(*args,**kwargs):
        assert claim.exists() and synced
        raise OSError('synthetic launch failure')
    monkeypatch.setattr(m.subprocess,'Popen',refuse)
    assert m.main()==1
    value=json.loads((folder/'launch_failure.json').read_text())
    assert value['observed_worker_exit_code'] is None and 'launch failure' in value['exception']
    assert value['measured_CPU_core_seconds']>=0 and not claim.exists()


def test_timeout_retains_observed_child_status_and_refuses_second_spawn(tmp_path,monkeypatch):
    m=importlib.import_module('scripts.run_proposal_tests');root=tmp_path/'project';(root/'scripts').mkdir(parents=True)
    child=root/'test_slow.py';child.write_text('import time\ndef test_slow():time.sleep(3)\n')
    monkeypatch.setattr(m,'__file__',str(root/'scripts/run_proposal_tests.py'))
    monkeypatch.setattr(sys,'argv',['runner','--timeout-seconds','.05','timeout',str(child)])
    assert m.main()==1
    p=root/'docs/results/2026-11-03_jepa_proposals/timeout.json';v=json.loads(p.read_text())
    assert v['termination_reason']=='timeout' and v['observed_worker_exit_code'] is not None
    with pytest.raises((ValueError,FileExistsError)):m.main()
