import json
from pathlib import Path
import pytest


def test_one_direct_wait_and_incremental_cpu_not_doublecounted(tmp_path,monkeypatch):
    from scripts import call_action_safety_collector as m
    run=tmp_path/'run';pub=tmp_path/'pub';run.mkdir();pub.mkdir();scripts=tmp_path/'scripts';scripts.mkdir()
    source="""import os,json,resource,pathlib
p=pathlib.Path(__file__).resolve().parents[1]/'pub'
a=resource.getrusage(resource.RUSAGE_SELF)
cpu=a.ru_utime+a.ru_stime
(p/'resources.json').write_text(json.dumps({'passed':True,'collector':{'model_pid':os.getpid(),'CPU_seconds_through_snapshot':cpu}}))
(p/'terminal_exit_receipt.json').write_text(json.dumps({'passed':True}))
"""
    (scripts/'close_action_safety.py').write_text(source);monkeypatch.setattr(m,'ROOT',tmp_path);monkeypatch.setattr(m,'RUN',run);monkeypatch.setattr(m,'PUB',pub);monkeypatch.setattr(m,'verify_freeze',lambda _:True);monkeypatch.setattr(m,'DEADLINE_EPOCH',10**12)
    assert m.main()==0
    r=json.loads((pub/'collector_OS_exit_receipt.json').read_text());assert r['observed_collector_OS_exit_code']==0 and r['collector_exit_directly_waited'] and r['parent_SELF_OS_exit_code'] is None
    assert r['total_incremental_CPU_seconds_after_resource_snapshot']==pytest.approx(r['collector_postsnapshot_CPU_seconds']+r['caller_SELF_CPU_seconds_through_sample'])
    assert r['collector_postsnapshot_CPU_seconds']>=0 and not r['overall_complete']
    with pytest.raises(ValueError):m.main()
