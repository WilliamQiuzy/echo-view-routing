from pathlib import Path
import importlib,json,sys
import pytest

def test_A62_claim_before_child_direct_wait_and_foreign_CPU_allowed(tmp_path,monkeypatch):
 m=importlib.import_module('scripts.supervise_event_risk');b=importlib.import_module('echo_routing.temporal.event_risk_budget');orig=m.subprocess.Popen
 def check(*a,**k):
  assert json.loads((tmp_path/'run/active_process.json').read_text())['state']=='durable_launch_intent'
  assert k['env']['CUDA_VISIBLE_DEVICES']==''
  assert k['env']['PYTHONDONTWRITEBYTECODE']=='1' and Path(k['env']['TMPDIR']).is_relative_to(tmp_path/'run') and Path(k['env']['MPLCONFIGDIR']).is_relative_to(tmp_path/'run')
  return orig(*a,**k)
 monkeypatch.setattr(m.subprocess,'Popen',check)
 row=m.execute('toy',[sys.executable,'-c','raise SystemExit(7)'],tmp_path/'run',tmp_path/'pub',b.totals([]),4,False,storage_probe=lambda:0,GPU_probe=lambda:[999999])
 assert row['observed_worker_exit_code']==7 and row['child_exit_code_directly_waited'] and row['preexisting_foreign_GPU_owners']==[999999] and row['measured_CPU_core_seconds']>0
 assert not (tmp_path/'run/active_process.json').exists()
 with b.ExclusiveQueue(tmp_path/'run/exclusive_supervisor.lock'):
  with pytest.raises(BlockingIOError):m.execute('bad',[],tmp_path/'run',tmp_path/'pub',b.totals([]),4,False)
 assert m.stage_exit_status({**row,'observed_worker_exit_code':0,'termination_reason':'guardian_fail'})==1

def test_locked_ledger_merge_and_GPU_foreign_refusal_has_no_child(tmp_path):
 m=importlib.import_module('scripts.supervise_event_risk');b=importlib.import_module('echo_routing.temporal.event_risk_budget');run=tmp_path/'r';pub=tmp_path/'p';run.mkdir();pub.mkdir()
 with pytest.raises(RuntimeError):m.execute('GPU',[sys.executable,'-c','pass'],run,pub,b.totals([]),3,True,storage_probe=lambda:0,GPU_probe=lambda:[123])
 assert not (run/'active_process.json').exists()
