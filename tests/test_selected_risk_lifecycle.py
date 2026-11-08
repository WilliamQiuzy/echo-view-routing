import importlib,json,sys
import pytest
from echo_routing.temporal.selected_risk_budget import totals

def mod():
 try:m=importlib.import_module('scripts.supervise_selected_risk')
 except ModuleNotFoundError:m=None
 assert m is not None,'new A63 sole guarded supervisor absent'
 return m

def test_real_toy_child_exit_seven_is_directly_waited_and_charged(tmp_path, synthetic_resource_window):
 synthetic_resource_window("echo_routing.temporal.selected_risk_budget")
 m=mod();r=tmp_path/'run';p=tmp_path/'pub'
 row=m.execute('toy',[sys.executable,'-c','raise SystemExit(7)'],r,p,totals([]),20,False,storage_probe=lambda:0,GPU_probe=lambda:[])
 assert row['observed_worker_exit_code']==7 and row['child_exit_code_directly_waited']
 assert row['measured_CPU_core_seconds']>0 and row['CUDA_visibility_hidden'] and not row['GPU_job']
 assert not (r/'active_process.json').exists() and json.loads((r/'process_toy_1.json').read_text())==row
 assert m.stage_exit_status(row)==1

def test_owned_queue_claim_not_overwritten_and_foreign_GPU_not_killed(tmp_path):
 m=mod();r=tmp_path/'run';p=tmp_path/'pub';r.mkdir();(r/'active_process.json').write_text('{"owner":"prior"}')
 with pytest.raises(RuntimeError):m.execute('toy',[sys.executable,'-c','pass'],r,p,totals([]),20,False,storage_probe=lambda:0,GPU_probe=lambda:[])
 assert (r/'active_process.json').read_text()=='{"owner":"prior"}'
 (r/'active_process.json').unlink()
 with pytest.raises(RuntimeError):m.execute('toy',[sys.executable,'-c','pass'],r,p,totals([]),20,True,storage_probe=lambda:0,GPU_probe=lambda:[123])
 assert not (r/'processes.jsonl').exists()

def test_guard_stage_path_rejected_before_any_log_mutation(tmp_path):
 m=mod();old=tmp_path/'old';old.mkdir();sentinel=old/'escape_1.stdout.log';sentinel.write_text('keep')
 with pytest.raises(ValueError):m.execute('../old/escape',[sys.executable,'-c','pass'],tmp_path/'run',tmp_path/'pub',totals([]),20,False,storage_probe=lambda:0,GPU_probe=lambda:[])
 assert sentinel.read_text()=='keep'

def test_guard_ancestor_symlink_rejected_before_queue_or_launch(tmp_path):
 m=mod();old=tmp_path/'old';old.mkdir();run=tmp_path/'run';run.symlink_to(old,target_is_directory=True)
 with pytest.raises(ValueError):m.execute('toy',[sys.executable,'-c','pass'],run,tmp_path/'pub',totals([]),20,False,storage_probe=lambda:0,GPU_probe=lambda:[])
 assert not list(old.iterdir())

def test_caps_already_over_storage_refuses_child_before_Popen(tmp_path):
 m=mod();run=tmp_path/'run'
 with pytest.raises(RuntimeError):m.execute('toy',[sys.executable,'-c','pass'],run,tmp_path/'pub',totals([]),20,False,storage_probe=lambda:1000000001,GPU_probe=lambda:[])
 assert not (run/'processes.jsonl').exists()

def test_caps_fast_child_final_storage_cannot_certify_pass(tmp_path, synthetic_resource_window):
 synthetic_resource_window("echo_routing.temporal.selected_risk_budget")
 m=mod();calls=[]
 def storage():
  calls.append(1);return 0 if len(calls)==1 else 1000000001
 row=m.execute('toy',[sys.executable,'-c','pass'],tmp_path/'run',tmp_path/'pub',totals([]),20,False,storage_probe=storage,GPU_probe=lambda:[])
 assert row['observed_worker_exit_code']==0 and m.stage_exit_status(row)==1
 assert row['maximum_sampled_tracked_storage_upper_bytes']>=1000000001
