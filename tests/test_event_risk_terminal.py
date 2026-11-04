import json,sys,time
from pathlib import Path
from echo_routing.temporal.anchor_cache import atomic_json

def setup(tmp_path,monkeypatch):
 import scripts.wait_event_risk_collector as m
 run=tmp_path/'run';pub=tmp_path/'pub';run.mkdir();pub.mkdir()
 for k,v in [('RUN',run),('PUB',pub),('ROOT',tmp_path)]:monkeypatch.setattr(m,k,v)
 monkeypatch.setattr(m,'write',atomic_json)
 child=tmp_path/'fake_child.py';child.write_text('import os,json,pathlib\np=pathlib.Path('+repr(str(pub))+')\nx={"collector_pid":os.getpid(),"collector_CPU_seconds_through_snapshot":0.0}\n(p/"resources.json").write_text(json.dumps(x))\nimport hashlib\n(p/"terminal_exit_receipt.json").write_text(json.dumps({"resources_sha256":hashlib.sha256((p/"resources.json").read_bytes()).hexdigest()}))\n')
 return m,run,pub,child

def test_actual_fake_child_wait_and_self_status(tmp_path,monkeypatch):
 m,r,p,c=setup(tmp_path,monkeypatch);assert m.run([sys.executable,str(c)])==0;row=json.loads((p/'collector_OS_exit_receipt.json').read_text())
 assert row['observed_collector_OS_exit_code']==0 and row['collector_exit_directly_waited'] and row['caller_SELF_OS_exit_code'] is None and not row['overall_complete']
 assert row['incremental_postsnapshot_and_caller_CPU_seconds']>=0

def test_already_exited_child_exception_still_directly_waits_true_OS(tmp_path,monkeypatch):
 m,r,p,c=setup(tmp_path,monkeypatch);real=m.subprocess.Popen;holder=[]
 class Child:
  def __init__(self,*a,**k):self.inner=real(*a,**k);self.pid=self.inner.pid;self.waits=0;holder.append(self)
  def poll(self):return self.inner.poll()
  def kill(self):return self.inner.kill()
  def wait(self,*a,**k):self.waits+=1;return self.inner.wait(*a,**k)
 monkeypatch.setattr(m.subprocess,'Popen',Child)
 def write(path,v):
  if v.get('state')=='actual_PID_bound':
   holder[0].inner.wait(timeout=5);raise OSError('toy PID-binding-write failure after child exit')
  atomic_json(path,v)
 monkeypatch.setattr(m,'write',write)
 assert m.run([sys.executable,str(c)])==1
 row=json.loads((p/'collector_OS_exit_receipt.json').read_text());assert holder[0].waits==1 and row['observed_collector_OS_exit_code']==0 and row['exception'] and not row['passed']

def test_source_attempt_costs_and_closure_pending():
 from scripts.package_event_risk import source_costs
 from scripts.close_event_risk import completion_state
 producer={'source_production_stage':'sources','records':[{'source':'a08','committed_sequences':1,'committed_logical_neural_units':2,'new_neural_units':2,'resumed_committed_prefix_sequences':0}]}
 rows=[{'stage':'sources','receipt_id':'sources:1','GPU_job':True,'CUDA_context_occupied':True,'full_process_wall_seconds':3.,'measured_CPU_core_seconds':2.,'observed_worker_exit_code':0,'child_exit_code_directly_waited':True}]
 assert source_costs(producer,rows)['actual_forward_units_across_all_attempts']['a08']==2
 rows.append({**rows[0],'receipt_id':'sources:2','observed_worker_exit_code':None})
 assert source_costs(producer,rows)['actual_forward_units_across_all_attempts'] is None
 assert not completion_state()['overall_complete']
