import json,sys,time
from pathlib import Path
from echo_routing.temporal.anchor_cache import atomic_json

def setup(tmp_path,monkeypatch):
 import scripts.wait_event_risk_collector as m
 run=tmp_path/'run';pub=tmp_path/'pub';run.mkdir();pub.mkdir()
 for k,v in [('RUN',run),('PUB',pub),('ROOT',tmp_path)]:monkeypatch.setattr(m,k,v)
 monkeypatch.setattr(m,'write',atomic_json);monkeypatch.setattr(m,'checked_output',lambda p:Path(p))
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

def test_exclusive_caller_claim_and_partial_status(tmp_path):
 import pytest
 from scripts.wait_event_risk_collector import reserve_intent
 from scripts.close_event_risk import completion_state
 p=tmp_path/'claim.json';reserve_intent(p,{'caller_pid':1});before=p.read_bytes()
 with pytest.raises(FileExistsError):reserve_intent(p,{'caller_pid':2})
 assert p.read_bytes()==before
 state=completion_state(False)
 assert not state['server_scientific_collection_complete'] and not state['overall_complete'] and state['scientific_incomplete']

def test_bad_attempt_refuses_before_freeze_or_mkdir(monkeypatch):
 import pytest
 import scripts.package_event_risk as m
 monkeypatch.setattr(m,'authorize',lambda a:True);monkeypatch.setattr(m,'verify_freeze',lambda n:(_ for _ in ()).throw(AssertionError('bad attempt reached freeze')))
 for a in ['/tmp/escape','../escape']:
  monkeypatch.setattr(sys,'argv',['package','--execute','--attempt',a])
  with pytest.raises(ValueError,match='attempt'):m.main()


def test_complete_report_cannot_be_written_from_partial_counts(monkeypatch):
 import pytest
 import scripts.report_event_risk as m
 monkeypatch.setattr(m,'verify_freeze',lambda n:{})
 values={'summary.json':{'paired_comparisons':[]},'outer_metrics.json':{'rows':[]},'inner_metrics.json':{'rows':[]},'model_audit.json':{'formal_optimizer_updates':0,'models':[],'fullstates':[]},'calibration.json':{'thresholds':[]},'action_space_diagnostic.json':{'new_full_oracle_count':0,'readonly_old_oracle_count':0,'support_count':0},'native_attribution.json':{'cell_count':0,'base_reference_count':0},'regression_summary.json':{'record_count':0},'figure_receipt.json':{'artifacts':[]}}
 monkeypatch.setattr(m,'read',lambda p:values[Path(p).name])
 with pytest.raises(ValueError,match='complete'):m.report()

def test_unspawned_launch_failure_and_nullable_actual_caps():
 from scripts.close_event_risk import has_retained_failure,actual_caps_status
 row={'observed_worker_exit_code':None,'child_exit_code_directly_waited':False,'termination_reason':'guardian_exception_fail_closed'}
 assert has_retained_failure([row],[],None,[])
 assert has_retained_failure([],[],{'state':'durable_launch_intent'},[])
 assert has_retained_failure([],[],None,['pending.json'])
 assert actual_caps_status(1,0,105000000,1793830000,True) is None
 assert actual_caps_status(21601,0,105000000,1793830000,False) is False


def test_complete_report_audit_acceptance_and_figure_crossbindings(monkeypatch):
 import pytest
 import scripts.report_event_risk as m
 from echo_routing.temporal.event_risk_metrics import plan
 p=plan();h='a'*64;summary={'paired_comparisons':p['comparisons'],**{k:h for k in ['outer_metrics_sha256','models_manifest_sha256','calibration_sha256','readout_plan_sha256','acceptance_sha256']}}
 values={'summary.json':summary,'outer_metrics.json':{'rows':p['rows']},'inner_metrics.json':{'rows':p['inner_rows']},'model_audit.json':{'passed':True,'formal_optimizer_updates':24000,'models':[{}]*4,'fullstates':[{}]*16},'calibration.json':{'thresholds':[{}]*6},'action_space_diagnostic.json':{'new_full_oracle_count':6,'readonly_old_oracle_count':30,'support_count':48},'native_attribution.json':{'cell_count':24,'base_reference_count':2},'regression_summary.json':{'record_count':16},'figure_receipt.json':{'artifacts':[{'path':f'figure{i}','sha256':h} for i in range(18)],**{k:h for k in ['summary_sha256','outer_metrics_sha256','inner_metrics_sha256','regression_sha256','native_sha256']}}}
 monkeypatch.setattr(m,'verify_freeze',lambda n:{});monkeypatch.setattr(m,'read',lambda p:values[Path(p).name]);monkeypatch.setattr(m,'sha',lambda p:h)
 assert m.validate_report_inputs()
 for obj,key,bad in [(summary,'acceptance_sha256','b'*64),(values['model_audit.json'],'passed',False),(values['figure_receipt.json'],'regression_sha256','b'*64)]:
  old=obj[key];obj[key]=bad
  with pytest.raises(ValueError):m.validate_report_inputs()
  obj[key]=old
