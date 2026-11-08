import pytest
from echo_routing.temporal.proposal_resources import totals,forecast,PRIOR_GPU_HOURS,PRIOR_MEASURED_CPU_HOURS,PRIOR_STORAGE_UPPER_BYTES
from scripts.supervise_jepa_proposals import execute

def test_inherited_a59_not_a58_and_original_deadline():
    t=totals([])
    assert PRIOR_GPU_HOURS==18.3349858850903 and PRIOR_MEASURED_CPU_HOURS==18.253671176388888
    assert PRIOR_STORAGE_UPPER_BYTES==34541577626
    f=forecast(t,1000,1000,1000000,1793760000,2000)
    assert f['all_caps_fit'] and f['closure_GPU_reserve_hours']==2
    assert not forecast(t,4*3600,0,0,1793760000,0)['all_caps_fit']
    assert not forecast(t,0,0,6e9,1793760000,0)['all_caps_fit']

def test_namespace_supervisor_refuses_stale_external_cost_and_pending_before_launch(tmp_path):
    import json,sys
    run=tmp_path/'run';pub=tmp_path/'pub';pub.mkdir();run.mkdir()
    p=run/'active_process.json';p.write_text('{}')
    with pytest.raises(RuntimeError):execute('fixture',[sys.executable,'-c','raise AssertionError()'],run,pub,totals([]),2,False,GPU_probe=lambda:[],storage_probe=lambda:0)
    p.unlink()
    (pub/'local_cpu_receipts.json').write_text(json.dumps({'receipts':[{'receipt_id':'bad'}]}))
    with pytest.raises((ValueError,KeyError)):execute('fixture',[sys.executable,'-c','raise AssertionError()'],run,pub,totals([]),2,False,GPU_probe=lambda:[],storage_probe=lambda:0)
    assert not p.exists()

def test_GPU_reserve_does_not_invent_shorter_CPU_wall_deadline():
    from echo_routing.temporal.proposal_resources import remaining_limits,DEADLINE_EPOCH
    t=totals([]);now=DEADLINE_EPOCH-3600
    assert forecast(t,0,10,0,now,100)['all_caps_fit']
    assert remaining_limits(t,now,potential_GPU_context=False)['stage_wall_seconds']==3600

def test_CPU_stage_does_not_claim_foreign_GPU_and_stops_own_context(tmp_path, synthetic_resource_window):
    synthetic_resource_window("echo_routing.temporal.proposal_resources")
    import json,sys
    from scripts.supervise_jepa_proposals import execute
    run=tmp_path/'cpu';pub=tmp_path/'pub'
    r=execute('CPU_fixture',[sys.executable,'-c','import time;time.sleep(.12)'],run,pub,totals([]),2,False,GPU_probe=lambda:[999999],storage_probe=lambda:0)
    assert r['observed_worker_exit_code']==0 and not r['CUDA_context_occupied'] and r['preexisting_foreign_GPU_owners']==[999999]
    def GPU():
        p=run/'active_process.json'
        return [json.loads(p.read_text())['model_pid']] if p.exists() and json.loads(p.read_text()).get('model_pid') else []
    r=execute('context_fixture',[sys.executable,'-c','import time;time.sleep(3)'],run,pub,totals([]),2,False,GPU_probe=GPU,storage_probe=lambda:0)
    assert r['termination_reason']=='unexpected_CPU_stage_CUDA_context' and r['CUDA_context_occupied']

def test_CPU_supervisor_bounds_use_CPU_and_original_deadline(tmp_path, synthetic_resource_window):
    synthetic_resource_window("echo_routing.temporal.proposal_resources")
    import sys
    r=execute('CPU_bounds',[sys.executable,'-c','pass'],tmp_path/'run',tmp_path/'pub',totals([]),14400,False,GPU_probe=lambda:[],storage_probe=lambda:0)
    assert r['effective_stage_wall_limit_seconds']==14400
    assert r['limits_at_launch']['stage_wall_seconds']>r['limits_at_launch']['remaining_GPU_seconds_before_reserve']

def test_entry_failure_is_metered_without_inventing_child_exit(tmp_path):
    from scripts.supervise_jepa_proposals import meter_entry
    import json
    def reject():raise SystemExit(2)
    assert meter_entry(reject,tmp_path/'run',tmp_path/'pub')==2
    rows=[json.loads(p.read_text()) for p in (tmp_path/'pub').glob('entry_CPU_*.json')]
    assert len(rows)==1 and rows[0]['measured_CPU_core_seconds']>=0
    assert rows[0]['observed_worker_exit_code'] is None and rows[0]['planned_CLI_exit_status']==2
    assert not rows[0]['new_worker_started']

def test_entry_complement_does_not_subtract_another_supervisor(tmp_path):
    import json,os,socket
    from scripts.supervise_jepa_proposals import meter_entry
    run=tmp_path/'run';run.mkdir();pub=tmp_path/'pub'
    def callback():
        (run/'processes.jsonl').write_text(json.dumps({'receipt_id':'foreign','measured_CPU_core_seconds':100,'model_pid':123,'supervisor_pid':os.getpid()+999,'measurement_host':socket.gethostname()})+'\n')
        return 0
    assert meter_entry(callback,run,pub)==0
    value=json.loads(next(pub.glob('entry_CPU_*.json')).read_text())
    assert value['already_charged_stage_CPU_seconds_subtracted']==0 and not value['new_worker_started']
