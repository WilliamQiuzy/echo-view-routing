import importlib,sys,json,os
import pytest
from scripts import supervise_structured_v2 as supervisor

def test_actual_closed_failed_worker_cost_and_host_provenance_are_retained(tmp_path, monkeypatch):
    import copy, time
    m = importlib.import_module('scripts.supervise_structured_evidence')
    original_read = m.read
    def synthetic_config(path):
        value = original_read(path)
        if path == m.CFG:
            value = copy.deepcopy(value)
            value['budget']['new_deadline'] = time.time() + 3 * 24 * 3600
            value['budget']['cumulative_deadline'] = value['budget']['new_deadline']
        return value
    monkeypatch.setattr(m, 'read', synthetic_config)
    r=m.execute('synthetic_failure',[sys.executable,'-c','sum(range(20000));raise SystemExit(3)'],False,10,tmp_path)
    assert r['exit_code']==3 and r['process_CPU_seconds']>=0 and r['end_epoch']>=r['start_epoch'] and r['measurement_host'] and isinstance(r['model_pid'],int)
    assert json.loads((tmp_path/'processes.jsonl').read_text())==r and not (tmp_path/'active_process.json').exists()

def test_live_owner_and_same_exclusive_lock_refuse_second_worker(tmp_path):
    m=importlib.import_module('scripts.supervise_structured_evidence');(tmp_path/'active_process.json').write_text(json.dumps({'model_pid':os.getpid()}))
    with pytest.raises(RuntimeError):m.execute('x',[sys.executable,'-c','pass'],False,10,tmp_path)
    (tmp_path/'active_process.json').unlink()
    from echo_routing.temporal.dense_resource_accounting import terminal_queue_lock
    with terminal_queue_lock(tmp_path/'exclusive_supervisor.lock'):
        with pytest.raises(RuntimeError):m.execute('x',[sys.executable,'-c','pass'],False,10,tmp_path)

def test_dual_budget_runtime_limit_retains_failed_cost_and_closure_reserve():
    m=importlib.import_module('scripts.supervise_structured_evidence');b={'new_GPUjob_h':16,'cumulative_GPUjob_h':24,'prior_GPUjob_h':14.3839654494,'closure_GPUreserve_h':2,'new_CPUcore_h':48,'cumulative_CPUcore_h':72,'prior_summed_measured_CPUreceipt_h':14.2846472567,'prior_setup_budget_allowance_h':.25,'new_deadline':200000,'cumulative_deadline':190000}
    rows=[{'GPU_job':True,'full_process_wall_seconds':3600,'process_CPU_seconds':100,'exit_code':1}];r=m.remaining_limits(b,rows,100000,True);assert abs(r['GPU_seconds']-(24-14.3839654494-1-2)*3600)<1e-6;assert r['wall_seconds']==90000
    with pytest.raises(RuntimeError):m.remaining_limits(b,rows,190001,True)

def test_guardian_failure_cannot_become_stage_success_when_child_exits_zero():
    assert supervisor.stage_exit_status({'exit_code':0,'termination_reason':None,'guardian_exception':None})==0
    for row in [{'exit_code':0,'termination_reason':'guarderror','guardian_exception':None},{'exit_code':0,'termination_reason':None,'guardian_exception':'error'},{'exit_code':-15,'termination_reason':None,'guardian_exception':None}]:assert supervisor.stage_exit_status(row)!=0
