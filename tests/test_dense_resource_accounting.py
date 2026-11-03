import importlib
import pytest

def test_closed_costs_count_failed_jobs_and_parallel_CPU_but_not_idle_or_allowances():
    a=importlib.import_module('echo_routing.temporal.dense_resource_accounting')
    rows=[{'model_pid':1,'GPU_job':True,'process_CPU_seconds':2.,'full_process_wall_seconds':3.,'start_epoch':0.,'end_epoch':3.,'exit_code':1},{'model_pid':2,'GPU_job':True,'process_CPU_seconds':7.,'full_process_wall_seconds':10.,'start_epoch':10.,'end_epoch':20.,'exit_code':0},{'model_pid':'localtest','GPU_job':False,'process_CPU_seconds':5.,'full_process_wall_seconds':4.,'ok':True}]
    r=a.closed_totals(rows);assert r['GPU_job_wall_seconds']==13 and r['measured_CPU_core_seconds']==14 and r['failed_GPU_job_wall_seconds']==3

def test_duplicate_workers_or_overlapping_GPU_jobs_are_rejected():
    a=importlib.import_module('echo_routing.temporal.dense_resource_accounting');one={'model_pid':1,'GPU_job':True,'process_CPU_seconds':1.,'full_process_wall_seconds':3.,'start_epoch':0.,'end_epoch':3.,'exit_code':0}
    with pytest.raises(ValueError):a.closed_totals([one,one])
    with pytest.raises(ValueError):a.closed_totals([one,{**one,'model_pid':2,'start_epoch':1.,'end_epoch':4.}])

def test_GPU_receipts_require_finite_closed_intervals_and_consistent_actual_wall():
    a=importlib.import_module('echo_routing.temporal.dense_resource_accounting');one={'model_pid':1,'GPU_job':True,'process_CPU_seconds':1.,'full_process_wall_seconds':3.,'start_epoch':0.,'end_epoch':3.,'exit_code':0}
    for change in [{'start_epoch':float('nan')},{'end_epoch':float('inf')},{'end_epoch':-1.},{'full_process_wall_seconds':1.}]:
        with pytest.raises(ValueError):a.closed_totals([{**one,**change}])

def test_real_PID_deduplication_survives_different_receipt_names_and_missing_provenance_is_disclosed():
    a=importlib.import_module('echo_routing.temporal.dense_resource_accounting');one={'model_pid':23,'measurement_host':'gpu-host','receipt_identity':'one.json','GPU_job':False,'process_CPU_seconds':2.}
    with pytest.raises(ValueError):a.closed_totals([one,{**one,'receipt_identity':'other.json'}])
    with pytest.raises(ValueError):a.closed_totals([one,{**one,'measurement_host':None,'receipt_identity':'other.json'}])
    r=a.closed_totals([one,{'model_pid':None,'receipt_identity':'old-no-PID.json','GPU_job':False,'process_CPU_seconds':3.}])
    assert r['measured_CPU_core_seconds']==5 and r['receipt_only_identity_records']==1 and r['complete_host_PID_start_provenance'] is False
    with pytest.raises(ValueError):a.closed_totals([{'model_pid':None,'GPU_job':False,'process_CPU_seconds':3.}])

def test_terminal_collection_holds_same_nonblocking_queue_lock_until_context_exit(tmp_path):
    a=importlib.import_module('echo_routing.temporal.dense_resource_accounting');path=tmp_path/'exclusive_supervisor.lock'
    with a.terminal_queue_lock(path):
        with pytest.raises(RuntimeError):
            with a.terminal_queue_lock(path):pass
    with a.terminal_queue_lock(path):pass
