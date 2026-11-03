import importlib,copy
import pytest

def row(pid=1):return {'stage':'cache_fit','model_pid':pid,'measurement_host':'host','start_epoch':10.,'end_epoch':20.,'full_process_wall_seconds':10.,'process_CPU_seconds':8.,'GPU_job':True,'exit_code':0,'child_exit_code_directly_waited':True}

def test_terminal_costs_retain_unknown_orphan_exit_and_separate_nonmeasured_allowance():
    m=importlib.import_module('echo_routing.temporal.structured_resources');r={**row(),'child_exit_code_directly_waited':False,'completion_verified_by_frozen_complete_cache_manifest':True,'unmeasured_completed_CPU_accounting_allowance_seconds':900};d=m.closed_costs([r]);assert d['GPU_job_wall_seconds']==10 and d['summed_measured_CPU_receipt_seconds']==8 and d['completed_unmeasured_CPU_accounting_allowance_seconds']==900 and d['worker_exit_statuses'][0]['observed_worker_exit_code'] is None and d['worker_exit_statuses'][0]['manifest_complete'] is True and d['unknown_worker_OS_exit_statuses']==1

def test_nonfinite_or_undercharged_or_overlapping_or_duplicate_receipts_fail_closed():
    m=importlib.import_module('echo_routing.temporal.structured_resources')
    for change in [{'end_epoch':float('nan')},{'end_epoch':9.},{'full_process_wall_seconds':1.},{'process_CPU_seconds':float('inf')}]:
        with pytest.raises(ValueError):m.closed_costs([{**row(),**change}])
    with pytest.raises(ValueError):m.closed_costs([row(),row()])
    with pytest.raises(ValueError):m.closed_costs([row(),{**row(2),'start_epoch':19.,'end_epoch':29.}])
