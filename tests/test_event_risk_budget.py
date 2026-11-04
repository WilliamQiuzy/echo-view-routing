import importlib
import pytest

def api():return importlib.import_module('echo_routing.temporal.event_risk_budget')

def test_absolute_authorized_window_dual_caps_and_no_CPU_GPU_wall_coupling():
 m=api();s=m.totals([]);assert s['cumulative_GPU_hours']==18.488818130029575
 assert s['cumulative_CPU_accounting_hours']==19.78341831
 assert m.DEADLINE_EPOCH==1793867561 and m.T0==1793824361
 f=m.forecast(s,GPU_seconds=700,CPU_seconds=3000,incremental_peak_bytes=100000000,now=m.T0,remaining_wall_seconds=4000)
 assert f['all_caps_fit'] and f['includes_GPU_reserve_seconds']==7200
 assert not m.forecast(s,GPU_seconds=1801,CPU_seconds=0,incremental_peak_bytes=1,now=m.T0,remaining_wall_seconds=1)['all_caps_fit']
 assert not m.forecast(s,GPU_seconds=0,CPU_seconds=21601,incremental_peak_bytes=1,now=m.T0,remaining_wall_seconds=1)['all_caps_fit']
 assert not m.forecast(s,GPU_seconds=0,CPU_seconds=1,incremental_peak_bytes=1250000001,now=m.T0,remaining_wall_seconds=1)['all_caps_fit']
 assert m.remaining_limits(s,m.DEADLINE_EPOCH-60,potential_GPU_context=False)['stage_wall_seconds']==60
 assert m.remaining_limits(s,m.DEADLINE_EPOCH-60,potential_GPU_context=True)['stage_wall_seconds']==60

def test_receipt_exactonce_fullwall_and_failed_CPU_costs_retained():
 m=api();r={'receipt_id':'bad','started_epoch':m.T0,'finished_epoch':m.T0+2,'full_process_wall_seconds':2,'measured_CPU_core_seconds':1.5,'GPU_job':True,'CUDA_context_occupied':False,'observed_worker_exit_code':1,'measurement_host':'toy','model_pid':1}
 s=m.totals([r]);assert s['new_GPU_wall_seconds']==2 and s['new_measured_CPU_seconds']==1.5 and s['failed_stages']==1
 with pytest.raises(ValueError):m.totals([r,r])
 with pytest.raises(ValueError):m.totals([{**r,'full_process_wall_seconds':1}])
 with pytest.raises(ValueError):m.totals([{**r,'measured_CPU_core_seconds':float('nan')}])
