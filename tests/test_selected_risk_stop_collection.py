import importlib
import copy
import pytest
import json
from pathlib import Path

def api():
    try:return importlib.import_module('scripts.a63_collect_selected_risk_stop')
    except ModuleNotFoundError:pytest.fail('resource-stop collector absent')

def row(i,code=0):
    return dict(receipt_id=i,measurement_host='host',model_pid=int(i[-1])+1,started_epoch=10.,finished_epoch=11.,full_process_wall_seconds=1.,measured_CPU_core_seconds=.5,GPU_job=False,CUDA_context_occupied=False,observed_worker_exit_code=code,child_exit_code_directly_waited=True)

def test_recovery_alias_merges_one_exact_actual_receipt_without_losing_failure():
    m=api();a=row('stage0');b=row('stage1',1)
    assert m.merge_receipts([a],[copy.deepcopy(a),b])==[a,b]
    bad=copy.deepcopy(a);bad['observed_worker_exit_code']=1
    with pytest.raises(ValueError):m.merge_receipts([a],[bad])
    with pytest.raises(ValueError):m.merge_receipts([a,a],[])

def test_resource_stop_preserves_unassessed_coverage_and_all_zero_science():
    m=api();source={k:0 for k in ['teacher_payloads','A08_windows','JEPA_chunks','teacher_forwards','aux_optimizer_updates','formal_optimizer_updates','C0_C1_instances']}
    stop={'mode':'resource-stopped','coverage_assessed':False,'source_counters':source,'scientific_queues_stopped':True,'baseline_not_reset':True}
    projection={'diagnostic_only_not_stage_admission':True,'remaining_wall_seconds':23684.,'now_epoch':1793874258.,'deadline_epoch':1793897040,'forecast_end_if_started_now':1793897942.}
    rows=[row('stage0'),row('stage1',1)]
    s=m.build_stop_state(stop,projection,rows,rows)
    assert s['coverage'] is None and s['coverage_passed'] is None and s['goal_met'] is False and s['overall_complete'] is False
    assert s['actual_counters']['aux_optimizer_updates']==0 and s['OS_summary']['nonzero_receipts']==['stage1']
    bad=copy.deepcopy(stop);bad['source_counters']['teacher_forwards']=1
    with pytest.raises(ValueError):m.build_stop_state(bad,projection,rows,rows)
    p=copy.deepcopy(projection);p['forecast_end_if_started_now']-=10000
    with pytest.raises(ValueError):m.build_stop_state(stop,p,rows,rows)

def test_recovered_parent_record_requires_actual_wait_and_zero_neural_scope():
    m=api();r={'actual_supervisor_OS_exit_code':0,'actual_supervisor_directly_waited':True,'error':None,'new_process_receipts':[row('stage0')],'new_entry_receipts':[],'new_MLP_module_forward_calls':0,'new_optimizer_updates':0,'teacher_forwards':0,'production_C0_C1_instances':0,'scope_is_admission_no_new_model_execution':True}
    assert m.recovered_parent_rows(r)==[row('stage0')]
    for key,value in [('actual_supervisor_directly_waited',False),('teacher_forwards',1),('error','failed')]:
        bad=copy.deepcopy(r);bad[key]=value
        with pytest.raises(ValueError):m.recovered_parent_rows(bad)

@pytest.mark.parametrize('exit_code',[0,7])
def test_local_caller_really_waits_one_fake_child_and_refuses_replay(monkeypatch,tmp_path,exit_code):
    import scripts.selected_risk_common as common
    import scripts.a63_collect_selected_risk_stop as collector
    caller=importlib.import_module('scripts.a63_wait_selected_risk_stop')
    root=tmp_path/'root';pub=root/'docs/results/2026-11-05_selected_action_risk';run=root/'runs/a63_selected_action_risk_v1';pub.mkdir(parents=True);run.mkdir(parents=True);(root/'scripts').mkdir()
    script="""import os,json,resource,pathlib
p=pathlib.Path('docs/results/2026-11-05_selected_action_risk')
a=resource.getrusage(resource.RUSAGE_SELF);b=resource.getrusage(resource.RUSAGE_CHILDREN)
(p/'local_resource_stop_CPU_snapshot.json').write_text(json.dumps({'process_PID':os.getpid(),'SELF_OS_exit_code':None,'whole_fresh_process_CPU_through_snapshot':a.ru_utime+a.ru_stime+b.ru_utime+b.ru_stime}))
for n in ['resources_resource_stopped.json','terminal_resource_stopped.json']:(p/n).write_text(json.dumps({'overall_complete':False}))
"""
    (root/'scripts/a63_collect_selected_risk_stop.py').write_text(script+'\nraise SystemExit('+str(exit_code)+')\n')
    for key,value in [('ROOT',root),('PUB',pub),('RUN',run)]:monkeypatch.setattr(common,key,value)
    monkeypatch.setattr(common,'checked_output',lambda p:Path(p));monkeypatch.setattr(collector,'verify_contract',lambda:{})
    monkeypatch.setattr(caller,'closing_prelaunch',lambda:{})
    assert caller.run()==(0 if exit_code==0 else 1)
    r=json.loads((pub/'local_resource_stop_OS_receipt.json').read_text())
    assert r['passed'] is (exit_code==0) and r['observed_collector_OS_exit_code']==exit_code and r['collector_exit_directly_waited'] is True
    assert r['collector_CPU_snapshot_subtracted_seconds']>0
    assert r['caller_SELF_OS_exit_code'] is None and r['separate_CPU_increment_outside_collector_snapshot']>=0
    with pytest.raises(ValueError):caller.run()

def test_closing_only_forecast_retains_reserves_and_refuses_boundaries():
    from scripts.a63_wait_selected_risk_stop import closing_forecast
    from echo_routing.temporal.selected_risk_budget import totals
    empty=totals([]);v=closing_forecast(empty,387545142,136049885,1793875000.)
    assert v['forecast']['forecast_incremental_tracked_storage_upper_bytes']==779595027
    assert v['forecast']['CPU_closure_seconds']==3600 and v['forecast_not_actual_spend'] is True
    with pytest.raises(ValueError):closing_forecast(empty,900000000,136049885,1793875000.)
    with pytest.raises(ValueError):closing_forecast(empty,387545142,136049885,1793897040-899.)

def test_intent_directory_is_durable_before_Popen_and_OS_publication_directory_after_link(monkeypatch,tmp_path):
    import os
    import scripts.selected_risk_common as common
    import scripts.a63_collect_selected_risk_stop as collector
    caller=importlib.import_module('scripts.a63_wait_selected_risk_stop')
    root=tmp_path/'root';pub=root/'docs/results/2026-11-05_selected_action_risk';run=root/'runs/a63_selected_action_risk_v1';pub.mkdir(parents=True);run.mkdir(parents=True)
    for key,value in [('ROOT',root),('PUB',pub),('RUN',run)]:monkeypatch.setattr(common,key,value)
    monkeypatch.setattr(common,'checked_output',lambda p:Path(p));monkeypatch.setattr(collector,'verify_contract',lambda:{})
    monkeypatch.setattr(caller,'closing_prelaunch',lambda:{})
    events=[];fds={};realopen=os.open;realfsync=os.fsync
    def opened(path,*a,**kw):
        fd=realopen(path,*a,**kw)
        if Path(path) in [pub,run]:fds[fd]=Path(path)
        return fd
    def synced(fd):
        if fd in fds:events.append(('fsync',fds[fd]))
        return realfsync(fd)
    def no_real_child(*a,**kw):
        events.append(('Popen',None))
        assert ('fsync',run) in events,'exclusive intent directory not durable before child'
        raise RuntimeError('order-only fixture, deliberately no child')
    monkeypatch.setattr(os,'open',opened);monkeypatch.setattr(os,'fsync',synced);monkeypatch.setattr(caller.subprocess,'Popen',no_real_child)
    assert caller.run()==1
    assert ('fsync',run) in events and events.index(('fsync',run))<events.index(('Popen',None))
    assert ('fsync',pub) in events and events.index(('fsync',pub))>events.index(('Popen',None))
