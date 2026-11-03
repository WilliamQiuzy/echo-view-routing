import importlib,sys,json,os
import pytest

def test_actual_closed_failed_worker_cost_and_host_provenance_are_retained(tmp_path):
    m=importlib.import_module('scripts.supervise_structured_evidence');r=m.execute('synthetic_failure',[sys.executable,'-c','sum(range(20000));raise SystemExit(3)'],False,10,tmp_path)
    assert r['exit_code']==3 and r['process_CPU_seconds']>=0 and r['end_epoch']>=r['start_epoch'] and r['measurement_host'] and isinstance(r['model_pid'],int)
    assert json.loads((tmp_path/'processes.jsonl').read_text())==r and not (tmp_path/'active_process.json').exists()

def test_live_owner_and_same_exclusive_lock_refuse_second_worker(tmp_path):
    m=importlib.import_module('scripts.supervise_structured_evidence');(tmp_path/'active_process.json').write_text(json.dumps({'model_pid':os.getpid()}))
    with pytest.raises(RuntimeError):m.execute('x',[sys.executable,'-c','pass'],False,10,tmp_path)
    (tmp_path/'active_process.json').unlink()
    from echo_routing.temporal.dense_resource_accounting import terminal_queue_lock
    with terminal_queue_lock(tmp_path/'exclusive_supervisor.lock'):
        with pytest.raises(RuntimeError):m.execute('x',[sys.executable,'-c','pass'],False,10,tmp_path)
