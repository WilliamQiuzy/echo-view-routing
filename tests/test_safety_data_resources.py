import importlib,sys,json
import numpy as np
import pytest


def api(n):return importlib.import_module('echo_routing.temporal.safety_'+n)


def test_full_fit_dataset_query_groups_positions_RMS_and_exact_seedplans():
    m=api('data');from echo_routing.temporal.safety_targets import effects
    qs=[{'kind':'native','labels':np.zeros(20,int)},{'kind':'multi','labels':np.r_[np.zeros(10,int),np.ones(10,int)]}]
    bs=[np.zeros(20,int),np.zeros(20,int)];a=[np.tile([.7,.1,.1,.05,.05],(20,1)) for _ in qs]
    j=[v.copy() for v in a];j[0][4:6]=[.1,.7,.1,.05,.05];j[1][10:]=[.1,.7,.1,.05,.05]
    d=m.prepare_training(qs,a,j,bs)
    assert d['features'].shape[1]==109 and d['position_features'].shape[1]==47
    assert np.array_equal(np.diff(d['position_offsets']),[len(x) for x in d['position_groups']])
    assert len(d['groups'])==2 and all(len(g)>0 for g in d['groups'])
    assert np.isclose(d['scale'],max(np.sqrt(np.mean(d['targets']**2)),1))
    assert set(d['kind_groups'])=={'native','constructed'}
    plans=m.plans(d);assert set(plans)=={'2026110461','2026110462'}
    assert m.validate(d,qs,a,j,bs,plans)['passed']
    with pytest.raises(ValueError):m.validate(d,qs,a,j,bs,{})
    with pytest.raises(ValueError):m.validate(d,qs,a,j,bs,{**plans,'02026110461':plans['2026110461']})


def test_inherited_A60_costs_and_CPU_deadline_not_GPU_reserve():
    m=api('resources');spent=m.totals([])
    assert spent['cumulative_GPU_hours']==18.41335994296604
    assert spent['cumulative_CPU_accounting_hours']==19.198515337222222
    assert m.PRIOR_STORAGE_UPPER_BYTES==35332702254
    assert m.remaining_limits(spent,m.DEADLINE_EPOCH-100,potential_GPU_context=False)['stage_wall_seconds']==100
    assert m.forecast(spent,0,1,100,m.DEADLINE_EPOCH-100,20)['all_caps_fit']


def test_supervisor_durable_claim_precedes_child_and_actual_failure_lock(tmp_path,monkeypatch):
    m=importlib.import_module('scripts.supervise_action_safety');r=api('resources')
    original=m.subprocess.Popen
    def checked(*a,**k):
        active=json.loads((tmp_path/'run/active_process.json').read_text())
        assert active['state']=='durable_launch_intent'
        return original(*a,**k)
    monkeypatch.setattr(m.subprocess,'Popen',checked)
    row=m.execute('synthetic',[sys.executable,'-c','raise SystemExit(7)'],tmp_path/'run',tmp_path/'pub',r.totals([]),3,False,storage_probe=lambda:0,GPU_probe=lambda:[])
    assert row['observed_worker_exit_code']==7 and row['child_exit_code_directly_waited'] and row['measured_CPU_core_seconds']>0
    assert not (tmp_path/'run/active_process.json').exists()
    with r.ExclusiveQueue(tmp_path/'run/exclusive_supervisor.lock'):
        with pytest.raises(BlockingIOError):m.execute('forbidden',[],tmp_path/'run',tmp_path/'pub',r.totals([]),3,False)
