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
