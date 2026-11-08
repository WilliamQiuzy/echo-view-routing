import hashlib,json,os
from types import SimpleNamespace
from pathlib import Path
import pytest
from scripts import a63_wait_resume_selected_risk as old
from scripts import a63_wait_measured_cache_ledger_selected_risk as new

def test_closed_cost_row_preserved_and_unclosed_normal_still_refused(tmp_path,monkeypatch):
    from echo_routing.temporal.selected_risk_resume_contract import RUN
    folder=tmp_path/RUN/'caller_intents';folder.mkdir(parents=True)
    normal=b'{"normal":true}';(folder/'a.json').write_bytes(normal);h=hashlib.sha256(normal).hexdigest()
    cost={'schema':'A63_cost_probe_outer_wait4_receipt_v1','stage':'cost_only_cache_probe'}
    monkeypatch.setattr(old,'_rows',lambda ctx,name:[({'path':'cost','sha256':'a'*64},cost),({}, {'caller_intent_sha256':h})])
    seen=[];monkeypatch.setattr(new,'validate_cost_row',lambda ctx,pin,row:seen.append((pin,row)),raising=False)
    ctx=SimpleNamespace(root=tmp_path,_parent=lambda p:(os.open((tmp_path/p).parent,os.O_RDONLY|os.O_DIRECTORY),Path(p).name))
    new.residual_callers(ctx)
    assert seen==[({'path':'cost','sha256':'a'*64},cost)]
    assert cost=={'schema':'A63_cost_probe_outer_wait4_receipt_v1','stage':'cost_only_cache_probe'}
    (folder/'b.json').write_bytes(b'unknown')
    with pytest.raises(ValueError,match='unclosed caller'):
        new.residual_callers(ctx)

def test_unknown_missing_normal_intent_is_not_cost_probe(tmp_path,monkeypatch):
    monkeypatch.setattr(old,'_rows',lambda ctx,name:[({}, {'stage':'unknown'})])
    with pytest.raises(ValueError,match='cost'):
        new.residual_callers(SimpleNamespace(root=tmp_path))

def test_real_cost_validator_complete_identity_and_negative_boundaries(monkeypatch):
    from echo_routing.temporal import selected_risk_resume_guardian as g
    pin=lambda name:{'path':name,'sha256':'a'*64}
    prefix='runs/a63_selected_action_risk_resume_v1/cost_only_synthetic_cache_probe/'+'b'*32
    parent=pin(prefix+'/outer_receipt.json');intentpin=pin(prefix+'/outer_intent.json');probepin=pin('probe');source=pin('source');contract=pin('contract');payload=pin('payload');current=pin('current')
    probe={'schema':'A63_full_cache_verifier_cost_probe_spec_v1','purpose':'cost_only_synthetic_nonadoptable','limits':{'CPU_seconds':600,'wall_seconds':600,'temp_bytes':80000000},'output_prefix':prefix,'runtime_authorization_sha256':'o','source_runtime_spec_pin':source,'source_Git_contract_pin':contract}
    row={'schema':'A63_cost_probe_outer_wait4_receipt_v1','stage':'cost_only_cache_probe','runtime_authorization_sha256':'o','payload_pin':payload,'probe_spec_pin':probepin,'source_runtime_spec_pin':source,'source_Git_contract_pin':contract,'receipt_id':'resume_cost_probe:'+'b'*32,'boot_id':'boot','measured_CPU_core_seconds':12.,'model_pid':20,'model_start_ticks':30,'termination_reason':None}
    row.update({k:True for k in ['passed','actual_child_directly_waited','whole_child_accounting_complete']});row.update({k:False for k in ['unresolved_worker_uncertainty','GPU_interval_unknown','CUDA_context_occupied','GPU_job']});row.update({k:0 for k in ['actual_child_OS_exit_code','observed_worker_exit_code','stage_exit_status','NN_calls','SGD_updates','model_instances']})
    intent={'schema':'A63_cost_only_probe_intent_v1','purpose':'cost_only_synthetic_nonadoptable','runtime_authorization_sha256':'o','probe_spec_pin':probepin,'source_runtime_spec_pin':source,'source_Git_contract_pin':contract,'limits':probe['limits'].copy(),'boot_id':'boot','parent_pid':10,'parent_start_ticks':11}
    spec={'expected_boot_id':'boot','arguments':{'measured_cache_probe_parent':parent,'measured_cache_probe':payload,'measured_cache_probe_outer_intent':intentpin}}
    objects={'current':spec,'probe':probe,parent['path']:row,intentpin['path']:intent,'source':{},'contract':{}}
    ctx=SimpleNamespace(_bound_runtime_spec_pin=current,owner_sha='o',read=lambda p:objects[p['path']],raw=lambda p:json.dumps(objects[p['path']]).encode())
    monkeypatch.setattr(g,'start_ticks',lambda p:None)
    assert new.validate_cost_row(ctx,parent,row)
    for obj,key,value in [(row,'actual_child_OS_exit_code',1),(row,'runtime_authorization_sha256','foreign'),(intent,'source_runtime_spec_pin',pin('other')),(intent,'parent_pid',-1),(probe,'purpose','other'),(probe,'limits',{'CPU_seconds':601,'wall_seconds':600,'temp_bytes':80000000})]:
        saved=obj[key];obj[key]=value
        with pytest.raises(ValueError):new.validate_cost_row(ctx,parent,row)
        obj[key]=saved
    monkeypatch.setattr(g,'start_ticks',lambda p:11 if p==10 else None)
    with pytest.raises(ValueError):new.validate_cost_row(ctx,parent,row)
