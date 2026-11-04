import importlib,json
import pytest

def test_all_current_review_schemas_dedup_pin_and_changed_raw_rejection(tmp_path):
 m=importlib.import_module('scripts.collect_event_risk_local_costs');p=tmp_path/'pub';p.mkdir()
 a={'CPU_SELF_plus_waitedchildren_seconds':.2,'scope':'root toy partial'};b={'known_measured_tool_CPU_seconds_to_snapshot':.3};c={'rusage_self_plus_waited_children_cpu_seconds_through_sample':.4}
 for name,v in [('root_a.json',a),('review_b.json',b),('review_c.json',c),('review_copy.json',c)]: (p/name).write_text(json.dumps(v))
 first=m.collect(p,tmp_path);second=m.collect(p,tmp_path)
 external=[r for r in second['receipts'] if r['receipt_id'].startswith('external:')]
 assert len(external)==3 and sum(r['measured_CPU_core_seconds'] for r in external)==pytest.approx(.9)
 assert len(second['file_pins'])==4 and len(second['receipts'])==len(first['receipts'])+1
 (p/'root_a.json').write_text(json.dumps({**a,'CPU_SELF_plus_waitedchildren_seconds':1.}))
 with pytest.raises(ValueError):m.collect(p,tmp_path)

def test_resource_admission_summaries_never_recharged_as_local_receipts(tmp_path):
 m=importlib.import_module('scripts.collect_event_risk_local_costs');p=tmp_path/'pub';p.mkdir()
 for name in ['resources.json','formal_admission.json','collector_OS_exit_receipt.json','source_physical_costs.json']:(p/name).write_text(json.dumps({'CPU_SELF_plus_waitedchildren_seconds':999,'scope':'cumulative/summarized existing costs'}))
 value=m.collect(p,tmp_path);assert not any(r['receipt_id'].startswith('external:') for r in value['receipts'])
