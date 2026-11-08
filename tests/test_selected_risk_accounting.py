import importlib,json,hashlib
import pytest
PREFIX='docs/results/2026-11-05_selected_action_risk/'

def mod():
 try:m=importlib.import_module('echo_routing.temporal.selected_risk_accounting')
 except ModuleNotFoundError:m=None
 assert m is not None,'A63 explicit registered exact-once localfee parser absent'
 return m

def put(root,name,value,field,kind='review',**extra):
 p=root/(PREFIX+name);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value));h=hashlib.sha256(p.read_bytes()).hexdigest()
 return {'path':PREFIX+name,'sha256':h,'charge_group':h,'CPU_field':field,'kind':kind,**extra}

def test_explicit_aggregate_once_and_component_field_alias_rejected(tmp_path):
 m=mod();a=put(tmp_path,'review_a.json',{'known_measured_tool_CPU_seconds_to_snapshot':.3,'component_measured_CPU_seconds':.2},'known_measured_tool_CPU_seconds_to_snapshot');b=put(tmp_path,'review_b.json',{'known_measured_tool_CPU_seconds_to_snapshot':.3,'component_measured_CPU_seconds':.2},'known_measured_tool_CPU_seconds_to_snapshot')
 out=m.collect(tmp_path,[a,b]);assert len(out['receipts'])==1 and out['known_measured_CPU_seconds']==.3 and len(out['file_pins'])==2
 with pytest.raises(ValueError):m.collect(tmp_path,[a,{**b,'CPU_field':'component_measured_CPU_seconds'}])

def test_no_process_admission_or_cumulative_totals_automatic_fee(tmp_path):
 m=mod();a=put(tmp_path,'sources_admission.json',{'CPU_SELF_plus_waitedchildren_seconds':100},'CPU_SELF_plus_waitedchildren_seconds',kind='admin')
 with pytest.raises(ValueError):m.collect(tmp_path,[a])
 a=put(tmp_path,'root_proof.json',{'measurement':{'CPU':.2}},'measurement/CPU',kind='admin')
 assert m.collect(tmp_path,[a])['known_measured_CPU_seconds']==.2

def test_unknown_admin_is_preserved_null_with_separate_allowance(tmp_path):
 m=mod();a=put(tmp_path,'admin_failure.json',{'actual_CPU':None},'actual_CPU',kind='admin_unknown',accounting_allowance_seconds=1.)
 out=m.collect(tmp_path,[a]);r=out['receipts'][0]
 assert r['measured_CPU_core_seconds'] is None and r['CPU_accounting_allowance_seconds']==1 and out['known_measured_CPU_seconds']==0

def test_canonical_refresh_retains_all_prior_costs_and_pins(tmp_path):
 m=mod();a=put(tmp_path,'review_a.json',{'CPU':.2},'CPU');b=put(tmp_path,'review_b.json',{'CPU':.3},'CPU')
 first=m.collect(tmp_path,[a]);second=m.collect(tmp_path,[b],prior=first)
 assert second['receipts'][0]==first['receipts'][0] and second['known_measured_CPU_seconds']==.5
 assert set(second['file_pins'])=={a['path'],b['path']}
 (tmp_path/a['path']).write_text('changed')
 with pytest.raises(ValueError):m.collect(tmp_path,[b],prior=first)

def test_old_closed_namespace_and_changed_expected_SHA_reject(tmp_path):
 m=mod();a=put(tmp_path,'review_a.json',{'CPU':.2},'CPU');old=tmp_path/'docs/results/2026-11-04_anchor_context_risk/root_old.json';old.parent.mkdir(parents=True);old.write_bytes((tmp_path/a['path']).read_bytes())
 with pytest.raises(ValueError):m.collect(tmp_path,[{**a,'path':str(old.relative_to(tmp_path))}])
 with pytest.raises(ValueError):m.collect(tmp_path,[{**a,'sha256':'0'*64}])

def test_different_raw_aggregate_and_covered_child_cannot_doublecharge(tmp_path):
 m=mod();child=put(tmp_path,'review_child.json',{'CPU':.2},'CPU');agg=put(tmp_path,'review_aggregate.json',{'CPU':.3},'CPU',covers_raw_sha256=[child['sha256']])
 for specs in [[agg,child],[child,agg]]:
  with pytest.raises(ValueError):m.collect(tmp_path,specs)
 with pytest.raises(ValueError):m.collect(tmp_path,[{**agg,'charge_group':child['charge_group']},child])

def test_two_aggregates_cannot_both_include_same_unregistered_child(tmp_path):
 m=mod();child='a'*64;a=put(tmp_path,'review_A.json',{'CPU':.3},'CPU',covers_raw_sha256=[child]);b=put(tmp_path,'review_B.json',{'CPU':.4},'CPU',covers_raw_sha256=[child])
 for left,right in [(a,b),(b,a)]:
  with pytest.raises(ValueError):m.collect(tmp_path,[left,right])
  prior=m.collect(tmp_path,[left])
  with pytest.raises(ValueError):m.collect(tmp_path,[right],prior=prior)

@pytest.mark.parametrize('bad',['A'*64,'g'*64])
def test_coverage_hash_canonical_lowercase_hex_only(tmp_path,bad):
 m=mod();a=put(tmp_path,'review_a.json',{'CPU':.3},'CPU',covers_raw_sha256=[bad])
 with pytest.raises(ValueError):m.collect(tmp_path,[a])
