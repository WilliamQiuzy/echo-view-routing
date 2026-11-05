import json,sys
from pathlib import Path
import pytest

def test_admission_metadata_collision_and_actual_failure_retained(tmp_path):
 from echo_routing.temporal.event_risk_terminal_recovery import admissions
 for name in ['engineering_sources_refined_admission_v1_identity.json','root_actual_source_admission_verification.json']:(tmp_path/name).write_text('{}')
 for stage,fit in [('sources',False),('formal',True)]:
  (tmp_path/f'{stage}_refined_admission_v1.json').write_text(json.dumps({'stage':stage,'version':1,'all_caps_fit':fit}))
 rows=admissions(tmp_path);assert len(rows)==2 and {r['stage']:r['all_caps_fit'] for r in rows}=={'sources':False,'formal':True}

def test_actual_admission_missing_or_mismatched_fields_fails(tmp_path):
 from echo_routing.temporal.event_risk_terminal_recovery import admissions
 p=tmp_path/'outer_refined_admission_v1.json'
 for value in [{},{'stage':'formal','version':1,'all_caps_fit':True},{'stage':'outer','version':1,'all_caps_fit':1}]:
  p.write_text(json.dumps(value))
  with pytest.raises(ValueError):admissions(tmp_path)
 p.unlink();(tmp_path/'sources_refined_admission_v0.json').write_text('{}')
 with pytest.raises(ValueError):admissions(tmp_path)

def test_recovery_caller_keeps_old_receipt_and_uses_new_exclusive_paths(tmp_path,monkeypatch):
 import scripts.wait_event_risk_terminal_recovery as m
 from echo_routing.temporal.anchor_cache import atomic_json
 run=tmp_path/'run';pub=tmp_path/'pub';run.mkdir();pub.mkdir();old=pub/'collector_OS_exit_receipt.json';old.write_text('{"passed":false}')
 for k,v in [('RUN',run),('PUB',pub),('ROOT',tmp_path)]:monkeypatch.setattr(m,k,v)
 monkeypatch.setattr(m,'write',atomic_json);monkeypatch.setattr(m,'checked_output',lambda p:Path(p))
 child=tmp_path/'child.py';child.write_text('import os,json,pathlib,hashlib\np=pathlib.Path('+repr(str(pub))+')\nq=p/"resources.json"\nq.write_text(json.dumps({"collector_pid":os.getpid(),"collector_CPU_seconds_through_snapshot":0.0}))\n(p/"terminal_exit_receipt.json").write_text(json.dumps({"resources_sha256":hashlib.sha256(q.read_bytes()).hexdigest()}))')
 assert m.run([sys.executable,str(child)])==0
 row=json.loads((pub/'collector_recovery_OS_exit_receipt.json').read_text());assert row['observed_collector_OS_exit_code']==0 and row['collector_exit_directly_waited'] and old.read_text()=='{"passed":false}'
 intent=json.loads((run/'terminal_recovery_v2_caller_intent.json').read_text());assert intent['nonce'] and intent['boot_id'] and intent['caller_start_ticks']>0
 with pytest.raises(ValueError):m.run([sys.executable,str(child)])


def test_recovery_keeps_retained_failure_predicate_and_base_fee_records(tmp_path):
 from scripts.close_event_risk_terminal_recovery import has_retained_failure
 assert not has_retained_failure([], [{'all_caps_fit':True}],None,[])
 assert has_retained_failure([], [{'all_caps_fit':False}],None,[])
 from scripts.collect_event_risk_recovery_costs import collect
 pub=tmp_path/'pub';pub.mkdir();base={'receipts':[{'receipt_id':'old','measured_CPU_core_seconds':1.}], 'file_pins':{}}
 (pub/'local_cpu_receipts.json').write_text(json.dumps(base));old=(pub/'local_cpu_receipts.json').read_bytes()
 (pub/'root_failed_terminal_fee.json').write_text(json.dumps({'CPU_SELF_plus_waitedchildren_seconds':4.862736}))
 v=collect(pub,tmp_path);assert (pub/'local_cpu_receipts.json').read_bytes()==old
 assert next(r for r in v['receipts'] if r['receipt_id']=='old')==base['receipts'][0]
 assert sum(r['measured_CPU_core_seconds'] for r in v['receipts'])>=5.862736


def test_original_science_and_cap_helpers_byte_ast_equivalent():
 import ast
 root=Path(__file__).resolve().parents[1]
 old=ast.parse((root/'scripts/close_event_risk.py').read_text());new=ast.parse((root/'scripts/close_event_risk_terminal_recovery.py').read_text())
 for name in ['completion_state','validate_science','has_retained_failure','actual_caps_status']:
  a=next(n for n in old.body if isinstance(n,ast.FunctionDef) and n.name==name);b=next(n for n in new.body if isinstance(n,ast.FunctionDef) and n.name==name)
  assert ast.dump(a)==ast.dump(b)
