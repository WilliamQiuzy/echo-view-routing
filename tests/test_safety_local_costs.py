import importlib,json
from pathlib import Path


def test_final_plan_review_schema_collected_once(tmp_path,monkeypatch):
    m=importlib.import_module('scripts.collect_safety_local_costs');pub=tmp_path/'public';pub.mkdir()
    old=tmp_path/'docs/results/2026-11-03_jepa_proposals';old.mkdir(parents=True);(old/'collector_OS_exit_receipt.json').write_text('{}')
    source=pub/'final_plan_review_raw.json';source.write_text(json.dumps({'stage':'final_plan_review','rusage_self_plus_waited_children_cpu_seconds_through_sample':.059016}))
    monkeypatch.setattr(m,'ROOT',tmp_path);monkeypatch.setattr(m,'PUB',pub)
    m.main();m.main();value=json.loads((pub/'local_cpu_receipts.json').read_text())
    rows=[r for r in value['receipts'] if r['stage']=='final_plan_review']
    assert len(rows)==1 and rows[0]['measured_CPU_core_seconds']==.059016
    assert value['file_pins']['public/final_plan_review_raw.json']==m.sha(source)
