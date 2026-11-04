import copy,importlib
import pytest


def test_precache_contract_excludes_headseal_and_preserves_true_production_point():
    m=importlib.import_module('echo_routing.temporal.safety_precache');classes=['PLAX','PSAX','A4C','A5C','SC4C'];cfg={'source':{s:{'sha256':s+'fixed'} for s in ['a08','jepa']},'channel_mapping':{'classes':classes},'training':{'irrelevant_head_status':'none'}};inputs={'roles':{r:{'sha256':r+'fixed','sequences':120} for r in m.ROLES}};freeze={'stage_kind':'producer_precache_geometry_only','objects':{'fixed_source.py':'fixed'}};records=[]
    for s in ['a08','jepa']:
        for r in m.ROLES:records.append({'source':s,'role':r,'identity':m.identity(cfg,inputs,freeze,s,r),'input_sha256':inputs['roles'][r]['sha256'],'complete':True,'committed_sequences':120,'path':m.folder(s,r),'journal_sha256':'journal','new_neural_units':1})
    manifest={'passed':True,'records':records,'formal_updates_at_production':0,'no_A61_action_heads_instantiated_or_forwarded':True,'no_outer_GT_metrics_or_utility_evaluated':True,'class_order':classes}
    assert len(m.validate(manifest,cfg,inputs,freeze))==4
    changed=copy.deepcopy(cfg);changed['training']['irrelevant_head_status']='60000finals'
    assert m.identity(changed,inputs,freeze,'a08',m.ROLES[0])==records[0]['identity']
    bound=m.binding(manifest,'sealed_models','actual_Git_head_proof',60000)
    assert bound['formal_updates_at_production']==0 and bound['formal_updates_at_readonly_binding']==60000 and bound['records']==records
    with pytest.raises(ValueError):m.binding(manifest,'m','p',0)
    bad=copy.deepcopy(manifest);bad['records'][0]['identity']='from_headseal_or_wrong_source'
    with pytest.raises(ValueError):m.validate(bad,cfg,inputs,freeze)


def test_precache_default_has_no_source_execution_and_fresh_head_reader_denies_early_only(tmp_path,monkeypatch):
    cli=importlib.import_module('scripts.precache_action_safety_geometry');calls=[];monkeypatch.setattr(cli,'execute_precache',lambda:calls.append(True))
    assert cli.main([])==0 and calls==[]
    source=importlib.import_module('scripts.cache_action_safety');monkeypatch.setattr(source,'PUB',tmp_path);(tmp_path/'producer_precache_manifest.json').write_text('{}')
    with pytest.raises(ValueError):source.require_precache_head_binding()
