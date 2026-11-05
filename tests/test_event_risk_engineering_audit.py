import json
from pathlib import Path
import pytest


def test_engineering_close_requires_exact_production_admission_bindings(tmp_path):
    from echo_routing.temporal.event_risk_engineering_audit import validate_bindings
    from echo_routing.temporal.anchor_cache import sha
    pub=tmp_path/'pub';run=tmp_path/'run';pub.mkdir();(run/'engineering_contracts').mkdir(parents=True)
    for n in ['input_manifest.json','cost_profile.json']:(pub/n).write_text('{}')
    probe={'stage':'refinement_probe_freeze.json','git_commit':'test_actualGit'}
    (pub/'cost_refinement_profile.json').write_text(json.dumps({'passed':True,'engineering_freeze_git':probe['git_commit'],'original_cost_profile_sha256':sha(pub/'cost_profile.json'),'input_manifest_sha256':sha(pub/'input_manifest.json')}))
    contracts={'refinement_probe_freeze.json':probe}
    for stage in ['sources','formal','outer']:
        name=f'{stage}_refined_admission_v1_freeze.json';p=run/'engineering_contracts'/name;p.write_text('{}');contracts[name]={'stage':name}
        (pub/f'{stage}_admission.json').write_text(json.dumps({'stage':stage,'version':1,'all_caps_fit':True,'independent_engineering_contract':{'path':str(p.relative_to(tmp_path)),'sha256':sha(p)}}))
    assert validate_bindings(tmp_path,run,pub,contracts)
    (pub/'outer_admission.json').write_text('{}')
    with pytest.raises(ValueError):validate_bindings(tmp_path,run,pub,contracts)


def test_engineering_close_rejects_changed_measurement_pin(tmp_path):
    from echo_routing.temporal.event_risk_engineering_audit import validate_bindings
    pub=tmp_path/'pub';pub.mkdir();(pub/'cost_refinement_profile.json').write_text(json.dumps({'passed':True,'engineering_freeze_git':'wrong','original_cost_profile_sha256':'x','input_manifest_sha256':'y'}))
    with pytest.raises(ValueError):validate_bindings(tmp_path,tmp_path/'run',pub,{'refinement_probe_freeze.json':{'git_commit':'expected'}})
