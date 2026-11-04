import copy
import pytest
from echo_routing.temporal.proposal_terminal import validate_history

def test_history_success_exact_coverage_and_object_identity_required():
    identity={'git_commit':'a'*40,'objects':{'x':'b'*64}}
    value={'passed':True,'validated_freezes':['preparation_freeze.json','training_prepare_freeze.json'],'historical_source_git_commit':'a'*40,'object_count':1,'all_actual_Git_objects_verified':True,'objects_SHA256':'c'*64,'preparation_identity_v3_sha256':'d'*64,'current_divergent_paths':['scripts/supervise_jepa_proposals.py']}
    assert validate_history(value,identity,'d'*64,'c'*64,['scripts/supervise_jepa_proposals.py'])
    for mutation in [{'passed':False},{'validated_freezes':['preparation_freeze.json']},{'object_count':0},{'all_actual_Git_objects_verified':False},{'objects_SHA256':'e'*64},{'current_divergent_paths':['unrelated.py']}]:
        bad={**value,**mutation}
        with pytest.raises(ValueError):validate_history(bad,identity,'d'*64,'c'*64,['scripts/supervise_jepa_proposals.py'])
