import pytest


def test_source_phase_requires_no_formal_and_audited_aux_for_confirmation(tmp_path):
    from echo_routing.temporal.selected_risk_resume_source import phase_rule
    fit=phase_rule('FIT',tmp_path,None,None)
    assert fit['payloads']==6000 and fit['aux_updates_at_production']==0
    (tmp_path/'aux').mkdir()
    with pytest.raises(ValueError):phase_rule('FIT',tmp_path,None,None)
    with pytest.raises(ValueError):phase_rule('CONFIRM',tmp_path,{'passed':True},{'passed':True})


def test_source_identity_binds_new_owner_and_actual_inputs():
    from echo_routing.temporal.selected_risk_resume_source import source_identity
    a=source_identity('a08','F0_mixed','a'*64,'b'*64,{'code':'c'*64},{'source':'d'*64},'FIT')
    b=source_identity('a08','F0_mixed','a'*64,'e'*64,{'code':'c'*64},{'source':'d'*64},'FIT')
    assert a!=b and len(a)==64
