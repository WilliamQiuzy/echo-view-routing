import pytest


def test_reader_has_no_public_boolean_adoption_bypass():
    from echo_routing.temporal.selected_risk_resume_seals import VerifiedOuterInputs
    reader=VerifiedOuterInputs.__new__(VerifiedOuterInputs)
    assert reader.outer_adoption_verified is False
    with pytest.raises(AttributeError):reader.outer_adoption_verified=True
    with pytest.raises(ValueError):reader.bank('a63_confirmation_selection',purpose='outer')


def test_head_core_rejects_fake_counts_or_empty_plan():
    from echo_routing.temporal.selected_risk_resume_seals import validate_head_core
    from echo_routing.temporal.selected_risk_metrics import readout_plan
    with pytest.raises(ValueError):validate_head_core({'passed':True},{'passed':True},readout_plan())
    with pytest.raises(ValueError):validate_head_core({'passed':True},{'passed':True},{})


def test_aux_identity_needs_exact_eight_seed_fold_roles():
    from echo_routing.temporal.selected_risk_resume_seals import validate_aux_core
    rows=[{'seed':s,'held_fold':h,'role':r,'step':6000} for s in [2026110461,2026110462] for h in [0,1] for r in ['utility','risk']]
    m={'models':rows,'aux_training_inputs_sha256':'a'*64};a={'passed':True,'aux_optimizer_updates':48000,'physical_models':8,'fullstate_count':32,'models':rows,'aux_training_inputs_sha256':'a'*64}
    assert validate_aux_core(m,a,'a'*64)
    a['physical_models']=7
    with pytest.raises(ValueError):validate_aux_core(m,a,'a'*64)


def _head_fixture():
    from echo_routing.temporal.selected_risk_resume_seals import validate_head_core
    from echo_routing.temporal.selected_risk_metrics import readout_plan
    p=readout_plan();rows=[{'seed':s,'arm':a,'step':6000,'parameters':6625,'path':f'{s}/{a}/step6000','sha256':'a'*64} for s in [2026110461,2026110462] for a in ['C0','C1']]
    states=[{'seed':r['seed'],'arm':r['arm'],'step':k,'kind':kind,'path':r['path'] if k==6000 and kind=='milestone' else r['path']+str(k)+kind,'sha256':'a'*64} for r in rows for k,kind in [(2000,'milestone'),(4000,'milestone'),(6000,'milestone'),(6000,'latest')]]
    thresholds=[{'seed':s,'policy':a,'tau':.1} for s in [2026110461,2026110462] for a in ['C0','C1']]
    cal={'passed':True,'thresholds':thresholds,'inner_records':p['inner_rows'],'legacy_thresholds':[{'seed':s,'policy':'L0','tau':.1} for s in [2026110461,2026110462]],'fixed_thresholds':[{'seed':s,'policy':a,'tau':.1} for s in [2026110461,2026110462] for a in ['M0','R1','R2']],'model_audit_sha256':'a'*64}
    models={**cal,'formal_optimizer_updates':24000,'physical_models':4,'fullstate_count':16,'models':rows,'fullstates':states,'fixed_reference_models':[{'seed':s,'arm':a} for s in [2026110461,2026110462] for a in ['S-L','risk','R1','R2']]}
    from echo_routing.temporal.selected_risk_resume_seals import RISK_REPORT_PATHS
    for i,(key,path) in enumerate(RISK_REPORT_PATHS.items(),1):models[key]=cal[key]={'path':path,'sha256':str(i)*64}
    return models,cal,p


def test_inner_count_cannot_be_filled_with_duplicate_endpoints():
    from echo_routing.temporal.selected_risk_resume_seals import validate_head_core
    models,cal,p=_head_fixture()
    assert validate_head_core(models,cal,p)
    cal['inner_records']=[p['inner_rows'][0]]*34;models['inner_records']=cal['inner_records']
    with pytest.raises(ValueError):validate_head_core(models,cal,p)


def test_head_core_requires_all_three_crossbound_risk_report_pins():
    from echo_routing.temporal.selected_risk_resume_seals import validate_head_core
    from echo_routing.temporal.selected_risk_resume_contract import PUB
    models,cal,plan=_head_fixture()
    pins={key:{'path':PUB+'/'+name,'sha256':str(i)*64} for i,(key,name) in enumerate([('FIT_risk_scores_manifest','FIT_risk_scores_manifest.json'),('FIT_risk_diagnostics','FIT_risk_diagnostics.json'),('risk_diagnostics','inner_risk_diagnostics.json')],1)}
    for key in pins:models.pop(key);cal.pop(key)
    # A complete formal/inner matrix does not stand in for the required FIT diagnostics.
    with pytest.raises(ValueError,match='risk'):validate_head_core(models,cal,plan)
    models.update(pins);cal.update(pins)
    assert validate_head_core(models,cal,plan)
    models['FIT_risk_diagnostics']={**pins['FIT_risk_diagnostics'],'sha256':'e'*64}
    with pytest.raises(ValueError):validate_head_core(models,cal,plan)
    models['FIT_risk_diagnostics']=pins['FIT_risk_diagnostics']
    cal['risk_diagnostics']=models['risk_diagnostics']={'path':'old_namespace/report.json','sha256':'f'*64}
    with pytest.raises(ValueError):validate_head_core(models,cal,plan)
