import importlib
import copy
import hashlib
import json
import pytest

def api():
    try:return importlib.import_module('echo_routing.temporal.selected_risk_resume_contract')
    except ModuleNotFoundError:pytest.fail('new preparation-only continuation contract absent')

def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def raw(v):return json.dumps(v,sort_keys=True,indent=2).encode()
def rawsha(v):return hashlib.sha256(raw(v)).hexdigest()

def auth():
    return {'schema':'A63_explicit_resume_execution_authorization_v1','execution_authorized':True,'preparation_only':False,'purpose':'continue_A63_same_science','approved_proposal_sha256':'a'*64,'scientific_card_sha256':'191acbf1204cd6bf3fbdb6066800a3c6e7d6208ff7ccc15fe752c7964daead7a','scientific_basis_sha256':'78b4f913064d7300238577c561384cc559ba09f2c54be84bffa8d876f9bbacbf','previous_public_manifest_sha256':'13fb3877c80a794ee706316867f8ae6632750288f540ae6da7ba9cc7a84f18e6','runtime_RUN':'runs/a63_selected_action_risk_resume_v1','runtime_PUB':'docs/results/2026-11-05_a63_resume_execution','T0_epoch':1800000000,'deadline_epoch':1800043200,'prior_T0_epoch':1793853840,'prior_deadline_epoch':1793897040,'new_budget_granted':False,'recorded_owner_user_authorization_sha256':'b'*64,'closed_baseline_account_sha256':'c'*64}

def test_missing_or_preparation_authority_refuses_even_old_window_current():
    m=api()
    with pytest.raises(ValueError):m.validate_execution_authority(None,None,'a'*64,'c'*64,1793895500.)
    old={'execution_authorized':True,'T0_epoch':1793853840,'deadline_epoch':1793897040}
    with pytest.raises(ValueError):m.validate_execution_authority(raw(old),rawsha(old),'a'*64,'c'*64,1793895500.)
    a=auth();a['preparation_only']=True
    with pytest.raises(ValueError):m.validate_execution_authority(raw(a),rawsha(a),'a'*64,'c'*64,1800000010.)

def test_explicit_fresh_window_binds_science_baseline_namespaces_and_exact12h():
    m=api();a=auth();assert rawsha(a)!=digest(a);assert m.validate_execution_authority(raw(a),rawsha(a),'a'*64,'c'*64,1800000010.)
    for key,value in [('T0_epoch',1793853840),('deadline_epoch',1800043201),('new_budget_granted',True),('runtime_RUN','runs/a63_selected_action_risk_v1'),('scientific_basis_sha256','d'*64),('closed_baseline_account_sha256','d'*64)]:
        bad=copy.deepcopy(a);bad[key]=value
        with pytest.raises(ValueError):m.validate_execution_authority(raw(bad),rawsha(bad),'a'*64,'c'*64,1800000010.)
    with pytest.raises(ValueError):m.validate_execution_authority(raw(a),'f'*64,'a'*64,'c'*64,1800000010.)
    with pytest.raises(ValueError):m.validate_execution_authority(raw(a),rawsha(a),'a'*64,'c'*64,1800043200.)
    with pytest.raises(ValueError):m.validate_execution_authority(a,digest(a),'a'*64,'c'*64,1800000010.)

def test_owner_duplicate_keys_or_noncanonical_boolean_integer_types_reject():
    m=api();a=auth();s=raw(a);dup=s[:-1]+b',"execution_authorized":true}';h=hashlib.sha256(dup).hexdigest()
    with pytest.raises(ValueError):m.validate_execution_authority(dup,h,'a'*64,'c'*64,1800000010.)
    for key,value in [('execution_authorized',1),('prior_T0_epoch',float(1793853840)),('T0_epoch',1800000000.)]:
        bad=copy.deepcopy(a);bad[key]=value
        with pytest.raises(ValueError):m.validate_execution_authority(raw(bad),rawsha(bad),'a'*64,'c'*64,1800000010.)

def test_complete_stage_graph_and_coverage_hard_barrier_cannot_be_a_passed_flag():
    m=api();g=m.stage_graph()
    assert set(g)=={'adopt_metadata','admit_FIT','source_FIT','source_FIT_adoption','observe_geometry','admit_aux','prepare_aux','train_aux','audit_aux','seal_aux','held_score','seal_selectors','coverage','admit_formal','prepare_formal','source_CONFIRM','admit_formal_after_CONFIRM','adopt_confirmation','train_formal','audit_formal','fit_risk_scores','calibrate','seal_heads','admit_outer','outer','aggregate','supports','forensics','package','scientific_review','resource_review','close_negative','close_full'}
    with pytest.raises(ValueError):m.require_phase('train_formal',{'coverage':{'passed':True}})
    with pytest.raises(ValueError):m.require_phase('source_CONFIRM',{'coverage':{'passed':True}})
    assert g['outer']['expected_outputs']==238 and g['aggregate']['expected_outputs']==140
    assert g['supports']['expected_outputs']==56 and g['forensics']['expected_outputs']=={'native':34,'regression':32}
    assert g['source_FIT_adoption']['requires'] == ['source_FIT']
    assert 'source_FIT_adoption' in g['observe_geometry']['requires']
    assert 'fit_risk_scores' in g['calibrate']['requires']
    assert g['train_aux']['optimizer_updates']==48000 and g['train_formal']['optimizer_updates']==24000

def test_stage_outputs_do_not_escape_or_modify_old_namespaces():
    m=api();assert m.output_relative('checkpoints/seed461/step6000.pt').startswith(m.RUN+'/')
    for p in ['../a63_selected_action_risk_v1/x','/tmp/x','a/../../x','']:
        with pytest.raises(ValueError):m.output_relative(p)

def test_coverage_floors_and_aux_seal_are_structural_not_23_true_booleans():
    m=api()
    from echo_routing.temporal.selected_risk_figures import FLOORS,CONCENTRATION
    row={'passed':True,'checks':{},'failed_checks':[]}
    for key,(minimum,group,field) in FLOORS.items():
        row.setdefault(group,{})[field]=minimum;row['checks'][key]={'actual':minimum,'minimum':minimum,'passed':True}
    for key in CONCENTRATION:row[key]=.1;row['checks'][key]={'actual':.1,'maximum':.2,'passed':True}
    cov={'passed':True,'source_isolation':{'passed':True},'aux_optimizer_updates':48000,'formal_optimizer_updates':0,'seeds':{str(s):copy.deepcopy(row) for s in [2026110461,2026110462]}}
    assert m.validate_coverage_pass(cov)
    def sealed(payload):return {'payload':payload,'payload_sha256':digest(payload),'git_commit':'1'*40,'object_proof_sha256':'2'*64,'runtime_authorization_sha256':'3'*64}
    deps=m.stage_graph()['train_formal']['requires'];e={k:sealed({'passed':True,'all_caps_fit':True}) for k in deps};e['prepare_formal']=sealed({'passed':True,'seeds':[2026110461,2026110462]});e['source_CONFIRM']=sealed({'complete':True,'phase':'CONFIRM','committed_payloads':480,'aux_updates_at_production':48000,'formal_updates_at_production':0});e['admit_formal_after_CONFIRM']=sealed({'passed':True,'all_caps_fit':True,'completed_CONFIRM_sha256':e['source_CONFIRM']['payload_sha256'],'prepared_formal_sha256':e['prepare_formal']['payload_sha256']});e['coverage']=sealed(cov);e['seal_aux']=sealed({'passed':True,'aux_optimizer_updates':48000,'physical_models':8,'fullstate_count':32})
    assert m.require_phase('train_formal',e,runtime_authorization_sha='3'*64)
    bad=copy.deepcopy(cov);bad['seeds']['2026110461']['checks']['D_danger30']['minimum']=1
    with pytest.raises(ValueError):m.validate_coverage_pass(bad)
    bad=copy.deepcopy(e);bad['coverage']['runtime_authorization_sha256']='4'*64
    with pytest.raises(ValueError):m.require_phase('train_formal',bad,runtime_authorization_sha='3'*64)
    bad=copy.deepcopy(e);bad['seal_aux']=sealed({'passed':True,'aux_optimizer_updates':48000,'physical_models':7,'fullstate_count':32})
    with pytest.raises(ValueError):m.require_phase('train_formal',bad,runtime_authorization_sha='3'*64)

def test_failed_or_incomplete_dependency_never_becomes_a_successful_git_barrier():
    m=api()
    def seal(p):return {'payload':p,'payload_sha256':digest(p),'git_commit':'1'*40,'object_proof_sha256':'2'*64,'runtime_authorization_sha256':'3'*64}
    for p in [{'passed':False,'aux_optimizer_updates':48000,'physical_models':8,'fullstate_count':32},{'passed':True,'aux_optimizer_updates':47999,'physical_models':8,'fullstate_count':32}]:
        with pytest.raises(ValueError):m.require_phase('calibrate',{'audit_formal':seal(p)},runtime_authorization_sha='3'*64)
    deps={'source_CONFIRM':seal({'complete':True,'phase':'CONFIRM','committed_payloads':479,'aux_updates_at_production':48000,'formal_updates_at_production':0}),'seal_heads':seal({'passed':True})}
    with pytest.raises(ValueError):m.require_phase('adopt_confirmation',deps,runtime_authorization_sha='3'*64)

def test_post_confirmation_fresh_admission_is_a_distinct_required_evidence():
    m=api();g=m.stage_graph()
    assert 'admit_formal_after_CONFIRM' in g['train_formal']['requires']
    assert set(g['admit_formal_after_CONFIRM']['requires'])=={'source_CONFIRM','prepare_formal'}
    def seal(p):return {'payload':p,'payload_sha256':digest(p),'git_commit':'1'*40,'object_proof_sha256':'2'*64,'runtime_authorization_sha256':'3'*64}
    deps={'source_CONFIRM':seal({'complete':True,'phase':'CONFIRM','committed_payloads':480,'aux_updates_at_production':48000,'formal_updates_at_production':0}),'prepare_formal':seal({'passed':True,'seeds':[2026110461,2026110462]})}
    assert m.require_phase('admit_formal_after_CONFIRM',deps,runtime_authorization_sha='3'*64)
    deps['source_CONFIRM']=seal({'complete':False,'phase':'CONFIRM','committed_payloads':480,'aux_updates_at_production':48000,'formal_updates_at_production':0})
    with pytest.raises(ValueError):m.require_phase('admit_formal_after_CONFIRM',deps,runtime_authorization_sha='3'*64)

def test_every_unconditional_driver_descriptor_is_a_declared_predecessor():
    g=api().stage_graph()
    assert {'held_score','seal_aux','admit_aux'}<=set(g['seal_selectors']['requires'])
    assert 'seal_selectors' in g['train_formal']['requires']
    assert {'coverage','seal_selectors'}<=set(g['audit_formal']['requires'])
