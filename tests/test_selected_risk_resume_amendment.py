import json,hashlib,importlib.util
from pathlib import Path
import pytest


def auth():
    path=Path(__file__).with_name('test_selected_risk_resume_contract.py');sp=importlib.util.spec_from_file_location('original_guard_fixture',path);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m.auth()


def test_explicit_storage_amendment_is_raw_bound_not_a_fresh_CPU_GPU_grant():
    from echo_routing.temporal import selected_risk_resume_contract as c
    a=auth();a.update(schema='A63_explicit_resume_execution_authorization_v2',approved_proposal_sha256=c.STORAGE_AMENDMENT_SHA,storage_lifecycle_cap_bytes=1300000000,storage_amendment_sha256=c.STORAGE_AMENDMENT_SHA,storage_approval_receipt=dict(c.STORAGE_APPROVAL));a['recorded_owner_user_authorization_sha256']=c.STORAGE_APPROVAL['sha256']
    raw=json.dumps(a,indent=2).encode();assert c.validate_execution_authority(raw,hashlib.sha256(raw).hexdigest(),c.STORAGE_AMENDMENT_SHA,'c'*64,1800000010.)
    assert c.storage_cap(a)==1300000000 and c.storage_cap(auth())==1000000000
    for key,value in [('storage_lifecycle_cap_bytes',1400000000),('storage_amendment_sha256','f'*64),('new_budget_granted',True)]:
        bad={**a,key:value};b=json.dumps(bad).encode()
        with pytest.raises(ValueError):c.validate_execution_authority(b,hashlib.sha256(b).hexdigest(),c.STORAGE_AMENDMENT_SHA,'c'*64,1800000010.)


def test_only_approved_storage_cap_changes_full_cap_math():
    from echo_routing.temporal.selected_risk_resume_contract import STORAGE_APPROVAL,STORAGE_AMENDMENT_SHA
    from echo_routing.temporal.selected_risk_resume_budget import forecast
    a=auth();a.update(schema='A63_explicit_resume_execution_authorization_v2',approved_proposal_sha256=STORAGE_AMENDMENT_SHA,storage_lifecycle_cap_bytes=1300000000,storage_amendment_sha256=STORAGE_AMENDMENT_SHA,storage_approval_receipt=dict(STORAGE_APPROVAL),recorded_owner_user_authorization_sha256=STORAGE_APPROVAL['sha256'])
    spent=dict(new_GPU_wall_seconds=0.,new_CPU_accounting_seconds=1000.,cumulative_GPU_hours=18.657838855,cumulative_CPU_accounting_hours=20.5)
    old=forecast(spent,4795.,20444.,1207032484,1800000010.,24044.,auth());new=forecast(spent,4795.,20444.,1207032484,1800000010.,24044.,a)
    assert old['checks']['A63_storage_cap'] is False and new['all_caps_fit'] is True
    assert new['forecast_incremental_CPU_seconds_including_closure']==old['forecast_incremental_CPU_seconds_including_closure']
    assert new['forecast_cumulative_GPU_hours_including_reserve']==old['forecast_cumulative_GPU_hours_including_reserve']


def test_CAL_original_N_counts_must_match_frozen_C6_before_any_threshold():
    from echo_routing.temporal.selected_risk_resume_readout import validate_CAL_parity
    assert validate_CAL_parity(2026110461,{'N_admitted':80,'unsafe_N_top':10,'total_axes':469})
    for key,value in [('N_admitted',79),('unsafe_N_top',9),('total_axes',468)]:
        r={'N_admitted':80,'unsafe_N_top':10,'total_axes':469,key:value}
        with pytest.raises(ValueError):validate_CAL_parity(2026110461,r)


def test_V2_context_reads_actual_raw_approval_and_amendment(tmp_path):
    import importlib.util,shutil
    from echo_routing.temporal.selected_risk_resume_context import Context
    from echo_routing.temporal import selected_risk_resume_contract as c
    root=Path(__file__).resolve().parents[1]
    for rel in [c.STORAGE_APPROVAL['path'],c.STORAGE_AMENDMENT_PATH]:
        target=tmp_path/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(root/rel,target)
    rawbase=b'{"closed_account":434.464139}';(tmp_path/'account.json').write_bytes(rawbase);bsha=hashlib.sha256(rawbase).hexdigest()
    a=auth();a.update(schema='A63_explicit_resume_execution_authorization_v2',approved_proposal_sha256=c.STORAGE_AMENDMENT_SHA,closed_baseline_account_sha256=bsha,storage_lifecycle_cap_bytes=1300000000,storage_amendment_sha256=c.STORAGE_AMENDMENT_SHA,storage_approval_receipt=dict(c.STORAGE_APPROVAL),recorded_owner_user_authorization_sha256=c.STORAGE_APPROVAL['sha256'])
    raw=json.dumps(a).encode();(tmp_path/'owner.json').write_bytes(raw);h=hashlib.sha256(raw).hexdigest()
    kwargs=(tmp_path,{'path':'owner.json','sha256':h},{'path':c.STORAGE_AMENDMENT_PATH,'sha256':c.STORAGE_AMENDMENT_SHA},{'path':'account.json','sha256':bsha})
    ctx=Context(*kwargs,now=1800000010.);assert c.storage_cap(ctx.owner)==1300000000
    (tmp_path/c.STORAGE_APPROVAL['path']).write_text('{}')
    with pytest.raises(ValueError):Context(*kwargs,now=1800000010.)
