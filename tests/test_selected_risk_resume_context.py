import hashlib
import json
from pathlib import Path

import pytest

from echo_routing.temporal.selected_risk_resume_contract import BASIS,CARD,OLD_MANIFEST,OLD_T0,OLD_DEADLINE,PUB,RUN,digest


def setup(tmp_path):
    from echo_routing.temporal.selected_risk_resume_context import Context
    def put(path,value):
        p=tmp_path/path;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value,indent=2))
        return {'path':path,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
    proposal=put('prep/proposal.json',{'preparation_only':True})
    baseline=put('prep/baseline.json',{'closed_A63_CPU_accounting_seconds':434.464139,'baseline_never_recharged':True})
    owner=put('owner.json',dict(schema='A63_explicit_resume_execution_authorization_v1',execution_authorized=True,preparation_only=False,purpose='continue_A63_same_science',approved_proposal_sha256=proposal['sha256'],scientific_card_sha256=CARD,scientific_basis_sha256=BASIS,previous_public_manifest_sha256=OLD_MANIFEST,runtime_RUN=RUN,runtime_PUB=PUB,T0_epoch=OLD_DEADLINE+60,deadline_epoch=OLD_DEADLINE+43260,prior_T0_epoch=OLD_T0,prior_deadline_epoch=OLD_DEADLINE,new_budget_granted=False,recorded_owner_user_authorization_sha256='a'*64,closed_baseline_account_sha256=baseline['sha256']))
    ctx=Context(tmp_path,owner,proposal,baseline,now=OLD_DEADLINE+61)
    spec={'schema':'A63_resume_runtime_spec_v1','stage':'adopt_metadata','owner':owner,'proposal':proposal,'baseline':baseline,'freeze':{'path':'future-freeze','sha256':'0'*64},'arguments':{}}
    pin=put('runtime.json',spec);ctx.bind_runtime_spec(pin,spec)
    return ctx,put


def final_bind(ctx,put,freeze_pin):
    spec=ctx.read(ctx._bound_runtime_spec_pin);spec['freeze']=freeze_pin
    pin=put('final-runtime.json',spec);ctx.bind_runtime_spec(pin,spec)


def test_absent_owner_never_creates_runtime_namespace(tmp_path):
    from echo_routing.temporal.selected_risk_resume_context import Context
    with pytest.raises(ValueError):Context(tmp_path,None,None,None)
    assert not (tmp_path/RUN).exists() and not (tmp_path/PUB).exists()


def test_nonclobber_new_namespace_only(tmp_path):
    ctx,_=setup(tmp_path)
    p=ctx.write_json('result.json',{'count':2},public=True)
    assert (tmp_path/p['path']).read_bytes()==b'{"count":2}'
    with pytest.raises(FileExistsError):ctx.write_json('result.json',{'count':3},public=True)
    for bad in ['../old.json','/old.json','a/../old.json','./old.json']:
        with pytest.raises(ValueError):ctx.write_json(bad,{})
    assert not (tmp_path/'old.json').exists()


def test_parent_symlink_rejected_without_external_write(tmp_path):
    ctx,_=setup(tmp_path);outside=tmp_path/'foreign';outside.mkdir()
    (tmp_path/'runs').symlink_to(outside,target_is_directory=True)
    with pytest.raises(ValueError):ctx.write_json('bad.json',{})
    assert list(outside.iterdir())==[]


def test_external_raw_owner_pin_cannot_be_relabelled(tmp_path):
    ctx,_=setup(tmp_path)
    (tmp_path/'owner.json').write_bytes((tmp_path/'owner.json').read_bytes()+b'\n')
    with pytest.raises(ValueError):ctx.revalidate()
    assert not (tmp_path/RUN).exists()


def test_stage_actual_proof_and_current_bytes_are_independent(tmp_path,monkeypatch):
    ctx,put=setup(tmp_path)
    code=put('echo_routing/temporal/new.py',{'a':1})
    proof=put('proof.json',{'proof':True})
    required={code['path']}
    contract={'schema':'A63_resume_stage_freeze_v1','stage':'adopt_metadata','git_commit':'a'*40,'objects':{code['path']:code['sha256']},'historical_frozen_imports':{},'actual_Git_object_proof':proof,'runtime_authorization_sha256':ctx.owner_sha,'runtime_spec_semantics_sha256':ctx.runtime_spec_semantics_sha256}
    pin=put('freeze.json',contract);final_bind(ctx,put,pin);seen=[]
    monkeypatch.setattr('echo_routing.temporal.selected_risk_resume_freeze.required',lambda *a:[code['path']]);monkeypatch.setattr(ctx,'_historical_expected',lambda:{})
    monkeypatch.setattr('echo_routing.temporal.safety_git.verify',lambda p,c,e:seen.append((p,c,e)) or {'passed':True})
    assert ctx.verify_stage('adopt_metadata',pin,required=required)['stage']=='adopt_metadata'
    assert seen[0][2]=={code['path']:code['sha256']}
    (tmp_path/code['path']).write_text('changed')
    with pytest.raises(ValueError):ctx.verify_stage('adopt_metadata',pin,required=required)


def test_missing_required_source_and_foreign_owner_fail(tmp_path,monkeypatch):
    ctx,put=setup(tmp_path);proof=put('proof.json',{})
    c={'schema':'A63_resume_stage_freeze_v1','stage':'adopt_metadata','git_commit':'a'*40,'objects':{},'historical_frozen_imports':{},'actual_Git_object_proof':proof,'runtime_authorization_sha256':ctx.owner_sha,'runtime_spec_semantics_sha256':ctx.runtime_spec_semantics_sha256}
    pin=put('freeze.json',c);final_bind(ctx,put,pin)
    monkeypatch.setattr('echo_routing.temporal.selected_risk_resume_freeze.required',lambda *a:['missing.py']);monkeypatch.setattr(ctx,'_historical_expected',lambda:{})
    with pytest.raises(ValueError):ctx.verify_stage('adopt_metadata',pin,required={'missing.py'})
    c['runtime_authorization_sha256']='b'*64;pin=put('foreign.json',c)
    with pytest.raises(ValueError):ctx.verify_stage('adopt_metadata',pin,required=set())


def test_evidence_requires_raw_git_coverage_and_same_owner(tmp_path,monkeypatch):
    ctx,put=setup(tmp_path);out=put('stage.json',{'all_caps_fit':True})
    proof=put('proof.json',{})
    called=[];monkeypatch.setattr('echo_routing.temporal.safety_git.verify',lambda p,c,e:called.append(e) or {'passed':True})
    descriptor={'payload_pin':out,'git_commit':'a'*40,'actual_Git_object_proof':proof,'runtime_authorization_sha256':ctx.owner_sha}
    evidence=ctx.evidence(descriptor)
    assert called==[{out['path']:out['sha256']}] and evidence['payload_sha256']==digest({'all_caps_fit':True})
    descriptor['runtime_authorization_sha256']='c'*64
    with pytest.raises(ValueError):ctx.evidence(descriptor)


def test_freeze_excluding_self_pin_still_binds_all_runtime_arguments(tmp_path):
    ctx,put=setup(tmp_path);spec=ctx.read(ctx._bound_runtime_spec_pin);spec['arguments']={'different':True};pin=put('changed-spec.json',spec)
    with pytest.raises(ValueError):ctx.bind_runtime_spec(pin,spec)


def test_shared_pool_expansion_rejects_a_wrong_referenced_oid(tmp_path):
    import base64
    ctx,put=setup(tmp_path);raw=b'blob 3\0abc';oid=hashlib.sha1(raw).hexdigest()
    proof=put('proof.json',{'objects':{oid:'@shared:'+oid}});pool=put(RUN+'/Git_proofs/shared_pool.json',{'object_format':'sha1','objects':{oid:base64.b64encode(raw).decode()}})
    contract={'proof_encoding':'A63_shared_Git_objects_v1','shared_object_pool':pool}
    assert ctx.contract_proof(contract,proof)['objects'][oid]==base64.b64encode(raw).decode()
    pool=put(RUN+'/Git_proofs/shared_pool.json',{'object_format':'sha1','objects':{oid:base64.b64encode(b'blob 3\0bad').decode()}});contract['shared_object_pool']=pool
    with pytest.raises(ValueError):ctx.contract_proof(contract,proof)


def test_postcoverage_gate_needs_complete_backend_reviews_and_exact_current_code(monkeypatch):
    from echo_routing.temporal.selected_risk_resume_context import Context
    from echo_routing.temporal.selected_risk_resume_contract import PUB
    import scripts.selected_risk_common as old
    for name in ['validate_config','validate_frozen_prior']:monkeypatch.setattr(old,name,lambda *a:None)
    monkeypatch.setattr(old,'read',lambda *a:{})
    monkeypatch.setattr('echo_routing.temporal.selected_risk_resume_context.require_phase',lambda *a,**k:True)
    code='echo_routing/temporal/selected_risk_resume_readout.py';objects={'owner':'a'*64,'proposal':'b'*64,'baseline':'c'*64,code:'d'*64}
    class C:
        owner_pin={'path':'owner','sha256':'a'*64};proposal_pin={'path':'proposal','sha256':'b'*64};baseline_pin={'path':'baseline','sha256':'c'*64};owner_sha='a'*64;owner={'schema':'A63_explicit_resume_execution_authorization_v1'};_bound_runtime_spec_pin={'path':'spec','sha256':'e'*64}
        value={'arguments':{}};ready={}
        def verify_stage(self,*a,**k):return {'objects':objects}
        def evidence(self,v):return v
        def read(self,p):return self.value if p==self._bound_runtime_spec_pin else self.ready
        def raw(self,p):return b''
    c=C()
    with pytest.raises(ValueError):Context.enter(c,'train_formal',{}, {},required=set())
    c.value={'arguments':{'complete_backend_readiness':{'path':PUB+'/ready.json','sha256':'f'*64}}}
    c.ready={'software_prepared':True,'scientific_runtime_completed':False,'stage_count':33,'SPEC_Approved':True,'QUALITY_Approved':True,'code_pins':{code:'d'*64},'review_receipts':[{'path':'spec_review','sha256':'1'*64},{'path':'quality_review','sha256':'2'*64}]}
    assert Context.enter(c,'train_formal',{}, {},required=set())[0]['objects']==objects
    c.ready['code_pins'][code]='0'*64
    with pytest.raises(ValueError):Context.enter(c,'train_formal',{}, {},required=set())


def test_missing_closed_publication_uses_only_pinned_mirror_and_never_conflict_fallback(tmp_path,monkeypatch):
    import echo_routing.temporal.selected_risk_resume_context as m
    ctx,put=setup(tmp_path);raw=b'closed-public-byte';h=hashlib.sha256(raw).hexdigest()
    rows=[{'path':'docs/results/closed/public'+str(i)+'.json','sha256':h,'bytes':len(raw)} for i in range(839)]
    alias='docs/results/2026-11-05_a63_resume_preparation/readonly_closed_publication_manifest.json'
    manifest=put(alias,{'file_count':839,'files':rows});monkeypatch.setattr(m,'OLD_MANIFEST',manifest['sha256'])
    for row in rows:
        p=tmp_path/RUN/'readonly_closed_publication'/row['path'];p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
    result=ctx.verify_closed_publication(manifest)
    assert result['original_missing_aliases_used']==[{'original_path':r['path'],'alias_pin':{'path':RUN+'/readonly_closed_publication/'+r['path'],'sha256':h}} for r in rows]
    original=tmp_path/rows[0]['path'];original.parent.mkdir(parents=True,exist_ok=True);original.write_bytes(b'conflict')
    with pytest.raises(ValueError):ctx.verify_closed_publication(manifest)
    original.unlink();original.symlink_to(tmp_path/'absent-target')
    with pytest.raises(ValueError):ctx.verify_closed_publication(manifest)
    original.unlink();original.parent.rmdir();original.parent.symlink_to(tmp_path/'absent-parent',target_is_directory=True)
    with pytest.raises(ValueError):ctx.verify_closed_publication(manifest)
    original.parent.unlink();original.parent.write_text('ancestor-is-a-file')
    with pytest.raises(ValueError):ctx.verify_closed_publication(manifest)


def test_observed_literal_admin_version_and_closed_snapshot_both_preserved(tmp_path,monkeypatch):
    import echo_routing.temporal.selected_risk_resume_context as m
    ctx,put=setup(tmp_path);closed=b'closed-snapshot';observed=b'earlier-server-admin';ch=hashlib.sha256(closed).hexdigest();oh=hashlib.sha256(observed).hexdigest()
    path='docs/results/2026-11-05_selected_action_risk/README.md';saved='docs/results/2026-11-05_a63_resume_preparation/observed_historical_README.md'
    rows=[{'path':path if i==0 else 'docs/results/closed/public'+str(i)+'.json','sha256':ch,'bytes':len(closed)} for i in range(839)]
    manifest=put('docs/results/2026-11-05_a63_resume_preparation/readonly_closed_publication_manifest.json',{'file_count':839,'files':rows});monkeypatch.setattr(m,'OLD_MANIFEST',manifest['sha256'])
    monkeypatch.setattr(m,'HISTORICAL_ADMIN_VERSIONS',{path:{'observed_sha256':oh,'closed_sha256':ch,'observation_path':saved}},raising=False)
    for r in rows:
        p=tmp_path/RUN/'readonly_closed_publication'/r['path'];p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(closed)
    original=tmp_path/path;original.parent.mkdir(parents=True,exist_ok=True);original.write_bytes(observed)
    snap=tmp_path/saved;snap.write_bytes(observed)
    result=ctx.verify_closed_publication(manifest)
    assert result['different_historical_administrative_snapshots']==[{'original_observed_pin':{'path':path,'sha256':oh},'observed_raw_snapshot_pin':{'path':saved,'sha256':oh},'expected_closed_snapshot_pin':{'path':RUN+'/readonly_closed_publication/'+path,'sha256':ch},'not_closed_or_current_budget_authority':True}]
    assert original.read_bytes()==observed and snap.read_bytes()==observed
    original.write_bytes(b'another-server-version')
    with pytest.raises(ValueError):ctx.verify_closed_publication(manifest)
    original.write_bytes(observed);snap.write_bytes(b'changed-observation')
    with pytest.raises(ValueError):ctx.verify_closed_publication(manifest)
