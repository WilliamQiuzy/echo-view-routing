"""Pure versioned FIT contract wiring, no source/SSH/GPU/NN or actual freeze."""
import copy
import hashlib
import importlib
import json

import pytest


def module():
    return importlib.import_module('scripts.a63_selected_risk_FIT_v2_contract')


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value, sort_keys=True))


def fixture(tmp_path, monkeypatch):
    m=module(); root=tmp_path; run=root/'runs/a63_selected_action_risk_v1'
    pub=root/'docs/results/2026-11-05_selected_action_risk'; cfg=root/'configs/experiments/selected_action_risk_v1.json'
    for k,v in dict(ROOT=root,RUN=run,PUB=pub,CFG=cfg).items():monkeypatch.setattr(m,k,v)
    write(pub/'input_manifest.json', {'fixture':True})
    admission=dict(all_caps_fit=True,stage='sources',source_phase='FIT',version=3,
                   admission_freeze_git='f558e0d1b0137d534446fdcc261623ca3234b1e4',
                   input_manifest_sha256=hashlib.sha256((pub/'input_manifest.json').read_bytes()).hexdigest(),
                   deadline_epoch=1793897040)
    write(pub/'sources_admission_v3.json',admission);write(pub/'sources_admission.json',admission)
    historical={};objects={}
    for i,p in enumerate(sorted(m.HISTORICAL_PATHS)):
        f=root/p;f.parent.mkdir(parents=True,exist_ok=True);f.write_text('# historical fixture '+str(i))
        h=hashlib.sha256(f.read_bytes()).hexdigest();objects[p]=h
        proof=dict(path='runs/a63_selected_action_risk_v1/proof'+str(i)+'.json',sha256='a'*64)
        write(root/proof['path'], {'fixture_historical':i})
        historical[p]=dict(git_commit='d'*40,blob_oid='e'*40,sha256=h,proof=proof)
    for name in ['input_manifest.json','sources_admission.json','sources_admission_v3.json']:
        p=str((pub/name).relative_to(root));objects[p]=hashlib.sha256((root/p).read_bytes()).hexdigest()
    write(cfg,{'source_refs':{'path':'source_refs.json','sha256':'a'*64}})
    write(root/'source_refs.json',{'historical_frozen_imports':{p:{k:v[k] for k in ['git_commit','blob_oid','sha256']} for p,v in historical.items()}})
    pool=dict(path='runs/a63_selected_action_risk_v1/pool.json',sha256='b'*64)
    packed=dict(path='runs/a63_selected_action_risk_v1/packed.json',sha256='c'*64)
    write(root/pool['path'],{'pool':True});write(root/packed['path'],{'packed':True})
    contract=dict(stage='source_FIT_v2',objects=objects,historical_frozen_imports=historical,
                  proof_encoding='A63_shared_Git_objects_v1',shared_object_pool=copy.deepcopy(pool),
                  actual_Git_object_proof=packed,git_commit='f'*40,fixed_T0_epoch=1793853840,
                  fixed_deadline_epoch=1793897040)
    write(run/'contracts/source_FIT_v2.json',contract)
    write(run/'contracts/source_FIT.json',{'original_prepared_contract':'never overwrite'})
    monkeypatch.setattr(m,'verify_freeze',lambda s: {'actual_Git_object_proof':copy.deepcopy(pool)} if s=='metadata' else pytest.fail('must not reuse old source_FIT'))
    monkeypatch.setattr(m,'required',lambda stage='source_FIT_v2':set(objects))
    monkeypatch.setattr(m,'verify_pin',lambda pin:root/pin['path'])
    monkeypatch.setattr(m,'validate_frozen_prior',lambda x:True)
    calls=[];monkeypatch.setattr(m,'verify_Git',lambda proof,commit,expected:calls.append((proof,commit,expected)))
    def expand(a,b):
        assert a=={'packed':True} and b=={'pool':True}
        return {'expanded_fixture':True}
    monkeypatch.setattr(m,'expand_proof',expand)
    return m,root,run,pub,contract,calls


def test_required_includes_old_producer_all_eight_imports_and_all_new_entrypoints(tmp_path,monkeypatch):
    m=module();monkeypatch.setattr(m,'ROOT',tmp_path);monkeypatch.setattr(m,'PUB',tmp_path/'docs/results/2026-11-05_selected_action_risk')
    stages=[];monkeypatch.setattr(m,'old_required',lambda s:stages.append(s) or {'scripts/cache_selected_risk.py'})
    monkeypatch.setattr(m,'source_closure',lambda root,paths:paths)
    result=m.required()
    assert stages==['source_FIT'] and set(m.HISTORICAL_PATHS)<=result
    assert {'scripts/a63_cache_selected_risk_FIT_v2.py','scripts/a63_freeze_selected_risk_FIT_v2.py',
            'scripts/a63_supervise_selected_risk_FIT_v2.py','scripts/a63_selected_risk_FIT_v2_contract.py',
            'scripts/cache_selected_risk.py','docs/results/2026-11-05_selected_action_risk/sources_admission_v3.json'}<=result
    with pytest.raises(ValueError):m.required('source_FIT')


def test_valid_versioned_binding_runs_old_pool_all_eight_historical_and_current_verifiers(tmp_path,monkeypatch):
    m,root,run,pub,c,calls=fixture(tmp_path,monkeypatch);before=(run/'contracts/source_FIT.json').read_bytes()
    assert m.verify_shared_freeze('source_FIT_v2')==c
    assert len(calls)==9 and calls[-1][0]=={'expanded_fixture':True}
    assert not set(m.HISTORICAL_PATHS)&set(calls[-1][2])
    assert (run/'contracts/source_FIT.json').read_bytes()==before


@pytest.mark.parametrize('change',['canonical_bytes','false','wrong_input','wrong_phase','wrong_stage','wrong_version','wrong_admission_git','missing_historical','current_drift','wrong_pool','wrong_contract_stage'])
def test_new_contract_never_adopts_old_false_or_unmatched_source_admission(tmp_path,monkeypatch,change):
    m,root,run,pub,c,calls=fixture(tmp_path,monkeypatch)
    if change=='canonical_bytes':(pub/'sources_admission.json').write_bytes((pub/'sources_admission_v3.json').read_bytes()+b'\n')
    elif change in ['false','wrong_input','wrong_phase','wrong_stage','wrong_version','wrong_admission_git']:
        d=json.loads((pub/'sources_admission.json').read_text())
        k,v={'false':('all_caps_fit',False),'wrong_input':('input_manifest_sha256','0'*64),
             'wrong_phase':('source_phase','CONFIRM'),'wrong_stage':('stage','outer'),
             'wrong_version':('version',2),'wrong_admission_git':('admission_freeze_git','0'*40)}[change]
        d[k]=v
        for n in ['sources_admission.json','sources_admission_v3.json']:write(pub/n,d)
        # Re-freezing a changed admission is insufficient: actual v3 identity is mandatory.
        for n in ['sources_admission.json','sources_admission_v3.json']:
            p=str((pub/n).relative_to(root));c['objects'][p]=hashlib.sha256((pub/n).read_bytes()).hexdigest()
    elif change=='missing_historical':c['historical_frozen_imports'].pop(next(iter(c['historical_frozen_imports'])))
    elif change=='current_drift':(root/next(iter(c['objects']))).write_text('changed')
    elif change=='wrong_pool':c['shared_object_pool']['sha256']='0'*64
    elif change=='wrong_contract_stage':c['stage']='source_FIT'
    write(run/'contracts/source_FIT_v2.json',c)
    with pytest.raises(ValueError):m.verify_shared_freeze('source_FIT_v2')

