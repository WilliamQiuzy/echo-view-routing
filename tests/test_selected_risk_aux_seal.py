import importlib,json,subprocess,hashlib
from pathlib import Path
import pytest
from echo_routing.temporal.selected_risk_contracts import stage_spec
from echo_routing.temporal.safety_git import build

def test_confirmation_freeze_mandatory_aux_core_and_actual_seal():
 _,inputs=stage_spec('source_CONFIRM');names={Path(p).name for p in inputs}
 assert {'aux_model_audit.json','aux_models_manifest.json','aux_training_inputs.json','aux_identity.json'}<=names

def test_actual_toy_Git_aux_seal_accepts_exact_eight_then_rejects_tamper(tmp_path,monkeypatch):
 m=importlib.import_module('scripts.selected_risk_common');f=getattr(m,'verify_aux_seal',None)
 assert callable(f),'actual aux Git/readable private pin barrier absent'
 root=tmp_path/'root';pub=root/'pub';pub.mkdir(parents=True);run=root/'run';run.mkdir();sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();dump=lambda p,v:p.write_text(json.dumps(v,sort_keys=True))
 monkeypatch.setattr(m,'ROOT',root);monkeypatch.setattr(m,'PUB',pub)
 rows=[]
 for seed in [2026110461,2026110462]:
  for fold in [0,1]:
   for role in ['utility','risk']:
    p=run/f'{seed}_{fold}_{role}.bin';p.write_bytes(f'FAKE no model state {seed}/{fold}/{role}'.encode());rows.append({'seed':seed,'held_fold':fold,'role':role,'step':6000,'path':str(p.relative_to(root)),'sha256':sha(p)})
 dump(pub/'aux_training_inputs.json',{'scope':'synthetic contract only; no training happened'})
 ih=sha(pub/'aux_training_inputs.json');models={'models':rows,'aux_training_inputs_sha256':ih};audit={'passed':True,'aux_optimizer_updates':48000,'physical_models':8,'fullstate_count':32,'models':rows,'aux_training_inputs_sha256':ih}
 dump(pub/'aux_models_manifest.json',models);dump(pub/'aux_model_audit.json',audit)
 def git(*args):return subprocess.check_output(['git','-C',str(root),*args],stderr=subprocess.PIPE).decode().strip()
 git('init','-q');git('config','user.name','toy');git('config','user.email','toy@example.invalid');git('add','pub');git('commit','-qm','synthetic aux contract')
 names=['pub/'+n for n in ['aux_models_manifest.json','aux_model_audit.json','aux_training_inputs.json']];commit=git('rev-parse','HEAD');proof=build(root,commit,names);dump(pub/'proof.json',proof)
 seal={'objects':{p:sha(root/p) for p in names},'git_commit':commit,'actual_Git_object_proof':{'path':'pub/proof.json','sha256':sha(pub/'proof.json')},'all_final_checkpoints':[{k:r[k] for k in ['path','sha256']} for r in rows]};dump(pub/'aux_identity.json',seal)
 assert f()==models
 audit['passed']=False;dump(pub/'aux_model_audit.json',audit)
 with pytest.raises(ValueError):f()
