import hashlib,json,subprocess,shutil
from pathlib import Path
import pytest

def test_relocated_public_context_has_exact_original_git_and_basis_bytes(tmp_path,monkeypatch):
 import echo_routing.temporal.event_risk_context as context
 from echo_routing.temporal.event_risk_context import verify,ORIGINALS
 from echo_routing.temporal.safety_git import build
 for k,v in [('user.name','Toy'),('user.email','toy@example.invalid')]:
  if k=='user.name':subprocess.run(['git','init','-q',str(tmp_path)],check=True)
  subprocess.run(['git','-C',str(tmp_path),'config',k,v],check=True)
 hashes={}
 for i,p in enumerate(ORIGINALS):
  f=tmp_path/p;f.parent.mkdir(parents=True,exist_ok=True);f.write_text(json.dumps({'synthetic':i}));hashes[p]=hashlib.sha256(f.read_bytes()).hexdigest()
 subprocess.run(['git','-C',str(tmp_path),'add',ORIGINALS[-1]],check=True);subprocess.run(['git','-C',str(tmp_path),'commit','-qm','scientific'],check=True);commit=subprocess.check_output(['git','-C',str(tmp_path),'rev-parse','HEAD'],text=True).strip()
 subprocess.run(['git','-C',str(tmp_path),'add','.'],check=True);subprocess.run(['git','-C',str(tmp_path),'commit','-qm','posthoc_archive'],check=True);archive=subprocess.check_output(['git','-C',str(tmp_path),'rev-parse','HEAD'],text=True).strip();proofs={commit:build(tmp_path,commit,[ORIGINALS[-1]]),archive:build(tmp_path,archive,ORIGINALS[:-1])};files={};monkeypatch.setattr(context,'ARCHIVE_GIT',archive)
 for i,p in enumerate(ORIGINALS):
  new=tmp_path/f'docs/results/2026-11-04_anchor_context_risk/readonly_context/context{i}.json';new.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(tmp_path/p,new);files[p]={'path':str(new.relative_to(tmp_path)),'sha256':hashes[p]}
 cfg={'prior_source_pins':hashes,'prior_scientific_git':commit};binding={'source_commits':{p:archive if p in ORIGINALS[:-1] else commit for p in ORIGINALS},'files':files};result=verify(tmp_path,cfg,binding,proofs);assert len(result)==4
 target=tmp_path/files[ORIGINALS[0]]['path'];target.write_text('changed')
 with pytest.raises(ValueError):verify(tmp_path,cfg,binding,proofs)

def test_readonly_context_parent_symlink_rejects_even_matching_bytes(tmp_path,monkeypatch):
 import echo_routing.temporal.event_risk_context as context
 from echo_routing.temporal.safety_git import build
 subprocess.run(['git','init','-q',str(tmp_path)],check=True)
 for k,v in [('user.name','Toy'),('user.email','toy@example.invalid')]:subprocess.run(['git','-C',str(tmp_path),'config',k,v],check=True)
 hashes={};files={}
 for i,p in enumerate(context.ORIGINALS):
  f=tmp_path/p;f.parent.mkdir(parents=True,exist_ok=True);f.write_text(json.dumps({'synthetic':i}));hashes[p]=hashlib.sha256(f.read_bytes()).hexdigest()
 subprocess.run(['git','-C',str(tmp_path),'add','.'],check=True);subprocess.run(['git','-C',str(tmp_path),'commit','-qm','toy'],check=True);commit=subprocess.check_output(['git','-C',str(tmp_path),'rev-parse','HEAD'],text=True).strip();monkeypatch.setattr(context,'ARCHIVE_GIT',commit)
 folder=tmp_path/'docs/results/2026-11-04_anchor_context_risk/readonly_context';folder.mkdir(parents=True)
 for i,p in enumerate(context.ORIGINALS):
  f=folder/f'file{i}.json';shutil.copyfile(tmp_path/p,f);files[p]={'path':str(f.relative_to(tmp_path)),'sha256':hashes[p]}
 binding={'source_commits':{p:commit for p in context.ORIGINALS},'files':files};cfg={'prior_scientific_git':commit,'prior_source_pins':hashes};proofs={commit:build(tmp_path,commit,context.ORIGINALS)}
 assert context.verify(tmp_path,cfg,binding,proofs)
 outside=tmp_path/'outside';folder.rename(outside);folder.symlink_to(outside,target_is_directory=True)
 with pytest.raises(ValueError):context.verify(tmp_path,cfg,binding,proofs)
