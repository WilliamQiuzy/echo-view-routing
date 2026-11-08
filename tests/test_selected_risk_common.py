import importlib,json
import pytest
from pathlib import Path

def api():
 try:m=importlib.import_module('scripts.selected_risk_common')
 except ModuleNotFoundError:m=None
 assert m is not None,'A63 explicit owner/namespace common absent'
 return m

def test_default_dryrun_and_exact_owner_config():
 m=api();assert m.validate_config(m.read(m.CFG)) is True
 p=m.execution_parser('test');a=p.parse_args([]);assert m.authorize(a) is False
 c=m.read(m.CFG);c['supply']['partition_prefix_bytes_hex']='00'
 with pytest.raises(ValueError):m.validate_config(c)

def test_qualified_output_namespace_parent_and_leaf_symlinks(tmp_path,monkeypatch):
 m=api();root=tmp_path/'root';run=root/'runs/a63_selected_action_risk_v1';pub=root/'docs/results/a63';run.mkdir(parents=True);pub.mkdir(parents=True)
 monkeypatch.setattr(m,'ROOT',root);monkeypatch.setattr(m,'RUN',run);monkeypatch.setattr(m,'PUB',pub)
 assert m.checked_output(run/'x')==run/'x'
 with pytest.raises(ValueError):m.checked_output(root/'old_namespace/x')
 outside=tmp_path/'outside';outside.mkdir();(run/'link').symlink_to(outside,target_is_directory=True)
 with pytest.raises(ValueError):m.checked_output(run/'link/x')

def test_alias_original_pin_cannot_be_changed_or_escape(tmp_path,monkeypatch):
 m=api();root=tmp_path/'root';root.mkdir();(root/'old').write_text('original');h=m.sha(root/'old');monkeypatch.setattr(m,'ROOT',root)
 assert m.verify_pin({'path':'old','sha256':h})==root/'old'
 (root/'old').write_text('tamper')
 with pytest.raises(ValueError):m.verify_pin({'path':'old','sha256':h})
 with pytest.raises(ValueError):m.verify_pin({'path':'../outside','sha256':h})

def test_old_dependency_current_git_proof_cannot_replace_approved_basis_sha():
 m=api();f=getattr(m,'validate_frozen_prior',None)
 assert callable(f),'independent original basis comparison missing from verifier'
 p='echo_routing/temporal/proposal_training.py'
 with pytest.raises(ValueError):f({p:'0'*64})
 cfg=m.read(m.CFG);basis=m.read(m.verify_pin(cfg['basis']))
 assert f({p:basis['predecessor_public_and_source_pins'][p]})

def test_historical_imports_crossbind_literal_source_commit_and_basis():
 m=api();f=getattr(m,'validate_source_refs',None)
 assert callable(f),'historical operational refs need independent basis binding'
 cfg=m.read(m.CFG);basis=m.read(m.verify_pin(cfg['basis']));refs=m.read(m.verify_pin(cfg['source_refs']))
 assert f(refs,basis)
 key=next(iter(refs['historical_frozen_imports']));refs['historical_frozen_imports'][key]['sha256']='0'*64
 with pytest.raises(ValueError):f(refs,basis)
