"""No scientific runtime: synthetic source closure and temporary toy Git only."""
import hashlib
import importlib
import json
import subprocess
from pathlib import Path

import pytest


def module():return importlib.import_module('echo_routing.temporal.selected_risk_resume_freeze')

@pytest.fixture(autouse=True)
def isolate_passive_scientific_prior_for_toy_roots(monkeypatch):
    monkeypatch.setattr(module(),'PASSIVE_SOURCE',{})
def pin(path,value):return {'path':path,'sha256':hashlib.sha256(json.dumps(value).encode()).hexdigest()}

def spec():
    return {'stage':'adopt_metadata','owner':pin('owner.json',{}),'proposal':pin('proposal.json',{}),
      'baseline':pin('baseline.json',{}),'metadata':pin('metadata.json',{}),'prep_ledger':pin('prep.json',{}),
      'local_storage':pin('local.json',{}),'FIT_producer':None,'CONFIRM_producer':None,
      'freeze':pin('contracts/not-yet-created.json',{}),'predecessors':{},'arguments':{}}


def test_spec_semantics_breaks_only_freeze_hash_cycle():
    m=module();v=spec();before=m.semantics(v);v['freeze']=pin('actual.json',{'new':1})
    assert m.semantics(v)==before
    v['arguments']={'stage_CPU_upper_seconds':1}
    assert m.semantics(v)!=before


def test_required_includes_real_payloads_core_and_imports_not_runtime_spec(tmp_path,monkeypatch):
    m=module();(tmp_path/'code.py').write_text('from echo_routing import pure\n')
    (tmp_path/'echo_routing').mkdir();(tmp_path/'echo_routing/pure.py').write_text('x=1\n')
    monkeypatch.setattr(m,'_base_paths',lambda root:({'code.py'},{}))
    s=spec();s['predecessors']={'adopt_metadata':{'payload_pin':pin('docs/results/receipt.json',{}),'actual_Git_object_proof':pin('runs/proof.json',{}),'git_commit':'a'*40,'runtime_authorization_sha256':'a'*64}}
    s['arguments']['core']={'models':pin('docs/results/models.json',{}),'git_commit':'a'*40,'actual_Git_object_proof':pin('runs/coreproof.json',{})}
    out=m.required(tmp_path,'adopt_metadata',pin('runtime-final.json',s),s)
    assert {'code.py','echo_routing/pure.py','docs/results/receipt.json','docs/results/models.json','owner.json','metadata.json'}<=set(out)
    assert 'runs/proof.json' not in out and 'runtime-final.json' not in out and s['freeze']['path'] not in out
    s['arguments']['core']['models']=pin('runs/weights.pt',{})
    with pytest.raises(ValueError):m.required(tmp_path,'adopt_metadata',pin('runtime-final.json',s),s)


def test_make_absent_new_owner_fails_before_git_or_directories(tmp_path,monkeypatch):
    m=module()
    class NoOwner:
        def revalidate(self):raise ValueError('new owner absent')
    monkeypatch.setattr(m,'required',lambda *args:(_ for _ in ()).throw(AssertionError('required called before authority')))
    with pytest.raises(ValueError):m.make(NoOwner(),'adopt_metadata',pin('spec.json',{}),spec(),'a'*40)
    assert list(tmp_path.iterdir())==[]


def test_builder_current_blob_is_checked_before_first_output(tmp_path,monkeypatch):
    m=module()
    from echo_routing.temporal.selected_risk_resume_context import Context as RealContext
    class Context(RealContext):
        root=tmp_path
        owner_sha='a'*64
        def revalidate(self):pass
        def bind_runtime_spec(self,*args):pass
        def raw(self,p):return (tmp_path/p['path']).read_bytes()
        def write_json(self,*args,**kwargs):raise AssertionError('uncommitted source reached output')
    subprocess.run(['git','init','-q',str(tmp_path)],check=True)
    (tmp_path/'code.py').write_text('original=1\n')
    subprocess.run(['git','-C',str(tmp_path),'add','code.py'],check=True)
    subprocess.run(['git','-C',str(tmp_path),'-c','user.name=Fixture','-c','user.email=fixture@example.invalid','-c','commit.gpgsign=false','commit','-qm','toy'],check=True)
    commit=subprocess.check_output(['git','-C',str(tmp_path),'rev-parse','HEAD'],text=True).strip()
    (tmp_path/'code.py').write_text('changed=2\n')
    monkeypatch.setattr(m,'required',lambda *a:['code.py']);monkeypatch.setattr(m,'_base_paths',lambda root:({'code.py'},{}))
    with pytest.raises(ValueError):m.make(object.__new__(Context),'adopt_metadata',pin('spec.json',{}),spec(),commit)
    assert not (tmp_path/'runs').exists()


def test_toy_actual_Git_pool_roundtrip_and_no_clobber(tmp_path,monkeypatch):
    m=module()
    from echo_routing.temporal.selected_risk_resume_context import Context as RealContext
    from echo_routing.temporal.safety_git import verify
    from echo_routing.temporal.selected_risk_storage_engineering import expand_proof
    class Context(RealContext):
        root=tmp_path
        owner_sha='a'*64
        def revalidate(self):pass
        def bind_runtime_spec(self,*args):return m.semantics(spec())
    ctx=object.__new__(Context)
    subprocess.run(['git','init','-q',str(tmp_path)],check=True)
    (tmp_path/'code.py').write_text('source=1\n')
    subprocess.run(['git','-C',str(tmp_path),'add','code.py'],check=True)
    subprocess.run(['git','-C',str(tmp_path),'-c','user.name=Fixture','-c','user.email=fixture@example.invalid','-c','commit.gpgsign=false','commit','-qm','toy'],check=True)
    commit=subprocess.check_output(['git','-C',str(tmp_path),'rev-parse','HEAD'],text=True).strip()
    monkeypatch.setattr(m,'required',lambda *a:['code.py']);monkeypatch.setattr(m,'_base_paths',lambda root:({'code.py'},{}))
    out=m.make(ctx,'adopt_metadata',pin('spec.json',{}),spec(),commit);contract=ctx.read(out)
    pool=ctx.read(contract['shared_object_pool']);packed=ctx.read(contract['actual_Git_object_proof'])
    assert verify(expand_proof(packed,pool),commit,contract['objects'])['passed'] is True
    assert contract['runtime_spec_semantics_sha256']==m.semantics(spec())
    assert set(packed['objects'].values())=={'@shared:'+k for k in packed['objects']}
    before={str(p.relative_to(tmp_path)):p.read_bytes() for p in (tmp_path/'runs').rglob('*') if p.is_file()}
    with pytest.raises(FileExistsError):m.make(ctx,'adopt_metadata',pin('spec.json',{}),spec(),commit)
    assert before=={str(p.relative_to(tmp_path)):p.read_bytes() for p in (tmp_path/'runs').rglob('*') if p.is_file()}


def test_real_Context_provisional_to_final_rawspec_and_actual_Git(tmp_path,monkeypatch):
    """Actual new reader/writer/Git; only the scientific prior table is toy data."""
    import runpy
    setup=runpy.run_path(str(Path(__file__).with_name('test_selected_risk_resume_context.py')))['setup']
    seeded,put=setup(tmp_path);m=module()
    from echo_routing.temporal.selected_risk_resume_context import Context
    from echo_routing.temporal.selected_risk_resume_contract import OLD_DEADLINE
    ctx=Context(tmp_path,seeded.owner_pin,seeded.proposal_pin,seeded.baseline_pin,now=OLD_DEADLINE+61)
    monkeypatch.setattr(m,'_base_paths',lambda root:({'code.py'},{}))
    monkeypatch.setattr(ctx,'_historical_expected',lambda:{})
    (tmp_path/'code.py').write_text('toy_source=1\n')
    s={'schema':'A63_resume_runtime_spec_v1','stage':'adopt_metadata','owner':ctx.owner_pin,
       'proposal':ctx.proposal_pin,'baseline':ctx.baseline_pin,'freeze':pin('future-contract.json',{}),
       'predecessors':{},'metadata':put('metadata.json',{}),'FIT_producer':None,'CONFIRM_producer':None,
       'prep_ledger':put('prep.json',{}),'local_storage':put('local.json',{}),'expected_boot_id':'toy',
       'arguments':{'stage_CPU_upper_seconds':1,'stage_wall_upper_seconds':1}}
    subprocess.run(['git','init','-q',str(tmp_path)],check=True)
    subprocess.run(['git','-C',str(tmp_path),'add','.'],check=True)
    subprocess.run(['git','-C',str(tmp_path),'-c','user.name=Fixture','-c','user.email=fixture@example.invalid','-c','commit.gpgsign=false','commit','-qm','toy'],check=True)
    commit=subprocess.check_output(['git','-C',str(tmp_path),'rev-parse','HEAD'],text=True).strip()
    provisional=put('spec.provisional.json',s);freeze=m.make(ctx,'adopt_metadata',provisional,s,commit)
    with pytest.raises(ValueError):ctx.verify_stage('adopt_metadata',freeze,required={'code.py'})
    s['freeze']=freeze;final=put('spec.final.json',s);ctx.bind_runtime_spec(final,s)
    assert ctx.verify_stage('adopt_metadata',freeze,required={'code.py'})['git_commit']==commit
    s['arguments']['stage_CPU_upper_seconds']=2
    changed=put('spec.changed.json',s)
    with pytest.raises(ValueError):ctx.bind_runtime_spec(changed,s)


def test_passive_calibration_source_is_pinned_without_importing_writing_cli(tmp_path,monkeypatch):
    m=module();(tmp_path/'code.py').write_text('x=1\n');(tmp_path/'scripts').mkdir();passive=tmp_path/'scripts/calibrate_event_risk.py';passive.write_text('import a_forbidden_writing_cli\n')
    import hashlib
    monkeypatch.setattr(m,'PASSIVE_SOURCE',{'scripts/calibrate_event_risk.py':hashlib.sha256(passive.read_bytes()).hexdigest()},raising=False)
    monkeypatch.setattr(m,'_base_paths',lambda root:({'code.py'},{}))
    required=m.required(tmp_path,'adopt_metadata',pin('spec.json',{}),spec())
    assert 'scripts/calibrate_event_risk.py' in required and 'a_forbidden_writing_cli' not in str(required)
    passive.write_text('changed=1\n')
    with pytest.raises(ValueError):m.required(tmp_path,'adopt_metadata',pin('spec.json',{}),spec())


def test_new_spec_semantics_preserves_all_prior_prepared_artifacts(tmp_path,monkeypatch):
    m=module()
    from echo_routing.temporal.selected_risk_resume_context import Context as RealContext
    class Context(RealContext):
        root=tmp_path;owner_sha='a'*64
        def revalidate(self):pass
        def bind_runtime_spec(self,*args):pass
    ctx=object.__new__(Context)
    subprocess.run(['git','init','-q',str(tmp_path)],check=True)
    (tmp_path/'code.py').write_text('x=1\n')
    subprocess.run(['git','-C',str(tmp_path),'add','code.py'],check=True)
    subprocess.run(['git','-C',str(tmp_path),'-c','user.name=Fixture','-c','user.email=fixture@example.invalid','-c','commit.gpgsign=false','commit','-qm','toy'],check=True)
    commit=subprocess.check_output(['git','-C',str(tmp_path),'rev-parse','HEAD'],text=True).strip()
    monkeypatch.setattr(m,'required',lambda *a:['code.py']);monkeypatch.setattr(m,'_base_paths',lambda root:({'code.py'},{}))
    first=m.make(ctx,'adopt_metadata',pin('spec.json',{}),spec(),commit)
    prior={str(p.relative_to(tmp_path)):p.read_bytes() for p in (tmp_path/'runs').rglob('*') if p.is_file()}
    newer=spec();newer['arguments']={'new_storage_pin':True}
    second=m.make(ctx,'adopt_metadata',pin('spec2.json',{}),newer,commit)
    assert second!=first
    assert all((tmp_path/p).read_bytes()==b for p,b in prior.items())
    assert m.semantics(newer)[:12] in second['path']
    assert ctx.read(first)['shared_object_pool']==ctx.read(second)['shared_object_pool']
