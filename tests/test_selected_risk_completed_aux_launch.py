import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import pytest

ROOT=Path(__file__).resolve().parents[1]

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,ROOT/path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def test_completed_aux_outer_uses_original_caller_and_exact_guardian(monkeypatch):
    m=module('completed_outer','scripts/a63_wait_completed_aux_selected_risk.py');seen=[]
    monkeypatch.setattr(m.subprocess,'Popen',lambda command,**kw:seen.append((command,kw)) or 'child')
    original=['python',str(ROOT/'scripts/a63_resume_selected_risk.py'),'--execute','--stage','audit_aux']
    assert m.spawn_guardian(ROOT,original,cwd=ROOT,env={'ticket':'raw'})=='child'
    assert seen==[(['python',str(ROOT/'scripts/a63_resume_completed_aux_selected_risk.py'),*original[2:]],{'cwd':ROOT,'env':{'ticket':'raw'}})]
    with pytest.raises(ValueError):m.spawn_guardian(ROOT,['python','foreign.py'])


def test_completed_aux_extra_sources_are_mandatory_before_parent_or_child(monkeypatch):
    m=module('completed_entry','scripts/a63_resume_completed_aux_selected_risk.py');calls=[]
    class Context:
        def verify_stage(self,stage,pin,required):calls.append((stage,pin,required));return {'objects':{}}
    m.verify_extra_sources(Context(),{'stage':'audit_aux','freeze':{'path':'x','sha256':'a'*64}})
    assert calls[0][2]==set(m.EXTRA_SOURCES)
    assert len(calls[0][2])==4
    import sys,types
    class Controller:
        def __init__(self):self.ctx=types.SimpleNamespace(root=ROOT)
        def admission(self):
            return {'all_caps_fit':False,'already_spent':{'receipt_records':[{'receipt_id':'old','actual_fresh_completed_aux_admission':{'older':'preserved only in original receipt'}}],'CPU':3},'stage_CPU_limit':600}
        def record(self,row,intent):return row
        def command(self,stage,intent,lock):
            return ['python',str(ROOT/'scripts/a63_resume_selected_risk.py'),'--child','--stage',stage]
    monkeypatch.setitem(sys.modules,'echo_routing.temporal.selected_risk_completed_aux_controller',types.SimpleNamespace(CompletedAuxRuntimeState=Controller))
    state=m.state_class()()
    assert state.command('audit_aux',{},3)[1]==str(ROOT/'scripts/a63_resume_selected_risk.py')
    assert state.command('admit_formal',{},3)[1]==str(ROOT/'scripts/a63_resume_completed_aux_selected_risk.py')
    original=state.admission();recorded=state.record({'stage':'audit_aux'},None)['actual_fresh_completed_aux_admission']
    assert recorded['all_caps_fit'] is False and recorded['stage_CPU_limit']==600
    assert original['already_spent']['receipt_records']
    assert 'receipt_records' not in recorded['already_spent'] and recorded['already_spent']['receipt_records_count']==1
    assert len(recorded['already_spent']['receipt_records_canonical_json_sha256'])==64
    assert recorded['already_spent']['CPU']==3


def test_completed_aux_freeze_augments_actual_git_and_preserves_base(tmp_path,monkeypatch):
    m=module('completed_freezer','scripts/a63_prepare_completed_aux_selected_risk.py')
    from echo_routing.temporal.safety_git import build,verify
    from echo_routing.temporal.selected_risk_storage_engineering import expand_proof
    from echo_routing.temporal.selected_risk_resume_contract import RUN
    def git(*args):return subprocess.check_output(['git','-C',str(tmp_path),*args]).decode().strip()
    git('init','-q');git('config','user.email','fixture@example.test');git('config','user.name','Fixture')
    for name in ['old.py',*m.EXTRA_SOURCES]:
        p=tmp_path/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('pass\n' if name!='old.py' else 'OLD=1\n')
    git('add','.');git('commit','-qm','frozen toy only');commit=git('rev-parse','HEAD');h=lambda p:hashlib.sha256((tmp_path/p).read_bytes()).hexdigest()
    class Context:
        root=tmp_path;owner_sha='a'*64
        def output(self,name):return tmp_path/RUN/name
        def raw(self,pin):
            b=(tmp_path/pin['path']).read_bytes();assert hashlib.sha256(b).hexdigest()==pin['sha256'];return b
        def read(self,pin):return json.loads(self.raw(pin))
        def write_json(self,name,value):
            p=self.output(name);p.parent.mkdir(parents=True,exist_ok=True);raw=json.dumps(value,sort_keys=True).encode()
            with p.open('xb') as f:f.write(raw)
            return {'path':str(p.relative_to(tmp_path)),'sha256':hashlib.sha256(raw).hexdigest()}
    c=Context();pool=build(tmp_path,commit,['old.py']);poolpin=c.write_json('Git_proofs/shared_pool.json',pool)
    base={'schema':'A63_resume_stage_freeze_v1','stage':'audit_aux','git_commit':commit,'objects':{'old.py':h('old.py')},'historical_frozen_imports':{},'actual_Git_object_proof':poolpin,'runtime_authorization_sha256':c.owner_sha,'runtime_spec_semantics_sha256':'b'*64,'proof_encoding':'A63_shared_Git_objects_v1','shared_object_pool':poolpin};basepin=c.write_json('contracts/base.json',base);rawbase=c.raw(basepin)
    monkeypatch.setattr(m.base,'make',lambda *args:basepin)
    out=m.make(c,'audit_aux',{'path':'spec.json','sha256':'c'*64},{'stage':'audit_aux','arguments':{}},commit);value=c.read(out)
    assert c.raw(basepin)==rawbase and value['historical_frozen_imports']==base['historical_frozen_imports']
    assert value['objects']=={p:h(p) for p in ['old.py',*m.EXTRA_SOURCES]}
    verify(expand_proof(c.read(value['actual_Git_object_proof']),c.read(poolpin)),commit,value['objects'])
    with pytest.raises(FileExistsError):m.make(c,'audit_aux',{'path':'spec.json','sha256':'c'*64},{'stage':'audit_aux','arguments':{}},commit)
