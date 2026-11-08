import ast
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def normalized(path):
    source=path.read_text().replace('completed_supply','completed_aux').replace('CompletedSupplyRuntimeState','CompletedAuxRuntimeState')
    tree=ast.parse(source)
    if tree.body and isinstance(tree.body[0],ast.Expr) and isinstance(tree.body[0].value,ast.Constant) and isinstance(tree.body[0].value.value,str):tree.body.pop(0)
    for node in tree.body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='EXTRA_SOURCES' for t in node.targets):
            seen=set();values=[]
            for value in node.value.elts:
                key=ast.dump(value)
                if key not in seen:values.append(value);seen.add(key)
            node.value.elts=values
    return ast.dump(tree,include_attributes=False)


def test_supply_thin_preserves_previous_control_flow_and_all_eight_sources():
    previous=json.loads((ROOT/'docs/results/2026-11-05_a63_resume_preparation/completed_aux_controller_quality_825_approved_v1.json').read_bytes())['file_pins']
    for name in ['echo_routing/temporal/selected_risk_completed_aux_controller.py',*[f'scripts/a63_{action}_completed_aux_selected_risk.py' for action in ['resume','wait','prepare']]]:
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==previous[name]['sha256']
    for action in ['resume','wait']:
        assert normalized(ROOT/f'scripts/a63_{action}_completed_supply_selected_risk.py')==normalized(ROOT/f'scripts/a63_{action}_completed_aux_selected_risk.py')
    path=ROOT/'scripts/a63_resume_completed_supply_selected_risk.py';spec=importlib.util.spec_from_file_location('supply_entry',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    expected={'echo_routing/temporal/selected_risk_completed_'+role+'_controller.py' for role in ['aux','supply']}
    expected|={f'scripts/a63_{action}_completed_{role}_selected_risk.py' for action in ['resume','wait','prepare'] for role in ['aux','supply']}
    assert set(m.EXTRA_SOURCES)==expected and len(m.EXTRA_SOURCES)==8
    seen=[]
    class Context:
        def verify_stage(self,stage,pin,required):seen.append(required);return {'objects':{}}
    m.verify_extra_sources(Context(),{'stage':'seal_aux','freeze':{'path':'new','sha256':'a'*64}})
    assert seen==[expected]


def _single_supply_setup(tmp_path,monkeypatch):
    import subprocess
    from echo_routing.temporal.selected_risk_resume_context import Context as RealContext
    from echo_routing.temporal.selected_risk_resume_contract import RUN
    from echo_routing.temporal.safety_git import build
    import scripts.a63_prepare_completed_supply_selected_risk as m
    def git(*args):return subprocess.check_output(['git','-C',str(tmp_path),*args],text=True).strip()
    git('init','-q');git('config','user.email','fixture@example.test');git('config','user.name','Fixture')
    for name in ['old.py','historical.py',*m.EXTRA_SOURCES]:
        p=tmp_path/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('X=1\n')
    git('add','.');git('commit','-qm','old toy only');commit=git('rev-parse','HEAD')
    class Context(RealContext):
        root=tmp_path;owner_sha='a'*64
        def revalidate(self):pass
        def bind_runtime_spec(self,*args):pass
    c=object.__new__(Context)
    h=lambda p:hashlib.sha256((tmp_path/p).read_bytes()).hexdigest()
    hp=build(tmp_path,commit,['historical.py']);hpin=c.write_json('Git_proofs/old/historical.py.json',hp)
    historical={'historical.py':{'git_commit':commit,'sha256':h('historical.py'),'blob_oid':hp['files']['historical.py']['blob_oid']}}
    pool=build(tmp_path,commit,['old.py']);poolpin=c.write_json('Git_proofs/shared_pool.json',pool)
    prior={'schema':'A63_resume_stage_freeze_v1','stage':'seal_aux','git_commit':commit,'objects':{'old.py':h('old.py'),'historical.py':h('historical.py')},'historical_frozen_imports':{'historical.py':{**historical['historical.py'],'proof':hpin}},'actual_Git_object_proof':poolpin,'runtime_authorization_sha256':c.owner_sha,'runtime_spec_semantics_sha256':'b'*64,'proof_encoding':'A63_shared_Git_objects_v1','shared_object_pool':poolpin}
    priorpin=c.write_json('contracts/old.json',prior)
    monkeypatch.setattr(m.base,'required',lambda *args:['old.py','historical.py'])
    monkeypatch.setattr(m.base,'_base_paths',lambda root:({'old.py','historical.py'},historical))
    monkeypatch.setattr(m.base,'make',lambda *args:(_ for _ in ()).throw(AssertionError('single supply freeze must not write a duplicate base freeze')))
    s={'stage':'seal_aux','arguments':{'completed_supply_prior_freeze':priorpin}}
    return m,c,commit,s,prior,pool,git


def test_single_supply_freeze_reuses_verified_history_and_writes_only_current_contract(tmp_path,monkeypatch):
    from echo_routing.temporal.safety_git import verify
    from echo_routing.temporal.selected_risk_storage_engineering import expand_proof
    from echo_routing.temporal.selected_risk_resume_contract import RUN
    import echo_routing.temporal.safety_git as g
    m,c,commit,s,prior,pool,git=_single_supply_setup(tmp_path,monkeypatch)
    before={str(p.relative_to(tmp_path)):p.read_bytes() for p in (tmp_path/RUN).rglob('*') if p.is_file()}
    calls=[];build=g.build
    def tracked(*args):calls.append(args);return build(*args)
    monkeypatch.setattr(g,'build',tracked)
    pin=m.make(c,'seal_aux',{'path':'spec.json','sha256':'c'*64},s,commit);out=c.read(pin)
    assert len(calls)==1
    assert out['historical_frozen_imports']==prior['historical_frozen_imports'] and out['shared_object_pool']==prior['shared_object_pool']
    current={p:h for p,h in out['objects'].items() if p not in out['historical_frozen_imports']}
    full=build(tmp_path,commit,sorted(current))
    assert expand_proof(c.read(out['actual_Git_object_proof']),pool)==full
    assert verify(full,commit,current)['passed'] is True
    after={str(p.relative_to(tmp_path)):p.read_bytes() for p in (tmp_path/RUN).rglob('*') if p.is_file()}
    assert set(after)-set(before)=={pin['path'],out['actual_Git_object_proof']['path']}
    assert all(after[p]==b for p,b in before.items())
    import pytest
    with pytest.raises(FileExistsError):m.make(c,'seal_aux',{'path':'spec.json','sha256':'c'*64},s,commit)
    assert after=={str(p.relative_to(tmp_path)):p.read_bytes() for p in (tmp_path/RUN).rglob('*') if p.is_file()}


def test_single_supply_freeze_bad_history_or_uncommitted_current_rejected_before_output(tmp_path,monkeypatch):
    from echo_routing.temporal.selected_risk_resume_contract import RUN
    import pytest
    m,c,commit,s,prior,pool,git=_single_supply_setup(tmp_path,monkeypatch)
    before={str(p.relative_to(tmp_path)):p.read_bytes() for p in (tmp_path/RUN).rglob('*') if p.is_file()}
    hp=tmp_path/prior['historical_frozen_imports']['historical.py']['proof']['path'];original=hp.read_bytes();hp.write_bytes(original+b' ')
    with pytest.raises(ValueError):m.make(c,'seal_aux',{'path':'spec.json','sha256':'c'*64},s,commit)
    hp.write_bytes(original)
    (tmp_path/'old.py').write_text('CHANGED=2\n')
    with pytest.raises(ValueError):m.make(c,'seal_aux',{'path':'spec.json','sha256':'c'*64},s,commit)
    assert before=={str(p.relative_to(tmp_path)):p.read_bytes() for p in (tmp_path/RUN).rglob('*') if p.is_file()}
