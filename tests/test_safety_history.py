import copy,hashlib,importlib,json,subprocess
import pytest
from echo_routing.temporal.safety_git import build


def test_history_requires_actual_git_all_objects_and_exact_config_projection(tmp_path):
    m=importlib.import_module('echo_routing.temporal.safety_history')
    subprocess.run(['git','init','-q',str(tmp_path)],check=True)
    old={'training':{'F_fit':'old','draw_RNG_domains':'old','kind_candidate_batch':[32,32],'sampling':'old','lr':.001},'source':'fixed'}
    new=copy.deepcopy(old);new['training'].update(F_fit='new',draw_RNG_domains={'utility':'declared','risk':'declared'},sampling='new');del new['training']['kind_candidate_batch']
    cfg=tmp_path/'config.json';source=tmp_path/'source.py';cfg.write_text(json.dumps(old));source.write_text('same-source\n')
    subprocess.run(['git','-C',str(tmp_path),'add','config.json','source.py'],check=True)
    subprocess.run(['git','-C',str(tmp_path),'-c','user.name=Fixture','-c','user.email=fixture@example.invalid','commit','-qm','old source'],check=True)
    commit=subprocess.check_output(['git','-C',str(tmp_path),'rev-parse','HEAD'],text=True).strip();proof=build(tmp_path,commit,['config.json','source.py']);objects={p:r['sha256'] for p,r in proof['files'].items()};identities={n:{'git_commit':commit,'objects':objects} for n in ['prep','fit','check']};expected={n:{'commit':commit,'objects':2,'identity_sha256':hashlib.sha256(json.dumps(v,sort_keys=True).encode()).hexdigest()} for n,v in identities.items()};raw={n:json.dumps(v,sort_keys=True).encode() for n,v in identities.items()};bundle={'contracts':{n:{'proof':proof} for n in identities}};cfg.write_text(json.dumps(new));correction={'old_config_sha256':objects['config.json'],'new_config_sha256':hashlib.sha256(cfg.read_bytes()).hexdigest()}
    result=m.validate(bundle,raw,tmp_path,expected=expected,config_path='config.json',correction=correction)
    assert result['passed'] and result['contract_count']==3 and result['historical_object_bindings']==6
    bad=copy.deepcopy(bundle);bad['contracts']['check']['proof']['objects'].pop(next(iter(proof['objects'])))
    with pytest.raises(ValueError):m.validate(bad,raw,tmp_path,expected=expected,config_path='config.json',correction=correction)
    with pytest.raises(ValueError):m.validate({'contracts':{'prep':bundle['contracts']['prep']}},raw,tmp_path,expected=expected,config_path='config.json',correction=correction)
    source.write_text('changed-source\n')
    with pytest.raises(ValueError):m.validate(bundle,raw,tmp_path,expected=expected,config_path='config.json',correction=correction)
    source.write_text('same-source\n');new['training']['lr']=.002;cfg.write_text(json.dumps(new));changed=copy.deepcopy(correction);changed['new_config_sha256']=hashlib.sha256(cfg.read_bytes()).hexdigest()
    with pytest.raises(ValueError):m.validate(bundle,raw,tmp_path,expected=expected,config_path='config.json',correction=changed)
