import importlib
import numpy as np
import pytest


def test_complete_models_taus_and_actual_seal_are_required_before_outer(tmp_path):
    import json,hashlib,subprocess,copy
    m=importlib.import_module('echo_routing.temporal.safety_barrier');g=importlib.import_module('echo_routing.temporal.safety_git')
    models,seal=barrier_fixture();prefix='docs/results/2026-11-04_action_safety_ranking/';folder=tmp_path/prefix;folder.mkdir(parents=True)
    payload={'calibration.json':{'thresholds':models['thresholds']},'readout_plan.json':{'count':314}}
    h=lambda b:hashlib.sha256(b).hexdigest()
    for name,value in payload.items():(folder/name).write_text(json.dumps(value,sort_keys=True))
    cal=h((folder/'calibration.json').read_bytes());plan=h((folder/'readout_plan.json').read_bytes());models.update(calibration_sha256=cal,readout_plan_sha256=plan)
    (folder/'models_manifest.json').write_text(json.dumps(models,sort_keys=True));model=h((folder/'models_manifest.json').read_bytes())
    subprocess.run(['git','init','-q',str(tmp_path)],check=True)
    subprocess.run(['git','-C',str(tmp_path),'add','docs'],check=True)
    subprocess.run(['git','-C',str(tmp_path),'-c','user.name=A61Fixture','-c','user.email=fixture@example.invalid','-c','commit.gpgsign=false','-c','core.hooksPath=/dev/null','commit','-qm','public fixture'],check=True)
    commit=subprocess.check_output(['git','-C',str(tmp_path),'rev-parse','HEAD'],text=True).strip();proof=g.build(tmp_path,commit,[prefix+n for n in ['models_manifest.json','calibration.json','readout_plan.json']])
    seal.update(git_commit=commit,models_manifest_sha256=model,calibration_sha256=cal,readout_plan_sha256=plan)
    assert m.validate(models,seal,model,cal,plan,proof)
    with pytest.raises(ValueError):m.validate({**models,'models':models['models'][:-1]},seal,model,cal,plan,proof)
    damaged=copy.deepcopy(proof);key=next(iter(damaged['objects']));damaged['objects'][key]='AAAA'
    with pytest.raises(ValueError):m.validate(models,seal,model,cal,plan,damaged)


def test_joint_cache_prefix_resume_empty_and_ambiguous_callback_no_extra_forward(tmp_path):
    m=importlib.import_module('echo_routing.temporal.safety_policy_cache');from echo_routing.temporal.safety_runtime import policies
    from echo_routing.temporal.proposal_actions import propose,partition
    b=[np.zeros(20,int) for _ in range(3)];a=[x.copy() for x in b];j=[x.copy() for x in b];j[1][3]=1
    c=m.SafetyCache(tmp_path,'a'*64,b,a,j,.4);calls=[]
    def produce(i):
        calls.append(i);allactions=propose(b[i],a[i],j[i]);actions,rejected=partition(b[i],allactions);score=np.tile([1.,.1,.8,.1,.1,.1],(len(actions),1))
        return {**policies(b[i],actions,score,.4),'score_matrix':score,'actions':actions,'cached_feature_heads_policy_seconds':.001}
    c.run_one(0,lambda:produce(0));d=m.SafetyCache(tmp_path,'a'*64,b,a,j,.4)
    assert d.cursor==1 and d.load(0)['score_matrix'].shape==(0,6)
    for i in range(d.cursor,3):d.run_one(i,lambda i=i:produce(i))
    assert calls==[0,1,2]
    with pytest.raises(ValueError):m.SafetyCache(tmp_path,'a'*64,b,a,j,.5)
    other=tmp_path/'failure';e=m.SafetyCache(other,'b'*64,b,a,j,.4)
    def failure():raise RuntimeError('interrupted after durable claim')
    with pytest.raises(RuntimeError):e.run_one(0,failure)
    with pytest.raises(ValueError):m.SafetyCache(other,'b'*64,b,a,j,.4)


def barrier_fixture():
    from echo_routing.temporal.safety_training import SEEDS,ARMS
    rows=[{'seed':s,'arm':a,'sha256':'a'*64,'path':f'seed{s}/{a}.pt'} for s in SEEDS for a in ['risk',*ARMS]]
    models={'passed':True,'formal_optimizer_updates':60000,'physical_models':10,'all10finals24inner8taus56sets_frozen':True,'inner_policy_sets':56,'models':rows,'inner_records':[{'seed':s,'arm':a,'step':k} for s in SEEDS for a in ARMS for k in [2000,4000,6000]],'thresholds':[{'seed':s,'arm':a,'tau':.2} for s in SEEDS for a in ARMS],'calibration_sha256':'c'*64,'readout_plan_sha256':'d'*64}
    seal={'models_manifest_sha256':'b'*64,'calibration_sha256':'c'*64,'readout_plan_sha256':'d'*64,'all_final_checkpoints':[{'path':r['path'],'sha256':r['sha256']} for r in rows]}
    return models,seal


def test_changed_plan_after_audit_cannot_be_resealed():
    m=importlib.import_module('echo_routing.temporal.safety_barrier');models,seal=barrier_fixture()
    with pytest.raises(ValueError):m.validate({**models,'readout_plan_sha256':'e'*64},seal,'b'*64,'c'*64,'d'*64)


def test_loose_seal_without_actual_git_commit_objects_is_rejected():
    m=importlib.import_module('echo_routing.temporal.safety_barrier');models,seal=barrier_fixture()
    with pytest.raises(ValueError):m.validate(models,seal,'b'*64,'c'*64,'d'*64)
