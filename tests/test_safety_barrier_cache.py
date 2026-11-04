import importlib
import numpy as np
import pytest


def test_complete_models_taus_and_actual_seal_are_required_before_outer():
    m=importlib.import_module('echo_routing.temporal.safety_barrier');from echo_routing.temporal.safety_training import SEEDS,ARMS
    rows=[{'seed':s,'arm':a,'sha256':'a'*64,'path':f'seed{s}/{a}.pt'} for s in SEEDS for a in ['risk',*ARMS]]
    inner=[{'seed':s,'arm':a,'step':k} for s in SEEDS for a in ARMS for k in [2000,4000,6000]]
    taus=[{'seed':s,'arm':a,'tau':.2} for s in SEEDS for a in ARMS]
    models={'passed':True,'formal_optimizer_updates':60000,'physical_models':10,'all10finals24inner8taus56sets_frozen':True,'inner_policy_sets':56,'models':rows,'inner_records':inner,'thresholds':taus,'calibration_sha256':'c'*64}
    seal={'models_manifest_sha256':'m'*64,'calibration_sha256':'c'*64,'readout_plan_sha256':'p'*64,'all_final_checkpoints':[{'path':r['path'],'sha256':r['sha256']} for r in rows]}
    assert m.validate(models,seal,'m'*64,'c'*64,'p'*64)
    with pytest.raises(ValueError):m.validate({**models,'models':rows[:-1]},seal,'m'*64,'c'*64,'p'*64)
    with pytest.raises(ValueError):m.validate(models,{**seal,'models_manifest_sha256':'x'*64},'m'*64,'c'*64,'p'*64)


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
