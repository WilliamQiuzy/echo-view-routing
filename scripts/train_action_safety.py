#!/usr/bin/env python3
"""A61 admitted two local-risk then eight matched complete-query utility models."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import sys,time,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from scripts.safety_common import *
from echo_routing.temporal.safety_training import Utility,Risk,optimizer,ARMS
from echo_routing.temporal.safety_loop import train_one
from echo_routing.temporal.safety_runtime import infer,policies
from echo_routing.temporal.safety_policy import calibrate
from echo_routing.temporal.safety_targets import effects
from echo_routing.temporal.proposal_actions import propose,partition
from echo_routing.temporal.anchor_edits import majority
from echo_routing.temporal.dense_runtime import metrics
from echo_routing.temporal.dense_analysis import short_metrics


def identity_for(role,seed,inputs,freeze):
    return {'code':digest(freeze['objects']),'config':sha(CFG),'sources':digest(inputs['source_cache_identities']),'input':inputs['dataset']['sha256'],'draw_plan':inputs['plans'][str(seed)]['sha256'],'normalizer':inputs['normalizers']['sha256'],'target_scales':digest({'scale':inputs['utility_scale']})}


def state_digest(model):
    h=hashlib.sha256()
    for name,t in model.state_dict().items():h.update(name.encode());h.update(t.detach().numpy().tobytes())
    return h.hexdigest()


def main():
    freeze=verify_freeze('formal_freeze.json');inputs=read(PUB/'training_inputs.json');cfg=read(CFG)
    if not read(PUB/'formal_admission.json')['all_caps_fit'] or not read(PUB/'FIT_checks.json')['passed']:raise ValueError('actual graph and fullmatrix admission required')
    if (PUB/'formal_completion.json').exists():raise ValueError('formal matrix already complete;never replay')
    data=np.load(verify_pin(inputs['dataset']),allow_pickle=True).item();all_plans={seed:np.load(verify_pin(pin),allow_pickle=True).item() for seed,pin in inputs['plans'].items()}
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    records=[];inner_records=[];initial_pins={};risks={};done=0
    def run(model,on_inner=None):
        nonlocal done
        role,seed,arm=model.role,model.seed,model.arm;plan=all_plans[str(seed)][role];folder=RUN/'models'/f'seed{seed}'/arm;identity=identity_for(role,seed,inputs,freeze);initial=state_digest(model)
        def progress(cursor,row):
            write(RUN/'runtimecoordination.json',{'phase':'formal_risk_then_utility6000','role':role,'seed':seed,'arm':arm,'committed_updates':cursor,'actual_Adam_updates':cursor,'prior_completed_model_updates':done,'model_pid':os.getpid(),'epoch':time.time(),'next_action':'complete2risk+8utility/24inner then8fixedcalibrations and allmodels actualGit seal beforeouter'})
        report=train_one(model,optimizer(model),data,plan,identity,folder,on_inner=on_inner,on_progress=progress)
        done+=report['committed_physical_updates'];record={'seed':seed,'arm':arm,'role':role,'initial_state_sha256':initial,'result':report,'identity':identity,'path':str((folder/'step6000.pt').relative_to(ROOT)),'sha256':sha(folder/'step6000.pt')};records.append(record)
        return record
    for seed in cfg['training']['seeds']:
        risk=Risk(seed);run(risk);risks[seed]=risk.eval()
    iq,ia,ij,ib,isources=role_values('inner_cal')
    if len(iq)!=928:raise ValueError('complete928 inner calibration/diagnostic axes required')
    for seed in cfg['training']['seeds']:
        for arm in ARMS:
            model=Utility(arm,seed);initial_pins.setdefault(str(seed),state_digest(model))
            if initial_pins[str(seed)]!=state_digest(model):raise ValueError('same-seed utility initial tensors differ')
            def inner(current,cursor,path):
                target=PUB/f'inner_seed{seed}_{arm}_step{cursor}.json'
                if target.exists():raise ValueError('unclaimed callback record exists')
                labels={p:{} for p in ['E','N']};scores=[];accepted={p:0 for p in labels}
                for q,a,j,b in zip(iq,ia,ij,ib):
                    result=infer(current,risks[seed],data['normalizers'],data['scale'],a,j,b);scores.append(result['score_matrix'])
                    for p in labels:
                        prediction=result[p]['prediction'];labels[p][q['id']]=prediction;accepted[p]+=result[p]['accepted'] is not None
                        if majority(prediction)!=majority(b):raise ValueError('complete-axis majority guard failed')
                private=RUN/'models'/f'seed{seed}'/arm/f'inner_step{cursor}.private.npy';save_array(private,{'labels':labels,'scores':scores})
                record={'seed':seed,'arm':arm,'step':cursor,'utility_weight_sha256':sha(path),'risk_weight_sha256':sha(RUN/'models'/f'seed{seed}'/'risk/step6000.pt'),'private_outputs':array_pin(private),'source_cache_identities':isources,'sequences':len(iq),'policies':{p:{**metrics(iq,pred)[0],**short_metrics(iq,pred,16)} for p,pred in labels.items()},'accepted_actions':accepted,'inner_role':'diagnostic/calibration;no checkpoint or hyperparameter selection','new_source_forwards':0}
                write(target,record);inner_records.append(array_pin(target));return array_pin(target)
            run(model,inner)
    if done!=60000 or len(records)!=10:raise ValueError('complete fixed matrix required before calibration')
    inner_records=[array_pin(PUB/f'inner_seed{seed}_{arm}_step{cursor}.json') for seed in cfg['training']['seeds'] for arm in ARMS for cursor in [2000,4000,6000]]
    thresholds=[]
    for seed in cfg['training']['seeds']:
        for arm in ARMS:
            record=read(PUB/f'inner_seed{seed}_{arm}_step6000.json');cached=np.load(verify_pin(record['private_outputs']),allow_pickle=True).item();rows=[];actions=[]
            if len(cached['scores'])!=928:raise ValueError('all928 cached top scores required')
            for a,j,b,q,score in zip(ia,ij,ib,iq,cached['scores']):
                domain,_=partition(b,propose(b,np.asarray(a).argmax(1),np.asarray(j).argmax(1)));actions.append(domain)
                result=policies(b,domain,score)['N'];top=result['top_index'];cost={} if top is None else effects(b,domain[top],q['labels'],data['F_fit'])
                rows.append({'N_admitted':result['N_admitted'],'r':result['r'],**cost})
            cal=calibrate(rows);pred={};accepted=0
            for q,b,domain,score in zip(iq,ib,actions,cached['scores']):
                result=policies(b,domain,score,cal['tau'])['C'];pred[q['id']]=result['prediction'];accepted+=result['accepted'] is not None
                if result['accepted'] is not None:
                    cost=effects(b,result['accepted'],q['labels'],data['F_fit'])
                    if cost['H']>0 or cost['dFP']>0:raise ValueError('C calibration empiricalunsafe accept')
            if accepted!=cal['C_accepted']:raise ValueError('calibration accept recount differs')
            private=RUN/'models'/f'seed{seed}'/arm/'inner_C_calibration.private.npy';save_array(private,{'predictions':pred,'calibration_rows':rows})
            target=PUB/f'calibration_seed{seed}_{arm}.json';value={'seed':seed,'arm':arm,**cal,'utility_weight_sha256':record['utility_weight_sha256'],'risk_weight_sha256':record['risk_weight_sha256'],'source_inner_record_sha256':sha(PUB/f'inner_seed{seed}_{arm}_step6000.json'),'source_cache_identities':isources,'private_outputs':array_pin(private),'C_metrics':{**metrics(iq,pred)[0],**short_metrics(iq,pred,16)}};write(target,value);thresholds.append({**array_pin(target),'seed':seed,'arm':arm,'tau':cal['tau']})
    if len(inner_records)!=24 or len(thresholds)!=8:raise ValueError('all24inner+8thresholds required')
    write(PUB/'calibration.json',{'passed':True,'thresholds':thresholds,'inner_records':inner_records,'inner_sequences':928,'inner_policy_sets':56,'zero_observed_unsafe_C_accepts':True,'independent_validation':False,'no_checkpoint_selection':True})
    write(PUB/'formal_completion.json',{'passed':True,'models':records,'physical_models':10,'formal_optimizer_updates':done,'same_seed_utility_initial_state_sha256':initial_pins,'risk_models_trained_before_all_utility_inner':True,'calibration_sha256':sha(PUB/'calibration.json'),'CUDA_initialized':torch.cuda.is_initialized(),'inner_used_for_gradients_or_checkpoint_selection':False})
    write(RUN/'runtimecoordination.json',{'phase':'all60000_and24inner8taus_complete_pending_audit_and_actualGit_outerseal','formal_optimizer_updates':done,'physical_models':10,'model_pid':os.getpid(),'epoch':time.time(),'next_action':'audit10finals/24inner/8taus/56sets then actualGit preouterfreeze;no outer yet'})
    print({'formal_updates':done,'models':len(records),'inner_records':len(inner_records),'taus':len(thresholds)},flush=True)
if __name__=='__main__':main()
