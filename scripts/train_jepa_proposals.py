#!/usr/bin/env python3
"""A60 actual cached-FIT preparation and fixed four CPU heads."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import sys,time,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from scripts.proposal_common import *
from scripts.cache_jepa_proposals import reader
from echo_routing.temporal.proposal_training import ProposalRegressor,optimizer,draw_plan
from echo_routing.temporal.proposal_loop import train_one
from echo_routing.temporal.proposal_features import fit_normalizer,fit_target_scales
from echo_routing.temporal.proposal_data import prepare_training
from echo_routing.temporal.proposal_runtime import infer
from echo_routing.temporal.anchor_exposure import summarize as summarize_exposure
from echo_routing.temporal.anchor_edits import majority
from echo_routing.temporal.dense_runtime import metrics
from echo_routing.temporal.dense_analysis import short_metrics
from echo_routing.temporal.baselines.registry import decode_with_prob


def role_values(role):
    qs=sequences(role);caches={s:reader(s,role,readonly=True) for s in ['a08','jepa']}
    values={s:[c.load(i) for i in range(len(qs))] for s,c in caches.items()}
    params=read(CFG)['source']['a08']['original_B3']
    base=[decode_with_prob('b3',p,{'decode':{'b3':params}})[0] for p in values['a08']]
    return qs,values['a08'],values['jepa'],base,{s:c.identity for s,c in caches.items()}


def prepare():
    freeze=verify_freeze('training_prepare_freeze.json')
    if not read(PUB/'source_cache_qualification.json')['passed'] or (PUB/'training_inputs.json').exists():raise ValueError('fresh preparation of qualified actual FIT cache required')
    start=time.perf_counter();qs,a,j,b,sources=role_values('fit');data=prepare_training(qs,a,j,b)
    if not len(data['targets']):
        write(PUB/'no_action_domain.json',{'genuine_no_action_domain':True,'formal_optimizer_updates':0});return
    normalizers={arm:fit_normalizer(data['features'][arm]) for arm in ['A','J']};scales=fit_target_scales(data['targets'])
    if any(not np.array_equal(normalizers['A'][k][:69],normalizers['J'][k][:69]) for k in ['mean','std']):raise ValueError('matched common69 normalizers differ')
    target=RUN/'training_dataset.private.npy';save_array(target,data);dataset=array_pin(target)
    norms={}
    for arm,norm in normalizers.items():
        target=RUN/f'normalizer_{arm}.private.json';write(target,{k:v.tolist() for k,v in norm.items()});norms[arm]=array_pin(target)
    target=RUN/'target_scales.private.json';write(target,{'scales':scales.tolist(),'centering':False,'floor':1.});scale_pin=array_pin(target)
    plans={}
    for seed in read(CFG)['training']['seeds']:
        target=RUN/f'draw_plan_seed{seed}.private.npy';save_array(target,draw_plan(*(len(data['kind_indices'][k]) for k in ['native','constructed']),seed));plans[str(seed)]=array_pin(target)
    exposure=summarize_exposure(qs,data['candidate_rows'])
    write(PUB/'training_inputs.json',{'passed':True,'dataset':dataset,'normalizers':norms,'target_scales':scale_pin,'target_scales_values':scales.tolist(),'plans':plans,'source_cache_identities':sources,'F_fit':data['F_fit'],'FIT_operating_counts':data['FIT_operating_counts'],'candidates':len(data['targets']),'all_candidates':data['all_candidates'],'G_rejected_candidates':data['G_rejected_candidates'],'kind_candidates':{k:len(v) for k,v in data['kind_indices'].items()},'features_each':109,'targets_each':3,'matched_common69_normalizer_equal':True,'candidate_eligibility':{k:v for k,v in exposure.items() if not k.startswith('actual_')},'targets_minmax':np.stack([data['targets'].min(0),data['targets'].max(0)]).tolist(),'features_nonconstant_columns':{arm:int(np.count_nonzero(data['features'][arm].std(0)>0)) for arm in ['A','J']},'full_FIT_prepare_wall_seconds':time.perf_counter()-start,'inner_or_outer_used_for_targets_normalizer_or_plan':False,'formal_optimizer_updates':0,'freeze_git':freeze['git_commit']})
    print({'actualFIT_candidates':len(data['targets']),'kind_candidates':{k:len(v) for k,v in data['kind_indices'].items()},'F_fit':data['F_fit']},flush=True)


def identity_for(arm,seed,inputs,freeze):
    return {'code':digest(freeze['objects']),'config':sha(CFG),'sources':digest(inputs['source_cache_identities']),'input':inputs['dataset']['sha256'],'draw_plan':inputs['plans'][str(seed)]['sha256'],'normalizer':inputs['normalizers'][arm]['sha256'],'target_scales':inputs['target_scales']['sha256']}


def formal():
    freeze=verify_freeze('formal_freeze.json');cfg=read(CFG);inputs=read(PUB/'training_inputs.json')
    if not read(PUB/'formal_admission.json')['all_caps_fit'] or not inputs['passed']:raise ValueError('whole actual graph/allmatrix admission required')
    data=np.load(verify_pin(inputs['dataset']),allow_pickle=True).item();scales=np.asarray(read(verify_pin(inputs['target_scales']))['scales'])
    iq,ia,ij,ib,isources=role_values('inner_cal');records=[];progress=[]
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    for seed in cfg['training']['seeds']:
        plan=np.load(verify_pin(inputs['plans'][str(seed)]),allow_pickle=True).item()
        for arm in ['A','J']:
            norm=read(verify_pin(inputs['normalizers'][arm]));identity=identity_for(arm,seed,inputs,freeze)
            model=ProposalRegressor(arm,seed);opt=optimizer(model);folder=RUN/'models'/f'seed{seed}'/arm
            def inner(current,cursor,path):
                target=PUB/f'inner_seed{seed}_{arm}_step{cursor}.json'
                if target.exists():raise ValueError('callback record unexpectedly exists without durable marker')
                predictions={p:{} for p in ['O','E','R']};accepted={p:0 for p in predictions};candidates=0;veto=0
                for q,a,j,b in zip(iq,ia,ij,ib):
                    result=infer(current,norm,scales,a,j,b,arm);candidates+=len(result['actions']);veto+=len(result['R']['risk_vetoed_indices'])
                    for policy in predictions:
                        labels=result[policy]['prediction'];predictions[policy][q['id']]=labels
                        accepted[policy]+=result[policy]['accepted'] is not None
                        if majority(labels)!=majority(b):raise ValueError('all-sequence G failed')
                target_private=folder/f'inner_step{cursor}_labels.private.npy';save_array(target_private,predictions)
                write(target,{'seed':seed,'arm':arm,'step':cursor,'weight_sha256':sha(path),'identity':identity,'source_cache_identities':isources,'private_predictions':array_pin(target_private),'policies':{p:{**metrics(iq,pred)[0],**short_metrics(iq,pred,16)} for p,pred in predictions.items()},'candidate_count':candidates,'accepted_actions':accepted,'risk_vetoes':veto,'all_sequence_guard_majorities_equal_Base':True,'diagnostic_only_no_checkpoint_or_threshold_selection':True})
                return array_pin(target)
            def update(cursor,row):
                write(RUN/'runtimecoordination.json',{'phase':'formal_three_output_heads','seed':seed,'arm':arm,'committed_updates':cursor,'actual_Adam_updates':cursor,'prior_completed_model_updates':sum(r['committed_physical_updates'] for r in progress),'model_pid':os.getpid(),'epoch':time.time(),'next_action':'finish four6000/12inner/36policy sets then actualGitfreeze before outer'})
            result=train_one(model,opt,data,plan,norm,scales,identity,folder,on_inner=inner,on_progress=update)
            rows=[json.loads(line) for line in (folder/'physical_updates.jsonl').read_text().splitlines()]
            result['actual_candidate_and_context_exposure']=summarize_exposure(sequences('fit'),data['candidate_rows'],[i for r in rows for i in r['candidate_indices']]);progress.append(result)
            records.append({'seed':seed,'arm':arm,'result':result,'final_path':str((folder/'step6000.pt').relative_to(ROOT)),'final_sha256':sha(folder/'step6000.pt'),'identity':identity})
    write(PUB/'formal_completion.json',{'passed':True,'models':records,'physical_models':4,'formal_optimizer_updates':sum(r['result']['committed_physical_updates'] for r in records),'CUDA_initialized':torch.cuda.is_initialized(),'inner_used_for_model_selection':False})

if __name__=='__main__':
    if sys.argv[1:]==['prepare']:prepare()
    elif sys.argv[1:]==['formal']:formal()
    else:raise ValueError('explicit prepare/formal required')
