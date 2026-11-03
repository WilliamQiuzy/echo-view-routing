#!/usr/bin/env python3
"""Independent model/exposure/inner-choice audit before any fresh outer result."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import sys,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from scripts.structured_common import ROOT,RUN,PUB,CFG,read,write,sha,verify_freeze,sequences
from scripts.structured_head_runtime import load_bank,head_identity,evaluate_inner
from echo_routing.temporal.structured_evidence import StructuredEvidence
from echo_routing.temporal.structured_lifecycle import audit_payload,state_digest,choose_inner
from echo_routing.temporal.structured_trace import audit_trace
from echo_routing.temporal.structured_predictions import LabelReadout
from echo_routing.temporal.structured_loop import _equal

def audit():
    freeze=verify_freeze('formal_freeze.json');verify_freeze('producer_freeze.json');cfg=read(CFG);admission=read(PUB/'admission.json');assert admission['all_caps_fit'] and admission['training_backend']=='cpu';torch.set_num_threads(1);manifest=read(PUB/'input_manifest.json');progress=read(PUB/'formal_progress.json');assert progress['completed_physical_optimizer_updates']==48000 and len(progress['completed_models'])==8;inner=load_bank('inner_cal');fit=sequences('fit');models=[];outer=[];choices=[];updates=positions=0
    for seed in cfg['training']['seeds']:
        pin=manifest['plans'][str(seed)];assert sha(ROOT/pin['path'])==pin['sha256'];plan=np.load(ROOT/pin['path'])
        for arm in cfg['training']['arms']:
            model=StructuredEvidence(arm,seed);initial={k:v.clone() for k,v in model.state_dict().items()};identity=head_identity(freeze,seed,arm,'cpu');folder=RUN/f'seed{seed}'/arm;assert not (folder/'pending.json').exists();trace=[json.loads(s) for s in (folder/'physical_updates.jsonl').read_text().splitlines()];names=[n for n,p in model.named_parameters() if p.requires_grad];physical=audit_trace(trace,plan,identity,[len(q['labels']) for q in fit],names);updates+=physical['physical_optimizer_updates'];positions+=physical['valid_original_positions'];records=[];checkpoints=[]
            for step in [2000,4000,6000]:
                path=folder/f'step{step}.pt';value=torch.load(path,map_location='cpu',weights_only=True);checked=audit_payload(model,value,identity);assert value['update']==step and checked['model_digest']==trace[step-1]['model_digest_after_update'];assert all(checked['nonzero_second_moment_tensors'].values());model.load_state_dict(value['model'],strict=True)
                row=read(PUB/'inner'/f'seed{seed}_{arm}_step{step}.json');cache=LabelReadout(folder/f'inner_step{step}',row['cache']['identity'],[len(q['labels']) for q in inner.sequences]);assert cache.cursor==len(inner.sequences);recount=evaluate_inner(model,inner,path,step,identity);assert recount==row;records.append(row);checkpoints.append({'step':step,'path':str(path.relative_to(ROOT)),'sha256':sha(path),'audit':checked})
            latest=torch.load(folder/'latest.pt',map_location='cpu',weights_only=True);final=torch.load(folder/'step6000.pt',map_location='cpu',weights_only=True);assert _equal(latest,final);assert state_digest(final['model'])==physical['final_model_digest'];drift={name:bool((final['model'][name]!=initial[name]).any()) for name in names};assert all(drift.values());written=next(r for r in progress['completed_models'] if r['seed']==seed and r['arm']==arm);assert written['initial_model_digest']==state_digest(initial) and written['final_model_digest']==physical['final_model_digest'];selected=choose_inner(records);choice={'seed':seed,'arm':arm,'step':selected['step'],'checkpoint_sha256':selected['checkpoint_sha256'],'selection':'innerSegmentF1_then_nativeMacroF1_then_fewer_short16FP_then_earlierStep_at1e-12','inner_only':True,'A08_historical_inner_exposure_not_OOF':True};choices.append(choice);modelrow={'seed':seed,'arm':arm,'head_identity':identity,'initial_model_digest':state_digest(initial),'physical':physical,'all_active_tensors_drifted':drift,'checkpoints':checkpoints,'selected_step':selected['step'],'backend':'cpu'};models.append(modelrow)
            for kind,step in [('fixed6000',6000),('inner_selected',selected['step'])]:
                checkpoint=next(r for r in checkpoints if r['step']==step);outer.append({'row_id':f'seed{seed}_{arm}_{kind}','seed':seed,'arm':arm,'kind':kind,'step':step,'path':checkpoint['path'],'sha256':checkpoint['sha256'],'head_identity':identity})
    assert updates==48000 and positions==sum(read(PUB/'admission.json')['exact_plan_position_totals'])*4;write(PUB/'inner_choices.json',{'rows':choices,'count':len(choices),'all_frozen_before_fresh_outer':True,'outer_or_stress_used_for_choice':False});result={'passed':True,'models':models,'models_count':len(models),'formal_optimizer_updates':updates,'formal_valid_original_positions':positions,'outer_rows':outer,'logical_outer_rows':len(outer),'unique_checkpoint_SHA_count':len({r['sha256'] for r in outer}),'inner_choices_sha256':sha(PUB/'inner_choices.json'),'formal_freeze_git':freeze['git_commit'],'producer_freeze_git':read(RUN/'producer_freeze.json')['git_commit'],'all_head_weights_and_inner_choices_frozen':True,'model_parameters_selected_by_outer':False,'all24inner_checkpoint_metrics_recounted_from_committed_labels':True,'full_active_Adam_moment_recipe_names_shapes_finiteness_and_actual6000_counters_passed':True,'no_silent_formal_optimizer_replay':True,'source_bodies_frozen_and_not_retrained':True,'new_protected_media_forwards':0,'epoch':time.time()};write(PUB/'models_manifest.json',result);write(RUN/'runtimecoordination.json',{'phase':'all8heads_and_inner_choices_audited_pending_git_freeze','formal_optimizer_updates':updates,'model_pid':None,'epoch':time.time(),'next_action':'commit all weights/choice hashes; freeze outer identity then fresh outer/stress source caches'});print(json.dumps({'passed':True,'formal_optimizer_updates':updates,'models':len(models),'logical_outer_rows':len(outer),'unique_checkpoints':result['unique_checkpoint_SHA_count']}),flush=True)

if __name__=='__main__':
    if sys.argv[1]=='audit':audit()
    else:raise ValueError('aggregate is a separately frozen cached-reader entrypoint')
