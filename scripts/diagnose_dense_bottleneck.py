#!/usr/bin/env python3
"""Bounded post-outer aggregate/cache-only diagnosis; no image reader/model build."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import json,resource,sys,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch,yaml
from scripts import train_dense_sequence as base
from scripts.aggregate_dense_sequence import evaluate
from echo_routing.temporal.dense_bottleneck import class_stats,position_diagnostics,correctness_cells
from echo_routing.temporal.dense_readout import SequenceReadout
from echo_routing.temporal.baselines.registry import decode_with_prob
from echo_routing.evaluate.window_target import nearest_centre
from echo_routing.evaluate.failure_units import session_of,match_segments
from echo_routing.temporal.segments import labels_to_segments
RUN=base.RUN;PUB=base.PUB/'diagnosis';read,write,sha=base.read,base.write,base.sha

def add(a,b):
    for k,v in b.items():
        if isinstance(v,dict):add(a.setdefault(k,{}),v)
        elif isinstance(v,list):a[k]=(np.asarray(a.get(k,np.zeros_like(v)))+np.asarray(v)).tolist()
        else:a[k]=a.get(k,0)+v

def summarize(seqs,pred,prob):
    out={};cm=np.zeros((5,5),np.int64);segments={str(k):{'true':0,'predicted':0,'true_unmatched':0,'predicted_unmatched':0,'true_short16':0,'FN_short16':0,'pred_short16':0,'FP_short16':0,'true_short10':0,'FN_short10':0,'pred_short10':0,'FP_short10':0} for k in range(5)}
    for q in seqs:
        if q['kind'] not in ['native','multi']:continue
        gt=np.asarray(q['labels']);p=pred[q['id']];add(out.setdefault(q['kind'],{}),position_diagnostics(gt,p,prob[q['id']]))
        if q['kind']=='native':cm[int(gt[0]),int(np.bincount(p,minlength=5).argmax())]+=1
        else:
            truth=labels_to_segments(gt);guess=labels_to_segments(p);matches=match_segments(guess,truth);used=set(matches.values())
            for i,s in enumerate(truth):
                d=segments[str(s.label)];d['true']+=1;d['true_unmatched']+=i not in matches
                for n in [16,10]:d[f'true_short{n}']+=s.length<=n;d[f'FN_short{n}']+=s.length<=n and i not in matches
            for i,s in enumerate(guess):
                d=segments[str(s.label)];d['predicted']+=1;d['predicted_unmatched']+=i not in used
                for n in [16,10]:d[f'pred_short{n}']+=s.length<=n;d[f'FP_short{n}']+=s.length<=n and i not in used
    for kind in out:
        for name in ['stable_interior','near_true_change_5samples']:
            d=out[kind][name];n=d['positions'];d['accuracy']=1-d['errors']/n if n else None
            for x in ['raw_true_class_probability','raw_true_minus_other_margin','raw_top2_confidence']:d[x+'_mean']=d[x+'_sum']/n if n else None
            d['per_class']=class_stats(d['confusion'])
    return {'native_majority_confusion':cm.tolist(),'native_majority_per_class':class_stats(cm),'position_bins':out,'multi_segments_by_class':segments}

def population(seqs):
    ns=[q for q in seqs if q['kind']=='native'];classes=[int(q['labels'][0]) for q in ns];sessions=[session_of(q['id']) for q in ns];lengths=[len(q['labels']) for q in ns]
    return {'native_clips':len(ns),'sessions':len(set(sessions)),'native_class_counts':np.bincount(classes,minlength=5).tolist(),'positions_per_class':[sum(len(q['labels']) for q in ns if q['labels'][0]==k) for k in range(5)],'sessions_with_each_class':[len({session_of(q['id']) for q in ns if q['labels'][0]==k}) for k in range(5)],'native_length_samples_quantiles':np.quantile(lengths,[0,.25,.5,.75,.95,1]).tolist()}

def exposure(fit,cal):
    f={q['id'] for q in fit if q['kind']=='native'};c={q['id'] for q in cal if q['kind']=='native'};assert len(f)==2736 and len(c)==628 and not f&c;raw=yaml.safe_load((ROOT/'configs/experiments/raw_video_context_v1.yaml').read_text())['raw_video_context'];out={}
    for seed in [0,1,2]:
        p=ROOT/f'runs/video_sequence_objective_v1/manifests/VSO_s{seed}.npz';z=np.load(p);v=set(z['videos'].tolist());run=read(ROOT/f'runs/video_within_view_v1/train/all_s{seed}/run_manifest.json');assert sha(p)==run['manifest_sha256'] and v<=f|c;pin=raw['source']['sc_manifests']['files'][seed];p0=ROOT/pin['path'];assert sha(p0)==pin['sha256'];v0=set(np.load(p0)['videos'].tolist());assert v0<=f|c;out[str(seed)]={'continuation_actual_unique_sources':len(v),'continuation_A57_inner_sources':len(v&c),'V01_initialization_actual_unique_sources':len(v0),'V01_A57_inner_sources':len(v0&c),'A08_run_manifest_sha256':sha(ROOT/f'runs/video_within_view_v1/train/all_s{seed}/run_manifest.json'),'continuation_manifest_sha256':sha(p),'V01_SC_manifest_sha256':sha(p0)}
    return {'protocol_train_allowlist':3364,'A57_actual_FIT_sources':2736,'A57_inner_withheld_from_A57_fit':628,'A08_actual_sources_by_seed':out,'A08_inner_comparison_valid':False,'note':'A08 inherited broader protocoltrain exposure; do not equate allowlist with everyrun exposing all3364'}

def main():
    resource.setrlimit(resource.RLIMIT_CPU,(1200,1205));freeze=read(RUN/'diagnosis_freeze.json')
    for p,h in freeze['objects'].items():assert sha(ROOT/p)==h,p
    models=read(base.PUB/'models_manifest.json');choices=read(base.PUB/'decoder_choices.json');pins=read(base.PUB/'outer_inputs.json');primary=read(base.PUB/'primary_outer_rows.json');assert len(primary['rows'])==8 and primary['model_manifest_sha256']==sha(base.PUB/'models_manifest.json');assert sha(RUN/'outer_metadata.private.npy')==pins['outer_metadata_sha256'];metadata=np.load(RUN/'outer_metadata.private.npy',allow_pickle=True).item();fit,cal=base.inputs('fit'),base.inputs('inner_cal');systems={};paired={};allcurves=[]
    for half in ['selection','calibration']:
        seqs=metadata[half]['sequences'];probsystems={};predsystems={};systems[half]={};paired[half]={}
        for seed in [0,1,2]:
            pin=pins['A08_cache_pins'][half][str(seed)];assert sha(ROOT/pin['path'])==pin['sha256'];probs={}
            with np.load(ROOT/pin['path']) as z:
                assert z['sequence_ids'].tolist()==[q['id'] for q in seqs];lp=torch.softmax(torch.from_numpy(z['logits']),-1).numpy()
                for i,q in enumerate(seqs):
                    ix=np.flatnonzero(z['seq']==i);p=lp[ix][nearest_centre(len(q['labels']),z['centre'][ix])];assert np.max(np.abs(p-z['sample_probs'][z['sample_offsets'][i]:z['sample_offsets'][i+1]]))<=1e-7;probs[q['id']]=p
            key=f'A08_s{seed}';probsystems[key]=probs;predsystems[key]={k:decode_with_prob('b3',v,{'decode':{'b3':pin['original_params']}})[0] for k,v in probs.items()}
        for seed in [2026110257,2026110258]:
            for arm in ['F','J-high']:
                row=next(q for q in models['outer_rows'] if q['seed']==seed and q['arm']==arm and q['kind']=='fixed_step' and q['step']==6000);path=RUN/'outer_readout'/row['sha256']/half;assert (path/'committed.jsonl').exists();cache=SequenceReadout(path,row['sha256']+'|'+pins['outer_metadata_sha256']+'|'+half,len(seqs),lengths=[len(q['labels']) for q in seqs]);assert cache.cursor==len(seqs);probs={q['id']:cache.load(i) for i,q in enumerate(seqs)}
                for mode in ['raw','inner_B3']:
                    key=f'{seed}_{arm}_{mode}';probsystems[key]=probs;params=choices['choices_by_checkpoint_sha256'][row['sha256']]['params'];predsystems[key]={k:(v.argmax(1) if mode=='raw' else decode_with_prob('b3',v,{'decode':{'b3':params}})[0]) for k,v in probs.items()}
        for key,pred in predsystems.items():
            m,_=evaluate(pred,half,metadata)
            if key.startswith('A08_'):assert m==read(base.PUB/'A08_reference_metrics.json')['systems'][half]['A08_original_B3_'+key[4:]]
            else:
                seed,arm,mode=key.split('_',2);row=f'seed{seed}_{arm}_fixed6000';old=next(x for x in primary['rows'] if x['row']==row and x['half']==half);assert m==old[mode]
            systems[half][key]={'canonical_metrics':m,**summarize(seqs,pred,probsystems[key])}
            if not key.startswith('A08_'):
                ref=predsystems['A08_s0'];nat=[q for q in seqs if q['kind']=='native'];cc=np.array([int(np.bincount(pred[q['id']],minlength=5).argmax())==q['labels'][0] for q in nat]);rr=np.array([int(np.bincount(ref[q['id']],minlength=5).argmax())==q['labels'][0] for q in nat]);paired[half][key]={'native_clip_majority':correctness_cells(cc,rr)}
                for kind in ['native','multi']:
                    subset=[q for q in seqs if q['kind']==kind];a=np.concatenate([pred[q['id']]==q['labels'] for q in subset]);b=np.concatenate([ref[q['id']]==q['labels'] for q in subset]);paired[half][key][kind+'_original_positions']=correctness_cells(a,b)
    for model in models['models']:
        folder=RUN/f"seed{model['seed']}"/model['arm'];trace=[json.loads(x) for x in (folder/'physical_updates.jsonl').read_text().splitlines()];curves=[read(folder/f'inner_step{step}/metrics.json') for step in range(500,model['updates']+1,500)];keys=['native_macro_f1','segment_f1_50_multi','boundary_precision_multi','boundary_recall_multi','native_internal_cuts_per_clip','short_true_unmatched_10samples'];allcurves.append({'seed':model['seed'],'arm':model['arm'],'mean_first100_trainCE':float(np.mean([x['loss'] for x in trace[:100]])),'mean_last100_trainCE':float(np.mean([x['loss'] for x in trace[-100:]])),'inner_selected_step':model['inner_chosen_step'],'inner_curves':[{'step':q['step'],**{k:q[k] for k in keys}} for q in curves]})
    result={'status':'post-outer exploratory cache diagnosis, no new model choice','systems':systems,'A08_s0_paired_correctness':paired,'native_populations':{'FIT':population(fit),'inner':population(cal),'protocol_train':population([q for q in fit+cal if q['kind']=='native']),**{h:population(metadata[h]['sequences']) for h in ['selection','calibration']}},'verified_training_exposure':exposure(fit,cal),'training_and_inner_curves':allcurves,'head_stage1_vs_stage2':'unavailable: saved caches contain laststage probabilities only; no new forwards authorized','bins':'distance<=5 original10Hz samples fromGTlabelchange inclusive;native positions allstable; artificial construction changes notclinical visibility annotations','source_freeze_git':freeze['git_commit'],'new_model_forwards':0,'new_media_reads':0,'new_training_updates':0,'protected_labels_used_for_training_or_selection':False,'epoch':time.time()};write(PUB/'summary.json',result);print(json.dumps({'cached_diagnosis_complete':True,'systems_perhalf':len(systems['selection']),'new_model_forwards':0}),flush=True)

if __name__=='__main__':
    start=time.time();ok=False;PUB.mkdir(parents=True,exist_ok=True)
    try:main();ok=True
    finally:
        r=resource.getrusage(resource.RUSAGE_SELF);receipt={'stage':'cached_bottleneck_diagnosis','model_pid':os.getpid(),'ok':ok,'start_epoch':start,'end_epoch':time.time(),'process_CPU_seconds':r.ru_utime+r.ru_stime,'full_process_wall_seconds':time.time()-start,'GPU_job':False};write(PUB/'process.json',receipt)
        with (RUN/'auxiliary_CPU_receipts.jsonl').open('a') as f:f.write(json.dumps(receipt)+'\n');f.flush();os.fsync(f.fileno())
