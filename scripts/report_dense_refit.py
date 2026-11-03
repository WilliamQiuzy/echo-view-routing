#!/usr/bin/env python3
"""Frozen full-data pair audit, once-per-sequence reader and paired cache analysis."""
import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES','0');os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8');os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd,torch
from scripts import train_dense_refit as run,train_dense_sequence as base
from scripts.aggregate_dense_sequence import evaluate,boot_pair
from echo_routing.temporal import dense_runtime as data
from echo_routing.temporal.multidepth_audit import verify_optimizer
from echo_routing.temporal.dense_sequence import TwoStageTCN
from echo_routing.temporal.dense_readout import SequenceReadout,ReadoutPixels
from echo_routing.temporal.dev_train import lr_factor
from echo_routing.temporal.baselines.registry import decode_with_prob
RUN,PUB=run.RUN,run.PUB;read,write,sha=base.read,base.write,base.sha

def frozen_reader():
    freeze=read(RUN/'outer_readout_freeze.json')
    for p,h in freeze['objects'].items():assert sha(ROOT/p)==h,p
    models=read(PUB/'models_manifest.json');meta=read(PUB/'input_metadata.json');transfer=read(PUB/'B3_transfer.json');assert models['passed'] and models['input_metadata_sha256']==sha(PUB/'input_metadata.json') and models['config_sha256']==sha(run.CFG) and models['feature_contract']==run.FEATURE and models['B3_transfer_sha256']==sha(PUB/'B3_transfer.json')==meta['B3_transfer_sha256'];return freeze,models,transfer

def audit():
    cfg,freeze=run.context(True);meta=read(PUB/'input_metadata.json');source=torch.load(ROOT/cfg['source_state'],map_location='cpu',weights_only=True,mmap=True);assert len(source)==158;head={'head.'+n:tuple(v.shape) for n,v in TwoStageTCN(768,128).state_dict().items()};models=[];rows=[]
    for arm in ['R-CE','R-ST']:
        folder=RUN/arm;assert not (folder/'pending.json').exists();trace=[json.loads(x) for x in (folder/'physical_updates.jsonl').read_text().splitlines()];assert [x['u'] for x in trace]==list(range(6000));assert all(np.isfinite(x['loss']) and np.isfinite(x['CE']) and np.isfinite(x['S']) and np.isfinite(x['gradient_norm']) and x['groups']['head']>0 and x['groups']['active_video']>0 and abs(x['weighted_S']-cfg['arms'][arm]*x['S'])<1e-7 for x in trace);z=torch.load(folder/'latest.pt',map_location='cpu',weights_only=True,mmap=True);assert z['update']==6000 and z['arm']==arm and z['coefficient']==cfg['arms'][arm] and z['schedule_total']==6000 and z['feature_contract']==run.FEATURE and z['plan_canonical_sha256']==meta['plan_canonical_sha256'] and z['config_sha256']==sha(run.CFG) and z['source_sha256']==sha(ROOT/cfg['source_state']);checks=verify_optimizer(z['model'],source,z['optimizer'],6000,True,read(base.PUB/'source.json')['inactive_video_names'],lr_factor(5999,200,6000),head);assert checks['active_video_optimizer_tensors']==149 and checks['head_optimizer_tensors']==56
        for step in [2000,6000]:
            identity=read(folder/f'step{step}.identity.json');assert identity['sha256']==sha(folder/f'step{step}.pt') and identity['feature_contract']==run.FEATURE and identity['arm']==arm and identity['step']==step and identity['config_sha256']==sha(run.CFG) and identity['plan_canonical_sha256']==meta['plan_canonical_sha256']
        rows.append({'row':arm+'_fixed6000','arm':arm,'step':6000,'seed':2026110257,'path':str((folder/'step6000.pt').relative_to(ROOT)),'sha256':sha(folder/'step6000.pt'),'feature_contract':run.FEATURE});models.append({'arm':arm,'physical_updates':len(trace),**checks,'latest_optimizer_sha256':sha(folder/'latest.pt'),'mean_first100_CE':float(np.mean([x['CE'] for x in trace[:100]])),'mean_last100_CE':float(np.mean([x['CE'] for x in trace[-100:]])),'mean_first100_S':float(np.mean([x['S'] for x in trace[:100]])),'mean_last100_S':float(np.mean([x['S'] for x in trace[-100:]]))});del z
    write(PUB/'models_manifest.json',{'passed':True,'models':models,'outer_rows':rows,'formal_physical_updates':12000,'source_sha256':sha(ROOT/cfg['source_state']),'input_metadata_sha256':sha(PUB/'input_metadata.json'),'config_sha256':sha(run.CFG),'B3_transfer_sha256':sha(PUB/'B3_transfer.json'),'feature_contract':run.FEATURE,'producer_git':freeze['git_commit'],'new_refit_outer_forwards':0,'former_inner_role':'TRAINING;no checkpoint or B3selection','epoch':time.time()});print(json.dumps({'refit_models_audit_passed':True,'physical_updates':12000}),flush=True)

def readout():
    cfg,_=run.context(True);freeze,models,transfer=frozen_reader();pins=read(base.PUB/'outer_inputs.json');assert sha(base.RUN/'outer_metadata.private.npy')==pins['outer_metadata_sha256'];metadata=np.load(base.RUN/'outer_metadata.private.npy',allow_pickle=True).item();results=[]
    for row in models['outer_rows']:
        assert sha(ROOT/row['path'])==row['sha256'];m=None
        for half in ['selection','calibration']:
            seqs=metadata[half]['sequences'];folder=RUN/'outer_readout'/row['sha256']/half;identity=row['sha256']+'|'+pins['outer_metadata_sha256']+'|'+half+'|'+sha(run.CFG);cache=SequenceReadout(folder,identity,len(seqs),lengths=[len(q['labels']) for q in seqs]);initial=cache.cursor;start=time.time();pixels=ReadoutPixels(seqs)
            if initial<len(seqs):
                if m is None:
                    m,o=run.build(cfg);del o;z=torch.load(ROOT/row['path'],map_location='cpu',weights_only=True);assert z['feature_contract']==run.FEATURE and z['arm']==row['arm'] and z['step']==6000;base.load_model_weights(m,ROOT/row['path']);m.eval();del z
                for i in range(initial,len(seqs)):
                    q=seqs[i];cache.run_one(i,lambda q=q:data.predict_sequence(m,q,pixels))
                    if (i+1)%100==0:base.budget_guard();write(base.RUN/'runtimecoordination.json',{'phase':'adaptive_refit_outer_readout','arm':row['arm'],'half':half,'sequences_done':i+1,'sequences':len(seqs),'model_pid':os.getpid(),'epoch':time.time(),'next_action':'complete fixed6000 raw/transferredB3 pair; no tuning'})
            pred={q['id']:cache.load(i).argmax(1) for i,q in enumerate(seqs)};metrics,_=evaluate(pred,half,metadata);cost={'new_sequences':len(seqs)-initial,'previously_committed_sequences':initial,'decode_model_stitch_cache_wall_seconds_this_process':time.time()-start,'GPU_peak_reserved_bytes':torch.cuda.max_memory_reserved()};write(folder/'inference_cost.json',cost);results.append({**row,'half':half,'raw_metrics':metrics,'inference_cost':cost});print(json.dumps({'refit_completed_outer_row':row['row'],'half':half,'raw_native_macro_f1':metrics['native_macro_f1'],'raw_segment_f1_50_multi':metrics['segment_f1_50_multi']}),flush=True)
        if m is not None:del m;torch.cuda.empty_cache()
    write(PUB/'raw_outer_results.json',{'model_manifest_sha256':sha(PUB/'models_manifest.json'),'rows':results,'readout_freeze_git':freeze['git_commit'],'new_training_updates_during_readout':0})

def aggregate():
    freeze,models,transfer=frozen_reader();pins=read(base.PUB/'outer_inputs.json');metadata=np.load(base.RUN/'outer_metadata.private.npy',allow_pickle=True).item();raw=read(PUB/'raw_outer_results.json');assert raw['model_manifest_sha256']==sha(PUB/'models_manifest.json');systems={};comparisons={};records=[]
    for half in ['selection','calibration']:
        seqs=metadata[half]['sequences'];systems[half]={};preds={};tables={}
        def keep(name,pred,fields):
            metrics,t=evaluate(pred,half,metadata);systems[half][name]=metrics;preds[name]=pred;tables[name]=t;records.append({'row':name,'half':half,**fields,**metrics});return metrics
        for row in models['outer_rows']:
            key=row['sha256'];cache=SequenceReadout(RUN/'outer_readout'/key/half,key+'|'+pins['outer_metadata_sha256']+'|'+half+'|'+sha(run.CFG),len(seqs),lengths=[len(q['labels']) for q in seqs]);assert cache.cursor==len(seqs);probs={q['id']:cache.load(i) for i,q in enumerate(seqs)}
            for mode in ['raw','transferred_parent_B3']:
                pred={k:(v.argmax(1) if mode=='raw' else decode_with_prob('b3',v,{'decode':{'b3':transfer['params']}})[0]) for k,v in probs.items()};metrics=keep(row['row']+'|'+mode,pred,{'arm':row['arm'],'decoding':mode,'checkpoint_sha256':key})
                if mode=='raw':assert metrics==next(x['raw_metrics'] for x in raw['rows'] if x['arm']==row['arm'] and x['half']==half)
        parentmodels=read(base.PUB/'models_manifest.json');parentchoices=read(base.PUB/'decoder_choices.json')
        for seed in [2026110257,2026110258]:
            row=next(r for r in parentmodels['outer_rows'] if r['row']==f'seed{seed}_J-high_fixed6000');key=row['sha256'];cache=SequenceReadout(base.RUN/'outer_readout'/key/half,key+'|'+pins['outer_metadata_sha256']+'|'+half,len(seqs),lengths=[len(q['labels']) for q in seqs]);assert cache.cursor==len(seqs);probs={q['id']:cache.load(i) for i,q in enumerate(seqs)}
            for mode in ['raw','inner_B3']:
                pred={k:(v.argmax(1) if mode=='raw' else decode_with_prob('b3',v,{'decode':{'b3':parentchoices['choices_by_checkpoint_sha256'][key]['params']}})[0]) for k,v in probs.items()};metrics=keep(f'parent_J_seed{seed}|'+mode,pred,{'decoding':mode});assert metrics==read(base.PUB/'summary.json')['systems'][half][row['row']+'|'+mode]
        from echo_routing.evaluate.window_target import nearest_centre
        for seed in [0,1,2]:
            pin=pins['A08_cache_pins'][half][str(seed)];assert sha(ROOT/pin['path'])==pin['sha256'];probs={}
            with np.load(ROOT/pin['path']) as z:
                assert z['sequence_ids'].tolist()==[q['id'] for q in seqs];logits=torch.softmax(torch.from_numpy(z['logits']),-1).numpy()
                for i,q in enumerate(seqs):
                    ix=np.flatnonzero(z['seq']==i);p=logits[ix][nearest_centre(len(q['labels']),z['centre'][ix])];assert np.max(np.abs(p-z['sample_probs'][z['sample_offsets'][i]:z['sample_offsets'][i+1]]))<=1e-7;probs[q['id']]=p
            metrics=keep(f'A08_original_B3_s{seed}',{k:decode_with_prob('b3',v,{'decode':{'b3':pin['original_params']}})[0] for k,v in probs.items()},{'decoding':'original_frozen_B3'});assert metrics==read(base.PUB/'A08_reference_metrics.json')['systems'][half][f'A08_original_B3_s{seed}']
        comparisons[half]={}
        for mode in ['raw','transferred_parent_B3']:
            ce='R-CE_fixed6000|'+mode;st='R-ST_fixed6000|'+mode;parent='parent_J_seed2026110257|'+('raw' if mode=='raw' else 'inner_B3');pairs=[(st,ce),(ce,parent),(st,parent),(ce,'A08_original_B3_s0'),(st,'A08_original_B3_s0')]
            for left,right in pairs:comparisons[half][left+' - '+right]={'delta':{k:systems[half][left][k]-systems[half][right][k] for k in ['native_macro_f1','segment_f1_50_multi','boundary_recall_multi','boundary_precision_multi','native_internal_cuts_per_clip','native_A4C_A5C_errors']},**boot_pair(preds[left],preds[right],seqs,tables[left],tables[right])}
    write(PUB/'summary.json',{'status':'adaptive exploratory reused development, fixed6000 pair completed','systems':systems,'paired_comparisons':comparisons,'model_manifest_sha256':sha(PUB/'models_manifest.json'),'input_metadata_sha256':sha(PUB/'input_metadata.json'),'B3_transfer_sha256':sha(PUB/'B3_transfer.json'),'outer_inputs_sha256':sha(base.PUB/'outer_inputs.json'),'raw_metrics_independently_reproduced':True,'all3_A08_and_both_parent_J_cached_metrics_reproduced':True,'primary':'R-ST minusR-CE fixed6000 raw, oneseed; transferredB3supplementary','adaptive_after_outer_feedback':True,'former628_inner_is_training':True,'protected_media_forwards':0,'clinical_independent_validation':False});pd.json_normalize(records).to_csv(PUB/'metrics.csv',index=False);print(json.dumps({'refit_summary_complete':True,'rows':len(records),'comparisons':20}),flush=True)
if __name__=='__main__':globals()[sys.argv[1]]()
