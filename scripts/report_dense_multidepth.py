#!/usr/bin/env python3
"""Independent MD audit, inner-only decoder, frozen reader and cache-only analysis."""
import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES','0');os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8');os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import copy,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd,torch
from scripts import train_dense_sequence as base,train_dense_multidepth as run
from echo_routing.temporal import dense_runtime as data,dense_multidepth as md
from echo_routing.temporal.multidepth_audit import verify_optimizer,verify_decoder_binding
from echo_routing.temporal.dense_sequence import TwoStageTCN
from echo_routing.temporal.dense_checkpoint_audit import compare_final_checkpoint
from echo_routing.temporal.dense_reference_plan import md_reference_pairs
from echo_routing.temporal.dense_readout import SequenceReadout,ReadoutPixels
from echo_routing.temporal.dense_analysis import validate_inner_cache
from echo_routing.temporal.dev_train import lr_factor
from echo_routing.temporal.baselines.b3_js_divergence import boundary_scores
from echo_routing.temporal.baselines.registry import decode_with_prob
from scripts.aggregate_dense_sequence import evaluate,boot_pair
RUN,PUB=run.RUN,run.PUB
read,write,sha=base.read,base.write,base.sha

def verify_freeze(name):
    freeze=read(RUN/name)
    for p,h in freeze['objects'].items():assert sha(ROOT/p)==h,p
    return freeze

def audit():
    cfg,freeze=run.context(True);admission=read(PUB/'admission.json');source=torch.load(ROOT/cfg['source_state'],map_location='cpu',weights_only=True,mmap=True);assert len(source)==158;head_shapes={'head.'+n:tuple(v.shape) for n,v in TwoStageTCN(3072,128).state_dict().items()};assert len(head_shapes)==56;models=[];rows=[];total=0
    for seed in admission['training_seeds']:
        for arm in ['F-MD','J-MD']:
            folder=RUN/f'seed{seed}'/arm;assert not (folder/'pending.json').exists();trace=[json.loads(s) for s in (folder/'physical_updates.jsonl').read_text().splitlines()];assert [x['u'] for x in trace]==list(range(6000)),'any replay requires an explicit independent physical accounting audit'
            assert all(np.isfinite(x['loss']) and np.isfinite(x['gradient_norm']) and x['groups']['head']>0 and (arm=='F-MD' or x['groups']['active_video']>0) for x in trace)
            z=torch.load(folder/'latest.pt',map_location='cpu',weights_only=True,mmap=True);assert z['update']==6000 and z['seed']==seed and z['arm']==arm and z['schedule_total']==6000 and z['feature_contract']==md.FEATURE_CONTRACT and z['source_sha256']==sha(ROOT/cfg['source_state'])
            assert z['plan_canonical_sha256']==read(base.PUB/'independent_input_exposure.json')['plans'][str(seed)]['canonical_little_endian_int64_plan_sha256']
            verified=verify_optimizer(z['model'],source,z['optimizer'],6000,arm=='J-MD',md.IMAGE_ONLY,lr_factor(5999,200,6000),head_shapes);assert verified['head_optimizer_tensors']==56 and verified['active_video_optimizer_tensors']==(155 if arm=='J-MD' else 0)
            inner=[]
            for step in range(500,6001,500):
                path=folder/f'step{step}.pt';identity=read(folder/f'step{step}.identity.json');record=read(folder/f'inner_step{step}/metrics.json');assert sha(path)==identity['sha256']==record['model_checkpoint_sha256'] and identity['feature_contract']==record['feature_contract']==md.FEATURE_CONTRACT and record['seed']==seed and record['arm']==arm and record['step']==step;inner.append(record)
            best=data.choose_inner(inner)
            final=torch.load(folder/'step6000.pt',map_location='cpu',weights_only=True,mmap=True);verified.update(compare_final_checkpoint(final,z['model'],arm=='J-MD',md.FEATURE_CONTRACT));del final
            for step,kind in [(2000,'fixed_step'),(6000,'fixed_step'),(best['step'],'inner_selected_package')]:
                innerpath=folder/f'inner_step{step}';rows.append({'row':f'seed{seed}_{arm}_'+(f'fixed{step}' if kind=='fixed_step' else 'inner_package'),'kind':kind,'arm':arm,'seed':seed,'step':step,'path':str((folder/f'step{step}.pt').relative_to(ROOT)),'sha256':sha(folder/f'step{step}.pt'),'feature_contract':md.FEATURE_CONTRACT,'inner_probabilities_sha256':sha(innerpath/'probabilities.private.npy'),'inner_predictions_sha256':sha(innerpath/'predictions.private.npy'),'inner_record_sha256':sha(innerpath/'metrics.json'),'inner_sequence_sha256':read(base.PUB/'input_metadata.json')['roles']['inner_cal']['sequence_sha256']})
            models.append({'seed':seed,'arm':arm,'updates':6000,'physical_rows':len(trace),**verified,'latest_optimizer_sha256':sha(folder/'latest.pt'),'inner_chosen_step':best['step']});total+=len(trace);del z
    assert total==12000*len(admission['training_seeds']) and len(rows)==6*len(admission['training_seeds']);write(PUB/'models_manifest.json',{'passed':True,'models':models,'outer_rows':rows,'formal_physical_updates':total,'source_sha256':sha(ROOT/cfg['source_state']),'feature_contract':md.FEATURE_CONTRACT,'producer_git':freeze['git_commit'],'all_models_frozen_before_MD_outer_forward':True,'new_MD_outer_forwards':0,'epoch':time.time()});print(json.dumps({'MD_audit_passed':True,'updates':total,'rows':len(rows)}),flush=True)

def calibrate():
    verify_freeze('decoder_stage_freeze.json');models=read(PUB/'models_manifest.json');assert models['passed'] and models['new_MD_outer_forwards']==0;seqs=base.inputs('inner_cal');expected=read(base.PUB/'input_metadata.json')['roles']['inner_cal']['sequence_sha256'];out=RUN/'decoder_calibration';out.mkdir(exist_ok=True);choices={}
    for row in models['outer_rows']:
        key=row['sha256']
        if key in choices:continue
        folder=RUN/f"seed{row['seed']}"/row['arm']/f"inner_step{row['step']}"
        for file,field in [('probabilities.private.npy','inner_probabilities_sha256'),('predictions.private.npy','inner_predictions_sha256'),('metrics.json','inner_record_sha256')]:assert sha(folder/file)==row[field]
        record=read(folder/'metrics.json');assert record['feature_contract']==md.FEATURE_CONTRACT and record['model_checkpoint_sha256']==key and record['arm']==row['arm'] and record['seed']==row['seed'] and record['step']==row['step'];probs=np.load(folder/'probabilities.private.npy',allow_pickle=True).item();validate_inner_cache(probs,seqs,expected,record);saved=np.load(folder/'predictions.private.npy',allow_pickle=True).item();assert set(saved)==set(probs) and all(np.array_equal(saved[k],v.argmax(1)) for k,v in probs.items());records=[]
        for half_window in [5,7,9,11,15]:
            scores=np.concatenate([boundary_scores(probs[q['id']],half_window)[1:] for q in seqs if q['kind']=='native'])
            for quantile in [.8,.85,.9,.95,.97,.99,.995,.999]:
                params={'half_window':half_window,'threshold_quantile':quantile,'threshold':float(np.quantile(scores,quantile)),'nms_radius':5};pred={k:decode_with_prob('b3',v,{'decode':{'b3':params}})[0] for k,v in probs.items()};metrics,_=data.metrics(seqs,pred);records.append({'order':len(records),'params':params,**metrics})
        best=max(records,key=lambda r:(r['segment_f1_50_multi'],r['native_macro_f1'],-r['native_A4C_A5C_errors'],-r['order']));write(out/(key+'.json'),{'records':records,'chosen_order':best['order']});choices[key]={'params':best['params'],'chosen_grid_order':best['order'],'inner_input_checksum':expected,'inner_probabilities_sha256':row['inner_probabilities_sha256'],'inner_grid_sha256':sha(out/(key+'.json')),'independent_raw_metric_recount_passed':True};print(json.dumps({'MD_inner_B3_chosen':row['row']}),flush=True)
    write(PUB/'decoder_choices.json',{'model_manifest_sha256':sha(PUB/'models_manifest.json'),'choices_by_checkpoint_sha256':choices,'feature_contract':md.FEATURE_CONTRACT,'grid':'originalV01 5halfwindow x8quantile NMS5','new_model_forwards':0,'MD_outer_reads_at_choices':0,'epoch':time.time()})

def readout():
    freeze=verify_freeze('outer_readout_freeze.json');cfg,_=run.context(True);models=read(PUB/'models_manifest.json');choices=read(PUB/'decoder_choices.json');verify_decoder_binding(models,choices,sha(PUB/'models_manifest.json'),md.FEATURE_CONTRACT);pins=read(base.PUB/'outer_inputs.json');assert sha(base.RUN/'outer_metadata.private.npy')==pins['outer_metadata_sha256'];metadata=np.load(base.RUN/'outer_metadata.private.npy',allow_pickle=True).item();completed={};results=[]
    for row in models['outer_rows']:
        assert sha(ROOT/row['path'])==row['sha256'] and row['feature_contract']==md.FEATURE_CONTRACT;model=None
        for half in ['selection','calibration']:
            key=row['sha256']+'|'+half
            if key in completed:results.append({**row,'half':half,'reused_same_checkpoint':True,'raw_metrics':completed[key]});continue
            start=time.time();seqs=metadata[half]['sequences'];pixels=ReadoutPixels(seqs);identity=row['sha256']+'|'+pins['outer_metadata_sha256']+'|'+half+'|'+sha(run.CFG);folder=RUN/'outer_readout'/row['sha256']/half;cache=SequenceReadout(folder,identity,len(seqs),lengths=[len(q['labels']) for q in seqs]);initial=cache.cursor
            if initial<len(seqs):
                if model is None:
                    model=md.build_model(cfg,row['arm'],row['seed']);z=torch.load(ROOT/row['path'],map_location='cpu',weights_only=True);assert z['feature_contract']==md.FEATURE_CONTRACT;base.load_model_weights(model,ROOT/row['path']);model.eval();del z
                for i in range(initial,len(seqs)):
                    q=seqs[i];cache.run_one(i,lambda q=q:data.predict_sequence(model,q,pixels))
                    if (i+1)%100==0:base.budget_guard();write(base.RUN/'runtimecoordination.json',{'phase':'multidepth_outer_readout','row':row['row'],'half':half,'sequences_done':i+1,'sequences':len(seqs),'model_pid':os.getpid(),'epoch':time.time(),'next_action':'finish frozen MD rows; no new optimization or choices'})
            pred={q['id']:cache.load(i).argmax(1) for i,q in enumerate(seqs)};metrics,_=evaluate(pred,half,metadata);cost={'new_sequences':len(seqs)-initial,'previously_committed_sequences':initial,'decode_model_stitch_cache_wall_seconds_this_process':time.time()-start,'GPU_peak_reserved_bytes':torch.cuda.max_memory_reserved()};write(folder/'inference_cost.json',cost);completed[key]=metrics;results.append({**row,'half':half,'reused_same_checkpoint':False,'raw_metrics':metrics,'inference_cost':cost})
        if model is not None:del model;torch.cuda.empty_cache()
    write(PUB/'raw_outer_results.json',{'model_manifest_sha256':sha(PUB/'models_manifest.json'),'feature_contract':md.FEATURE_CONTRACT,'rows':results,'readout_freeze_git':freeze['git_commit'],'new_training_updates_during_outer':0});print(json.dumps({'MD_outer_complete':len(results),'unique_checkpoint_halves':len(completed)}),flush=True)

def aggregate():
    verify_freeze('outer_readout_freeze.json');models=read(PUB/'models_manifest.json');pins=read(base.PUB/'outer_inputs.json');choices=read(PUB/'decoder_choices.json');verify_decoder_binding(models,choices,sha(PUB/'models_manifest.json'),md.FEATURE_CONTRACT);raw=read(PUB/'raw_outer_results.json');assert raw['model_manifest_sha256']==sha(PUB/'models_manifest.json');metadata=np.load(base.RUN/'outer_metadata.private.npy',allow_pickle=True).item();systems={};comparisons={};records=[]
    for half in ['selection','calibration']:
        seqs=metadata[half]['sequences'];systems[half]={};comparisons[half]={};predictions={};tables={};done={}
        for row in models['outer_rows']:
            key=row['sha256']
            if key not in done:
                cache=SequenceReadout(RUN/'outer_readout'/key/half,key+'|'+pins['outer_metadata_sha256']+'|'+half+'|'+sha(run.CFG),len(seqs),lengths=[len(q['labels']) for q in seqs]);assert cache.cursor==len(seqs);done[key]={q['id']:cache.load(i) for i,q in enumerate(seqs)}
            for mode in ['raw','inner_B3']:
                params=choices['choices_by_checkpoint_sha256'][key]['params'];name=row['row']+'|'+mode;pred={k:(v.argmax(1) if mode=='raw' else decode_with_prob('b3',v,{'decode':{'b3':params}})[0]) for k,v in done[key].items()};m,t=evaluate(pred,half,metadata);systems[half][name]=m;predictions[name]=pred;tables[name]=t;records.append({**row,'half':half,'decoding':mode,**m})
                if mode=='raw':assert m==next(x['raw_metrics'] for x in raw['rows'] if x['row']==row['row'] and x['half']==half)
        from echo_routing.evaluate.window_target import nearest_centre
        for seed in [0,1,2]:
            pin=pins['A08_cache_pins'][half][str(seed)];assert sha(ROOT/pin['path'])==pin['sha256'];probs={}
            with np.load(ROOT/pin['path']) as z:
                assert z['sequence_ids'].tolist()==[q['id'] for q in seqs];p=torch.softmax(torch.from_numpy(z['logits']),-1).numpy()
                for i,q in enumerate(seqs):
                    ix=np.flatnonzero(z['seq']==i);v=p[ix][nearest_centre(len(q['labels']),z['centre'][ix])];assert np.max(np.abs(v-z['sample_probs'][z['sample_offsets'][i]:z['sample_offsets'][i+1]]))<=1e-7;probs[q['id']]=v
            name=f'A08_original_B3_s{seed}';pred={k:decode_with_prob('b3',v,{'decode':{'b3':pin['original_params']}})[0] for k,v in probs.items()};metrics,t=evaluate(pred,half,metadata);assert metrics==read(base.PUB/'A08_reference_metrics.json')['systems'][half][name];systems[half][name]=metrics;predictions[name]=pred;tables[name]=t;records.append({'row':name,'half':half,'decoding':'original_frozen_B3',**metrics})
        for pair in md_reference_pairs(read(PUB/'admission.json')['training_seeds']):
            left,right=pair['left'],pair['right'];comparisons[half][left+' - '+right]={'comparison_kind':pair['kind'],'delta':{k:systems[half][left][k]-systems[half][right][k] for k in ['native_macro_f1','segment_f1_50_multi','boundary_recall_multi','boundary_precision_multi','native_internal_cuts_per_clip','native_A4C_A5C_errors']},**boot_pair(predictions[left],predictions[right],seqs,tables[left],tables[right])}
    write(PUB/'summary.json',{'status':'exploratory reused development, cost-admitted MD tier completed','systems':systems,'paired_comparisons':comparisons,'model_manifest_sha256':sha(PUB/'models_manifest.json'),'decoder_choices_sha256':sha(PUB/'decoder_choices.json'),'core_outer_inputs_sha256':sha(base.PUB/'outer_inputs.json'),'A08_reference_metrics_sha256':sha(base.PUB/'A08_reference_metrics.json'),'feature_contract':md.FEATURE_CONTRACT,'raw_reader_metrics_independently_reproduced':True,'all3_original_A08_metrics_reproduced':True,'primary_comparison':'unchanged within-four-depth matched6000 J-MD minusF-MD raw byseed;innerB3supplementary;fixedA08s0reference addendum beforeMDouteroutputs','clinical_independent_validation':False,'protected_media_forwards':0});pd.json_normalize(records).to_csv(PUB/'metrics.csv',index=False);print(json.dumps({'MD_summary_complete':True,'rows':len(records),'paired_comparisons':sum(map(len,comparisons.values()))}),flush=True)

if __name__=='__main__':globals()[sys.argv[1]]()
