#!/usr/bin/env python3
"""All frozen A58 main/stress systems; original-axis baselines and same-score MAP."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import sys,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from scripts.structured_common import ROOT,RUN,PUB,CFG,read,write,sha,sequences,verify_freeze
from scripts.structured_head_runtime import load_bank,digest
from scripts.produce_structured_cache import reader
from echo_routing.temporal.structured_evidence import StructuredEvidence
from echo_routing.temporal.structured_lifecycle import audit_payload
from echo_routing.temporal.structured_evaluation import decode_bank
from echo_routing.temporal.structured_metrics import evaluate,stress_context,compose_stress_predictions,verify_reference
from echo_routing.temporal.baselines.registry import decode_with_prob
from echo_routing.temporal.structured_inputs import check_probabilities
from echo_routing.evaluate.window_target import nearest_centre

HALVES=['selection','calibration']
TABLES=['fixed6000','inner_selected','duration_stress_fixed6000']

def original_a08(half,qs,seed,cfg):
    pins=read(ROOT/cfg['data']['outer_inputs']);assert sha(ROOT/cfg['data']['outer_inputs'])==cfg['data']['outer_inputs_sha256'];pin=pins['A08_cache_pins'][half][str(seed)];path=ROOT/pin['path'];assert sha(path)==pin['sha256'];values={}
    with np.load(path) as z:
        assert z['sequence_ids'].tolist()==[q['id'] for q in qs];assert str(z['identity'])==pin['identity'];prob=torch.softmax(torch.from_numpy(z['logits']),1).numpy()
        for i,q in enumerate(qs):
            indices=np.flatnonzero(z['seq']==i);p=prob[indices][nearest_centre(len(q['labels']),z['centre'][indices])];saved=z['sample_probs'][z['sample_offsets'][i]:z['sample_offsets'][i+1]];assert np.max(np.abs(p-saved))<=1e-7;check_probabilities(p,len(q['labels']));values[q['id']]=p
    return values,pin['original_params']

def from_cache(source,role,qs):
    cache=reader(source,role);assert cache.cursor==len(qs);return {q['id']:cache.load(i) for i,q in enumerate(qs)}

def baseline_predictions(half,main,stress,mainbank,stressbank,cfg):
    raw=lambda p:{k:v.argmax(1).astype(np.int8) for k,v in p.items()};b3=lambda p,params:{k:decode_with_prob('b3',v,{'decode':{'b3':params}})[0].astype(np.int8) for k,v in p.items()};mainp={};a0=None
    for seed in [0,1,2]:
        p,params=original_a08(half,main['sequences'],seed,cfg);mainp[f'A08_s{seed}_original_B3']=b3(p,params)
        if seed==0:a0=p;assert params==cfg['source']['a08']['original_B3']
    ad={q['id']:mainbank.values[i][0].numpy() for i,q in enumerate(mainbank.sequences)};jp={q['id']:mainbank.values[i][1].numpy() for i,q in enumerate(mainbank.sequences)};mainp.update({'A08_s0_original_raw':raw(a0),'A08_s0_dense_raw':raw(ad),'A08_s0_dense_original_B3_transfer':b3(ad,cfg['source']['a08']['original_B3']),'JEPA_core_fixed6000_raw':raw(jp)});original=from_cache('a08_original','stress_'+half,stress);dense={q['id']:stressbank.values[i][0].numpy() for i,q in enumerate(stress)};jepa={q['id']:stressbank.values[i][1].numpy() for i,q in enumerate(stress)};new={'A08_s0_original_raw':raw(original),'A08_s0_original_B3':b3(original,cfg['source']['a08']['original_B3']),'A08_s0_dense_raw':raw(dense),'A08_s0_dense_original_B3_transfer':b3(dense,cfg['source']['a08']['original_B3']),'JEPA_core_fixed6000_raw':raw(jepa)};stressp={name:compose_stress_predictions(mainp[name],p,main['sequences'],stress) for name,p in new.items()};return mainp,stressp

def execution_order(rows):
    return sorted(rows,key=lambda r:(r['kind']!='fixed6000',r['arm'] not in ['D1','D2'],r['seed'],r['arm']))

def main():
    freeze=verify_freeze('readout_freeze.json');verify_freeze('outer_freeze.json');verify_freeze('producer_freeze.json');models=read(PUB/'models_manifest.json');assert models['passed'] and models['formal_optimizer_updates']==48000 and models['all_head_weights_and_inner_choices_frozen'];assert models['inner_choices_sha256']==sha(PUB/'inner_choices.json');outercache=read(PUB/'cache_outer_manifest.json');assert outercache['passed'] and all(r['complete'] for r in outercache['records']);cfg=read(CFG);torch.set_num_threads(1);torch.use_deterministic_algorithms(True);metadata=np.load(ROOT/cfg['data']['old_outer_metadata'],allow_pickle=True).item();assert sha(ROOT/cfg['data']['old_outer_metadata'])==cfg['data']['old_outer_metadata_sha256'];order=execution_order(models['outer_rows']);assert read(PUB/'outer_execution_order.json')['row_ids']==[r['row_id'] for r in order];banks={};contexts={};predictions={table:{h:{} for h in HALVES} for table in TABLES};rows=[];headcaches={};baseline_reference=read(ROOT/'docs/results/2026-11-02_dense_target_sequence/A08_reference_metrics.json')['systems']
    def publish(name,table,half,pred,info):
        metric,tables=evaluate(pred,contexts[table][half]);row={'system':name,'table':table,'half':half,**info,**metric};rows.append(row);predictions[table][half][name]=pred;write(PUB/'completed_outer_rows.json',{'rows':rows,'metric_rows':len(rows),'provisional_until_independent_aggregate_audit':True,'all8heads_and_choices_frozen_before_any_outer':True,'readout_freeze_git':freeze['git_commit']});print(json.dumps({'completed_outer_row':name,'table':table,'half':half,'native_macro_f1':metric['native_macro_f1'],'segment_f1_50_multi':metric['segment_f1_50_multi'],'short16_FN':metric['short_true_unmatched_16samples'],'short16_FP':metric['short_spurious_segments_16samples']}),flush=True);return metric
    for table in TABLES:contexts[table]={}
    for half in HALVES:
        mainbank=load_bank(half);stressqs=sequences('stress_'+half);stressbank=load_bank('stress_'+half);banks[half]=(mainbank,stressbank);contexts['fixed6000'][half]=contexts['inner_selected'][half]=metadata[half];contexts['duration_stress_fixed6000'][half]=stress_context(metadata[half],stressqs);mainp,stressp=baseline_predictions(half,metadata[half],stressqs,mainbank,stressbank,cfg)
        for name,pred in mainp.items():
            metric=publish(name,'fixed6000',half,pred,{'role':'reference','new_head_seed':None,'original_A08_parameter_search':False});predictions['inner_selected'][half][name]=pred
            if name.startswith('A08_s') and name.endswith('_original_B3'):
                seed=name.split('_')[1][1:];verify_reference(metric,baseline_reference[half][f'A08_original_B3_s{seed}'])
        for name,pred in stressp.items():publish(name,'duration_stress_fixed6000',half,pred,{'role':'reference','new_head_seed':None,'native_pair_predictions_reused_identically':True,'fresh_original_stride5_stress_forward':name.startswith('A08_s0_original')})
    for row in order:
        path=ROOT/row['path'];assert sha(path)==row['sha256'];model=StructuredEvidence(row['arm'],row['seed']);payload=torch.load(path,map_location='cpu',weights_only=True);audit_payload(model,payload,row['head_identity']);model.load_state_dict(payload['model'],strict=True);model.eval()
        for half in HALVES:
            bank,_=banks[half];binding=digest({'checkpoint_sha256':row['sha256'],'head_identity':row['head_identity'],'posterior_sources':bank.source_identities,'role':half,'input_sha256':read(PUB/'input_manifest.json')['roles'][half]['sha256']});pred,cache=decode_bank(model,bank,RUN/'head_readout'/row['sha256']/half,binding);headcaches[(row['sha256'],half)]=cache;publish(row['row_id'],row['kind'],half,pred,{'role':'head',**row,'same_score_MAP':True,'postprocessing':None,'head_label_cache':cache});write(RUN/'runtimecoordination.json',{'phase':'fixed_outer_head_MAP','model_pid':os.getpid(),'row':row['row_id'],'half':half,'completed_metric_rows':len(rows),'epoch':time.time(),'next_action':'complete all16logical main rows then fixed6000duration stress; no outer-led choices'})
    for row in [r for r in order if r['kind']=='fixed6000']:
        model=StructuredEvidence(row['arm'],row['seed']);model.load_state_dict(torch.load(ROOT/row['path'],map_location='cpu',weights_only=True)['model'],strict=True);model.eval()
        for half in HALVES:
            _,bank=banks[half];role='stress_'+half;binding=digest({'checkpoint_sha256':row['sha256'],'head_identity':row['head_identity'],'posterior_sources':bank.source_identities,'role':role,'input_sha256':read(PUB/'input_manifest.json')['roles'][role]['sha256']});new,cache=decode_bank(model,bank,RUN/'head_readout'/row['sha256']/role,binding);headcaches[(row['sha256'],role)]=cache;pred=compose_stress_predictions(predictions['fixed6000'][half][row['row_id']],new,metadata[half]['sequences'],bank.sequences);publish(row['row_id'],'duration_stress_fixed6000',half,pred,{'role':'head',**row,'same_score_MAP':True,'postprocessing':None,'head_label_cache':cache,'native_pair_predictions_reused_identically':True})
    assert len(rows)==72 and sum(r['role']=='head' for r in rows)==48;path=RUN/'readout_predictions.private.npy';temp=path.with_suffix('.npy.tmp')
    with temp.open('wb') as f:np.save(f,predictions,allow_pickle=True);f.flush();os.fsync(f.fileno())
    temp.replace(path);write(PUB/'outer_metrics.json',{'rows':rows,'metric_rows':len(rows),'head_metric_rows':48,'reference_metric_rows':24,'readout_freeze_git':freeze['git_commit'],'models_manifest_sha256':sha(PUB/'models_manifest.json'),'source_outer_cache_manifest_sha256':sha(PUB/'cache_outer_manifest.json'),'private_prediction_bundle_sha256':sha(path),'unique_head_cache_cells':len(headcaches),'actual_new_same_score_MAP_sequences':sum(c['sequences'] for c in headcaches.values()),'head_weights_and_choices_frozen_before_outer':True,'no_head_B3_or_postprocessing':True,'historical_A08_all3_reference_reproduced':True,'stress_native_pair_reused_not_new_samples':True,'new_protected_media_forwards':0,'epoch':time.time()});write(RUN/'runtimecoordination.json',{'phase':'all_main_and_stress_readouts_complete_pending_independent_aggregate','model_pid':None,'metric_rows':len(rows),'epoch':time.time(),'next_action':'cached-only independent recount and all132paired5000bootstrap comparisons'});print(json.dumps({'all_readouts_complete':True,'metric_rows':len(rows),'head_rows':48}),flush=True)
if __name__=='__main__':main()
