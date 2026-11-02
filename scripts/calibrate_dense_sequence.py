#!/usr/bin/env python3
"""A57 optional B3 control on new learned probabilities, only fixed inner calibration."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.train_dense_sequence import RUN,PUB,read,write,sha,inputs
from echo_routing.temporal import dense_runtime as data
from echo_routing.temporal.baselines.b3_js_divergence import boundary_scores
from echo_routing.temporal.baselines.registry import decode_with_prob
from echo_routing.temporal.dense_analysis import validate_inner_cache

def main():
 models=read(PUB/'models_manifest.json');assert models['passed'] and models['new_external_development_forwards']==0;seqs=inputs('inner_cal');choices={};out=RUN/'decoder_calibration';out.mkdir(exist_ok=True)
 for row in models['outer_rows']:
  key=row['sha256']
  if key in choices:continue
  folder=RUN/f"seed{row['seed']}"/row['arm']/f"inner_step{row['step']}";p=folder/'probabilities.private.npy';assert sha(p)==row['inner_probabilities_sha256'] and sha(folder/'metrics.json')==row['inner_record_sha256'] and sha(folder/'predictions.private.npy')==row['inner_predictions_sha256'];record=read(folder/'metrics.json');assert record['arm']==row['arm'] and record['seed']==row['seed'] and record['step']==row['step'];probs=np.load(p,allow_pickle=True).item();validate_inner_cache(probs,seqs,row['inner_sequence_sha256'],record);pred=np.load(folder/'predictions.private.npy',allow_pickle=True).item();assert set(pred)==set(probs) and all(np.array_equal(pred[k],v.argmax(1)) for k,v in probs.items());records=[];order=0
  for half_window in [5,7,9,11,15]:
   scores=np.concatenate([boundary_scores(probs[q['id']],half_window)[1:] for q in seqs if q['kind']=='native'])
   for quantile in [.80,.85,.90,.95,.97,.99,.995,.999]:
    params={'half_window':half_window,'threshold_quantile':quantile,'threshold':float(np.quantile(scores,quantile)),'nms_radius':5};pred={k:decode_with_prob('b3',v,{'decode':{'b3':params}})[0] for k,v in probs.items()};metrics,_=data.metrics(seqs,pred);records.append({'order':order,'params':params,**metrics});order+=1
  best=max(records,key=lambda r:(r['segment_f1_50_multi'],r['native_macro_f1'],-r['native_A4C_A5C_errors'],-r['order']));write(out/(key+'.json'),{'records':records,'chosen_order':best['order']});choices[key]={'params':best['params'],'chosen_grid_order':best['order'],'inner_native_macro_f1':best['native_macro_f1'],'inner_segment_f1_50_multi':best['segment_f1_50_multi'],'inner_A4C_A5C_errors':best['native_A4C_A5C_errors'],'inner_probabilities_sha256':sha(p),'inner_grid_sha256':sha(out/(key+'.json')),'independent_raw_metric_recount_passed':True,'inner_input_checksum':row['inner_sequence_sha256']};print(json.dumps({'inner_B3_chosen_for':row['row'],'grid_count':40}),flush=True)
 write(PUB/'decoder_choices.json',{'rule':'inner multiSegmentF1 thennativeMacroF1 thenfewerA4C/A5Cbidirectionalerrors thensmallergridorder','grid':'V01original5halfwindowx8quantilex1NMS','model_manifest_sha256':sha(PUB/'models_manifest.json'),'choices_by_checkpoint_sha256':choices,'external_development_reads_at_choices':0,'new_model_forwards':0,'epoch':time.time()})
if __name__=='__main__':main()
