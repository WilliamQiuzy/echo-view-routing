#!/usr/bin/env python3
"""A57 cache-only complete-task/B3/reference summaries and paired uncertainty."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import json,resource,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd,torch
from scripts.train_dense_sequence import RUN,PUB,read,write,sha
from echo_routing.evaluate.dev_metrics import evaluate_predictions
from echo_routing.temporal import dense_runtime as data
from echo_routing.temporal.baselines.registry import decode_with_prob
from echo_routing.temporal.dense_readout import SequenceReadout
from echo_routing.evaluate.failure_units import session_of,match_segments
from echo_routing.temporal.segments import labels_to_segments
from echo_routing.temporal.dense_analysis import short_metrics

def evaluate(preds,half,metadata):
 ev=metadata[half]['eval_set'];seqs=metadata[half]['sequences'];m,t=evaluate_predictions(preds,ev,10,.5);subset=[q for q in seqs if q['kind'] in ['native','multi']];e,_=data.metrics(subset,{q['id']:preds[q['id']] for q in subset});m.update({k:v for k,v in e.items() if k.startswith('short_') or k.startswith('native_A4C') or k.startswith('native_A5C')});m.update(short_metrics(seqs,preds,16));return m,t

def boot_pair(left,right,seqs,left_tables,right_tables,seed=2026110257,n_boot=2000):
 # Stream resampling is conditional on this fixed constructed bank: shared source
 # clips make these intervals descriptive, not independent-patient uncertainty.
 a=left_tables['multi_seg'];b=right_tables['multi_seg'];assert a.stream.tolist()==b.stream.tolist();n_multi=len(a);aa=a[['seg_matched','seg_pred','seg_true']].to_numpy();bb=b[['seg_matched','seg_pred','seg_true']].to_numpy();rng=np.random.default_rng(seed);d=[]
 for _ in range(n_boot):
  idx=rng.integers(0,len(a),len(a));x,y=aa[idx].sum(0),bb[idx].sum(0);d.append(2*x[0]/(x[1]+x[2])-2*y[0]/(y[1]+y[2]))
 nat=[q for q in seqs if q['kind']=='native'];groups={}
 for q in nat:groups.setdefault(session_of(q['id']),[]).append(q)
 clusters=list(groups.values());nf=[]
 def macro(conf):
  den=conf.sum(0)+conf.sum(1);return float(np.mean(np.divide(2*np.diag(conf),den,out=np.zeros(5,float),where=den>0)))
 confs=[]
 for group in clusters:
  c=np.zeros((2,5,5),int)
  for q in group:
   lab=int(q['labels'][0]);c[0,lab,int(np.bincount(left[q['id']],minlength=5).argmax())]+=1;c[1,lab,int(np.bincount(right[q['id']],minlength=5).argmax())]+=1
  confs.append(c)
 confs=np.asarray(confs)
 for _ in range(n_boot):
  c=confs[rng.integers(0,len(clusters),len(clusters))].sum(0);nf.append(macro(c[0])-macro(c[1]))
 exact={'native_left_only_correct':0,'native_right_only_correct':0,'native_both_wrong':0,'native_both_correct':0,'multi_true_segments_recovered':0,'multi_true_segments_lost':0,'short_true_segments_recovered_10samples':0,'short_true_segments_lost_10samples':0,'short_true_segments_recovered_16samples':0,'short_true_segments_lost_16samples':0}
 for q in seqs:
  a,b=left[q['id']],right[q['id']]
  if q['kind']=='native':
   x=int(np.bincount(a,minlength=5).argmax())==int(q['labels'][0]);y=int(np.bincount(b,minlength=5).argmax())==int(q['labels'][0]);name='native_both_correct' if x and y else 'native_both_wrong' if not x and not y else 'native_left_only_correct' if x else 'native_right_only_correct';exact[name]+=1
  elif q['kind']=='multi':
   truth=labels_to_segments(q['labels']);ma=match_segments(labels_to_segments(a),truth);mb=match_segments(labels_to_segments(b),truth);recovered=set(ma)-set(mb);lost=set(mb)-set(ma);exact['multi_true_segments_recovered']+=len(recovered);exact['multi_true_segments_lost']+=len(lost);exact['short_true_segments_recovered_10samples']+=sum(truth[i].end-truth[i].start<=10 for i in recovered);exact['short_true_segments_lost_10samples']+=sum(truth[i].end-truth[i].start<=10 for i in lost);exact['short_true_segments_recovered_16samples']+=sum(truth[i].end-truth[i].start<=16 for i in recovered);exact['short_true_segments_lost_16samples']+=sum(truth[i].end-truth[i].start<=16 for i in lost)
 return {'paired_conditional_stream_bootstrap_segmentF1_delta95':np.quantile(d,[.025,.975]).tolist(),'paired_session_bootstrap_nativeMacroF1_delta95':np.quantile(nf,[.025,.975]).tolist(),'bootstrap':{'replicates':n_boot,'seed':seed,'native_sessions':len(clusters),'multi_streams':n_multi,'scope':'native grouped sessions (notpatients); constructedstreams conditional fixedbank, sharedsources not independent'},'exact_disagreements':exact}

def main():
 models=read(PUB/'models_manifest.json');pins=read(PUB/'outer_inputs.json');choices=read(PUB/'decoder_choices.json');assert choices['model_manifest_sha256']==sha(PUB/'models_manifest.json');assert read(PUB/'raw_outer_results.json')['model_manifest_sha256']==sha(PUB/'models_manifest.json');metadata=np.load(RUN/'outer_metadata.private.npy',allow_pickle=True).item();systems={};tables={};predictions={};rawcheck=read(PUB/'raw_outer_results.json');basehistorical=read(ROOT/'docs/results/2026-10-16_video_within_view/summary.json');records=[]
 for half in ['selection','calibration']:
  seqs=metadata[half]['sequences'];systems[half]={};tables[half]={};predictions[half]={};done={}
  for row in models['outer_rows']:
   key=row['sha256']
   if key not in done:
    cursor=SequenceReadout(RUN/'outer_readout'/key/half,key+'|'+pins['outer_metadata_sha256']+'|'+half,len(seqs),lengths=[len(q['labels']) for q in seqs]);assert cursor.cursor==len(seqs);probs={q['id']:cursor.load(i) for i,q in enumerate(seqs)};done[key]=probs
   probs=done[key];params=choices['choices_by_checkpoint_sha256'][key]['params']
   for mode in ['raw','inner_B3']:
    name=row['row']+'|'+mode;pred={k:(v.argmax(1) if mode=='raw' else decode_with_prob('b3',v,{'decode':{'b3':params}})[0]) for k,v in probs.items()};m,t=evaluate(pred,half,metadata);systems[half][name]=m;tables[half][name]=t;predictions[half][name]=pred;records.append({**row,'half':half,'decoding':mode,**m})
    folder=RUN/'aggregate'/half/name;folder.mkdir(parents=True,exist_ok=True)
    for k,v in t.items():v.to_csv(folder/(k+'.private.csv'),index=False)
    if mode=='raw':
     old=next(r['raw_metrics'] for r in rawcheck['rows'] if r['row']==row['row'] and r['half']==half)
     assert all(m[k]==v for k,v in old.items()),'independent cached raw metrics must reproduce reader metrics exactly;16sample diagnostic is additive'
  # Original A08 frozen cache and original B3 parameters: never retuned here.
  for seed in [0,1,2]:
   pin=pins['A08_cache_pins'][half][str(seed)];assert sha(ROOT/pin['path'])==pin['sha256']
   with np.load(ROOT/pin['path']) as z:
    ids=z['sequence_ids'].tolist();assert ids==[q['id'] for q in seqs];probs={};logit_prob=torch.softmax(torch.from_numpy(z['logits']),-1).numpy();offset=z['sample_offsets'];saved=z['sample_probs']
    from echo_routing.evaluate.window_target import nearest_centre
    for i,q in enumerate(seqs):
     rows=np.flatnonzero(z['seq']==i);p=logit_prob[rows][nearest_centre(len(q['labels']),z['centre'][rows])];assert np.max(np.abs(p-saved[offset[i]:offset[i+1]]))<=1e-7;probs[q['id']]=p
   pred={k:decode_with_prob('b3',v,{'decode':{'b3':pin['original_params']}})[0] for k,v in probs.items()};m,t=evaluate(pred,half,metadata);want=basehistorical['systems'][f'A + B3 s{seed}'][half]
   for key in ['native_macro_f1','segment_f1_50_multi','boundary_recall_multi','boundary_precision_multi','boundary_f1_multi','native_internal_cuts_per_clip','same_class_cut_rate_multi','native_frame_accuracy','multi_frame_accuracy','segment_f1_50_pairs']:
    assert abs(m[key]-want[key])<1e-12,('A08 historical row mismatch',key)
   name=f'A08_original_B3_s{seed}';systems[half][name]=m;tables[half][name]=t;predictions[half][name]=pred;records.append({'row':name,'half':half,'decoding':'original_frozen_B3',**m})
 comparisons={};selected=read(PUB/'inner_lr_selection.json')['selected_joint_arm']
 for half in ['selection','calibration']:
  comparisons[half]={}
  for seed in [2026110257,2026110258]:
   for mode in ['raw','inner_B3']:
    left=f'seed{seed}_{selected}_fixed6000|{mode}';right=f'seed{seed}_F_fixed6000|{mode}';key=left+' - '+right;comparisons[half][key]={'delta':{k:systems[half][left][k]-systems[half][right][k] for k in ['native_macro_f1','segment_f1_50_multi','boundary_recall_multi','native_internal_cuts_per_clip','native_A4C_A5C_errors']},**boot_pair(predictions[half][left],predictions[half][right],metadata[half]['sequences'],tables[half][left],tables[half][right])}
   left=f'seed{seed}_{selected}_fixed6000|inner_B3';right='A08_original_B3_s0';key=left+' - '+right;comparisons[half][key]={'delta':{k:systems[half][left][k]-systems[half][right][k] for k in ['native_macro_f1','segment_f1_50_multi','boundary_recall_multi','native_internal_cuts_per_clip','native_A4C_A5C_errors']},**boot_pair(predictions[half][left],predictions[half][right],metadata[half]['sequences'],tables[half][left],tables[half][right])}
 write(PUB/'summary.json',{'status':'exploratory reused development, full matrix completed','systems':systems,'paired_comparisons':comparisons,'model_manifest_sha256':sha(PUB/'models_manifest.json'),'decoder_choices_sha256':sha(PUB/'decoder_choices.json'),'outer_inputs_sha256':sha(PUB/'outer_inputs.json'),'A08_historical_metrics_reproduced_all3seeds_bothhalves':True,'raw_reader_metrics_independently_reproduced':True,'primary_comparison':'matched6000 J-F raw bybothseeds;innerB3supplementary;A08all3reference (s0prespecifiedpairedref)','clinical_independent_validation':False,'test_Pediatric_Unity_quarantine_opened':False});pd.json_normalize(records).to_csv(PUB/'metrics.csv',index=False);print(json.dumps({'summary_written':True,'rows':len(records),'paired_comparisons':sum(map(len,comparisons.values()))}),flush=True)
if __name__=='__main__':main()
