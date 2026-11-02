#!/usr/bin/env python3
"""A57 frozen-model entire-timeline outer readout with safe committed-prefix resume."""
import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES','0');os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8');os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import copy,json,resource,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from scripts.train_dense_sequence import RUN,PUB,CFG,read,write,sha,build,load_model_weights,budget_guard
from echo_routing.temporal import dense_runtime as data,jepa21_source as source
from echo_routing.temporal.dense_readout import SequenceReadout,ReadoutPixels
from echo_routing.evaluate.dev_metrics import evaluate_predictions

def main():
 start=time.time();torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
 freeze=read(RUN/'outer_readout_freeze.json')
 for p,h in freeze['objects'].items():assert sha(ROOT/p)==h,p
 src=read(PUB/'source.json')
 for p,h in src['loaded_model_subset_SHA'].items():assert sha(source.UPSTREAM/p)==h
 cfg=read(CFG);assert sha(ROOT/cfg['source_state'])==src['target_encoder_sha256'];models=read(PUB/'models_manifest.json');pins=read(PUB/'outer_inputs.json');assert models['source_sha256']==src['target_encoder_sha256'];assert pins['model_manifest_sha256']==sha(PUB/'models_manifest.json') and sha(RUN/'outer_metadata.private.npy')==pins['outer_metadata_sha256'];metadata=np.load(RUN/'outer_metadata.private.npy',allow_pickle=True).item();completed={};results=[]
 for row in models['outer_rows']:
  assert sha(ROOT/row['path'])==row['sha256'];local=copy.deepcopy(cfg);local['seed']=row['seed'];model=None
  for half in ['selection','calibration']:
   key=row['sha256']+'|'+half
   if key in completed:results.append({**row,'half':half,'reused_same_checkpoint':True,'raw_metrics':completed[key]});continue
   half_start=time.time();seqs=metadata[half]['sequences'];pixels=ReadoutPixels(seqs);identity=row['sha256']+'|'+pins['outer_metadata_sha256']+'|'+half;folder=RUN/'outer_readout'/row['sha256']/half;cursor=SequenceReadout(folder,identity,len(seqs),lengths=[len(q['labels']) for q in seqs]);initial_cursor=cursor.cursor
   if cursor.cursor<len(seqs):
    if model is None:model=build(row['arm'],local);load_model_weights(model,ROOT/row['path']);model.eval()
    for i in range(cursor.cursor,len(seqs)):
     q=seqs[i];cursor.run_one(i,lambda q=q:data.predict_sequence(model,q,pixels))
     if (i+1)%100==0:budget_guard();write(RUN/'runtimecoordination.json',{'phase':'outer_readout','row':row['row'],'half':half,'model_pid':os.getpid(),'sequences_done':i+1,'sequences':len(seqs),'epoch':time.time(),'next_action':'finish frozen rows; no training or new choices from outer results'})
   cost={'new_sequences':len(seqs)-initial_cursor,'previously_committed_sequences':initial_cursor,'decode_model_stitch_cache_wall_seconds_this_process':time.time()-half_start,'includes_model_coldstart_on_first_half':half=='selection','full_fresh_readout':initial_cursor==0,'GPU_peak_reserved_bytes':torch.cuda.max_memory_reserved()};write(folder/'inference_cost.json',cost)
   predictions={q['id']:cursor.load(i).argmax(1) for i,q in enumerate(seqs)};metrics,tables=evaluate_predictions(predictions,metadata[half]['eval_set'],10,.5);subset=[q for q in seqs if q['kind'] in ['native','multi']];extras,_=data.metrics(subset,{q['id']:predictions[q['id']] for q in subset});metrics.update({k:v for k,v in extras.items() if k.startswith('short_') or k.startswith('native_A4C') or k.startswith('native_A5C')});write(folder/'raw_metrics.json',metrics)
   for name,table in tables.items():table.to_csv(folder/(name+'.private.csv'),index=False)
   completed[key]=metrics;results.append({**row,'half':half,'reused_same_checkpoint':False,'raw_metrics':metrics,'inference_cost':cost})
  if model is not None:del model;torch.cuda.empty_cache()
 write(PUB/'raw_outer_results.json',{'status':'complete fixed frozen exploratory rows','rows':results,'model_manifest_sha256':sha(PUB/'models_manifest.json'),'readout_freeze_git':freeze['git_commit'],'wall_seconds':time.time()-start,'new_training_updates_during_outer':0});print(json.dumps({'outer_rows_completed':len(results),'unique_checkpoint_halves':len(completed)}),flush=True)
if __name__=='__main__':
 start=time.time();ok=False
 try:main();ok=True
 finally:
  r=resource.getrusage(resource.RUSAGE_SELF);write(RUN/'outer_readout_latest_process.json',{'ok':ok,'pid':os.getpid(),'CPU_core_seconds':r.ru_utime+r.ru_stime,'GPU_job_wall_seconds':time.time()-start})
