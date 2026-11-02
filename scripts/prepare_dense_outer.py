#!/usr/bin/env python3
"""A57 post-model-freeze metadata and source-cache qualification; CPU only."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import json,sys,time
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from scripts.train_dense_sequence import RUN,PUB,read,write,sha

def prepare():
 models=read(PUB/'models_manifest.json');assert models['passed'] and models['formal_physical_updates']==32000 and models['new_external_development_forwards']==0
 assert read(PUB/'decoder_choices.json')['model_manifest_sha256']==sha(PUB/'models_manifest.json')
 for q in models['outer_rows']:assert sha(ROOT/q['path'])==q['sha256']
 # Reuse the unchanged metadata-only half adapter; it rejects non-validation sources.
 from scripts import report_video_within_view as a08,trained_token_readout as ttr
 from scripts._common import load_cfg
 from echo_routing.temporal.dense_sequence import chunk_starts
 from echo_routing.temporal.dense_runtime import verify_partition
 name='configs/experiments/video_within_view_v1.yaml';ctx=a08.Ctx(load_cfg(SimpleNamespace(config=name,set=[])),name);choices=read(ROOT/ctx.relpath);historical=read(ROOT/'docs/results/2026-10-16_video_within_view/summary.json');out={};private={};basepins={}
 for half in ['selection','calibration']:
  ev,seqs=ctx.half(half,a08.DevAccess({half}));streams={s.recipe.recipe_id:s for s in [*ev.pairs,*ev.multi]}
  for q in seqs:q['joins']=np.zeros(0,int) if q['kind']=='native' else streams[q['id']].joins
  verify_partition(seqs,set(ctx.proto()['validation'][half]['clips']));private[half]={'eval_set':ev,'sequences':seqs};W=ttr.half_windows(seqs,5);ids=[q['id'] for q in seqs];basepins[half]={}
  for seed in [0,1,2]:
   key=f'all_s{seed}';path=ROOT/f'runs/video_within_view_v1/evaluation/{half}/{key}.npz';assert sha(ROOT/f'models_frozen/video_within_view_v1/{key}/final.pt')==choices['provenance']['runs'][key]['final_sha256']
   with np.load(path) as z:
    assert str(z['identity'])==choices['provenance']['runs'][key]['identity'];ttr.check_windows(W,ids,z,'A08cached output');assert z['logits'].shape==(len(W['seq']),5) and np.isfinite(z['logits']).all();assert np.array_equal(z['order'],np.broadcast_to(np.arange(16,dtype=np.int8),z['order'].shape))
   basepins[half][str(seed)]={'path':str(path.relative_to(ROOT)),'sha256':sha(path),'identity':choices['provenance']['runs'][key]['identity'],'original_params':choices['choices'][key]['params'],'original_model_sha256':choices['provenance']['runs'][key]['final_sha256']}
  out[half]={'sequences':len(seqs),'native_clips':len(ev.natives),'multi_streams':len(ev.multi),'pair_streams':len(ev.pairs),'original_positions':sum(len(q['labels']) for q in seqs),'chunks':sum(len(chunk_starts(len(q['labels']))) for q in seqs),'sessions':len({v.rsplit('_',1)[0] for v in ctx.proto()['validation'][half]['clips']})}
 np.save(RUN/'outer_metadata.private.npy',private,allow_pickle=True);total=sum(v['chunks'] for v in out.values());assert total<=14000,'reforecast remaining caps before any new outer model forward'
 record={'passed':True,'model_manifest_sha256':sha(PUB/'models_manifest.json'),'outer_metadata_sha256':sha(RUN/'outer_metadata.private.npy'),'counts':out,'total_chunks_both_halves':total,'A08_cache_pins':basepins,'A08_choices_sha256':sha(ROOT/ctx.relpath),'historical_A08_summary_sha256':sha(ROOT/'docs/results/2026-10-16_video_within_view/summary.json'),'metadata_pins':{k:ctx.a03src()[k]['sha256'] for k in ['protocol','clip_index','window_plans']},'new_model_forwards':0,'official_test_Pediatric_Unity_quarantine_reads':0,'development_reuse':'open exploratory reuse, not fresh validation','epoch':time.time()};write(PUB/'outer_inputs.json',record);print(json.dumps({'outer_metadata_ready':True,'counts':out,'chunks':total}),flush=True)
if __name__=='__main__':prepare()
