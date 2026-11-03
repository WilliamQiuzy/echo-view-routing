#!/usr/bin/env python3
"""Early cache-only A08 reproduction after all new-model inner choices freeze."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import sys,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from scripts.train_dense_sequence import RUN,PUB,read,write,sha
from scripts.aggregate_dense_sequence import evaluate
from echo_routing.evaluate.window_target import nearest_centre
from echo_routing.temporal.baselines.registry import decode_with_prob

def main():
    models=read(PUB/'models_manifest.json');choices=read(PUB/'decoder_choices.json');assert models['passed'] and choices['model_manifest_sha256']==sha(PUB/'models_manifest.json') and choices['external_development_reads_at_choices']==0
    pins=read(PUB/'outer_inputs.json');assert sha(RUN/'outer_metadata.private.npy')==pins['outer_metadata_sha256'];metadata=np.load(RUN/'outer_metadata.private.npy',allow_pickle=True).item();historical=read(ROOT/'docs/results/2026-10-16_video_within_view/summary.json');systems={}
    for half in ['selection','calibration']:
        seqs=metadata[half]['sequences'];systems[half]={}
        for seed in [0,1,2]:
            pin=pins['A08_cache_pins'][half][str(seed)];assert sha(ROOT/pin['path'])==pin['sha256'];probs={}
            with np.load(ROOT/pin['path']) as z:
                assert z['sequence_ids'].tolist()==[q['id'] for q in seqs];p=torch.softmax(torch.from_numpy(z['logits']),-1).numpy();offset=z['sample_offsets'];saved=z['sample_probs']
                for i,q in enumerate(seqs):
                    rows=np.flatnonzero(z['seq']==i);v=p[rows][nearest_centre(len(q['labels']),z['centre'][rows])];assert np.max(np.abs(v-saved[offset[i]:offset[i+1]]))<=1e-7;probs[q['id']]=v
            pred={k:decode_with_prob('b3',v,{'decode':{'b3':pin['original_params']}})[0] for k,v in probs.items()};metrics,_=evaluate(pred,half,metadata);want=historical['systems'][f'A + B3 s{seed}'][half]
            for k in ['native_macro_f1','segment_f1_50_multi','boundary_recall_multi','boundary_precision_multi','boundary_f1_multi','native_internal_cuts_per_clip','same_class_cut_rate_multi','native_frame_accuracy','multi_frame_accuracy','segment_f1_50_pairs']:assert abs(metrics[k]-want[k])<1e-12,(half,seed,k)
            systems[half][f'A08_original_B3_s{seed}']=metrics
    write(PUB/'A08_reference_metrics.json',{'status':'pinned original model/cache/B3 reproduced on identical outer axes before candidate summary','all3seeds_bothhalves_historical_metrics_reproduced':True,'systems':systems,'outer_inputs_sha256':sha(PUB/'outer_inputs.json'),'decoder_choices_sha256':sha(PUB/'decoder_choices.json'),'new_model_forwards':0,'primary_paired_reference':'A08_original_B3_s0','epoch':time.time()});print(json.dumps({'A08_reproduced_3seeds_2halves':True,'new_model_forwards':0}),flush=True)
if __name__=='__main__':main()
