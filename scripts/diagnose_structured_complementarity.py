#!/usr/bin/env python3
"""Once-only A58 OPEN-cache complementarity; no model/media or deployable oracle."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import sys,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from scripts.structured_common import ROOT,RUN,PUB,CFG,read,write,sha,verify_freeze
from scripts.aggregate_dense_sequence import evaluate
from echo_routing.temporal.structured_inputs import error_overlap,check_probabilities
from echo_routing.evaluate.window_target import nearest_centre
from echo_routing.temporal.baselines.registry import decode_with_prob
from echo_routing.temporal.segments import labels_to_segments
from echo_routing.evaluate.failure_units import match_segments

def old_probabilities(half,seqs,manifest,cfg):
    pins=manifest['diagnostic_inputs'];pin=pins['a08'][half];assert sha(ROOT/pin['path'])==pin['sha256'] and pin['original_model_sha256']==cfg['source']['a08']['sha256'];a={}
    with np.load(ROOT/pin['path']) as z:
        assert z['sequence_ids'].tolist()==[q['id'] for q in seqs];assert str(z['identity'])==pin['identity'];p=torch.softmax(torch.from_numpy(z['logits']),1).numpy()
        for i,q in enumerate(seqs):
            ix=np.flatnonzero(z['seq']==i);v=p[ix][nearest_centre(len(q['labels']),z['centre'][ix])];saved=z['sample_probs'][z['sample_offsets'][i]:z['sample_offsets'][i+1]];assert np.max(np.abs(v-saved))<=1e-7;check_probabilities(v,len(q['labels']));a[q['id']]=v
    pin=pins['jepa'][half];folder=ROOT/pin['path'];assert sha(folder/'committed.jsonl')==pin['journal_sha256'] and sha(folder/'identity.json')==pin['identity_sha256'];identity=cfg['source']['jepa']['sha256']+'|'+cfg['data']['old_outer_metadata_sha256']+'|'+half;assert read(folder/'identity.json')=={'identity':identity,'total':len(seqs)};rows=[json.loads(s) for s in (folder/'committed.jsonl').read_text().splitlines()];assert len(rows)==len(seqs) and not (folder/'pending.json').exists();b={}
    for i,(row,q) in enumerate(zip(rows,seqs)):
        path=folder/f'sequence{i:06d}.private.npz';assert row['index']==i and row['identity']==identity and sha(path)==row['sha256'];assert row['positions']==len(q['labels'])
        with np.load(path) as z:v=z['prob'].copy()
        check_probabilities(v,len(q['labels']));b[q['id']]=v
    return a,b

def paired_counts(left,right,seqs):
    out={'native_both_correct':0,'native_both_wrong':0,'native_left_only_correct':0,'native_right_only_correct':0,'multi_true_segments_left_only_recovered':0,'multi_true_segments_right_only_recovered':0,'short16_left_only_recovered':0,'short16_right_only_recovered':0,'short10_left_only_recovered':0,'short10_right_only_recovered':0}
    for q in seqs:
        if q['kind']=='native':
            x=int(np.bincount(left[q['id']],minlength=5).argmax())==int(q['labels'][0]);y=int(np.bincount(right[q['id']],minlength=5).argmax())==int(q['labels'][0]);key='native_both_correct' if x and y else 'native_both_wrong' if not x and not y else 'native_left_only_correct' if x else 'native_right_only_correct';out[key]+=1
        elif q['kind']=='multi':
            truth=labels_to_segments(q['labels']);a=set(match_segments(labels_to_segments(left[q['id']]),truth));b=set(match_segments(labels_to_segments(right[q['id']]),truth));out['multi_true_segments_left_only_recovered']+=len(a-b);out['multi_true_segments_right_only_recovered']+=len(b-a)
            for n in [10,16]:out[f'short{n}_left_only_recovered']+=sum(truth[i].end-truth[i].start<=n for i in a-b);out[f'short{n}_right_only_recovered']+=sum(truth[i].end-truth[i].start<=n for i in b-a)
    return out

def main():
    freeze=verify_freeze('diagnosis_freeze.json');cfg=read(CFG);manifest=read(PUB/'input_manifest.json');path=ROOT/cfg['data']['old_outer_metadata'];assert sha(path)==cfg['data']['old_outer_metadata_sha256'];metadata=np.load(path,allow_pickle=True).item();halves={}
    for half in ['selection','calibration']:
        seqs=metadata[half]['sequences'];a,b=old_probabilities(half,seqs,manifest,cfg);pred={'A08_original_stride5_raw':{k:v.argmax(1) for k,v in a.items()},'JEPA_core_fixed6000_raw':{k:v.argmax(1) for k,v in b.items()},'A08_original_stride5_B3':{k:decode_with_prob('b3',v,{'decode':{'b3':cfg['source']['a08']['original_B3']}})[0] for k,v in a.items()}};overlaps={};paired={}
        for leftname in ['A08_original_stride5_raw','A08_original_stride5_B3']:
            left,right=pred[leftname],pred['JEPA_core_fixed6000_raw'];groups={};oracle={}
            for q in seqs:
                y=np.asarray(q['labels']);l,rp=left[q['id']],right[q['id']];oracle[q['id']]=np.where((l!=y)&(rp==y),rp,l)
                if q['kind']=='native':masks={'native':np.ones(len(y),bool)}
                elif q['kind']=='multi':
                    changes=np.flatnonzero(y[1:]!=y[:-1])+1;near=np.zeros(len(y),bool) if not len(changes) else np.abs(np.arange(len(y))[:,None]-changes[None,:]).min(1)<=5;masks={'multi_interior_distance_gt5':~near,'multi_near_change_distance_le5':near}
                else:continue
                for group,mask in masks.items():
                    overlap=error_overlap(l,rp,y,mask);bucket=groups.setdefault(group,{k:0 for k in overlap})
                    for k,v in overlap.items():bucket[k]+=v
            overlaps[leftname+'_versus_JEPA_raw']=groups;paired[leftname+'_versus_JEPA_raw']=paired_counts(left,right,seqs);pred['label_assisted_position_oracle_'+leftname]=oracle
        metrics={name:evaluate(p,half,metadata)[0] for name,p in pred.items()};reference=read(ROOT/'docs/results/2026-11-02_dense_target_sequence/A08_reference_metrics.json')['systems'][half]['A08_original_B3_s0'];assert metrics['A08_original_stride5_B3']==reference;halves[half]={'systems':metrics,'position_error_overlap':overlaps,'paired_native_and_fixed_IoU50_segments':paired,'native_sessions':36,'multi_streams':120}
    out={'status':'completed cache-only exploratory complementarity','halves':halves,'source_freeze_git':freeze['git_commit'],'source_input_manifest_sha256':sha(PUB/'input_manifest.json'),'A08_reference_metrics_exactly_reproduced':True,'oracle_is_position_accuracy_upper_envelope_only':True,'oracle_is_segmentF1_upper_bound':False,'oracle_is_deployable_or_evidence_of_learnable_gain':False,'oracle_note':'label-assistedpositionchoicesmayalterRLEandshortsegmentsnonmonotonically','no_performance_gate_cancels_admitted_matrix':True,'model_forwards':0,'optimizer_updates':0,'media_decodes':0,'clinical_independent_validation':False,'epoch':time.time()};write(PUB/'complementarity.json',out);print(json.dumps({'complementarity_completed':True,'new_model_forwards':0,'position_overlaps':{h:x['position_error_overlap'] for h,x in halves.items()}}),flush=True)
if __name__=='__main__':main()
