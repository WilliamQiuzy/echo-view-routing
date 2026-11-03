#!/usr/bin/env python3
"""A58 frozen producers on genuine complete-stream context, never concatenated logits."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import sys,time,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from scripts.structured_common import ROOT,RUN,PUB,CFG,read,write,sha,verify_freeze,sequences
from echo_routing.temporal.structured_inputs import windows,check_probabilities
from echo_routing.evaluate.window_target import nearest_centre,window_frames
from echo_routing.vit import target_mvit as tm
from echo_routing.temporal.dense_sequence import DenseSequence,chunk_starts,chunk_positions,stitch_probabilities
from echo_routing.temporal import jepa21_source
from echo_routing.temporal.dense_readout import SequenceReadout,ReadoutPixels

def determinism():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False

def build_a08(device='cpu'):
    pin=read(CFG)['source']['a08'];assert sha(ROOT/pin['path'])==pin['sha256'];state=torch.load(ROOT/pin['path'],map_location='cpu',weights_only=True,mmap=True);model=tm.build_model(None,tm.initial_new_state(0));model.load_state_dict(state,strict=True)
    assert all(torch.equal(model.state_dict()[k],v) for k,v in state.items())
    for p in model.parameters():p.requires_grad_(False)
    return model.eval().to(device)

def build_jepa(device='cpu'):
    pin=read(CFG)['source']['jepa'];assert sha(ROOT/pin['path'])==pin['sha256'];source=read(ROOT/pin['source_manifest']);assert sha(ROOT/pin['source_manifest'])==pin['source_manifest_sha256']
    for p,h in source['loaded_model_subset_SHA'].items():assert sha(jepa21_source.UPSTREAM/p)==h
    model=DenseSequence(jepa21_source.build_encoder(),'J-high',seed=2026110257,inactive_names=source['inactive_video_names']);z=torch.load(ROOT/pin['path'],map_location='cpu',weights_only=True,mmap=True);model.load_state_dict(z['model'],strict=True);assert all(torch.equal(model.state_dict()[k],v) for k,v in z['model'].items())
    for p in model.parameters():p.requires_grad_(False)
    return model.eval().to(device)

@torch.no_grad()
def a08_windows(model,q,pixels,centres,device='cuda',batch=32):
    centres=np.asarray(centres,int);out=[]
    for begin in range(0,len(centres),batch):
        cs=centres[begin:begin+batch];ss=[window_frames(int(c),len(q['refs']),16,16) for c in cs];u=np.stack([[pixels.frame(*q['refs'][int(i)]) for i in s] for s in ss]);markers=torch.tensor([tm.marker_slot(s,int(c)) for s,c in zip(ss,cs)],device=device);out.append(model(tm.to_rgb(torch.from_numpy(u).to(device)),tm.marker_planes(markers)).float().cpu().numpy())
    return np.concatenate(out)

@torch.no_grad()
def a08_sequence(model,q,pixels,device='cuda',batch=32):
    n=len(q['refs']);w=windows(n,2);z=a08_windows(model,q,pixels,w['centre'],device,batch);p=torch.softmax(torch.from_numpy(z),1).numpy()[nearest_centre(n,w['centre'])];check_probabilities(p,n);return p

@torch.no_grad()
def jepa_sequence(model,q,pixels,device='cuda'):
    n=len(q['refs']);starts=chunk_starts(n);chunks=[]
    for begin in range(0,len(starts),2):
        frames=[];masks=[]
        for start in starts[begin:begin+2]:
            indices,mask=chunk_positions(n,start);frames.append(np.stack([pixels.frame(*q['refs'][int(i)]) for i in indices]));masks.append(mask)
        chunks.extend(model.probabilities(torch.from_numpy(np.stack(frames)).to(device),torch.from_numpy(np.stack(masks)).to(device)).cpu().numpy())
    p=stitch_probabilities(chunks,n).astype(np.float32);check_probabilities(p,n);return p
