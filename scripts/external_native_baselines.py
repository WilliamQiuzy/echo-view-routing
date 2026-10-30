#!/usr/bin/env python3
"""STFM's full-native original test procedure on all 888 EV9V test cines.

Existing numeric-order s100 weights; legacy released fusion and corrected
prior-once fusion are both predeclared. This is NOT a newly trained 61ea9b6 model.
"""
from pathlib import Path
import argparse,json,sys,time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision.transforms import v2
from echo_routing.evaluate.external_baselines import grouped_probabilities,stfm_probabilities
from echo_routing.evaluate.external_native import native_metrics
from echo_routing.ingest.frames import list_frame_paths
from echo_routing.temporal.windowed import RAW9_ORDER,RAW9_TO_FAMILY5
import scripts.external_original_tests as e

RUN=e.RUN;PRIVATE=e.PRIVATE;RESULT=e.RESULT
WPATH=ROOT/'models_frozen/stfm_numeric/s100/model.ckpt'


def prepare():
    df=pd.read_csv(ROOT/'data/manifests/ev9v_native.csv');df=df[df.split=='test']
    rows=[dict(id=str(q.video_id),true9=int(q.raw9_index),true5=None if pd.isna(q.family5_index) else int(q.family5_index),
        path=str(q.frames_dir),n_frames=int(q.n_frames)) for q in df.itertuples()]
    assert len(rows)==888 and sum(q['true5'] is not None for q in rows)==800
    e.write(PRIVATE/'baseline_manifest.json',rows)
    p={'created_utc':e.now(),'weights':str(WPATH.relative_to(ROOT)),'weights_sha256':e.sha(WPATH),
       'manifest_sha256':e.sha(PRIVATE/'baseline_manifest.json'),'code_sha256':e.sha(__file__),
       'mapping_sha256':e.sha(ROOT/'echo_routing/evaluate/external_baselines.py'),
       'conditions':['legacy_per_key_prior','corrected_prior_once_same_old_weights'],
       'input':'10 full-native evenly spaced key frames; 5 raw frames at interval5 per key; numeric ordering; authors ImageNet transform',
       'shared_task':'sum PSAX probabilities; preserve unsupported PMPALA as sixth error category; no mass deletion',
       'model_identity':'existing project STFM numeric s100, not latest-recipe retraining'}
    f=RUN/'baseline_protocol.json'
    if f.exists():
        old=json.loads(f.read_text())
        if any(old[k]!=p[k] for k in ['weights_sha256','manifest_sha256','code_sha256','mapping_sha256']):raise RuntimeError('immutable baseline identity changed')
    else:e.write(f,p)
    e.write(RESULT/'baseline_preparation.json',json.loads(f.read_text()))


def run():
    t0,cpu=time.time(),time.process_time();p=json.loads((RUN/'baseline_protocol.json').read_text())
    if e.sha(WPATH)!=p['weights_sha256'] or e.sha(__file__)!=p['code_sha256']:raise RuntimeError('baseline identity changed')
    rows=json.loads((PRIVATE/'baseline_manifest.json').read_text());device=e.setup()
    sys.path.insert(0,str(ROOT/'third_party/stfm_numeric'))
    from models_enhanced import create_enhanced_stfm
    net=create_enhanced_stfm(num_classes=9,backbone='resnet18',hidden_size=512,num_layers=2,embed_dims=128)
    ck=torch.load(WPATH,map_location='cpu',weights_only=False);net.load_state_dict(ck['state_dict'],strict=True);net.to(device).eval()
    for par in net.parameters():par.requires_grad_(False)
    tf=v2.Compose([v2.Resize(256),v2.CenterCrop(224),v2.ToImage(),v2.ToDtype(torch.float32,scale=True),v2.Normalize([.485,.456,.406],[.229,.224,.225])])
    out=PRIVATE/'baseline_predictions';out.mkdir(exist_ok=True);key=e.sha(RUN/'baseline_protocol.json');gpu=0
    for i,q in enumerate(rows):
        dest=out/f'{i:05d}.npz'
        if dest.exists():
            with np.load(dest) as z:
                if str(z['key'])!=key or str(z['id'])!=q['id']:raise RuntimeError('baseline cache mismatch')
            continue
        files=list_frame_paths(Path(q['path']));n=len(files)
        if n!=q['n_frames']:raise RuntimeError('baseline frame count differs')
        keys=torch.linspace(0,n-1,min(10,n)).long().tolist();keys += [n-1]*(10-len(keys))
        clips=[[min(max(k-10+j*5,0),n-1) for j in range(5)] for k in keys]
        needed=sorted(set(keys+[j for c in clips for j in c]));x={}
        for j in needed:
            with Image.open(files[j]) as im:x[j]=tf(im.convert('RGB'))
        a=torch.stack([x[j] for j in keys]).to(device);b=torch.stack([torch.stack([x[j] for j in c]) for c in clips]).to(device)
        torch.cuda.synchronize();t=time.time()
        with torch.no_grad():lg=net((a,b)).float().cpu().numpy()
        torch.cuda.synchronize();gpu+=time.time()-t
        tmp=dest.with_suffix('.tmp.npz');np.savez_compressed(tmp,key=key,id=q['id'],logits=lg);tmp.replace(dest)
        if (i+1)%50==0:print(f'STFM native {i+1}/888 GPU {gpu:.1f}s',flush=True)
    report();e.resource('STFM-original-native',t0,cpu,gpu_forward_seconds=gpu)


def report():
    rows=json.loads((PRIVATE/'baseline_manifest.json').read_text());key=e.sha(RUN/'baseline_protocol.json');result={}
    mapping={j:RAW9_TO_FAMILY5[n] for j,n in enumerate(RAW9_ORDER) if RAW9_TO_FAMILY5[n] is not None}
    for condition in ['legacy_per_key_prior','corrected_prior_once_same_old_weights']:
        cm=np.zeros((9,9),int);shared=[]
        for i,q in enumerate(rows):
            with np.load(PRIVATE/'baseline_predictions'/f'{i:05d}.npz') as z:
                if str(z['key'])!=key:raise RuntimeError('baseline identity mismatch')
                prob=stfm_probabilities(z['logits'],condition.startswith('corrected'))
            cm[q['true9'],prob.argmax()]+=1
            if q['true5'] is not None:
                k=int(grouped_probabilities(prob[None],mapping).argmax(1)[0]);pred=-1 if k==5 else k
                shared.append(dict(true=q['true5'],pred=pred,n_samples=1,cuts=0,duration_s=q['n_frames']/30))
        tp=np.diag(cm);den=cm.sum(0)+cm.sum(1);f1=np.divide(2*tp,den,out=np.zeros(9,float),where=den>0)
        result[condition]={'original_nine_code':{'n_clips':888,'accuracy':float(tp.sum()/888),'macro_f1':float(f1.mean()),'confusion':cm.tolist()},
            'shared_five_family':native_metrics(shared),'cut_note':'Whole-file classification emits no internal cut by construction; not a stability finding.'}
    e.write(RESULT/'stfm_native_results.json',result)
    print(json.dumps({k:{'nine_acc':v['original_nine_code']['accuracy'],'five_macro':v['shared_five_family']['five_class_macro_f1']} for k,v in result.items()}),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['prepare','run','report'],required=True);a=ap.parse_args()
    {'prepare':prepare,'run':run,'report':report}[a.stage]()
