#!/usr/bin/env python3
"""Frozen A08 original-native and auxiliary external-source evaluation (A50).

Private input manifests/predictions never leave the server. No training or target tuning.
"""
from __future__ import annotations

import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'third_party/external_test_pydeps'))

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image

from echo_routing.evaluate.external_native import CLASSES, majority_label, native_metrics, sample_indices, window_plan
from echo_routing.evaluate.window_target import nearest_centre
from echo_routing.ingest.frames import list_frame_paths
from echo_routing.temporal.joint_visual import crop_uint8, load_frame_u8
from echo_routing.temporal.baselines.b3_js_divergence import detect_boundaries, labels_from_boundaries
from echo_routing.vit import target_mvit as tm

CONFIG = ROOT / 'configs/experiments/external_original_test_v1.yaml'
RUN = ROOT / 'runs/external_original_test_v1'
PRIVATE = RUN / 'private'
RESULT = ROOT / 'docs/results/2026-10-30_external_original_tests'


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def write(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp'); tmp.write_text(json.dumps(obj, indent=2)); tmp.replace(path)


def now():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def resource(stage, start, cpu, **extra):
    r = {'stage': stage, 'ended_utc': now(), 'wall_seconds': time.time()-start,
         'process_cpu_seconds': time.process_time()-cpu, **extra}
    RUN.mkdir(parents=True, exist_ok=True)
    with (RUN/'resources.jsonl').open('a') as f: f.write(json.dumps(r)+'\n')


def prepare():
    t0, cpu = time.time(), time.process_time()
    cfg = yaml.safe_load(CONFIG.read_text()); datasets = {}
    source_files = [ROOT/'data/manifests/ev9v_native.csv']
    ev = pd.read_csv(source_files[0]); ev = ev[(ev.split=='test') & ev.family5_index.notna()]
    datasets['ev9v'] = [dict(id=str(q.video_id), group=str(q.video_id).rsplit('_',1)[0],
        true=int(q.family5_index), kind='jpeg', path=str(q.frames_dir), fps=30., n_frames=int(q.n_frames)) for q in ev.itertuples()]
    dr = ROOT/'data/echonet/dynamic/EchoNet-Dynamic'; f = dr/'FileList.csv'; source_files.append(f)
    dy = pd.read_csv(f); dy = dy[dy.Split=='TEST']
    datasets['dynamic'] = [dict(id=str(q.FileName), group=str(q.FileName), true=2, kind='video',
        path=str(dr/'Videos'/f'{q.FileName}.avi'), fps=float(q.FPS), n_frames=int(q.NumberOfFrames)) for q in dy.itertuples()]
    lr=ROOT/'data/echonet/lvh'; f=lr/'MeasurementsList.csv'; source_files.append(f); lv=pd.read_csv(f)
    if (lv.groupby('HashedFileName')['split'].nunique()>1).any(): raise RuntimeError('LVH recording crosses official splits')
    lv=lv[lv['split']=='test'].drop_duplicates('HashedFileName'); paths={p.name:p for p in lr.rglob('*.avi')}
    datasets['lvh']=[dict(id=str(q.HashedFileName),group=str(q.HashedFileName),true=0,kind='video',
        path=str(paths[str(q.HashedFileName) if str(q.HashedFileName).endswith('.avi') else str(q.HashedFileName)+'.avi']),
        fps=float(q.FPS),n_frames=int(q.Frames)) for q in lv.itertuples()]
    cr=ROOT/'data/camus/CAMUS_public'; f=cr/'database_split/subgroup_testing.txt'; source_files.append(f)
    cam=[]
    for line in f.read_text().splitlines():
        ident=line.strip()
        if not ident: continue
        p=cr/'database_nifti'/ident/f'{ident}_4CH_half_sequence.nii.gz'
        info=p.parent/f'{ident}_Info_4CH.cfg'
        if not info.exists(): info=p.parent/'Info_4CH.cfg'
        fields={k.strip():v.strip() for l in info.read_text().splitlines() if ':' in l for k,v in [l.split(':',1)]}
        if 'FrameRate' not in fields: raise RuntimeError('CAMUS source frame rate absent')
        cam.append(dict(id=ident,group=ident,true=2,kind='nifti',path=str(p),fps=float(fields['FrameRate']),n_frames=None))
        source_files.append(info)
    datasets['camus']=cam
    for name, rows in datasets.items():
        if len({q['id'] for q in rows}) != len(rows): raise RuntimeError('duplicate recording')
        for q in rows:
            p=Path(q['path'])
            if not p.exists(): raise FileNotFoundError(f'{name}: source media missing')
            if p.is_file(): q['source_size']=p.stat().st_size; q['source_mtime_ns']=p.stat().st_mtime_ns
    assert len(datasets['ev9v'])==800 and len(datasets['dynamic'])==1277
    manifest={'created_utc':now(),'datasets':datasets,'source_files':{str(p.relative_to(ROOT)):sha(p) for p in source_files}}
    f=PRIVATE/'manifest.json'
    if f.exists():
        old=json.loads(f.read_text())
        if old['datasets']!=manifest['datasets'] or old['source_files']!=manifest['source_files']: raise RuntimeError('immutable source manifest changed')
        manifest=old
    else: write(f,manifest)
    choices=json.loads((ROOT/cfg['choices']).read_text())['choices']
    weights={str(s):{'path':cfg['model_pattern'].format(seed=s),'sha256':sha(ROOT/cfg['model_pattern'].format(seed=s)),
        'b3':choices[f'all_s{s}']['params']} for s in cfg['seeds']}
    protocol={'created_utc':now(),'config_sha256':sha(CONFIG),'manifest_sha256':sha(f),'weights':weights,
        'counts':{k:len(v) for k,v in datasets.items()}, 'code_sha256':{str(p.relative_to(ROOT)):sha(p) for p in
        [Path(__file__),ROOT/'echo_routing/evaluate/external_native.py',ROOT/'echo_routing/vit/target_mvit.py',ROOT/'echo_routing/temporal/joint_visual.py']},
        'targets':{'ev9v':'original STFM source, shared five-family native endpoint', 'dynamic':'auxiliary A4C source check; NOT LVEF',
                   'lvh':'auxiliary PLAX source check; NOT wall thickness','camus':'auxiliary 4CH-coded source check; NOT quality/segmentation'}}
    pf=RUN/'protocol.json'
    if pf.exists():
        old=json.loads(pf.read_text())
        for k in ['config_sha256','manifest_sha256','weights','counts','code_sha256']:
            if old[k]!=protocol[k]: raise RuntimeError('frozen run protocol changed: '+k)
        protocol=old
    else: write(pf,protocol)
    write(RESULT/'preparation.json',protocol); resource('prepare',t0,cpu)
    print(json.dumps({'counts':protocol['counts'],'protocol_sha256':sha(pf)},indent=2),flush=True)


def load_frames(q):
    """Only media/timing fields are consumed. Reference labels never enter this path."""
    import cv2
    kind=q['kind']; fps=float(q['fps']); p=Path(q['path'])
    if kind=='jpeg':
        files=list_frame_paths(p); n=len(files)
        if n!=q['n_frames']: raise RuntimeError('JPEG frame count differs')
        idx=sample_indices(n,fps); frames=np.stack([load_frame_u8(files[int(i)],1.) for i in idx])
    elif kind=='video':
        cap=cv2.VideoCapture(str(p)); actual_fps=cap.get(cv2.CAP_PROP_FPS); n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if n!=int(q['n_frames']) or abs(actual_fps-fps)>max(.02,.001*fps):
            cap.release(); raise RuntimeError('video frame-count/rate metadata mismatch')
        idx=sample_indices(n,fps); wanted=set(idx.tolist()); selected={}; count=0
        while True:
            ok,im=cap.read()
            if not ok: break
            if count in wanted: selected[count]=crop_uint8(Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB)))
            count+=1
        cap.release()
        if count!=n or len(selected)!=len(wanted): raise RuntimeError('incomplete video decoding')
        frames=np.stack([selected[int(i)] for i in idx])
    else:
        import nibabel as nib
        im=nib.load(str(p)); a=np.asanyarray(im.dataobj)
        if a.ndim!=3: raise RuntimeError('CAMUS sequence is not a 3-D x/y/time array')
        a=a.transpose(2,1,0); n=len(a); idx=sample_indices(n,fps)
        if a.min()<0 or a.max()>255: raise RuntimeError('CAMUS values need an unspecified rescaling')
        frames=np.stack([crop_uint8(Image.fromarray(a[int(i)].astype(np.uint8)).convert('RGB')) for i in idx])
    return frames,{'n_frames':int(n),'fps':fps,'duration_s':n/fps,'sample_frame_indices':idx.tolist(),
                   'input_pixels_sha256':hashlib.sha256(frames.tobytes()).hexdigest()}


def model(seed, protocol, device):
    p=ROOT/protocol['weights'][str(seed)]['path']
    if sha(p)!=protocol['weights'][str(seed)]['sha256']: raise RuntimeError('A08 weight changed')
    m=tm.TargetMViT(tm.build_source_encoder(None));m.load_state_dict(torch.load(p,map_location='cpu',weights_only=True),strict=True)
    m.to(device).eval()
    for x in m.parameters():x.requires_grad_(False)
    return m


@torch.no_grad()
def predict(m, frames, device, batch=16):
    c,w,s=window_plan(len(frames)); out=[]; seconds=0.
    for b in range(0,len(c),batch):
        u8=torch.from_numpy(frames[w[b:b+batch]]).to(device)
        marker=tm.marker_planes(torch.as_tensor(s[b:b+batch],device=device))
        torch.cuda.synchronize(); t=time.time(); z=m(tm.to_rgb(u8),marker)
        torch.cuda.synchronize(); seconds+=time.time()-t;out.append(z.cpu().numpy())
    return np.concatenate(out),c,seconds


def labels_of(logits,centres,n,params):
    x=np.asarray(logits,float); x=x-x.max(1,keepdims=True);p=np.exp(x);p/=p.sum(1,keepdims=True)
    p=p[nearest_centre(n,centres)]
    cuts=detect_boundaries(p,int(params['half_window']),float(params['threshold']),int(params['nms_radius']))
    return {'argmax':p.argmax(1),'frozen_B3':labels_from_boundaries(p,cuts)}


def setup():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.cuda.set_per_process_memory_fraction(.6);return torch.device('cuda')


def check():
    """Read development parity only, never tune on target outcomes."""
    t0,cpu=time.time(),time.process_time();p=json.loads((RUN/'protocol.json').read_text());device=setup()
    import scripts.report_video_switch_head as rsh
    import scripts.report_raw_video_context as rr
    import scripts.prepare_raw_video_context as prv
    from echo_routing.config import load_paths
    from scripts.train_joint_visual_temporal import FramePaths
    a8=rsh.a08_ctx()
    import scripts.report_segment_objective as rso
    ev,seqs=a8.half('calibration',rso.DevAccess({'calibration'})); refs=sorted({r for q in seqs for r in q['refs']});at={r:i for i,r in enumerate(refs)}
    W=rr.half_windows(seqs,at,5); take=np.arange(min(16,len(W['seq'])))
    used=sorted(set(W['ref'][take].ravel().tolist())); chosen=[refs[i] for i in used]
    fp=FramePaths(prv.frame_dirs(load_paths(),{'source':{'frame_manifest_columns':['video_id','frames_dir']}},{r[0] for r in chosen}))
    fr=rr.decode_frames(chosen,fp,2); by={v:i for i,v in enumerate(used)}
    frames=np.stack([fr[[by[int(i)] for i in row]] for row in W['ref'][take]])
    result={}
    for seed in [0,1,2]:
        m=model(seed,p,device)
        with torch.no_grad():
            got=m(tm.to_rgb(torch.from_numpy(frames).to(device)),tm.marker_planes(torch.from_numpy(W['marker'][take]).to(device))).cpu().numpy()
        stored=ROOT/f'runs/video_within_view_v1/evaluation/calibration/all_s{seed}.npz'
        if not stored.exists():
            options=list((ROOT/'runs/video_within_view_v1').rglob(f'all_s{seed}.npz'))
            raise RuntimeError('stored development parity path must be bound: '+str([str(q.relative_to(ROOT)) for q in options]))
        with np.load(stored) as z:
            keys=list(z.files); expected=z['logits'][take]
        err=float(np.max(np.abs(got-expected)))
        if err>1e-4 or not np.array_equal(got.argmax(1),expected.argmax(1)):raise RuntimeError('development parity failed')
        result[str(seed)]={'windows':len(take),'max_logit_abs':err,'argmax_equal':True}
        del m;torch.cuda.empty_cache()
    write(RESULT/'checks.json',{'created_utc':now(),'development_parity':result,'protocol_sha256':sha(RUN/'protocol.json')})
    resource('check',t0,cpu);print(json.dumps(result),flush=True)


def run(datasets):
    t0,cpu=time.time(),time.process_time();p=json.loads((RUN/'protocol.json').read_text())
    if sha(CONFIG)!=p['config_sha256'] or sha(PRIVATE/'manifest.json')!=p['manifest_sha256']:raise RuntimeError('protocol identity changed')
    for f,h in p['code_sha256'].items():
        if sha(ROOT/f)!=h:raise RuntimeError('frozen code changed')
    if not (RESULT/'checks.json').exists():raise RuntimeError('development check required')
    manifest=json.loads((PRIVATE/'manifest.json').read_text()); device=setup(); models={s:model(s,p,device) for s in [0,1,2]}
    key=sha(RUN/'protocol.json');gpu=0.;windows=0
    for name in datasets:
        records=manifest['datasets'][name];out=PRIVATE/'predictions'/name;out.mkdir(parents=True,exist_ok=True)
        for i,q in enumerate(records):
            f=out/f'{i:05d}.npz'
            if f.exists():
                with np.load(f) as z:
                    if str(z['key'])!=key or str(z['id'])!=q['id']:raise RuntimeError('cached prediction identity mismatch')
                continue
            frames,receipt=load_frames(q); saved={'key':np.asarray(key),'id':np.asarray(q['id']),'receipt':np.asarray(json.dumps(receipt))}
            for s,m in models.items():
                lg,c,sec=predict(m,frames,device);gpu+=sec;windows+=len(c);saved[f'logits_s{s}']=lg
                labs=labels_of(lg,c,len(frames),p['weights'][str(s)]['b3'])
                for dec,lab in labs.items():saved[f'{dec}_s{s}']=lab
            saved['centres']=c
            tmp=f.with_suffix('.tmp.npz');np.savez_compressed(tmp,**saved);tmp.replace(f)
            write(RUN/'progress.json',{'phase':'running','dataset':name,'done':i+1,'total':len(records),'gpu_seconds':gpu,'windows':windows,'updated_utc':now(),'pid':os.getpid()})
            if (i+1)%20==0: print(f'{name}: {i+1}/{len(records)}, GPU forward {gpu:.1f}s',flush=True)
        report([name])
    for s in models:
        if sha(ROOT/p['weights'][str(s)]['path'])!=p['weights'][str(s)]['sha256']:raise RuntimeError('frozen weight file changed')
    resource('run',t0,cpu,gpu_forward_seconds=gpu,windows=windows,datasets=datasets)
    write(RUN/'progress.json',{'phase':'complete','datasets':datasets,'updated_utc':now(),'pid':os.getpid()})


def report(datasets):
    manifest=json.loads((PRIVATE/'manifest.json').read_text());key=sha(RUN/'protocol.json');summary={}
    for name in datasets:
        rows={f'{dec}_s{s}':[] for dec in ['argmax','frozen_B3'] for s in [0,1,2]}
        for i,q in enumerate(manifest['datasets'][name]):
            f=PRIVATE/'predictions'/name/f'{i:05d}.npz'
            with np.load(f) as z:
                if str(z['key'])!=key or str(z['id'])!=q['id']:raise RuntimeError('evaluation cache identity mismatch')
                rec=json.loads(str(z['receipt']))
                for k in rows:
                    lab=z[k];rows[k].append({'true':q['true'],'pred':majority_label(lab),'n_samples':len(lab),
                        'cuts':int((lab[1:]!=lab[:-1]).sum()),'duration_s':rec['duration_s']})
        summary[name]={k:native_metrics(v) for k,v in rows.items()}
    dest=RESULT/'native_results.json';old=json.loads(dest.read_text()) if dest.exists() else {}
    old.update(summary);write(dest,old);print(json.dumps({'completed':{k:len(manifest['datasets'][k]) for k in datasets}}),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--stage',required=True,choices=['prepare','check','run','report'])
    ap.add_argument('--datasets',nargs='+',default=['ev9v','dynamic','lvh','camus']);a=ap.parse_args()
    if a.stage=='prepare':prepare()
    elif a.stage=='check':check()
    elif a.stage=='run':run(a.datasets)
    else:report(a.datasets)
