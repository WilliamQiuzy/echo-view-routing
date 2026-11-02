#!/usr/bin/env python3
"""A57 split-pure metadata/plans and a fixed actual2x64 FIT smoke batch."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse,hashlib,json,resource,sys,time
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts import prepare_video_feature_temporal as legacy
from scripts import train_interval_visual_adaptation as pixels
from scripts._common import load_cfg
from echo_routing.temporal import dense_sequence as dense,video_sequence_features as refs
from echo_routing.compose.recipe_builder import Pool,make_multi_recipes
from echo_routing.evaluate.failure_units import session_of
RUN=ROOT/'runs/a57_dense_sequence_v1';PUB=ROOT/'docs/results/2026-11-02_dense_target_sequence';SEED=2026110257
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(16*1024*1024),b''):h.update(b)
 return h.hexdigest()
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2)+'\n')
def legacy_data():
 name='configs/experiments/video_feature_temporal_v1.yaml';ctx=legacy.Ctx(load_cfg(SimpleNamespace(config=name,set=[])),name);return ctx,legacy.TrainingData(ctx)
def metadata():
 assert not (PUB/'input_metadata.json').exists(),'metadata alreadysealed'
 ctx,data=legacy_data();assert len(data.clips)==3364;session=sorted({session_of(v) for v in data.clips});perm=np.random.default_rng(SEED).permutation(len(session));cal_sessions={session[i] for i in perm[:round(.2*len(session))]};roles={'fit':[v for v in data.clips if session_of(v) not in cal_sessions],'inner_cal':[v for v in data.clips if session_of(v) in cal_sessions]};idx=refs.training_index(data.dir/'index.csv');rc,mc=ctx.cfg['recipes'],ctx.cfg.get('multi',{});groups={};counts={}
 for role,clips in roles.items():
  allow=set(clips);sub=idx[idx.video_id.isin(allow)];sequences=[]
  for v in sorted(clips):
   f=data.frame_idx(v);sequences.append({'id':v,'kind':'native','refs':refs.native_refs(v,f),'labels':np.full(len(f),data.label[v],np.int64),'joins':np.array([],np.int64)})
  recipes=make_multi_recipes(Pool.from_index(sub.video_id,sub.label_index,sub.n_samples),1500 if role=='fit' else 300,np.random.default_rng([SEED,57,int(role=='inner_cal')]),n_fragments=tuple(mc.get('n_fragments',(4,8))),edit_variant=rc['edit_variant'],edit_prob=float(mc.get('edit_prob',.5)),min_len=rc['min_len'],max_len=rc['max_len'],reuse=True,p_same_next=float(mc.get('p_same_next',.35)))
  for i,r in enumerate(recipes):
   labels,joins=refs.stream_labels_joins(r.fragments);sequences.append({'id':f'a57_{role}_{i:04d}','kind':'mixed','refs':refs.stream_refs(r.fragments,data.frame_idx),'labels':labels,'joins':joins})
  assert all(len(q['refs'])==len(q['labels']) and {r[0] for r in q['refs']}<=allow for q in sequences)
  groups[role]=sequences;np.save(RUN/(role+'_sequences.private.npy'),np.array(sequences,dtype=object),allow_pickle=True);counts[role]={'native_clips':len(clips),'sessions':len({session_of(v) for v in clips}),'native_class_counts':[sum(data.label[v]==c for v in clips) for c in range(5)],'mixed_streams':len(recipes),'derived_positions':sum(len(q['labels']) for q in sequences),'source_crossrole_violations':0,'sequence_sha256':sha(RUN/(role+'_sequences.private.npy'))}
 assert not ({session_of(v) for v in roles['fit']}&{session_of(v) for v in roles['inner_cal']})
 seqs=groups['fit'];native=[np.array([i for i,q in enumerate(seqs) if q['kind']=='native' and int(q['labels'][0])==c],np.int64) for c in range(5)];rng=np.random.default_rng([SEED,58]);cycles=[]
 for pool in native:
  stream=[]
  while len(stream)<1200:stream.extend(rng.permutation(pool).tolist())
  cycles.append(stream[:1200])
 mixed=np.array([i for i,q in enumerate(seqs) if q['kind']=='mixed']);changing=np.array([i for i in mixed if len(refs.label_changes(seqs[i]['labels']))]);plan=np.empty((6000,2),np.int64);starts=np.empty_like(plan);valid_total=0;positions=set()
 for u in range(6000):
  ni=cycles[u%5][u//5];mi=int(rng.choice(mixed if u%2==0 else changing));plan[u]=[ni,mi]
  for j,i in enumerate(plan[u]):
   q=seqs[int(i)];n=len(q['labels']);high=max(0,n-64)
   start=int(rng.integers(high+1)) if j==0 or u%2==0 else int(np.clip(int(rng.choice(refs.label_changes(q['labels'])))-32,0,high));starts[u,j]=start;indices,mask=dense.chunk_positions(n,start);valid_total+=int(mask.sum());assert {q['refs'][k][0] for k in indices}<=set(roles['fit'])
 np.savez(RUN/'plan.private.npz',sequence=plan,start=starts);record={'run_id':'A57','seed':SEED,'protocol_train_clips':3364,'protocol_train_sessions':len(session),'roles':counts,'session_overlap':0,'updates_planned':6000,'chunks_per_update':2,'original_positions_per_chunk':64,'tubelets_per_chunk':32,'native_class_planned_exposures':[1200]*5,'native_clips_exposed_in6000':len(set(plan[:,0])),'valid_original_FIT_positions_planned':valid_total,'padded_original_positions_planned':6000*128-valid_total,'whole64_position_footprints_checked':12000,'plan_sha256':sha(RUN/'plan.private.npz'),'native_sampling':'classescycle0..4;perclassshuffledcycles;uniformfullchunkstart','mixed_sampling':'evenupdatesuniformstart;oddupdatestruechange-centered;FITonly','source_membership_inputs_are_not_model_inputs':True,'model_forwards':0};assert record['native_clips_exposed_in6000']==len(roles['fit']);write(PUB/'input_metadata.json',record);print(json.dumps(record),flush=True)
def smoke_pixels():
 frozen=json.loads((RUN/'metadata_freeze.json').read_text());assert frozen['actual_git_objects_verified'] and frozen['metadata_sha256']==sha(PUB/'input_metadata.json');seqs=np.load(RUN/'fit_sequences.private.npy',allow_pickle=True).tolist();plan=np.load(RUN/'plan.private.npz');chosen=[seqs[int(i)] for i in plan['sequence'][0]];fp=pixels.frame_paths(chosen);frames=[];labels=[];valids=[]
 for j,q in enumerate(chosen):
  indices,valid=dense.chunk_positions(len(q['labels']),int(plan['start'][0,j]));memo={};out=[]
  for index in indices:
   ref=tuple(q['refs'][int(index)])
   if ref not in memo:
    # Validated numeric RGB/crop/gamma loader; no new image transform.
    v,f,g=ref;p=fp(v);memo[ref]=pixels.load_frame_u8(p[min(int(f),len(p)-1)],round(float(g),6))
   out.append(memo[ref])
  frames.append(np.stack(out));labels.append(q['labels'][indices]);valids.append(valid)
 np.savez(RUN/'FIT_smoke.private.npz',frames=np.stack(frames),labels=np.stack(labels),mask=np.stack(valids));write(PUB/'smoke_input.json',{'FIT_smoke_sha256':sha(RUN/'FIT_smoke.private.npz'),'shape':[2,64,224,224,3],'valid_original_positions':int(np.sum(valids)),'source':'fixedplanupdate0;nativethenconstructed;FITonly;all64footprintsverified','model_forwards':0});print(json.dumps({'smoke_pixels_prepared':True,'valid_positions':int(np.sum(valids))}),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['metadata','smoke_pixels']);a=p.parse_args();RUN.mkdir(parents=True,exist_ok=True);PUB.mkdir(parents=True,exist_ok=True);start=time.time();ok=False
 try:globals()[a.stage]();ok=True
 finally:
  u=resource.getrusage(resource.RUSAGE_SELF);write(RUN/(a.stage+'_process.json'),{'pid':os.getpid(),'ok':ok,'CPU_core_seconds':u.ru_utime+u.ru_stime,'wall_seconds':time.time()-start,'GPU_job_wall_seconds':0})
