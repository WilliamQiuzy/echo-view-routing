#!/usr/bin/env python3
"""A57 independent metadata-only split and exact whole-plan exposure recount."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import hashlib,json,resource,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.train_dense_sequence import RUN,PUB,read,write,sha,make_plan
from echo_routing.temporal.dense_runtime import verify_partition
from echo_routing.temporal.dense_sequence import chunk_positions
from echo_routing.evaluate.failure_units import session_of

def digest(plan):
 h=hashlib.sha256()
 for k in ['sequence','start']:h.update(k.encode());h.update(np.asarray(plan[k],dtype='<i8').tobytes())
 return h.hexdigest()

def main():
 meta=read(PUB/'input_metadata.json');roles={}
 for role in ['fit','inner_cal']:
  p=RUN/(role+'_sequences.private.npy');assert sha(p)==meta['roles'][role]['sequence_sha256'];roles[role]=np.load(p,allow_pickle=True).tolist()
 native={k:{q['id'] for q in v if q['kind']=='native'} for k,v in roles.items()};proto=read(ROOT/'runs/20261002-051140-model_development_protocol-fa45e7f4-s0/protocol.json');assert native['fit'].isdisjoint(native['inner_cal']) and native['fit']|native['inner_cal']==set(proto['train']['clips']);sessions=sorted({session_of(v) for v in proto['train']['clips']});perm=np.random.default_rng(2026110257).permutation(len(sessions));cal={sessions[i] for i in perm[:round(.2*len(sessions))]};assert {session_of(v) for v in native['inner_cal']}==cal and not ({session_of(v) for v in native['fit']}&cal)
 for k,q in roles.items():verify_partition(q,native[k])
 results={};fit=roles['fit']
 for seed in [2026110257,2026110258]:
  plan=make_plan(fit,seed)
  if seed==2026110257:
   saved=np.load(RUN/'plan.private.npz');assert all(np.array_equal(plan[k],saved[k]) for k in plan)
  record={'canonical_little_endian_int64_plan_sha256':digest(plan),'updates':6000,'whole64_footprints':12000,'native_unique_clips':len(set(plan['sequence'][:,0].tolist())),'valid_positions_by_domain_class':{'native':[0]*5,'mixed':[0]*5},'padded_positions_by_domain':{'native':0,'mixed':0}};unique={};seen={}
  for limit in [2000,6000]:
   unique_images=set();unique_variants=set();source_clips=set()
   for u in range(limit):
    for j,index in enumerate(plan['sequence'][u]):
     q=fit[int(index)];indices,mask=chunk_positions(len(q['labels']),int(plan['start'][u,j]));references=[q['refs'][int(i)] for i in indices];assert {r[0] for r in references}<=native['fit'];source_clips.update(r[0] for r in references);unique_images.update((r[0],r[1]) for r in references);unique_variants.update(tuple(r) for r in references)
     if limit==6000:
      domain='native' if j==0 else 'mixed';b=np.bincount(q['labels'][indices][mask],minlength=5);record['valid_positions_by_domain_class'][domain]=[int(a+b) for a,b in zip(record['valid_positions_by_domain_class'][domain],b)];record['padded_positions_by_domain'][domain]+=int((~mask).sum())
   record[str(limit)+'step_sources']={'distinct_source_clips':len(source_clips),'distinct_source_sessions':len({session_of(v) for v in source_clips}),'unique_original_image_refs':len(unique_images),'unique_frame_gamma_refs':len(unique_variants)}
  record['valid_original_positions']=sum(sum(v) for v in record['valid_positions_by_domain_class'].values());assert record['valid_original_positions']+sum(record['padded_positions_by_domain'].values())==6000*128;results[str(seed)]=record
 assert results['2026110257']['valid_original_positions']==meta['valid_original_FIT_positions_planned']
 write(PUB/'independent_input_exposure.json',{'passed':True,'protocol_train_union_exact':3364,'sessions_fit':len({session_of(v) for v in native['fit']}),'sessions_inner':len(cal),'session_overlap':0,'source_footprints_outside_role':0,'no_model_forwards':True,'same_partitions_and_constructed_banks_for_both_training_seeds':True,'plans':results,'interpretation':'image refs/derived positions/sourceclips/sessions are distinct quantities; sessions are not patients'})
 print(json.dumps({'independent_input_exposure_passed':True,'seeds':list(results)}),flush=True)
if __name__=='__main__':
 start=time.time();ok=False
 try:main();ok=True
 finally:
  r=resource.getrusage(resource.RUSAGE_SELF);write(RUN/'input_exposure_process.json',{'ok':ok,'CPU_core_seconds':r.ru_utime+r.ru_stime,'wall_seconds':time.time()-start,'GPU_job_wall_seconds':0})
