#!/usr/bin/env python3
"""Independent direct-arithmetic recount; never imports production audit helpers."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from collections import defaultdict
import hashlib,json,resource,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];O=ROOT/'runs/post_a52_temporal_evidence_v1';P=ROOT/'docs/results/2026-11-01_post_a52_temporal_evidence'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 t=time.time();r=read(P/'audit.json');raw=read(O/'tokens.private.json');assert sha(O/'tokens.private.json')==r['private_token_table_sha256'];source=ROOT/'runs/a51_discriminative_evidence_v1/source';meta=read(source/'private_metadata_marker.json');chrono=read(source/'private_chronology.json');prior=ROOT/'runs/a51_verified_direct_view_pilot_v1';b=read(prior/'bound.private.json');assert [z['id'] for z in raw['tokens']]==[z['id'] for z in b['requests']];indices={(v['folder'],v['video']):v['native_pts_indices'] for v in chrono['videos']};recs=defaultdict(list)
 for a in meta['records']:recs[a['directory_label'],a['video']].append(a)
 for q,z in zip(b['requests'],raw['tokens']):
  key=q['directory_label'],q['video'];manual=[];refs=[]
  for slot in range(0,16,2):
   ordinal=q['frames'][slot];pts=indices[key][ordinal];answers=[]
   for origin,inclusive in [(0,False),(0,True),(1,False),(1,True)]:
    for pos in [ordinal,pts]:answers.append({a['true_label'] for a in recs[key] if (pos+origin)>=a['start'] and ((pos+origin)<=a['end'] if inclusive else (pos+origin)<a['end'])})
   valid=all(a==answers[0] for a in answers) and len(answers[0])==1;label=next(iter(answers[0])) if valid else None;manual.append({'A4C':0,'A5C':1}.get(label));refs.append([*key,ordinal])
  assert manual==z['labels'] and refs==z['references'];seen={k:{tuple(refs[j]) for j,x in enumerate(manual) if x==k} for k in [0,1]};known=[v for v in manual if v is not None];flag=len(known)>=4 and sum(v!=z['marker_label'] for v in known)>=2 and min(map(len,seen.values()))>=2;assert flag==z['informative']
 counts={}
 for role in ['fit','diagnostic']:
  zz=[z for z in raw['tokens'] if z['role']==role];info=[z for z in zz if z['informative']];counts[role]={'requests':len(info),'patients':len({z['patient'] for z in info}),'videos':len({tuple(z['video']) for z in info}),'marker_classes':len({z['marker_label'] for z in info})};assert counts[role]==r['by_role'][role]['informative'];assert sum(r['by_role'][role]['token_label_counts'].values())==len(zz)*8
 gate=all(counts[k][a]>=v for k,m in {'fit':{'requests':80,'patients':8,'videos':8,'marker_classes':2},'diagnostic':{'requests':20,'patients':2,'videos':2,'marker_classes':2}}.items() for a,v in m.items());assert gate==r['dense_supervision_qualification']['passed']
 v=np.load(prior/'source_scores.private.npz');pp=ROOT/'runs/a52_point_detail_v1';p=np.load(pp/'scores.private.npz');scores={'G':v['G'],'L':v['L'],'W':v['W'],'H':p['H'],'R':p['R'],'video_geometry':v['geometry'],'video_constant':v['constant'],'point_geometry':p['geometry'],'point_constant':p['constant']};comparisons=0
 for role in ['fit','diagnostic']:
  rr=[z for z in b['requests'] if z['role']==role];y=np.array([z['true_label']=='A5C' for z in rr]);all_scores=scores if role=='diagnostic' else {a:np.load(root/'source_train'/a/'fit_final.npy') for root,arms in [(prior,['L','W']),(pp,['H','R'])] for a in arms}
  for arm,s in all_scores.items():
   pairs=[(i,j,float(s[i]>s[j])+.5*float(s[i]==s[j])) for i in np.flatnonzero(y) for j in np.flatnonzero(~y)]
   for group in ['video','patient']:
    get=(lambda z:z['directory_label']+'/'+z['video']) if group=='video' else (lambda z:z['patient']);g=[get(z) for z in rr];within=[w for i,j,w in pairs if g[i]==g[j]];between=[w for i,j,w in pairs if g[i]!=g[j]];out=r['score_discrimination'][role][arm][group]
    assert len(within)==out['within_pairs'] and len(between)==out['between_pairs'];assert abs(sum(w for _,_,w in pairs)/len(pairs)-out['global_AUC'])<1e-12
    for values,key in [(within,'within_AUC'),(between,'between_AUC')]:assert (out[key] is None) if not values else abs(sum(values)/len(values)-out[key])<1e-12
    assert abs((sum(within)+sum(between))/len(pairs)-out['reconstructed_AUC'])<1e-12;comparisons+=1
 out={'passed':True,'independent_all702x8_token_conventions_and_physical_reference_rule':True,'same_source_marker_manual_labels_every_model':True,'dense_qualification_recomputed':gate,'all_model_group_AUC_partitions_recomputed':comparisons,'score_and_token_audit_sha256':sha(P/'audit.json'),'new_model_image_or_GPU_operations':0,'new_training':0,'wall_seconds':time.time()-t}
 P.mkdir(parents=True,exist_ok=True);(P/'independent_audit.json').write_text(json.dumps(out,indent=2)+'\n');u=resource.getrusage(resource.RUSAGE_SELF);(O/'verify_process.json').write_text(json.dumps({'pid':os.getpid(),'CPU_seconds':u.ru_utime+u.ru_stime,'wall_seconds':time.time()-t,'GPU_job_wall_seconds':0},indent=2)+'\n');print(json.dumps(out))
if __name__=='__main__':main()
