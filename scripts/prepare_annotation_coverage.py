#!/usr/bin/env python3
"""A55 prospective metadata inventory and new centre-supervision binding."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse,json,hashlib,resource,sys,time
from collections import defaultdict,Counter
from fractions import Fraction
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,av
from sklearn.linear_model import LogisticRegression
from echo_routing.temporal import annotation_coverage as ac,direct_view_training as dt
OUT=ROOT/'runs/a55_long_iteration_v1';PUB=ROOT/'docs/results/2026-11-01_annotation_coverage';OLD=ROOT/'runs/a51_verified_direct_view_pilot_v1';SRC=ROOT/'runs/a51_discriminative_evidence_v1/source';CARD='docs/superpowers/plans/2026-11-01-five-hour-evidence-iteration.md'
write=dt.write_json

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def j(p):return json.loads(Path(p).read_text())
def arsha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def key(r):return r['directory_label'],r['video']
def ikey(r):return (*key(r),r['start'],r['end'],r['true_label'])
def codepins():return {p:sha(ROOT/p) for p in ['echo_routing/temporal/annotation_coverage.py','scripts/prepare_annotation_coverage.py',CARD]}

def metadata():
 assert not (OUT/'metadata.private.json').exists(),'metadata already committed; no silent reselection'
 b=j(OLD/'bound.private.json');sel=j(OLD/'selection.private.json');m=j(SRC/'private_metadata_marker.json');fit=[r for r in b['requests'] if r['role']=='fit'];diag=[r for r in b['requests'] if r['role']=='diagnostic'];assert len(fit)==569 and len(diag)==133
 labels=defaultdict(list);anchors=defaultdict(list);excluded=defaultdict(set)
 for r in m['records']:labels[key(r)].append(r)
 for r in fit:anchors[ikey(r)].append(r['centre'])
 for r in sel['excluded_supervised_requests']:excluded[key(r)].add(r['centre'])
 axes={(v['folder'],v['video']):v['native_pts_indices'] for v in b['videos']};pool={k:ac.eligible_points(labels[k[:2]],axes[k[:2]],k[2],k[3],k[4],a,excluded[k[:2]]) for k,a in anchors.items()};parents=[]
 for r in fit:
  pp=pool[ikey(r)];assert r['centre'] in pp,'original goodmarker missing from conservativepool';parents.append({'source':r,'key':list(key(r)),'patient':r['patient'],'label':int(r['true_label']=='A5C'),'centre':r['centre'],'points':pp})
 plan,dense=ac.make_plan(parents);counts=ac.coverage(parents,plan,dense);old_context={(key(r),i) for r in fit for i in r['frames']};used={(tuple(parents[int(i)]['key']),int(c)) for row,centres in zip(plan,dense) for i,c in zip(row,centres)};counts['scheduled_points_not_in_any_original_context']=len(used-old_context);counts['supervised_marker_novelty_not_fresh_pixel_or_patient_claim']=True
 np.save(OUT/'metadata_parent_plan.npy',plan);np.save(OUT/'metadata_dense_centres.npy',dense);dest={'parents':parents,'diagnostic':diag,'videos':b['videos'],'labels':[{'key':list(k),'records':v} for k,v in labels.items() if k in axes],'excluded':[{'key':list(k),'ordinals':sorted(v)} for k,v in excluded.items()],'original_bound_sha256':sha(OLD/'bound.private.json'),'original_metadata_sha256':sha(SRC/'private_metadata_marker.json'),'original_selection_sha256':sha(OLD/'selection.private.json')};write(OUT/'metadata.private.json',dest)
 public={'run_id':'A55','stage':'metadata_only_no_new_pixels_or_neural_model','old_bound_sha256':dest['original_bound_sha256'],'old_selection_sha256':dest['original_selection_sha256'],'metadata_private_sha256':sha(OUT/'metadata.private.json'),'parent_plan_sha256':sha(OUT/'metadata_parent_plan.npy'),'dense_plan_sha256':sha(OUT/'metadata_dense_centres.npy'),'counts':counts,'source_original':{'fit':569,'diagnostic':133,'patients':[36,10],'videos':[55,13],'raw_dimensions':[500,500]},'geometry_factor':'Ffull224resize versus exactA08short256centercrop224; originalW/Pidenticalfactorrejected','code':codepins(),'new_neural_forwards':0,'formal_updates':0};write(PUB/'metadata_manifest.json',public);print(json.dumps({'metadata_quantity_pass':counts['quantity_pass'],'counts':counts}),flush=True)

def binding():
 m=j(PUB/'metadata_manifest.json');fr=j(OUT/'metadata_freeze.json');assert fr['actual_git_objects_verified'] and fr['metadata_manifest_sha256']==sha(PUB/'metadata_manifest.json');assert m['counts']['quantity_pass'];assert not (OUT/'bound.private.json').exists(),'binding alreadysealed'
 for p,h in m['code'].items():assert sha(ROOT/p)==h,p
 q=j(OUT/'metadata.private.json');assert sha(OUT/'metadata.private.json')==m['metadata_private_sha256'];parents=q['parents'];diag=q['diagnostic'];fit=[p['source'] for p in parents];labels={tuple(x['key']):x['records'] for x in q['labels']};requested=defaultdict(set);supervised=defaultdict(dict)
 for r in fit+diag:
  requested[key(r)].update(r['frames']);supervised[key(r)][r['centre']]=(r['role'],int(r['true_label']=='A5C'))
 for p in parents:
  k=tuple(p['key']);n=next(v['decoded_frames'] for v in q['videos'] if (v['folder'],v['video'])==k)
  for c in p['points']:
   requested[k].update(max(0,min(n-1,c+3*(s-8))) for s in range(16));assert c not in supervised[k] or supervised[k][c]==('fit',p['label']);supervised[k][c]=('fit',p['label'])
 caches=[];points=[];sparse=[];all_geo={a:[] for a in ['F','C']};cache_dir=OUT/'pixels';cache_dir.mkdir(exist_ok=True)
 for vi,v in enumerate(q['videos']):
  k=v['folder'],v['video'];path=SRC/'videos'/k[0]/k[1];assert sha(path)==v['sha256'];ordinals=sorted(requested[k]);lookup={x:i for i,x in enumerate(ordinals)};dest=cache_dir/(hashlib.sha256(('/'.join(k)).encode()).hexdigest()+'.npz');receipt=dest.with_suffix('.json')
  if receipt.exists():
   rec=j(receipt);assert sha(dest)==rec['cache_sha256'];caches.append(rec);points.extend(rec['points']);sparse.extend(rec['sparse']);continue
  F,C=[],[];pts=[];raw_hash={};pointstats={};ordinal_set=set(ordinals)
  with av.open(str(path)) as cc:
   ss=cc.streams.video[0];ss.thread_type='NONE';assert abs(float(ss.average_rate)-30)<.001 and (ss.width,ss.height)==(v['width'],v['height'])
   for i,frame in enumerate(cc.decode(ss)):
    position=Fraction(frame.pts)*Fraction(frame.time_base)*30;assert position.denominator==1;pts.append(position.numerator)
    if i not in ordinal_set:continue
    rgb=frame.to_ndarray(format='rgb24');f,c=ac.geometry(rgb,'F'),ac.geometry(rgb,'C');F.append(f);C.append(c)
    if i in supervised[k]:
     raw_hash[i]=hashlib.sha256(rgb.tobytes()).hexdigest();pointstats[i]={a:[v['width'],v['height'],float(ss.average_rate),v['decoded_frames'],float(x.mean()),float(x.std()),float((x.max(-1).astype(float)-x.min(-1)).mean()),float((x.mean(-1)>10).mean())] for a,x in [('F',f),('C',c)]}
  assert pts==v['native_pts_indices'] and len(F)==len(ordinals);F=np.stack(F);C=np.stack(C);tmp=dest.with_suffix('.tmp')
  with tmp.open('wb') as f:np.savez_compressed(f,ordinals=np.asarray(ordinals,np.int64),F=F,C=C);f.flush();os.fsync(f.fileno())
  tmp.replace(dest);dt.sync_directory(dest.parent);pp=[];sp=[]
  for ordinal,(role,label) in sorted(supervised[k].items()):
   at=lookup[ordinal];pp.append({'key':list(k),'ordinal':ordinal,'role':role,'label':label,'raw_sha256':raw_hash[ordinal],'F_sha256':arsha(F[at]),'C_sha256':arsha(C[at])})
  for r in fit+diag:
   if key(r)!=k:continue
   ids=np.array([lookup[i] for i in r['frames']]);assert arsha(F[ids])==r['pixel_sha256'],'Fsourcewindow must exactlyreproduce originalA51pixels';sp.append({'id':r['id'],'key':list(k),'centre':r['centre'],'role':r['role'],'patient':r['patient'],'label':int(r['true_label']=='A5C'),'F_window_sha256':arsha(F[ids]),'C_window_sha256':arsha(C[ids]),'geometry':pointstats[r['centre']]})
  rec={'key':list(k),'raw_video_sha256':sha(path),'cache':str(dest.relative_to(ROOT)),'cache_sha256':sha(dest),'ordinals':len(ordinals),'points':pp,'sparse':sp,'native_PTS_reproduced':True};write(receipt,rec);caches.append(rec);points.extend(pp);sparse.extend(sp);write(OUT/'binding_progress.json',{'pid':os.getpid(),'complete_videos':vi+1,'total':len(q['videos']),'source_neural_forwards':0,'epoch':time.time()});print(json.dumps({'bound_videos':vi+1,'total':len(q['videos'])}),flush=True)
 by_hash=defaultdict(list)
 for p in points:by_hash[p['raw_sha256']].append(p)
 crossrole=[h for h,ps in by_hash.items() if len({p['role'] for p in ps})>1];conflicts={h for h,ps in by_hash.items() if len({p['label'] for p in ps})>1};original={(tuple(p['key']),p['centre']) for p in parents}|{(key(r),r['centre']) for r in diag};bad_original=any((tuple(p['key']),p['ordinal']) in original for h in conflicts for p in by_hash[h]);assert not crossrole and not bad_original,'sourcepartition/pinnedmarker pixel conflict: qualificationblocked, no replacement'
 bad={(tuple(p['key']),p['ordinal']) for h in conflicts for p in by_hash[h]};filtered=[{**p,'points':[c for c in p['points'] if (tuple(p['key']),c) not in bad]} for p in parents];assert all(p['points'] for p in filtered),'originalfixedparent lost allDpoints';plan,dense=ac.make_plan(filtered);counts=ac.coverage(filtered,plan,dense);point_by={(tuple(p['key']),p['ordinal']):p for p in points};used={(tuple(filtered[int(i)]['key']),int(c)) for row,cs in zip(plan,dense) for i,c in zip(row,cs)};old_hash={point_by[(tuple(p['key']),p['centre'])]['raw_sha256'] for p in filtered};used_hash={point_by[x]['raw_sha256'] for x in used};used_by_class={str(y):len({point_by[x]['raw_sha256'] for x in used if point_by[x]['label']==y}) for y in (0,1)};counts['scheduled_distinct_raw_pixel_hashes']=len(used_hash);counts['scheduled_hashes_not_original_supervised_point_hashes']=len(used_hash-old_hash);counts['scheduled_effective_classes']=used_by_class;counts['pixel_quantity_pass']=bool(len(used_hash)>=2276 and len(used_hash-old_hash)>=1707 and min(used_by_class.values())>=800)
 sparse_by={x['id']:x for x in sparse};fit_sparse=[sparse_by[r['id']] for r in fit];different=[r for r in fit_sparse if r['F_window_sha256']!=r['C_window_sha256']];geom_counts={'different_FIT_windows':len(different),'FIT_windows':569,'different_fraction':len(different)/569,'patients_with_difference':len({r['patient'] for r in different})};geom_counts['factor_identifiable']=bool(geom_counts['different_fraction']>=.2 and geom_counts['patients_with_difference']>=8)
 np.save(OUT/'parent_plan.npy',plan);np.save(OUT/'dense_centres.npy',dense);geometry={};y=np.array([r['label'] for r in fit_sparse])
 for a in ['F','C']:
  x=np.array([r['geometry'][a] for r in fit_sparse]);mu=x.mean(0);sd=x.std(0);sd[sd<1e-12]=1;model=LogisticRegression(C=1.,solver='lbfgs',max_iter=1000,random_state=ac.SEED).fit((x-mu)/sd,y);geometry[a]={'fit_mean':mu.tolist(),'fit_scale':sd.tolist(),'coef':model.coef_[0].tolist(),'intercept':float(model.intercept_[0]),'constant_prior':float(y.mean())}
 private={'parents':filtered,'diagnostic':diag,'videos':q['videos'],'caches':caches,'points':points,'sparse':sparse,'geometry':geometry,'conflicting_new_point_hashes_excluded':sorted(conflicts),'metadata_sha256':sha(OUT/'metadata.private.json')};write(OUT/'bound.private.json',private)
 passed=counts['quantity_pass'] and counts['pixel_quantity_pass'] and geom_counts['factor_identifiable'];pub={'run_id':'A55','passed':bool(passed),'source_CNN_forwards':0,'fit_geometry_CPU_models':2,'counts':counts,'geometry':geom_counts,'all68videoSHA_nativePTS_reproduced':True,'original_all702F_windows_exact_pixel_parity':True,'crossrole_exact_point_hashes':len(crossrole),'conflicting_old_markers':int(bad_original),'newconflicting_hash_groups_excluded':len(conflicts),'private_bound_sha256':sha(OUT/'bound.private.json'),'parent_plan_sha256':sha(OUT/'parent_plan.npy'),'dense_centres_sha256':sha(OUT/'dense_centres.npy'),'metadata_manifest_sha256':sha(PUB/'metadata_manifest.json'),'cache_sha256':{hashlib.sha256('/'.join(x['key']).encode()).hexdigest():x['cache_sha256'] for x in caches},'cached_processed_frame_rows':sum(x['ordinals'] for x in caches),'new_cache_bytes':sum((ROOT/x['cache']).stat().st_size for x in caches),'geometry_control_definitions':['width','height','fps','decodedN','processed_point_mean','std','mean_channel_range','grayscale_fraction_gt10'],'source_models_or_diagnostic_scores':0,'code':codepins(),'versions':{'numpy':np.__version__,'PyAV':av.__version__}};write(PUB/'qualification.json',pub);print(json.dumps({'qualification_passed':bool(passed),'counts':counts,'geometry':geom_counts}),flush=True)

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('stage',choices=['metadata','binding']);args=a.parse_args();OUT.mkdir(exist_ok=True,parents=True);PUB.mkdir(exist_ok=True,parents=True);lock=dt.acquire_lock(OUT/'owner.lock');start=time.time();ok=False
 try:globals()[args.stage]();ok=True
 finally:
  u=resource.getrusage(resource.RUSAGE_SELF);write(OUT/f'{args.stage}_worker_process.json',{'pid':os.getpid(),'ok':ok,'CPU_seconds':u.ru_utime+u.ru_stime,'wall_seconds':time.time()-start,'GPU_job_wall_seconds':0});lock.close()
