#!/usr/bin/env python3
"""A55 independent raw binding, complete state/journal, and cached source metric audits."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse,json,hashlib,resource,sys,time,math
from pathlib import Path
from fractions import Fraction
from collections import defaultdict
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import av,numpy as np,torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from scripts.prepare_annotation_coverage import OUT,PUB,OLD,SRC,sha,j,arsha,write

def crop(x,mode):
 im=Image.fromarray(x)
 if mode=='F':return np.asarray(im.resize((224,224),Image.Resampling.BILINEAR)).copy()
 w,h=im.size;short=min(w,h);nw,nh=max(224,round(w*256/short)),max(224,round(h*256/short));im=im.resize((nw,nh),Image.Resampling.BILINEAR);a,b=(nw-224)//2,(nh-224)//2;return np.asarray(im.crop((a,b,a+224,b+224))).copy()

def binding():
 q=j(PUB/'qualification.json');b=j(OUT/'bound.private.json');old=j(OLD/'bound.private.json');meta=j(SRC/'private_metadata_marker.json');selection=j(OLD/'selection.private.json');assert sha(OUT/'bound.private.json')==q['private_bound_sha256'];labels=defaultdict(list)
 for r in meta['records']:labels[(r['directory_label'],r['video'])].append(r)
 def agrees(k,ordinal,pts,label):
  for frame in [ordinal,pts]:
   for shift in [0,1]:
    for inc in [0,1]:
     got=[r['true_label'] for r in labels[k] if r['start']-shift<=frame<r['end']-shift+inc]
     if set(got)!={label}:return False
  return True
 prohibited={(r['directory_label'],r['video'],r['centre']) for r in selection['excluded_supervised_requests']};pointmap={(tuple(x['key']),x['ordinal']):x for x in b['points']};cachemap={tuple(x['key']):x for x in b['caches']};axes={};pixel_checks=0
 for v in b['videos']:
  k=v['folder'],v['video'];path=SRC/'videos'/k[0]/k[1];assert sha(path)==v['sha256'];c=cachemap[k];assert sha(ROOT/c['cache'])==c['cache_sha256'];zz=np.load(ROOT/c['cache']);z={a:zz[a] for a in ['ordinals','F','C']};lookup={int(i):at for at,i in enumerate(z['ordinals'])};pts=[]
  with av.open(str(path)) as con:
   st=con.streams.video[0];st.thread_type='NONE';assert st.width==500 and st.height==500
   for ordinal,fr in enumerate(con.decode(st)):
    p=Fraction(fr.pts)*Fraction(fr.time_base)*30;assert p.denominator==1;pts.append(p.numerator)
    if ordinal not in lookup:continue
    raw=fr.to_ndarray(format='rgb24');at=lookup[ordinal]
    for mode in ['F','C']:assert np.array_equal(crop(raw,mode),z[mode][at]);pixel_checks+=1
    if (k,ordinal) in pointmap:
     point=pointmap[k,ordinal];assert hashlib.sha256(raw.tobytes()).hexdigest()==point['raw_sha256'];assert agrees(k,ordinal,p.numerator,'A5C' if point['label'] else 'A4C')
  assert pts==v['native_pts_indices'];axes[k]=pts
 parents=b['parents'];plan=np.load(OUT/'parent_plan.npy');dense=np.load(OUT/'dense_centres.npy');assert plan.shape==dense.shape==(3600,8);yy=np.array([p['label'] for p in parents]);rng=np.random.default_rng(2026110155);streams=[]
 for y in [0,1]:
  values=[];pool=np.flatnonzero(yy==y)
  while len(values)<14400:values+=rng.permutation(pool).tolist()
  streams.append(np.array(values[:14400],np.int64).reshape(3600,4))
 assert np.array_equal(plan,np.concatenate(streams,axis=1));assert set(plan.ravel())==set(range(569));offset=np.random.default_rng(2026110156).integers(0,2**31,size=569);visit=np.zeros(569,int);used=set();poolset=set();orig={(tuple(p['key']),p['centre']) for p in parents}
 for p in parents:
  k=tuple(p['key'])
  for c in p['points']:
   assert (*k,c) not in prohibited and agrees(k,c,axes[k][c],'A5C' if p['label'] else 'A4C');poolset.add((k,c))
 for row,cs in zip(plan,dense):
  assert yy[row].sum()==4
  for i,c in zip(row,cs):
   p=parents[int(i)];assert c==p['points'][(int(offset[i])+int(visit[i]))%len(p['points'])];visit[i]+=1;used.add((tuple(p['key']),int(c)))
 for name,points in [('pool',poolset),('scheduled',used)]:
  got={'physical_points':len(points),'not_original_centres':len(points-orig),'classes':{str(y):sum(pointmap[k,c]['label']==y for k,c in points) for y in [0,1]}};assert got==q['counts'][name]
 hashes={pointmap[x]['raw_sha256'] for x in used};oldhashes={pointmap[x]['raw_sha256'] for x in orig};assert len(hashes)==q['counts']['scheduled_distinct_raw_pixel_hashes']>=2276 and len(hashes-oldhashes)==q['counts']['scheduled_hashes_not_original_supervised_point_hashes']>=1707
 fitrows=[r for r in old['requests'] if r['role']=='fit'];diagrows=[r for r in old['requests'] if r['role']=='diagnostic'];assert [p['source']['id'] for p in parents]==[r['id'] for r in fitrows] and [r['id'] for r in b['diagnostic']]==[r['id'] for r in diagrows];assert len({r['patient'] for r in fitrows})==36 and len({r['patient'] for r in diagrows})==10 and not {r['patient'] for r in fitrows}&{r['patient'] for r in diagrows}
 sparse={r['id']:r for r in b['sparse']};diff=[r for r in fitrows if sparse[r['id']]['F_window_sha256']!=sparse[r['id']]['C_window_sha256']];assert len(diff)==q['geometry']['different_FIT_windows'] and len(diff)>=.2*569 and len({r['patient'] for r in diff})>=8
 for mode in ['F','C']:
  X=np.array([sparse[r['id']]['geometry'][mode] for r in fitrows]);y=np.array([r['true_label']=='A5C' for r in fitrows]);mu=X.mean(0);sd=X.std(0);sd[sd<1e-12]=1;model=LogisticRegression(C=1.,solver='lbfgs',max_iter=1000,random_state=2026110155).fit((X-mu)/sd,y);g=b['geometry'][mode];assert np.max(abs(mu-g['fit_mean']))<1e-10 and np.max(abs(sd-g['fit_scale']))<1e-10 and np.max(abs(model.coef_[0]-g['coef']))<1e-10 and abs(model.intercept_[0]-g['intercept'])<1e-10
 assert q['passed'];write(PUB/'independent_binding.json',{'passed':True,'qualification_sha256':sha(PUB/'qualification.json'),'all68raw_video_SHA_PTS_recounted':True,'all_cached_processed_frame_arrays_independently_equal':True,'processed_pixel_arrays_compared':pixel_checks,'all_scheduled_centres_and_labels_reconstructed':True,'plan_actual_point_minima_pass':True,'original_parent_and_patient_allocation_unchanged':True,'fit_geometry_controls_independently_refitted':True,'new_source_CNN_forwards':0,'target_readouts':0});print(json.dumps({'binding_audit_passed':True,'processed_arrays':pixel_checks}),flush=True)

def binding_supplement():
 b=j(OUT/'bound.private.json');q=j(PUB/'qualification.json');prior=j(PUB/'independent_binding.json');assert prior['passed'] and prior['qualification_sha256']==sha(PUB/'qualification.json');old=j(OLD/'bound.private.json');sparse={r['id']:r for r in b['sparse']};by_video=defaultdict(list)
 for r in old['requests']:by_video[(r['directory_label'],r['video'])].append(r)
 geometric={g:{} for g in ['F','C']};checks=0
 for rec in b['caches']:
  assert sha(ROOT/rec['cache'])==rec['cache_sha256'];zz=np.load(ROOT/rec['cache']);z={a:zz[a] for a in ['ordinals','F','C']};lookup={int(o):i for i,o in enumerate(z['ordinals'])};k=tuple(rec['key']);v=next(v for v in b['videos'] if (v['folder'],v['video'])==k)
  for r in by_video[k]:
   ix=[lookup[i] for i in r['frames']];assert arsha(z['F'][ix])==r['pixel_sha256']==sparse[r['id']]['F_window_sha256'];assert arsha(z['C'][ix])==sparse[r['id']]['C_window_sha256'];checks+=1
   for g in ['F','C']:
    x=z[g][lookup[r['centre']]];features=[v['width'],v['height'],v['fps'],v['decoded_frames'],float(x.mean()),float(x.std()),float((x.max(-1).astype(float)-x.min(-1)).mean()),float((x.mean(-1)>10).mean())];assert np.array_equal(features,sparse[r['id']]['geometry'][g]);geometric[g][r['id']]=features
 hh=defaultdict(list)
 for r in b['points']:hh[r['raw_sha256']].append(r)
 assert all(len({r['role'] for r in v})==1 for v in hh.values());bad={h for h,v in hh.items() if len({r['label'] for r in v})>1};assert bad==set(b['conflicting_new_point_hashes_excluded']);assert len(bad)==q['newconflicting_hash_groups_excluded'];processed={}
 for field in ['raw_sha256','F_sha256','C_sha256']:
  groups=defaultdict(list)
  for point in b['points']:groups[point[field]].append(point)
  cross=sum(len({x['role'] for x in rows})>1 for rows in groups.values());conflict={h for h,rows in groups.items() if len({x['label'] for x in rows})>1};assert cross==0 and not conflict,'processedpoint collision requires same-set admission repair before anymodel, no replacement';fit_values={p[field] for p in b['points'] if p['role']=='fit'};original_values={p[field] for p in b['points'] if p['role']=='fit' and (tuple(p['key']),p['ordinal']) in {(tuple(v['key']),v['centre']) for v in b['parents']}};classes={str(y):len({p[field] for p in b['points'] if p['role']=='fit' and p['label']==y}) for y in [0,1]};assert len(fit_values)>=2276 and len(fit_values-original_values)>=1707 and min(classes.values())>=800;processed[field]={'unique_all':len(groups),'unique_fit':len(fit_values),'new_supervised_fit_hashes':len(fit_values-original_values),'classes':classes,'crossrole_groups':cross,'conflicting_label_groups':len(conflict)}
 write(PUB/'processed_integrity.json',{'passed':True,'qualification_sha256':sha(PUB/'qualification.json'),'raw_F_C':processed,'same_D_point_set_all_arms':True,'additional_processed_exclusions_required':0});assert not any(r['raw_sha256'] in bad for r in b['points'] if (tuple(r['key']),r['ordinal']) in {(tuple(p['key']),p['centre']) for p in b['parents']})
 write(PUB/'independent_binding_supplement.json',{'passed':True,'qualification_sha256':sha(PUB/'qualification.json'),'prior_binding_audit_sha256':sha(PUB/'independent_binding.json'),'original702windowFparity_and_Cwindow_hashes':checks,'all_geometry_control_inputs_independently_recomputed_from_verified_pixels':True,'pixel_duplicate_role_and_label_exclusions_recounted':True,'source_CNN_forwards':0})

def models():
 from scripts import train_annotation_coverage as t
 manifest=t.verify();record=j(PUB/'source_models_manifest.json');assert record['input_manifest_sha256']==sha(PUB/'input_manifest.json');b=j(OUT/'bound.private.json');parents=b['parents'];plan=np.load(OUT/'parent_plan.npy');dense=np.load(OUT/'dense_centres.npy');cache={tuple(c['key']):{g:z for g,z in ((g,np.load(ROOT/c['cache'])[g]) for g in ['ordinals','F','C'])} for c in b['caches']};index={k:{int(o):i for i,o in enumerate(z['ordinals'])} for k,z in cache.items()};nframes={(v['folder'],v['video']):v['decoded_frames'] for v in b['videos']};windowhash={};fit_frames={}
 def window(p,c,g):
  k=tuple(p['key']);ids=[index[k][max(0,min(nframes[k]-1,int(c)+3*(x-8)))] for x in range(16)];return cache[k][g][ids]
 def digest(p,c,g):
  k=tuple(p['key']),int(c),g
  if k not in windowhash:windowhash[k]=arsha(window(p,c,g))
  return windowhash[k]
 results={}
 for arm in ['SF','SC','DF','DC']:
  rd=OUT/'source_train'/arm;e=record['models'][arm];bundle=torch.load(rd/'final.pt',map_location='cpu',weights_only=True);st=torch.load(rd/'optimizer_final.pt',map_location='cpu',weights_only=True);assert sha(rd/'final.pt')==e['bundle_sha256'] and sha(rd/'optimizer_final.pt')==e['optimizer_final_sha256'] and sha(rd/'updates.private.jsonl')==e['journal_sha256'];m,h=t.old.restore(bundle,'cpu');flat=t.old.flat(m,h);assert t.state_sha(flat)==bundle['full_state_sha256'];assert t.state_sha({**{'m.'+k:v for k,v in st['model'].items()},**{'h.'+k:v for k,v in st['head'].items()}})==bundle['full_state_sha256'];assert st['update']==3600 and st['identity']==e['identity'];opt=st['opt'];assert len(opt['state'])==e['optimizer_state_count'] and {int(v['step']) for v in opt['state'].values()}=={3600};logs=[json.loads(s) for s in (rd/'updates.private.jsonl').read_text().splitlines()];assert [x['u'] for x in logs]==list(range(3600));sequence=[x['pid'] for x in logs];transitions=[(i,sequence[i]) for i in range(len(sequence)) if i==0 or sequence[i]!=sequence[i-1]];assert len({p for i,p in transitions})==len(transitions),'PIDsegments must not interleave';events=[json.loads(x) for x in (rd/'resume_events.private.jsonl').read_text().splitlines()] if (rd/'resume_events.private.jsonl').exists() else [];assert all(any(e['start_update']==i and e['pid']==p and e['journal_length']==i and e['safe_continuity_validated'] for e in events) for i,p in transitions[1:])
  for u,(r,row) in enumerate(zip(logs,plan)):
   centres=np.array([dense[u,j] if arm[0]=='D' else parents[int(i)]['centre'] for j,i in enumerate(row)],np.int64);expected=hashlib.sha256(''.join(digest(parents[int(i)],c,arm[1]) for i,c in zip(row,centres)).encode()).hexdigest();assert r['arm']==arm and r['schedule_index']==u and r['plan_row_sha256']==arsha(row) and r['actual_centres_sha256']==arsha(centres) and r['input_window_sha256']==expected and r['physical_forwards']==r['physical_backwards']==8 and np.isfinite(r['loss']) and r['device']=='NVIDIA L40S'
  G=t.old.t44.groups_of(m,h);canonical=[p for ps in G.values() for p in ps];assert len({id(p) for p in canonical})==len(canonical)==len(opt['state']);used_ids=[]
  for group,(name,params) in zip(opt['param_groups'],G.items()):
   assert group['name']==name and len(group['params'])==len(params) and group['base']==manifest['rates'][name] and tuple(group['betas'])==(.9,.999) and group['eps']==1e-8 and group['weight_decay']==1e-4;expected_lr=group['base']*.5*(1+math.cos(math.pi*(3599-360)/(3600-360)));assert abs(group['lr']-expected_lr)<1e-16
   for pid,param in zip(group['params'],params):
    assert pid not in used_ids;used_ids.append(pid);v=opt['state'][pid]
    for k in ['exp_avg','exp_avg_sq']:assert v[k].shape==param.shape and v[k].dtype==param.dtype and torch.isfinite(v[k]).all()
    assert (v['exp_avg_sq']>=0).all()
  assert used_ids==list(range(len(canonical))) and set(used_ids)==set(opt['state']) and st['cpu_rng'].dtype==torch.uint8 and st['cpu_rng'].ndim==1 and isinstance(st['cuda_rng'],list) and len(st['cuda_rng'])==1 and all(x.dtype==torch.uint8 and x.ndim==1 for x in st['cuda_rng']);base_m,base_h=t.old.build('cpu');base=t.old.flat(base_m,base_h);live=set(bundle['live']);assert t.state_sha(base)==manifest['reference_A51_initial_state_sha256'];assert all(torch.equal(flat[k],base[k]) for k in flat if k not in live);assert t.state_sha({k:base[k] for k in base if k not in live})==bundle['frozen_state_sha256'];actual_drift={k:float((flat[k].double()-base[k].double()).norm()) for k in flat};assert all(abs(actual_drift[k]-e['drift_by_tensor'][k])<=1e-10*(1+actual_drift[k]) for k in flat);del base_m,base_h,base
  assert all(any(r['groups'][k]>0 for r in logs[1:]) for k in G);assert all(logs[0]['groups'][k]==0 for k in G if k!='scalar_head');assert j(rd/'update_guard.json')=={'pending':None,'last_committed':3599}
  # Model hashes are the primary exact check; fresh CPU FIT checks validate physical restoration independently.
  fit=np.load(rd/'fit_original.npy');errs=[]
  with torch.no_grad():
   for i in [0,1,567,568]:errs.append(abs(float(t.score(m,h,window(parents[i],parents[i]['centre'],arm[1]),device='cpu'))-float(fit[i])))
  assert max(errs)<1e-4;assert all(e['drift_by_tensor'][k]==0 for k in e['drift_by_tensor'] if k.startswith('m.classifier.'));assert sha(t.old.BASE)==manifest['Base_A08_sha256'];results[arm]={'updates':3600,'physical_training_forwards':28800,'physical_training_backwards':28800,'optimizer_all_steps':3600,'full_state_exact':True,'every_journal_input_plan_centre_and_physical_forward_verified':True,'fresh_CPU_FIT_forwards':4,'fresh_CPU_GPU_fit_max_abs':max(errs)};del m,h,st,bundle
 write(PUB/'independent_models.json',{'passed':True,'source_models_manifest_sha256':sha(PUB/'source_models_manifest.json'),'models':results,'all14400updates_verified':True,'fresh_CPU_FIT_forwards':16,'new_source_diagnostic_forwards':0,'target_readouts':0});print(json.dumps({'all_source_models_audited':True}),flush=True)

def final():
 r=j(PUB/'source_readout.json');z=np.load(OUT/'source_scores.private.npz');assert sha(OUT/'source_scores.private.npz')==r['raw_score_cache_sha256'];old=j(OLD/'bound.private.json');rows=[x for x in old['requests'] if x['role']=='diagnostic'];y=np.array([x['true_label']=='A5C' for x in rows]);d=np.array([x['true_label']!=x['directory_label'] for x in rows]);assert (int(y.sum()),int((~y).sum()))==(39,94);metrics={}
 for arm in z.files:
  s=z[arm];pred=s>0;tp=int((pred&y).sum());fp=int((pred&~y).sum());tn=int((~pred&~y).sum());fn=int((~pred&y).sum());ba=.5*(tp/39+tn/94);auc=sum(1 if a>b else .5 if a==b else 0 for a in s[y] for b in s[~y])/(39*94);v=r['metrics'][arm];assert [tp,tn,fp,fn]==[v[k] for k in ['tp','tn','fp','fn']] and abs(ba-v['balanced_accuracy'])<1e-12 and abs(auc-v['AUC'])<1e-12;metrics[arm]={'tp':tp,'fp':fp,'BA':ba,'AUC':auc}
 for aa,bb in [('DC','SC'),('DF','SF'),('DC','DF'),('SC','SF')]:
  for group,mask in [('A5C_correct',y),('A4C_errors',~y)]:
   a=mask&(z[aa]>0);b=mask&(z[bb]>0);assert r['paired'][aa+'-'+bb][group]=={'shared':int((a&b).sum()),'A_only':int((a&~b).sum()),'B_only':int((~a&b).sum())}
 absolute={a:bool(v['BA']>=.8 and v['AUC']>=max(.85,metrics['G-F']['AUC']+.03) and v['fp']<=5 and v['tp']>=28 and v['BA']>=max(metrics['geometry-'+a[1]]['BA'],metrics['constant']['BA'])+.1) for a,v in metrics.items() if a in ['SF','SC','DF','DC']};dc,sc=metrics['DC'],metrics['SC'];inc=bool(dc['BA']>=sc['BA']+.05 and dc['AUC']>=sc['AUC']+.03 and dc['fp']<=sc['fp'] and dc['tp']>=sc['tp']+4);pair='C' if any(absolute[a] for a in ['DC','SC']) else 'F' if any(absolute[a] for a in ['DF','SF']) else None;assert absolute==r['absolute'] and inc==r['DC_minus_SC_source_increment'] and (absolute['DC'] and inc)==r['coverage_mechanism_supported'] and pair==r['target_pair_geometry'] and (pair is not None)==r['source_qualified_for_target_card'];fr=j(OUT/'models_freeze.json');assert fr['actual_git_objects_verified'];rec=j(OUT/'process_readout_1.json');assert rec['start_epoch']>fr['freeze_epoch'] and rec['exit_code']==0
 write(PUB/'independent_source_final.json',{'passed':True,'raw_score_cache_sha256':sha(OUT/'source_scores.private.npz'),'all_counts_auc_paired_source_gates_reconstructed':True,'metrics':metrics,'absolute':absolute,'DC_minus_SC_source_increment':inc,'source_qualified_for_target_card':pair is not None,'coverage_mechanism_supported':absolute['DC'] and inc,'target_pair_geometry':pair,'new_source_CNN_forwards':0,'new_target_readouts':0,'scope':'Cached source view classification; not targetA5Crecovery/protection/F1 or clinical evidence.'});print(json.dumps({'source_final_audit_pass':True,'source_qualified':pair is not None}),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['binding','binding_supplement','models','final']);a=p.parse_args();start=time.time();ok=False
 try:globals()[a.stage]();ok=True
 finally:
  u=resource.getrusage(resource.RUSAGE_SELF);write(OUT/f'audit_{a.stage}_worker_process.json',{'pid':os.getpid(),'ok':ok,'CPU_seconds':u.ru_utime+u.ru_stime,'wall_seconds':time.time()-start,'GPU_job_wall_seconds':0})
