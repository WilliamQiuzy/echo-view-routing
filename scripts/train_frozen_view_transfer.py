#!/usr/bin/env python3
"""A53: frozen source-feature supplement; six real CPU-only linear heads."""
import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES','0');os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8');os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse,copy,hashlib,json,resource,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
import echo_routing.vit.target_mvit as tm
from echo_routing.temporal import frozen_view_transfer as ft,direct_view_training as dt
from echo_routing.temporal import pw_transfer_live as reference
from echo_routing.temporal.pw_transfer import state_sha
from scripts import train_direct_view_evidence as src
from scripts.prepare_direct_view_evidence import sha
OUT=ROOT/'runs/a53_frozen_view_transfer_v1';PUB=ROOT/'docs/results/2026-11-01_frozen_direct_view_transfer';PRIOR=ROOT/'runs/a51_verified_direct_view_pilot_v1'
ARMS=('G','L','W');t=src.t44;write=dt.write_json;CARD='docs/superpowers/plans/2026-11-01-frozen-direct-view-evidence-transfer.md'

def digest(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def ctx():return src.ctx()
def source_manifest():return json.loads((ROOT/'docs/results/2026-11-01_direct_view_evidence/pilot/source_models_manifest.json').read_text())
def files():
 paths=['configs/experiments/video_risk_constrained_a5c_v1.yaml','configs/experiments/video_session_correction_v1.yaml','configs/experiments/video_within_view_v1.yaml','configs/experiments/video_router_data_v1.yaml','configs/experiments/video_interval_recognition_v1.yaml','configs/experiments/a5c_source_external_evidence_v1.yaml',CARD,'scripts/train_frozen_view_transfer.py','scripts/audit_frozen_view_transfer.py','scripts/supervise_frozen_view_transfer.py','echo_routing/temporal/frozen_view_transfer.py']
 for m in list(sys.modules.values()):
  p=getattr(m,'__file__',None)
  if p:
   p=Path(p).resolve()
   if p.exists() and p.is_relative_to(ROOT) and '.venv' not in p.parts and p.suffix=='.py':paths.append(str(p.relative_to(ROOT)))
 return {p:sha(ROOT/p) for p in sorted(set(paths))}

def prepare():
 old=None
 if (PUB/'input_manifest.json').exists():
  assert not (PUB/'feature_manifest.json').exists() and not list(OUT.glob('features_*.json')) and not list(OUT.glob('features_*.npy')) and not list((OUT/'heads').glob('*/claim.json')),'cannot revise prepared provenance after formal work'
  old=json.loads((PUB/'input_manifest.json').read_text());history=PUB/'prelaunch_history';history.mkdir(exist_ok=True);(history/(sha(PUB/'input_manifest.json')+'.input.json')).write_bytes((PUB/'input_manifest.json').read_bytes())
  if (PUB/'checks.json').exists():(history/(sha(PUB/'checks.json')+'.checks.json')).write_bytes((PUB/'checks.json').read_bytes())
 c=ctx();folds={};sm=source_manifest()
 for f in (0,1):
  F=t.Fold(c,0,f);O=t.t31.outputs(c.a31,0,f);assert sha(c.a31.out/f'outputs_s0_f{f}.npz')==t.PINS['outputs'][0][str(f)];assert len(F.need)==[5030,5437][f];assert (len(F.L['positives']),len(F.L['negatives']),len(F.L['other']))==[(34,714,1),(69,699,3)][f]
  folds[str(f)]={'rows':len(F.need),'rows_sha256':digest(F.need),'plan_sha256':F.psha,'outputs_sha256':t.PINS['outputs'][0][str(f)],'pool_sha256':sha(c.a31.out/f'pool_{f}.npy'),'runs_sha256':sha(c.a31.out/'data'/f'runs_s0_f{f}.npz'),'request_grid_sha256':digest(np.column_stack([F.W['seq'],F.W['marker'],F.W['s']])),'positive_runs':len(F.L['positives']),'protected_negative_runs':len(F.L['negatives']),'other_runs':len(F.L['other']),'training_occurrences':sum(len(F.occ(u)[0]) for u in range(600))}
 rec={'run_id':'A53','arms':ARMS,'source_bundles':{a:sm['models'][a]['bundle_sha256'] for a in ['L','W']},'source_manifest_sha256':sha(ROOT/'docs/results/2026-11-01_direct_view_evidence/pilot/source_models_manifest.json'),'Base_A08_sha256':sha(src.BASE),'folds':folds,'code':files(),'source_heads_discarded':True,'source_backbones_frozen':True,'new_source_classifier_gate_pass':False,'head_dimension':1024,'head_parameters':1025,'head_precision':'float64','head_steps_each':600,'G_supplement_structurally_zero':True,'prior_gates_not_revised':True,'target_efficacy_scores_at_preparation':0}
 
 rec=json.loads(json.dumps(rec))
 if old:
  for key in ['source_bundles','source_manifest_sha256','Base_A08_sha256','folds','arms','head_dimension','head_parameters','head_precision','head_steps_each']:assert old[key]==rec[key],'scientific input change during provenance repair: '+key
 write(PUB/'input_manifest.json',rec);print(json.dumps({k:v for k,v in rec.items() if k!='code'}))

def verify():
 m=json.loads((PUB/'input_manifest.json').read_text());assert sha(src.BASE)==m['Base_A08_sha256']
 for p,h in m['code'].items():assert sha(ROOT/p)==h,p
 assert sha(ROOT/'docs/results/2026-11-01_direct_view_evidence/pilot/source_models_manifest.json')==m['source_manifest_sha256']
 for a,h in m['source_bundles'].items():assert sha(PRIOR/'source_train'/a/'final.pt')==h
 c=ctx()
 for f in (0,1):
  F=t.Fold(c,0,f);entry=m['folds'][str(f)];assert F.psha==entry['plan_sha256'] and digest(F.need)==entry['rows_sha256'];assert sha(c.a31.out/f'outputs_s0_f{f}.npz')==entry['outputs_sha256'] and sha(c.a31.out/f'pool_{f}.npy')==entry['pool_sha256'] and sha(c.a31.out/'data'/f'runs_s0_f{f}.npz')==entry['runs_sha256'];assert digest(np.column_stack([F.W['seq'],F.W['marker'],F.W['s']]))==entry['request_grid_sha256']
 return m

def encoder(arm):
 if arm=='G':m,h=src.build()
 else:m,h=src.restore(torch.load(PRIOR/'source_train'/arm/'final.pt',map_location='cpu',weights_only=True))
 del h
 for p in m.parameters():p.requires_grad_(False)
 assert not any(p.requires_grad for p in m.parameters());return m.eval()

@torch.no_grad()
def embeddings(m,F,rows):
 fr,idx,mk=F.frames(np.asarray(rows));fr=fr.cuda();idx=idx.cuda();mk=mk.cuda();out=[]
 for k in range(len(rows)):out.append(m.embed(tm.to_rgb(fr[idx[k:k+1]]),tm.marker_planes(mk[k:k+1])).cpu().numpy()[0])
 return np.asarray(out,np.float32)

def check():
 man=verify();c=ctx();F=t.Fold(c,0,0);O=t.t31.outputs(c.a31,0,0);rows=F.need[np.linspace(0,len(F.need)-1,4,dtype=int)];m=encoder('G');start=time.time();e=embeddings(m,F,rows);parity=float(abs(e-O['emb'][rows]).max());assert parity<3e-5;del m
 times={}
 for a in ['L','W']:
  m=encoder(a);torch.cuda.synchronize();start=time.time();embeddings(m,F,rows);torch.cuda.synchronize();times[a]=(time.time()-start)/len(rows);del m;torch.cuda.empty_cache()
 # Exact double-precision linear pooled/window equivalence on actual G data.
 R=F.pl[0];look=np.arange(len(O['emb']));mu,sd=ft.scaler(O['emb'],F.occ(0)[0]);mean=np.stack([w@O['emb'][r].astype(float) for r,w in [t.rc.members(F.pk,int(i),F.off) for i in R]]);x=ft.design(mean,None,(mu,sd),None);h=ft.new_head()
 with torch.no_grad():h.weight[:,:512].fill_(.001)
 z=h(torch.from_numpy(x)).flatten();loss=ft.objective(z);loss.backward();grads=[p.grad.clone() for p in h.parameters()]
 h.zero_grad();zz=[]
 for i in R:
  rr,w=t.rc.members(F.pk,int(i),F.off);xx=ft.design(O['emb'][rr],None,(mu,sd),None);zz.append((h(torch.from_numpy(xx)).flatten()*torch.from_numpy(w)).sum())
 ft.objective(torch.stack(zz)).backward();gradient=max(float(abs(p.grad-g).max()) for p,g in zip(h.parameters(),grads));assert gradient<1e-10
 def execute(stop,st=None):
  h=ft.new_head();o=torch.optim.AdamW(h.parameters(),lr=1e-3,weight_decay=1e-4);begin=0
  if st:
   h.load_state_dict(st['head']);o.load_state_dict(st['optimizer']);begin=st['update']
   if 'CPU_rng' in st:torch.set_rng_state(st['CPU_rng'])
  for u in range(begin,stop):ft.step(h,o,x,u)
  return h,o
 a,oa=execute(6);b,ob=execute(6);assert all(torch.equal(u,v) for u,v in zip(a.parameters(),b.parameters()));b,ob=execute(3);save(OUT/'check_resume.pt',{'head':b.state_dict(),'optimizer':ob.state_dict(),'update':3,'CPU_rng':torch.get_rng_state()});st=torch.load(OUT/'check_resume.pt',map_location='cpu',weights_only=True);(OUT/'check_resume.pt').unlink();b,ob=execute(6,st);assert all(torch.equal(u,v) for u,v in zip(a.parameters(),b.parameters()))
 result={'passed':True,'input_manifest_sha256':sha(PUB/'input_manifest.json'),'G_fresh_fit_subset_vs_A31_embedding_max_abs':parity,'linear_explicit_window_vs_runmean_gradient_error':gradient,'repeat_and_forced_identical_features_exact':True,'disk3plus3_resume_exact':True,'seconds_per_source_window_including_decode':times,'complete_extraction_forecast_seconds':max(times.values())*20934*1.3+120,'head_training_is_only_CPU_linear':True,'new_target_classifier_scores':0,'fit_only_GPU_checks':12,'backbone_parameters_require_grad':False}
 write(PUB/'checks.json',result);print(json.dumps(result))

def save(path,obj):
 tmp=path.with_suffix('.tmp')
 with tmp.open('wb') as f:torch.save(obj,f);f.flush();os.fsync(f.fileno())
 tmp.replace(path);dt.sync_directory(path.parent)

def extract():
 man=verify();freeze=json.loads((OUT/'extract_freeze.json').read_text());assert freeze['input_manifest_sha256']==sha(PUB/'input_manifest.json') and freeze['actual_git_objects_verified'];checks=json.loads((PUB/'checks.json').read_text());assert checks['passed'] and checks['input_manifest_sha256']==sha(PUB/'input_manifest.json');c=ctx();entries={}
 for a in ['L','W']:
  model=encoder(a);frozen_before=state_sha(model.state_dict())
  for f in (0,1):
   F=t.Fold(c,0,f);key=f'{a}_f{f}';path=OUT/f'features_{key}.npy';progress=OUT/f'features_{key}.json';ident={'arm':a,'fold':f,'rows_sha256':digest(F.need),'source_bundle_sha256':man['source_bundles'][a],'input_manifest_sha256':sha(PUB/'input_manifest.json')}
   if progress.exists():
    rec=json.loads(progress.read_text());assert rec['identity']==ident and rec['pending'] is None,'inflight uncertain feature forward';E=np.load(path,mmap_mode='r+')
   else:
    assert not path.exists();E=np.lib.format.open_memmap(path,mode='w+',dtype=np.float32,shape=(len(F.need),512));E[:]=np.nan;E.flush();rec={'identity':ident,'done':0,'pending':None};write(progress,rec)
   for lo in range(rec['done'],len(F.need),32):
    rr=F.need[lo:lo+32];fr,idx,mk=F.frames(rr);fr=fr.cuda();idx=idx.cuda();mk=mk.cuda()
    for j in range(len(rr)):
     k=lo+j;rec['pending']=k;write(progress,rec)
     with torch.no_grad():v=model.embed(tm.to_rgb(fr[idx[j:j+1]]),tm.marker_planes(mk[j:j+1])).cpu().numpy()[0]
     assert np.isfinite(v).all();E[k]=v;E.flush()
     with path.open('r+b') as handle:os.fsync(handle.fileno())
     rec.update(done=k+1,pending=None);write(progress,rec)
    if rec['done']%256<32:write(OUT/'progress.json',{'stage':'extract','arm':a,'fold':f,'windows_done':rec['done'],'windows_total':len(F.need),'pid':os.getpid(),'device':torch.cuda.get_device_name(0),'updated_epoch':time.time()})
   assert E.shape==(len(F.need),512) and np.isfinite(E).all();entries[key]={'identity':ident,'rows':len(F.need),'feature_sha256':sha(path),'physical_frozen_model_forwards':len(F.need),'classifier_scores':0};del E
  assert state_sha(model.state_dict())==frozen_before
  for f in (0,1):entries[f'{a}_f{f}']['encoder_state_sha256_before_after']=frozen_before
  del model;torch.cuda.empty_cache()
 write(PUB/'feature_manifest.json',{'input_manifest_sha256':sha(PUB/'input_manifest.json'),'features':entries,'new_frozen_forward_count':20934,'backbone_optimizer_updates':0,'new_target_classifier_scores':0})

def raw_features(F,c):
 O=t.t31.outputs(c.a31,0,F.f);g=np.asarray(O['emb'][F.need],np.float64);out={'G':g}
 for a in ['L','W']:
  p=OUT/f'features_{a}_f{F.f}.npy';fm=json.loads((PUB/'feature_manifest.json').read_text());assert sha(p)==fm['features'][f'{a}_f{F.f}']['feature_sha256'];out[a]=np.asarray(np.load(p),np.float64)
 return out

def setup(F,c,arm,scales=None):
 feat=raw_features(F,c);index=np.full(len(F.W['seq']),-1,np.int64);index[F.need]=np.arange(len(F.need));trainrows=np.unique(np.concatenate([F.occ(u)[0] for u in range(600)]));assert (index[trainrows]>=0).all()
 if scales is None:
  mu,sd=ft.scaler(feat['G'],index[trainrows]);smu,ssd=(np.zeros(512),np.ones(512)) if arm=='G' else ft.scaler(feat[arm],index[trainrows]);scales=(mu,sd,smu,ssd)
 mu,sd,smu,ssd=scales;window=ft.design(feat['G'],None if arm=='G' else feat[arm],(mu,sd),(smu,ssd));X=np.zeros((len(F.pk['base']),1024),np.float64)
 for k in np.flatnonzero(F.el):
  rr,w=t.rc.members(F.pk,int(k),F.off);assert abs(w.sum()-1)<1e-10 and (index[rr]>=0).all();X[k]=w@window[index[rr]]
 return X,window,scales

def seal_features():
 verify();assert not (PUB/'fit_design_manifest.json').exists();c=ctx();items={};arrays={}
 for f in (0,1):
  F=t.Fold(c,0,f)
  for arm in ARMS:
   X,_,sc=setup(F,c,arm);key=f'{arm}_f{f}';arrays[key]=np.stack(sc);items[key]={'scales_sha256':digest(np.stack(sc)),'G_scales_sha256':digest(np.stack(sc[:2])),'full_fit_pooled_X_sha256':digest(X),'plan_sha256':F.psha}
  assert len({items[f'{a}_f{f}']['G_scales_sha256'] for a in ARMS})==1
 np.savez(OUT/'fit_scalers.private.npz',**arrays);write(PUB/'fit_design_manifest.json',{'items':items,'feature_manifest_sha256':sha(PUB/'feature_manifest.json'),'fit_scalers_sha256':sha(OUT/'fit_scalers.private.npz'),'head_updates':0})

def train():
 man=verify();fr=json.loads((OUT/'head_train_freeze.json').read_text());assert fr['feature_manifest_sha256']==sha(PUB/'feature_manifest.json') and fr['fit_design_manifest_sha256']==sha(PUB/'fit_design_manifest.json') and fr['actual_git_objects_verified'];designs=json.loads((PUB/'fit_design_manifest.json').read_text());assert sha(OUT/'fit_scalers.private.npz')==designs['fit_scalers_sha256'];scalers=np.load(OUT/'fit_scalers.private.npz');c=ctx();entries={}
 for f in (0,1):
  F=t.Fold(c,0,f);Gscale=None
  for arm in ARMS:
   X,_,sc=setup(F,c,arm,tuple(scalers[f'{arm}_f{f}']));assert digest(X)==designs['items'][f'{arm}_f{f}']['full_fit_pooled_X_sha256']
   if Gscale is None:Gscale=sc[:2]
   else:assert all(np.array_equal(u,v) for u,v in zip(Gscale,sc[:2]))
   key=f'{arm}_f{f}';rd=OUT/'heads'/key;rd.mkdir(parents=True,exist_ok=True);ident={'arm':arm,'fit_fold':f,'plan_sha256':F.psha,'input_manifest_sha256':sha(PUB/'input_manifest.json'),'feature_manifest_sha256':sha(PUB/'feature_manifest.json')};h=ft.new_head();opt=torch.optim.AdamW(h.parameters(),lr=1e-3,weight_decay=1e-4);rp=rd/'resume.pt';jp=rd/'updates.private.jsonl';logs=[];start=0
   if (rd/'manifest.json').exists():
    e=json.loads((rd/'manifest.json').read_text());assert e['identity']==ident and sha(rd/'final.pt')==e['head_sha256'] and sha(rd/'optimizer_final.pt')==e['optimizer_final_sha256'];entries[key]=e;continue
   if (rd/'claim.json').exists():
    assert json.loads((rd/'claim.json').read_text())==ident
    if not rp.exists():rp=rd/'optimizer_final.pt'
    assert rp.exists(),'claimed head missing safe full state';st=torch.load(rp,map_location='cpu',weights_only=True);h.load_state_dict(st['head']);opt.load_state_dict(st['optimizer']);torch.set_rng_state(st['CPU_rng']);start=st['update'];logs=[json.loads(x) for x in jp.read_text().splitlines()];dt.validate_recovery(rd,start,len(logs))
   else:write(rd/'claim.json',ident)
   initial=float(torch.nn.functional.softplus(torch.tensor(2.,dtype=torch.float64))+torch.nn.functional.softplus(torch.tensor(-2.,dtype=torch.float64)))
   with jp.open('a') as jf:
    def persist(z):jf.write(json.dumps(z)+'\n');jf.flush();os.fsync(jf.fileno());logs.append(z)
    for u in range(start,600):
     def compute():
      rec=ft.step(h,opt,X[F.pl[u]],u);rec.update(plan_row_sha256=digest(F.pl[u]),pid=os.getpid(),device='CPU');return rec
     dt.claimed_update(rd,u,compute,persist)
     if (u+1)%150==0:save(rp,{'head':h.state_dict(),'optimizer':opt.state_dict(),'update':u+1,'identity':ident,'CPU_rng':torch.get_rng_state()})
   assert len(logs)==600 and {int(v['step']) for v in opt.state.values()}=={600}
   if arm=='G':assert not torch.count_nonzero(h.weight[:,512:])
   save(rd/'final.pt',{'head':h.state_dict(),'scales':[torch.from_numpy(x) for x in sc],'identity':ident,'updates':600});
   if rp!=rd/'optimizer_final.pt':rp.replace(rd/'optimizer_final.pt')
   dt.sync_directory(rd)
   e={'identity':ident,'updates':600,'head_parameters':1025,'active_parameters':513 if arm=='G' else 1025,'head_sha256':sha(rd/'final.pt'),'optimizer_final_sha256':sha(rd/'optimizer_final.pt'),'journal_sha256':sha(jp),'scales_sha256':digest(np.stack(sc)),'G_scaler_sha256':digest(np.stack(sc[:2])),'pooled_run_logits':2400,'loss_backwards':600,'CNN_backwards':0,'fit_initial_first_batch_loss':initial,'fit_final_first_batch_loss':float(ft.objective(h(torch.from_numpy(X[F.pl[0]])).flatten()).detach())};write(rd/'manifest.json',e);entries[key]=e;print(json.dumps({'head':key,'updates':600}),flush=True)
 write(PUB/'heads_manifest.json',{'run_id':'A53','models':entries,'input_manifest_sha256':sha(PUB/'input_manifest.json'),'feature_manifest_sha256':sha(PUB/'feature_manifest.json'),'head_updates':3600,'backbone_updates':0})

def readout():
 verify();man=json.loads((PUB/'heads_manifest.json').read_text());assert man['input_manifest_sha256']==sha(PUB/'input_manifest.json') and man['feature_manifest_sha256']==sha(PUB/'feature_manifest.json')
 for key,e in man['models'].items():assert e['identity']['input_manifest_sha256']==man['input_manifest_sha256'] and e['identity']['feature_manifest_sha256']==man['feature_manifest_sha256'] and e['identity']['fit_fold']==int(key[-1])
 fr=json.loads((OUT/'heads_freeze.json').read_text());assert fr['heads_manifest_sha256']==sha(PUB/'heads_manifest.json') and fr['actual_git_objects_verified'];audit=json.loads((PUB/'independent_heads.json').read_text());assert audit['passed'] and audit['heads_manifest_sha256']==sha(PUB/'heads_manifest.json');designs=json.loads((PUB/'fit_design_manifest.json').read_text());assert designs['feature_manifest_sha256']==man['feature_manifest_sha256'];c=ctx();cache={}
 for held in (0,1):
  F=t.Fold(c,0,held);cache[f'rows_f{held}']=F.need
  for arm in ARMS:
   key=f'{arm}_f{1-held}';entry=man['models'][key];path=OUT/'heads'/key/'final.pt';assert sha(path)==entry['head_sha256'];ck=torch.load(path,map_location='cpu',weights_only=True);assert ck['identity']==entry['identity'] and ck['updates']==600 and digest(np.stack([v.numpy() for v in ck['scales']]))==designs['items'][key]['scales_sha256']==entry['scales_sha256'];assert ck['identity']['plan_sha256']==man['models'][key]['identity']['plan_sha256'];_,X,_=setup(F,c,arm,tuple(v.numpy() for v in ck['scales']));cell=OUT/f'scores_{arm}_f{held}.npy';guard=cell.with_suffix('.json');identity={'head_sha256':sha(path),'feature_manifest_sha256':sha(PUB/'feature_manifest.json'),'rows_sha256':digest(F.need),'rows':len(F.need)}
   if guard.exists():
    r=json.loads(guard.read_text());assert r['identity']==identity and r['state']=='complete','uncertain once-only scorecell';assert sha(cell)==r['sha256'];scores=np.load(cell)
   else:
    assert not cell.exists();write(guard,{'identity':identity,'state':'pending'});scores=X@ck['head']['weight'].numpy()[0]+float(ck['head']['bias'][0]);assert np.isfinite(scores).all();np.save(cell,scores)
    with cell.open('r+b') as handle:os.fsync(handle.fileno())
    dt.sync_directory(OUT);write(guard,{'identity':identity,'state':'complete','sha256':sha(cell),'classifier_window_scores':len(scores)})
   cache[f'scores_{arm}_f{held}']=scores
 dest=OUT/'readout.private.npz'
 with dest.open('xb') as f:np.savez_compressed(f,**cache);f.flush();os.fsync(f.fileno())
 write(PUB/'raw_complete.json',{'cache_sha256':sha(dest),'classifier_window_scores':31401,'heads_freeze':fr,'source_heads_used':False,'backbone_updates':0})

def evaluate():
 # Reuse unchanged reference evaluator, replace only cache filename/receipt schema expected by its adapter.
 raw=json.loads((PUB/'raw_complete.json').read_text());cache=np.load(OUT/'readout.private.npz');c=ctx();a8=t.rsh.a08_ctx();data=t.t31.vpr.training_data(c.a31.a15.a10.a07p);old=json.loads((ROOT/'docs/results/2026-10-28_risk_constrained_a5c/readout_s0.json').read_text());rec={'cells':{},'oof':{},'raw_cache_sha256':raw['cache_sha256']};sets={a:[] for a in ARMS};sources={a:set() for a in ARMS}
 for held in (0,1):
  F=t.Fold(c,0,held);D=t.d32.fold_data(c.d,0,held);D['M']=t.d32.set_masks(D);P={'seqs':D['seqs'],'n_of':dict(data.n),'label_of':dict(data.label)};evaluation=t.bc.evalset(P,list(range(len(D['seqs']))),f'heldout_f{held}');base=t.tables_pred(F.pk['base'],D,a8.cfg,P,evaluation);bp=base.pop('preds');base.pop('pred');base['boundaries']=t.bounds(D,bp);base.update(t.a5c_extras(D,F.pk['base'])[0]);assert base['sequence_row']==old['cells'][f'heldout_f{held}']['Base']['sequence_row'];cell={'Base':base};assert np.array_equal(cache[f'rows_f{held}'],F.need)
  for arm in ARMS:
   s=cache[f'scores_{arm}_f{held}'];full=np.full(len(F.W['seq']),np.nan);full[F.need]=s;r=t.rc.run_weighted_mean(F.pk,full,F.off,F.el);pred=t.rc.one_way(F.pk['base'],r);ref_r,ref_p=reference.reference_labels(F.pk,F.need,s,F.off);assert np.array_equal(pred,ref_p) and np.max(abs(ref_r[F.el]-r[F.el]))<1e-10
   table=t.tables_pred(pred,D,a8.cfg,P,evaluation);p=table.pop('preds');table.pop('pred');extra,vid=t.a5c_extras(D,pred);table.update(extra);sources[arm]|=vid;fix=(F.pk['base']==2)&(F.pk['labels']==3)&(pred==3);harm=(F.pk['base']==2)&(F.pk['labels']==2)&(pred==3);sets[arm].append((fix,harm));table.update(S1_fixed_count=int(fix.sum()),A4C_harmed_count=int(harm.sum()),net_fixed_count=int(fix.sum()-harm.sum()),S1_source_count=len(vid),non_A4C_labels_unchanged=bool(np.array_equal(pred[~F.el],F.pk['base'][~F.el])),boundaries=t.bounds(D,p));cell[arm]=table
  rec['cells'][str(held)]=cell
 for arm in ARMS:rec['oof'][arm]={'A5C_fixed':sum(int(a.sum()) for a,b in sets[arm]),'A4C_harmed':sum(int(b.sum()) for a,b in sets[arm]),'S1_source_count':len(sources[arm]),'F1_mean':float(np.mean([rec['cells'][str(h)][arm]['sequence_row']['segment_f1_50_multi'] for h in (0,1)]))}
 rec['paired']={}
 for a,b in [('L','W'),('L','G'),('W','G')]:rec['paired'][a+'_vs_'+b]={kind:{'shared':sum(int((sets[a][h][j]&sets[b][h][j]).sum()) for h in (0,1)),'A_only':sum(int((sets[a][h][j]&~sets[b][h][j]).sum()) for h in (0,1)),'B_only':sum(int((~sets[a][h][j]&sets[b][h][j]).sum()) for h in (0,1))} for j,kind in enumerate(['fix','harm'])}
 rec['gates']=gates(rec);write(PUB/'readout.json',rec);print(json.dumps({'oof':rec['oof'],'gates':rec['gates']}))

def gates(r):
 absolute={}
 for a in ARMS:
  v=[r['cells'][str(h)][a] for h in (0,1)];b=[r['cells'][str(h)]['Base'] for h in (0,1)];delta=lambda key:[x['sequence_row'][key]-z['sequence_row'][key] for x,z in zip(v,b)]
  c={'fix':r['oof'][a]['A5C_fixed']>=11 and all(x['S1_fixed_count']>=1 and x['S1_source_count']>=1 for x in v) and r['oof'][a]['S1_source_count']>=3,'harm':r['oof'][a]['A4C_harmed']<=14,'net':all(x['net_fixed_count']>0 for x in v) and r['oof'][a]['A5C_fixed']-r['oof'][a]['A4C_harmed']>=5,'F1':min(delta('segment_f1_50_multi'))>0 and np.mean(delta('segment_f1_50_multi'))>=.003,'native':min(delta('native_macro_f1'))>=-.002,'boundary':min(delta('boundary_recall_multi'))>=-.005,'short':sum(x['counts']['fn_short'] for x in v)<=79,'structure':all(x['non_A4C_labels_unchanged'] for x in v)};absolute[a]={**c,'all':all(c.values())}
 contribution={}
 for control in ['G','W']:contribution['L_vs_'+control]=r['oof']['L']['A5C_fixed']>=r['oof'][control]['A5C_fixed']+5 and r['oof']['L']['A4C_harmed']<=r['oof'][control]['A4C_harmed'] and all(r['cells'][str(h)]['L']['sequence_row']['segment_f1_50_multi']>r['cells'][str(h)][control]['sequence_row']['segment_f1_50_multi'] for h in (0,1))
 return {'absolute':absolute,'source_contribution':contribution,'source_label_package_pass':absolute['L']['all'] and all(contribution.values())}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','check','extract','seal_features','train','readout','evaluate']);a=ap.parse_args();torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False;OUT.mkdir(parents=True,exist_ok=True);lock=dt.acquire_lock(OUT/'exclusive_worker.lock');start=time.time();ok=False
 try:globals()[a.stage]();ok=True
 finally:
  lock.close();u=resource.getrusage(resource.RUSAGE_SELF);write(OUT/f'{a.stage}_worker_process.json',{'pid':os.getpid(),'ok':ok,'stage':a.stage,'CPU_seconds':u.ru_utime+u.ru_stime,'wall_seconds':time.time()-start,'GPU_job_wall_seconds':time.time()-start if a.stage in ['check','extract'] else 0})
if __name__=='__main__':main()
