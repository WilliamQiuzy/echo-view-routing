#!/usr/bin/env python3
"""A55 real full-body matched coverage/geometry source continuations."""
import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES','0');os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8');os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse,copy,hashlib,json,resource,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
import torch.nn.functional as TF
from scripts import train_direct_view_evidence as old
from scripts.prepare_annotation_coverage import OUT,PUB,SRC,CARD,sha,j,arsha,write
from echo_routing.temporal import direct_view_training as dt
from echo_routing.temporal.pw_transfer import state_sha
from echo_routing.temporal.dev_train import lr_factor
import echo_routing.vit.target_mvit as tm
ARMS=('SF','SC','DF','DC');U=3600
ADDENDUM='docs/superpowers/plans/2026-11-02-a55-source-package-clarification.md'

def pins():
 paths={Path(CARD),Path(ADDENDUM),Path('scripts/train_annotation_coverage.py'),Path('scripts/audit_annotation_coverage.py'),Path('scripts/supervise_annotation_coverage.py'),Path('scripts/prepare_annotation_coverage.py'),Path('echo_routing/temporal/annotation_coverage.py')}
 for mod in list(sys.modules.values()):
  f=getattr(mod,'__file__',None)
  if f:
   p=Path(f).resolve()
   if p.is_relative_to(ROOT) and '.venv' not in p.parts and p.suffix=='.py':paths.add(p.relative_to(ROOT))
 for p in (ROOT/'configs/experiments').glob('*.yaml'):
  if p.name in ['video_risk_constrained_a5c_v1.yaml','video_within_view_v1.yaml','video_interval_recognition_v1.yaml','video_router_data_v1.yaml','video_session_correction_v1.yaml']:paths.add(p.relative_to(ROOT))
 return {str(p):sha(ROOT/p) for p in sorted(paths)}

def save(path,state):
 tmp=path.with_suffix('.tmp')
 with tmp.open('wb') as f:torch.save(state,f);f.flush();os.fsync(f.fileno())
 tmp.replace(path);dt.sync_directory(path.parent)

class Inputs:
 def __init__(self):
  self.b=j(OUT/'bound.private.json');self.parents=self.b['parents'];self.fit=[p['source'] for p in self.parents];self.diag=self.b['diagnostic'];self.cache={};self.axis={(v['folder'],v['video']):v['native_pts_indices'] for v in self.b['videos']};self.plan=np.load(OUT/'parent_plan.npy');self.dense=np.load(OUT/'dense_centres.npy');assert self.plan.shape==self.dense.shape==(U,8)
  for c in self.b['caches']:
   p=ROOT/c['cache'];assert sha(p)==c['cache_sha256'];z=np.load(p);self.cache[tuple(c['key'])]={'index':{int(o):i for i,o in enumerate(z['ordinals'])},'F':z['F'],'C':z['C']}
 def window(self,parent,centre,geom):
  key=tuple(parent['key']);cache=self.cache[key];n=len(self.axis[key]);idx=[cache['index'][max(0,min(n-1,int(centre)+3*(i-8)))] for i in range(16)];return cache[geom][np.asarray(idx)]
 def diagnostic(self,r,geom):return self.window({'key':[r['directory_label'],r['video']]},r['centre'],geom)
 def batch(self,u,arm):
  for k,i in enumerate(self.plan[u]):
   p=self.parents[int(i)];centre=int(self.dense[u,k]) if arm[0]=='D' else p['centre'];yield p,self.window(p,centre,arm[1]),centre

def prepare():
 assert not list((OUT/'source_train').glob('*/claim.json')) and not list((OUT/'source_cells').glob('*')),'cannot revise input manifest after formal model/readout claim'
 if (PUB/'input_manifest.json').exists():raise RuntimeError('input manifest alreadyprepared; explicit prelaunch archive/identity review required, no silent overwrite')
 q=j(PUB/'qualification.json');a=j(PUB/'independent_binding.json');extra=j(PUB/'independent_binding_supplement.json');integrity=j(PUB/'processed_integrity.json');assert integrity['passed'] and integrity['qualification_sha256']==sha(PUB/'qualification.json');assert extra['passed'] and extra['qualification_sha256']==sha(PUB/'qualification.json');assert q['passed'] and a['passed'] and a['qualification_sha256']==sha(PUB/'qualification.json');assert sha(OUT/'bound.private.json')==q['private_bound_sha256'];data=Inputs();assert len(data.parents)==569 and len(data.diag)==133
 rec={'run_id':'A55','arms':list(ARMS),'updates_each':U,'batch':8,'fit':569,'diagnostic':133,'bound_sha256':sha(OUT/'bound.private.json'),'qualification_sha256':sha(PUB/'qualification.json'),'parent_plan_sha256':sha(OUT/'parent_plan.npy'),'dense_centres_sha256':sha(OUT/'dense_centres.npy'),'Base_A08_sha256':sha(old.BASE),'reference_A51_initial_state_sha256':j(old.PUB/'source_models_manifest.json')['models']['L']['initial_full_state_sha256'],'reference_A51_models_manifest_sha256':sha(old.PUB/'source_models_manifest.json'),'reference_source_scores_sha256':sha(old.OUT/'source_scores.private.npz'),'reference_source_readout_sha256':sha(old.PUB/'source_readout.json'),'initial_state_seed':'A44source0/fold0build, allarmsidentical','rates':old.t44.LR,'warmup':360,'code':pins(),'source_diagnostic_forwards':0,'source_CNN_updates':0,'versions':{'torch':torch.__version__,'numpy':np.__version__},'raw_source_patients':[36,10],'not_fresh_clinical_validation':True};write(PUB/'input_manifest.json',rec);print(json.dumps({'prepared':True,'updates_each':U,'arms':list(ARMS)}))

def verify():
 m=j(PUB/'input_manifest.json');assert sha(OUT/'bound.private.json')==m['bound_sha256'] and sha(OUT/'parent_plan.npy')==m['parent_plan_sha256'] and sha(OUT/'dense_centres.npy')==m['dense_centres_sha256'] and sha(old.BASE)==m['Base_A08_sha256'];assert sha(old.OUT/'source_scores.private.npz')==m['reference_source_scores_sha256'] and sha(old.PUB/'source_models_manifest.json')==m['reference_A51_models_manifest_sha256'];assert torch.__version__==m['versions']['torch'] and np.__version__==m['versions']['numpy']
 for p,h in m['code'].items():assert sha(ROOT/p)==h,p
 return m

def build(device='cuda'):
 torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;return old.build(device)

def score(m,h,fr,device='cuda',base=False):
 x=tm.to_rgb(torch.from_numpy(fr[None]).to(device));marker=tm.marker_planes(torch.tensor([8],device=device));e=m.embed(x,marker)
 if base:
  z=m.classifier(e);return z[:,3]-z[:,2]
 return h(e).reshape(-1)

def step(m,h,opt,data,u,arm,force=None,schedule_index=None):
 old.t44.t31.sched(opt,u if schedule_index is None else schedule_index,{'updates':U,'warmup':360});opt.zero_grad(set_to_none=True);loss_sum=0.;hashes=[];centres=[];a=force or arm
 for p,fr,c in data.batch(u,a):
  hashes.append(arsha(fr));centres.append(c);s=score(m,h,fr);y=s.new_tensor([float(p['label'])]);loss=TF.binary_cross_entropy_with_logits(s,y)/8;loss.backward();loss_sum+=float(loss.detach())
 groups={k:old.t44.sq_norm([p.grad for p in pp]) for k,pp in old.t44.groups_of(m,h).items()};norm=float(torch.nn.utils.clip_grad_norm_(old.t44.trainable(m,h),1.));assert np.isfinite(norm) and np.isfinite(loss_sum);opt.step()
 return {'u':int(u),'schedule_index':int(u if schedule_index is None else schedule_index),'arm':arm,'loss':loss_sum,'groups':groups,'gradient_norm':norm,'plan_row_sha256':arsha(data.plan[u]),'actual_centres_sha256':arsha(np.asarray(centres,np.int64)),'input_window_sha256':hashlib.sha256(''.join(hashes).encode()).hexdigest(),'physical_forwards':8,'physical_backwards':8,'pid':os.getpid(),'device':torch.cuda.get_device_name(0)}

def ckpt(m,h,opt,u,identity):return {**old.t44.state_of(m,h,opt,1.),'identity':identity,'update':int(u)}

def check():
 manifest=verify();data=Inputs();m,h=build();opt=old.t44.make_opt(m,h);initial={k:v.clone() for k,v in old.flat(m,h).items()};assert state_sha(initial)==manifest['reference_A51_initial_state_sha256'];groups=old.t44.groups_of(m,h);assert {id(p) for g in opt.param_groups for p in g['params']}=={id(p) for pp in groups.values() for p in pp};batch0=list(data.batch(0,'SC'));rows=[batch0[i] for i in (0,4,1,5)];yy=np.array([p['label'] for p,fr,c in rows],float);z=torch.cat([score(m,h,fr) for p,fr,c in rows]);assert np.array_equal(z.detach().cpu().numpy(),np.full(4,-2.));loss=TF.binary_cross_entropy_with_logits(z,torch.as_tensor(yy,dtype=torch.float32,device='cuda'));loss.backward();ferr=abs(float(loss.detach())-float(np.logaddexp(0,-2.)+2*yy.mean()));berr=abs(float(h.fc.bias.grad)-(1/(1+np.exp(2))-yy.mean()));assert max(ferr,berr)<1e-5
 with torch.no_grad():h.fc.weight.fill_(.001)
 opt.zero_grad(set_to_none=True);z=torch.cat([score(m,h,fr) for p,fr,c in rows]);TF.binary_cross_entropy_with_logits(z,torch.as_tensor(yy,dtype=torch.float32,device='cuda')).backward();gr={n:p.grad.detach().clone() for n,p in list(m.named_parameters())+[("H."+n,p) for n,p in h.named_parameters()] if p.requires_grad};opt.zero_grad(set_to_none=True)
 for p,fr,c in rows:(TF.binary_cross_entropy_with_logits(score(m,h,fr),torch.tensor([float(p['label'])],device='cuda'))/4).backward()
 errors=[float((p.grad-gr[n]).abs().max())/(1+float(gr[n].abs().max())) for n,p in list(m.named_parameters())+[("H."+n,p) for n,p in h.named_parameters()] if p.requires_grad];assert max(errors)<1e-4;assert all(old.t44.sq_norm([p.grad for p in pp])>0 for pp in groups.values());del m,h,opt,gr;torch.cuda.empty_cache()
 def execute(arm,force=None,state=None,n=6):
  m,h=build();o=old.t44.make_opt(m,h);start=0
  if state is not None:old.t44.load_state(m,h,o,state);start=state['update']
  logs=[step(m,h,o,data,u,arm,force,schedule_index=360+u) for u in range(start,n)];return m,h,o,logs
 m,h,o,logs=execute('SC');expected={k:v.clone() for k,v in old.flat(m,h).items()};assert all(logs[0]['groups'][k]==0 for k in logs[0]['groups'] if k!='scalar_head');assert all(any(x['groups'][k]>0 for x in logs[1:]) for k in groups);del m,h,o
 for a,force in [('SC',None),('DC','SC'),('SF','SC')]:
  m,h,o,_=execute(a,force);assert state_sha(old.flat(m,h))==state_sha(expected);del m,h,o
 m,h,o,_=execute('SC',n=3);save(OUT/'check_resume.pt',ckpt(m,h,o,3,{}));del m,h,o;st=torch.load(OUT/'check_resume.pt',map_location='cpu',weights_only=True);m,h,o,_=execute('SC',state=st);assert state_sha(old.flat(m,h))==state_sha(expected);del m,h,o,st;(OUT/'check_resume.pt').unlink()
 m,h=build();o=old.t44.make_opt(m,h);torch.cuda.synchronize();start=time.time()
 for u in range(40):
  step(m,h,o,data,u,'DC',schedule_index=360+u)
  if (u+1)%20==0:save(OUT/'timed_check_resume.pt',ckpt(m,h,o,u+1,{}))
 torch.cuda.synchronize();elapsed=time.time()-start;(OUT/'timed_check_resume.pt').unlink();current=old.flat(m,h);drift={k:float((current[k].double()-initial[k].double()).norm()) for k in initial};assert all(torch.equal(current[k],initial[k]) for k in initial if k.startswith('m.classifier.'));assert all(any(drift['m.'+n]>0 for n,p in m.named_parameters() if p is g) or k=='scalar_head' for k,pp in old.t44.groups_of(m,h).items() for g in pp[:1])
 v={'passed':True,'input_manifest_sha256':sha(PUB/'input_manifest.json'),'independent_BCE_formula_error':ferr,'bias_gradient_error':berr,'production_grouped_FP32_relative_gradient_error':max(errors),'repeat_forced_geometry_density_and_disk3plus3_resume':True,'all_trunk_marker_projection_group_gradients':True,'old5class_frozen_Base_unchanged':sha(old.BASE)==manifest['Base_A08_sha256'],'timed_steps':40,'timed_wall_seconds':elapsed,'timed_seconds_per_step_including20update_checkpoint':elapsed/40,'source_four_model_forecast_GPU_hours':4*U*elapsed/40*1.25/3600,'GPU_peak_allocated_bytes':torch.cuda.max_memory_allocated(),'GPU_peak_reserved_bytes':torch.cuda.max_memory_reserved(),'trainable_parameter_count':sum(p.numel() for p in old.t44.trainable(m,h)),'check_schedule_indices':list(range(360,400)),'source_diagnostic_forwards':0,'FIT_physical_forwards':572,'FIT_physical_backwards':572,'fit_check_mutable_models':8,'formal_models_trained':0};write(PUB/'checks.json',v);print(json.dumps(v),flush=True)

def fit_panel(m,h,data,arm):
 ids=np.linspace(0,len(data.parents)-1,32,dtype=int);result=[]
 with torch.no_grad():
  for i in ids:
   p=data.parents[int(i)];result.append(float(score(m,h,data.window(p,p['centre'],arm[1]))))
 y=np.array([data.parents[int(i)]['label'] for i in ids]);s=np.asarray(result);return {'BCE':float(np.mean(np.logaddexp(0,s)-y*s)),'A5C_correct':int(((s>0)&(y==1)).sum()),'A4C_harmed':int(((s>0)&(y==0)).sum()),'A5C_total':int(y.sum()),'A4C_total':int(len(y)-y.sum()),'rows':32,'fit_forwards':32}

def final_fit(m,h,data,arm,rd):
 values={};model_sha=state_sha(old.flat(m,h));original_centres={(tuple(p['key']),p['centre']) for p in data.parents}
 with torch.no_grad():
  for role in ['original','additional_centre_panel']:
   path=rd/('fit_'+role+'.npy');seal=rd/('fit_'+role+'.json')
   if seal.exists():
    previous=j(seal);assert previous['model_full_state_sha256']==model_sha and sha(path)==previous['scores_sha256'];values[role]=previous['metrics'];continue
   ss=[]
   for p in data.parents:
    c=p['centre'] if role=='original' else next((x for x in p['points'] if (tuple(p['key']),x) not in original_centres),p['centre']);ss.append(float(score(m,h,data.window(p,c,arm[1]))))
   a=np.asarray(ss,np.float32);tmp=path.with_suffix('.tmp')
   with tmp.open('wb') as f:np.save(f,a);f.flush();os.fsync(f.fileno())
   tmp.replace(path);dt.sync_directory(rd);y=np.array([p['label'] for p in data.parents]);v={'A5C_correct':int(((a>0)&(y==1)).sum()),'A5C_total':int(y.sum()),'A4C_harmed':int(((a>0)&(y==0)).sum()),'A4C_total':int((y==0).sum()),'BCE':float(np.mean(np.logaddexp(0,a)-y*a)),'scores_sha256':sha(path)};values[role]=v;write(seal,{'model_full_state_sha256':model_sha,'scores_sha256':sha(path),'metrics':v,'committed_FIT_forwards':len(data.parents)})
 return values

def train():
 manifest=verify();check=j(PUB/'checks.json');fr=j(OUT/'input_freeze.json');assert check['passed'] and check['input_manifest_sha256']==sha(PUB/'input_manifest.json') and fr['actual_git_objects_verified'] and fr['input_manifest_sha256']==sha(PUB/'input_manifest.json');program=j(OUT/'coordination.json');ledger=j(OUT/'resource_ledger.json');assert check['source_four_model_forecast_GPU_hours']+ledger['new_phase_GPU_job_wall_hours']+.3<program['new_phase_caps']['GPU_job_wall_hours'],'completeforecastdoesnotfit';data=Inputs();entries={}
 for arm in ARMS:
  rd=OUT/'source_train'/arm;rd.mkdir(exist_ok=True,parents=True);identity={'arm':arm,'input_manifest_sha256':sha(PUB/'input_manifest.json'),'parent_plan_sha256':sha(OUT/'parent_plan.npy'),'dense_centres_sha256':sha(OUT/'dense_centres.npy'),'updates':U}
  if (rd/'manifest.json').exists():
   e=j(rd/'manifest.json');assert e['identity']==identity and sha(rd/'final.pt')==e['bundle_sha256'] and sha(rd/'optimizer_final.pt')==e['optimizer_final_sha256'] and sha(rd/'updates.private.jsonl')==e['journal_sha256'];entries[arm]=e;continue
  m,h=build();opt=old.t44.make_opt(m,h);initial={n:v.clone() for n,v in old.flat(m,h).items()};claim=rd/'claim.json';rp=rd/'resume.pt';jp=rd/'updates.private.jsonl';start=0;logs=[];fit_forwards=0
  if claim.exists():
   assert j(claim)==identity;checkpoint_path=rp if rp.exists() else rd/'optimizer_final.pt';logs=[json.loads(s) for s in jp.read_text().splitlines()] if jp.exists() else []
   if checkpoint_path.exists():
    st=torch.load(checkpoint_path,map_location='cpu',weights_only=True);assert st['identity']==identity;old.t44.load_state(m,h,opt,st);start=st['update'];dt.validate_recovery(rd,start,len(logs))
    with (rd/'resume_events.private.jsonl').open('a') as ef:ef.write(json.dumps({'start_update':start,'checkpoint_sha256':sha(checkpoint_path),'pid':os.getpid(),'journal_length':len(logs),'safe_continuity_validated':True})+'\n');ef.flush();os.fsync(ef.fileno())
   else:
    guard=j(rd/'update_guard.json') if (rd/'update_guard.json').exists() else {'pending':None,'last_committed':-1};assert not logs and guard=={'pending':None,'last_committed':-1},'cannot repeat potentially executed uncheckpointed updates';start=0
   fit_forwards=j(rd/'fit_monitor.json')['total_fit_forwards'] if (rd/'fit_monitor.json').exists() else 0
  else:write(claim,identity)
  print(json.dumps({'formal_model_start':arm,'pid':os.getpid(),'device':torch.cuda.get_device_name(0),'start_update':start,'total':U}),flush=True)
  with jp.open('a') as jf:
   def persist(v):jf.write(json.dumps(v)+'\n');jf.flush();os.fsync(jf.fileno());logs.append(v)
   for u in range(start,U):
    rec=dt.claimed_update(rd,u,lambda:step(m,h,opt,data,u,arm),persist)
    if (u+1)%20==0:save(rp,ckpt(m,h,opt,u+1,identity))
    if (u+1)%25==0:
     write(OUT/'progress.json',{'stage':'formal_source_training','arm':arm,'pid':os.getpid(),'device':rec['device'],'update':u+1,'total':U,'completed_models':list(entries),'epoch':time.time()});print(json.dumps({'arm':arm,'update':u+1,'loss':rec['loss'],'pid':os.getpid()}),flush=True)
    if (u+1)%300==0:
     panel=fit_panel(m,h,data,arm);fit_forwards+=32;write(rd/'fit_monitor.json',{'last_update':u+1,'panel':panel,'total_fit_forwards':fit_forwards})
    if u+1 in (1200,2400):save(rd/f'model_only_{u+1}.pt',old.make_bundle(m,h,{**identity,'checkpoint_update':u+1,'FIT_diagnostics_only':True}))
    if time.time()>program['hard_wall_deadline_epoch']:raise RuntimeError('authorizedwallwindow ended; retainedexactcheckpoint,no automaticvariant')
  assert len(logs)==U and set(int(s['step']) for s in opt.state.values())=={U};final=old.flat(m,h);assert all(torch.equal(final[n],x) for n,x in initial.items() if n.startswith('m.classifier.'));fit_results=final_fit(m,h,data,arm,rd);fit_forwards+=2*len(data.parents);bundle=old.make_bundle(m,h,identity);save(rd/'final.pt',bundle);save(rp,ckpt(m,h,opt,U,identity));rp.replace(rd/'optimizer_final.pt');dt.sync_directory(rd);drift={n:float((final[n].double()-x.double()).norm()) for n,x in initial.items()};cm,ch=old.restore(torch.load(rd/'final.pt',map_location='cpu',weights_only=True),'cpu');assert state_sha(old.flat(cm,ch))==bundle['full_state_sha256'];e={'identity':identity,'updates':U,'bundle_sha256':sha(rd/'final.pt'),'optimizer_final_sha256':sha(rd/'optimizer_final.pt'),'full_state_sha256':bundle['full_state_sha256'],'frozen_state_sha256':bundle['frozen_state_sha256'],'journal_sha256':sha(jp),'physical_training_forwards':U*8,'physical_training_backwards':U*8,'fit_monitor_and_final_forwards':fit_forwards,'fit_forward_count_scope':'Successfullycommitteddiagnostics; failed/incompleteattempts separatelycharged byfullprocessreceipt','model_only_FIT_checkpoint_sha256':{str(u):sha(rd/f'model_only_{u}.pt') for u in (1200,2400)},'fit_results':fit_results,'optimizer_state_count':len(opt.state),'optimizer_steps':sorted({int(v['step']) for v in opt.state.values()}),'drift_by_tensor':drift,'old5class_head_frozen':True};write(rd/'manifest.json',e);entries[arm]=e;del m,h,opt,cm,ch,bundle,initial;torch.cuda.empty_cache()
 write(PUB/'source_models_manifest.json',{'run_id':'A55','models':entries,'input_manifest_sha256':sha(PUB/'input_manifest.json'),'Base_A08_sha256':sha(old.BASE),'formal_source_updates':U*4,'physical_training_forwards':U*4*8,'physical_training_backwards':U*4*8,'source_diagnostic_forwards':0,'code':pins()});print(json.dumps({'all_four_source_models_complete':True,'formal_updates':U*4}),flush=True)

def readout():
 m=verify();models=j(PUB/'source_models_manifest.json');fr=j(OUT/'models_freeze.json');audit=j(PUB/'independent_models.json');assert models['input_manifest_sha256']==sha(PUB/'input_manifest.json') and models['Base_A08_sha256']==m['Base_A08_sha256'] and all(e['identity']['input_manifest_sha256']==models['input_manifest_sha256'] and e['identity']['arm']==arm and e['updates']==U for arm,e in models['models'].items());assert fr['actual_git_objects_verified'] and fr['source_models_manifest_sha256']==sha(PUB/'source_models_manifest.json') and audit['passed'] and audit['source_models_manifest_sha256']==sha(PUB/'source_models_manifest.json');data=Inputs();values={}
 for arm in (*ARMS,'G-C'):
  hsha=m['Base_A08_sha256'] if arm=='G-C' else models['models'][arm]['bundle_sha256'];cell=dt.open_cell(OUT/'source_cells',arm,{'source_model_sha256':hsha,'bound_sha256':m['bound_sha256'],'rows':133},133)
  if arm=='G-C':model,head=build()
  else:
   p=OUT/'source_train'/arm/'final.pt';assert sha(p)==hsha;model,head=old.restore(torch.load(p,map_location='cpu',weights_only=True))
  with torch.no_grad():
   while (i:=dt.claim_row(cell)) is not None:dt.finish_row(cell,i,float(score(model,head,data.diagnostic(data.diag[i],'C' if arm=='G-C' else arm[1]),base=arm=='G-C')))
  dt.seal_cell(cell);values[arm]=np.asarray(cell['scores']).copy();dt.close_cell(cell);del model,head;torch.cuda.empty_cache()
 old_scores=np.load(old.OUT/'source_scores.private.npz');values['G-F']=old_scores['G'];ys=np.array([r['true_label']=='A5C' for r in data.diag]);discord=np.array([r['true_label']!=r['directory_label'] for r in data.diag]);sparse={x['id']:x for x in data.b['sparse']}
 for geom in ['F','C']:
  z=data.b['geometry'][geom];x=np.array([sparse[r['id']]['geometry'][geom] for r in data.diag]);values['geometry-'+geom]=((x-z['fit_mean'])/z['fit_scale'])@np.array(z['coef'])+z['intercept']
 prior=data.b['geometry']['F']['constant_prior'];values['constant']=np.full(133,np.log(prior/(1-prior)))
 if (OUT/'source_scores.private.npz').exists():
  existing=np.load(OUT/'source_scores.private.npz');assert set(existing.files)==set(values) and all(np.array_equal(existing[k],v) for k,v in values.items())
 else:
  with (OUT/'source_scores.private.npz').open('xb') as f:np.savez(f,**values);f.flush();os.fsync(f.fileno())
  dt.sync_directory(OUT)
 metrics={}
 for arm,s in values.items():
  v=dt.source_metrics(ys,s,discord);v['AUC']=float(np.mean((s[ys,None]>s[None,~ys])+.5*(s[ys,None]==s[None,~ys])));metrics[arm]=v
 absolute={}
 for arm in ARMS:
  v=metrics[arm];absolute[arm]=bool(v['balanced_accuracy']>=.8 and v['AUC']>=max(.85,metrics['G-F']['AUC']+.03) and v['fp']<=5 and v['tp']>=28 and v['balanced_accuracy']>=max(metrics['geometry-'+arm[1]]['balanced_accuracy'],metrics['constant']['balanced_accuracy'])+.1)
 dc,sc=metrics['DC'],metrics['SC'];increment=bool(dc['balanced_accuracy']>=sc['balanced_accuracy']+.05 and dc['AUC']>=sc['AUC']+.03 and dc['fp']<=sc['fp'] and dc['tp']>=sc['tp']+4);paired={}
 for aa,bb in [('DC','SC'),('DF','SF'),('DC','DF'),('SC','SF')]:
  paired[aa+'-'+bb]={}
  for group,mask in [('A5C_correct',ys),('A4C_errors',~ys)]:
   a=mask&(values[aa]>0);b=mask&(values[bb]>0);paired[aa+'-'+bb][group]={'shared':int((a&b).sum()),'A_only':int((a&~b).sum()),'B_only':int((~a&b).sum())}
 pair='C' if any(absolute[a] for a in ['DC','SC']) else 'F' if any(absolute[a] for a in ['DF','SF']) else None
 write(PUB/'source_readout.json',{'metrics':metrics,'absolute':absolute,'DC_minus_SC_source_increment':increment,'coverage_mechanism_supported':absolute['DC'] and increment,'target_pair_geometry':pair,'source_qualified_for_target_card':pair is not None,'paired':paired,'raw_score_cache_sha256':sha(OUT/'source_scores.private.npz'),'new_source_diagnostic_forwards':665,'original_G_F_scores_reused':133,'new_target_readout':0,'scope':'Reused sourcepatientdiagnostic, not LVOT/anatomy/EV9Vclinicalvalidation.'});print(json.dumps({'source_absolute':absolute,'DC_minus_SC':increment,'target_card_qualified':pair is not None,'target_pair_geometry':pair}),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','check','train','readout']);a=p.parse_args();OUT.mkdir(exist_ok=True,parents=True);PUB.mkdir(exist_ok=True,parents=True);lock=dt.acquire_lock(OUT/'owner.lock');start=time.time();ok=False
 try:globals()[a.stage]();ok=True
 finally:
  u=resource.getrusage(resource.RUSAGE_SELF);write(OUT/f'{a.stage}_worker_process.json',{'pid':os.getpid(),'ok':ok,'CPU_seconds':u.ru_utime+u.ru_stime,'wall_seconds':time.time()-start,'GPU_job_wall_seconds':time.time()-start if a.stage in ['check','train','readout'] else 0});lock.close()
