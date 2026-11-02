#!/usr/bin/env python3
"""A56 source-only matched five-class replay; no target evaluation path."""
import os
for k,v in {'CUDA_VISIBLE_DEVICES':'0','CUBLAS_WORKSPACE_CONFIG':':4096:8','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}.items():os.environ.setdefault(k,v)
import argparse,copy,hashlib,json,resource,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
import torch.nn.functional as F
from scripts import train_annotation_coverage as a55
from scripts import train_direct_view_evidence as a51
from scripts import train_session_correction as a31
from scripts import train_interval_visual_adaptation as a28
from scripts.prepare_annotation_coverage import sha,j,arsha,write
from echo_routing.temporal import source_retention as sr,direct_view_training as dt
from echo_routing.temporal.pw_transfer import state_sha
from echo_routing.temporal.dev_train import lr_factor
from echo_routing.evaluate.failure_units import session_of
import echo_routing.vit.target_mvit as tm
OUT=ROOT/'runs/a56_source_retention_v1';PUB=ROOT/'docs/results/2026-11-02_source_retention'
CARD='docs/superpowers/plans/2026-11-02-fiveclass-source-retention.md';U=1200;ARMS=('Z','R')
POOL_SHA=('96d1fb683e1a261d5be9a3cc7ec393637577bbf83b56140cec41f544c4f3b486','08255192dae3a90644ea062880bf6b2b298181ed12e9890d56ca04630f5adef4')
COUNTS={'forwards':0,'backwards':0,'builds':0}
LOGIT_SHA=('5884435770e8b65959b22ff77f74b0c2053cadf7fd2a5a28c842104df2f1a512','0a99be35b0cd78d650f5014c8ddc783173e6fb849ee556d6fa2a52c73f33d360')


def pins():
 paths={Path(CARD),Path('scripts/train_source_retention.py'),Path('scripts/supervise_source_retention.py'),Path('echo_routing/temporal/source_retention.py')}
 for mod in list(sys.modules.values()):
  f=getattr(mod,'__file__',None)
  if f:
   p=Path(f).resolve()
   if p.is_relative_to(ROOT) and '.venv' not in p.parts and p.suffix=='.py' and p.is_file():paths.add(p.relative_to(ROOT))
 for name in ('video_session_correction_v1.yaml','video_router_data_v1.yaml','video_interval_recognition_v1.yaml','video_risk_constrained_a5c_v1.yaml','video_within_view_v1.yaml'):paths.add(Path('configs/experiments')/name)
 return {str(p):sha(ROOT/p) for p in sorted(paths)}


def native_data():
 ctx=a51.ctx().a31;data,clips=a31.fit_clips(ctx);allowed={v:data.label[v] for v in clips};allseq=[];original=[]
 for f in (0,1):
  p=ROOT/f'runs/video_session_correction_v1/pool_{f}.npy';assert sha(p)==POOL_SHA[f];seqs=np.load(p,allow_pickle=True).tolist();native=sr.native_allowlist(seqs,allowed);allseq.extend(native);original.append(seqs)
 assert len(allseq)==1487 and len({q['id'] for q in allseq})==1487
 return allseq,a31.pool_W(allseq),original,allowed


def prepare():
 assert not (PUB/'input_manifest.json').exists(),'input manifest immutable'
 data=a55.Inputs();seqs,W,original,allowed=native_data();y=np.array([p['label'] for p in data.parents]);sp,rp=sr.exposure_plans(y,seqs,U);visits=np.zeros(len(seqs),np.int64);rng=np.random.default_rng([sr.SEED,57]);offset=rng.integers(0,2**31,len(seqs));rows=[np.flatnonzero(W['seq']==i) for i in range(len(seqs))];rr=np.empty_like(rp)
 for u in range(U):
  for k,i in enumerate(rp[u]):
   rr[u,k]=rows[i][(offset[i]+visits[i])%len(rows[i])];visits[i]+=1
 panel=np.concatenate([np.flatnonzero(W['label']==c)[np.linspace(0,int((W['label']==c).sum())-1,50,dtype=int)] for c in range(5)])
 fitpatients={p['patient'] for p in data.parents};diagpatients={r['patient'] for r in data.diag};assert not fitpatients&diagpatients
 assert j(a55.PUB/'processed_integrity.json')['passed'] and j(a55.PUB/'qualification.json')['crossrole_exact_point_hashes']==0
 np.savez(OUT/'plan.private.npz',source=sp,replay=rr,replay_native=rp,panel=panel)
 private={'native_ids':[q['id'] for q in seqs],'panel_rows':panel.tolist(),'panel_exact_windows':[{'id':seqs[int(W['seq'][r])]['id'],'centre':int(W['centre'][r]),'marker':int(W['marker'][r]),'label':int(W['label'][r]),'refs':[list(seqs[int(W['seq'][r])]['refs'][int(k)]) for k in W['s'][r]]} for r in panel],'source_fit_ids':[p['source']['id'] for p in data.parents],'source_diagnostic_ids':[r['id'] for r in data.diag]};write(OUT/'selection.private.json',private)
 rec={'run_id':'A56','mode':'source_only_no_target_OOF_ever','arms':list(ARMS),'updates_each':U,'source_replay_batch':[8,8],'seed':sr.SEED,'code':pins(),'Base_A08_sha256':sha(a51.BASE),'A55_bound_sha256':sha(a55.OUT/'bound.private.json'),'A55_qualification_sha256':sha(a55.PUB/'qualification.json'),'A55_cache_manifest_digest':j(a55.PUB/'qualification.json')['complete_private_cache_manifest_digest'],'A31_pool_sha256':list(POOL_SHA),'A31_split_sha256':sha(ROOT/'runs/video_session_correction_v1/split.json'),'plan_sha256':sha(OUT/'plan.private.npz'),'selection_sha256':sha(OUT/'selection.private.json'),'source_fit_diagnostic':[569,133],'source_patient_counts':[len(fitpatients),len(diagpatients)],'source_patients_shared':0,'replay_native_recordings':len(seqs),'replay_sessions':len({session_of(q['id']) for q in seqs}),'replay_class_recordings':np.bincount([int(q['labels'][0]) for q in seqs],minlength=5).tolist(),'replay_class_exposures':np.bincount(W['label'][rr].ravel(),minlength=5).tolist(),'replay_distinct_windows':len(set(rr.ravel().tolist())),'replay_recordings_exposed':len(set(rp.ravel().tolist())),'source_parents_exposed':len(set(sp.ravel().tolist())),'all_replay_recordings_in_reconstructed_A31_fit_allowlist':all(q['id'] in allowed for q in seqs),'FIT_retention_panel':250,'diagnostic_neural_forwards':0,'new_optimizer_updates':0,'versions':{'torch':torch.__version__,'numpy':np.__version__},'frozen_source_reference_readout_sha256':sha(a55.PUB/'source_readout.json'),'inherited_A55_finalreceipt_sha256':sha(a55.OUT/'final_resource_receipt.json')}
 rec['A31_raw_five_logit_cache_sha256']=list(LOGIT_SHA);rec['A31_cache_receipt_sha256']=sha(ROOT/'runs/video_session_correction_v1/cache/prefix_s0.json');rec['panel_exact_window_identity_sha256']=hashlib.sha256(json.dumps(private['panel_exact_windows'],sort_keys=True).encode()).hexdigest()
 for f in (0,1):assert sha(ROOT/f'runs/video_session_correction_v1/outputs_s0_f{f}.npz')==LOGIT_SHA[f]
 assert rec['source_parents_exposed']==569 and rec['replay_recordings_exposed']==1487 and rec['replay_class_exposures']==[1920]*5;write(PUB/'input_manifest.json',rec);print(json.dumps(rec),flush=True)


class Inputs:
 def __init__(self):
  self.source=a55.Inputs();self.seqs,self.W,self.original,_=native_data();self.fp=a28.frame_paths(self.seqs);self.p=np.load(OUT/'plan.private.npz');self.plan=self.p['source'];self.replay=self.p['replay'];self.panel=self.p['panel']
 def source_window(self,i):
  p=self.source.parents[int(i)];return self.source.window(p,p['centre'],'C'),8,int(p['label'])+2
 def replay_window(self,row):
  r=int(row);q=self.seqs[int(self.W['seq'][r])];refs=[tuple(q['refs'][int(k)]) for k in self.W['s'][r]]
  return a28.load_clip(refs,self.fp),int(self.W['marker'][r]),int(self.W['label'][r])
 def batch(self,u):return [self.source_window(i) for i in self.plan[u]]+[self.replay_window(i) for i in self.replay[u]]
 def original_panel_logits(self):
  at={}
  for f,seqs in enumerate(self.original):
   W=a31.pool_W(seqs);p=ROOT/f'runs/video_session_correction_v1/outputs_s0_f{f}.npz';old=np.load(p);receipt=j(ROOT/'runs/video_session_correction_v1/cache/prefix_s0.json');assert sha(p)==receipt['outputs_sha256'][str(f)]==LOGIT_SHA[f];assert np.array_equal(old['seq'],W['seq']) and np.array_equal(old['centre'],W['centre'])
   for i,q in enumerate(seqs):
    if q['kind']=='native':
     for row in np.flatnonzero(W['seq']==i):at[(q['id'],int(W['centre'][row]))]=old['logits'][row]
  return np.stack([at[(self.seqs[int(self.W['seq'][r])]['id'],int(self.W['centre'][r]))] for r in self.panel])


def verify():
 m=j(PUB/'input_manifest.json');assert sha(a51.BASE)==m['Base_A08_sha256'];assert sha(a55.OUT/'bound.private.json')==m['A55_bound_sha256'];assert sha(OUT/'plan.private.npz')==m['plan_sha256'];assert sha(OUT/'selection.private.json')==m['selection_sha256'];assert sha(a55.PUB/'source_readout.json')==m['frozen_source_reference_readout_sha256']
 for p,h in m['code'].items():assert sha(ROOT/p)==h,p
 return m


def build(device='cuda'):
 COUNTS['builds']+=1
 torch.set_num_threads(1);torch.manual_seed(sr.SEED);torch.cuda.manual_seed_all(sr.SEED);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
 m,h=a55.build(device);del h
 torch.manual_seed(sr.SEED);torch.cuda.manual_seed_all(sr.SEED)
 for p in m.parameters():p.requires_grad_(True)
 return m.eval()


def groups(m):return {'encoder':list(m.encoder.parameters()),'marker':list(m.marker_stem.parameters()),'classifier':list(m.classifier.parameters())}
def optimizer(m):return torch.optim.AdamW([{'params':pp,'lr':1e-5 if k=='encoder' else 1e-4,'base_lr':1e-5 if k=='encoder' else 1e-4,'name':k} for k,pp in groups(m).items()],betas=(.9,.999),eps=1e-8,weight_decay=1e-4)
def flat(m):return {k:v.detach().cpu() for k,v in m.state_dict().items()}
def logits(m,fr,marker,device='cuda'):
 COUNTS['forwards']+=1
 q=m.classifier(m.embed(tm.to_rgb(torch.from_numpy(fr[None]).to(device)),tm.marker_planes(torch.tensor([marker],device=device))))
 if q.requires_grad:q.register_hook(lambda g:COUNTS.__setitem__('backwards',COUNTS['backwards']+1))
 return q
def checkpoint(m,o,u,identity):return {'model':flat(m),'optimizer':o.state_dict(),'cpu_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'update':u,'identity':identity}
def load(m,o,st):m.load_state_dict(st['model']);o.load_state_dict(st['optimizer']);torch.set_rng_state(st['cpu_rng']);torch.cuda.set_rng_state_all(st['cuda_rng'])


def step(m,o,data,u,coefficient,schedule_index=None):
 k=u if schedule_index is None else schedule_index
 for g in o.param_groups:g['lr']=g['base_lr']*lr_factor(k,120,U)
 o.zero_grad(set_to_none=True);b=data.batch(u);rec=sr.accumulate(lambda i:logits(m,*b[i][:2]),[x[2] for x in b[:8]],[x[2] for x in b[8:]],coefficient)
 rec.update({'u':u,'coefficient':coefficient,'schedule_index':k,'source_plan_sha256':arsha(data.plan[u]),'replay_plan_sha256':arsha(data.replay[u]),'input_window_sha256':hashlib.sha256(''.join(arsha(x[0]) for x in b).encode()).hexdigest(),'groups':{k:a51.t44.sq_norm([p.grad for p in pp]) for k,pp in groups(m).items()},'gradient_norm':float(torch.nn.utils.clip_grad_norm_(list(m.parameters()),1.)),'pid':os.getpid(),'device':torch.cuda.get_device_name(0)})
 assert all(np.isfinite(rec[k]) for k in ['loss','source_loss','replay_loss','gradient_norm']);o.step();return rec


def fit_scores(m,data,source=True):
 rows=np.linspace(0,568,32,dtype=int) if source else data.panel;out=[]
 with torch.no_grad():
  for r in rows:out.append(logits(m,*(data.source_window(r)[:2] if source else data.replay_window(r)[:2])).cpu().numpy()[0])
 return np.asarray(out,np.float32)


def check():
 manifest=verify();fr=j(OUT/'input_freeze.json');assert fr['actual_git_objects_verified'] and fr['input_manifest_sha256']==sha(PUB/'input_manifest.json');data=Inputs();m=build();initial=copy.deepcopy(flat(m));initial_sha=state_sha(initial);o=optimizer(m);assert {id(p) for g in o.param_groups for p in g['params']}=={id(p) for p in m.parameters()};z=fit_scores(m,data,False);parity=float(np.max(np.abs(z-data.original_panel_logits())));assert parity<=1e-4
 write(OUT/'initial_panel.private.json',{'logits':z.tolist(),'rows':data.panel.tolist()})
 # Four FIT windows independently redecoded; source diagnostic pixels/forwards excluded.
 import av
 from PIL import Image
 from echo_routing.temporal.joint_visual import crop_uint8
 source_pixel_checks=[]
 for i in (0,189,378,568):
  parent=data.source.parents[i];request=parent['source'];path=a55.SRC/'videos'/request['directory_label']/request['video'];wanted=set(request['frames']);decoded={}
  with av.open(str(path)) as cc:
   cc.streams.video[0].thread_type='NONE'
   for ordinal,frame in enumerate(cc.decode(cc.streams.video[0])):
    if ordinal in wanted:decoded[ordinal]=crop_uint8(Image.fromarray(frame.to_ndarray(format='rgb24')))
    if ordinal>=max(wanted):break
  fresh=np.stack([decoded[k] for k in request['frames']]);cached,marker,_=data.source_window(i);assert np.array_equal(fresh,cached)
  with torch.no_grad():a=logits(m,fresh,marker);bq=logits(m,cached,marker)
  assert torch.equal(a,bq);source_pixel_checks.append({'FIT_index':i,'exact_C_window_sha256':arsha(fresh),'fresh_logit_max_abs':float((a-bq).abs().max())})
 # Explicit independent CE and accumulated/grouped production gradients on four FIT windows.
 b=data.batch(0);pick=[0,4,8,9];zz=torch.cat([logits(m,*b[i][:2]) for i in pick]);yy=torch.tensor([b[i][2] for i in pick],device='cuda');ce=F.cross_entropy(zz,yy,reduction='sum')/8;ind=np.asarray(zz.detach().cpu(),float);maximum=ind.max(1);ref=np.mean(maximum+np.log(np.exp(ind-maximum[:,None]).sum(1))-ind[np.arange(4),yy.cpu().numpy()])/2;assert abs(float(ce.detach())-ref)<1e-5;ce.backward();expected={n:p.grad.clone() for n,p in m.named_parameters()};o.zero_grad(set_to_none=True)
 for i in pick:(F.cross_entropy(logits(m,*b[i][:2]),torch.tensor([b[i][2]],device='cuda'))/8).backward()
 ge=max(float((p.grad-expected[n]).abs().max())/(1+float(expected[n].abs().max())) for n,p in m.named_parameters());assert ge<1e-4
 # Replay coefficient zero truly contributes zero; replay graphs still backpropagate.
 o.zero_grad(set_to_none=True)
 for i in range(8):(F.cross_entropy(logits(m,*b[i][:2]),torch.tensor([b[i][2]],device='cuda'))/8).backward()
 source_grads={n:p.grad.clone() for n,p in m.named_parameters()};count=[0]
 for i in range(8,16):
  q=logits(m,*b[i][:2]);q.register_hook(lambda g:count.__setitem__(0,count[0]+int(torch.count_nonzero(g)==0)));(0.*F.cross_entropy(q,torch.tensor([b[i][2]],device='cuda'))/8).backward()
 assert count[0]==8 and all(torch.equal(p.grad,source_grads[n]) for n,p in m.named_parameters());del m,o,expected,source_grads;torch.cuda.empty_cache()
 def run(coef,n=6,st=None):
  m=build();o=optimizer(m);start=0
  if st is not None:load(m,o,st);start=st['update']
  recs=[step(m,o,data,u,coef,schedule_index=120+u) for u in range(start,n)];return m,o,recs
 m,o,rs=run(0.);expected=copy.deepcopy(flat(m));expected_opt=copy.deepcopy(o.state_dict());del m,o
 for c in (0.,0.):
  m,o,_=run(c);assert state_sha(flat(m))==state_sha(expected);del m,o
 m,o,_=run(0.,n=3);a55.save(OUT/'check_resume.pt',checkpoint(m,o,3,{}));del m,o;st=torch.load(OUT/'check_resume.pt',map_location='cpu',weights_only=True);m,o,_=run(0.,st=st);assert state_sha(flat(m))==state_sha(expected)
 for k,v in o.state_dict()['state'].items():
  for name,val in v.items():assert torch.equal(val.cpu(),expected_opt['state'][k][name].cpu())
 del m,o,st;(OUT/'check_resume.pt').unlink();m=build();o=optimizer(m);torch.cuda.synchronize();start=time.time();timed=[]
 for u in range(20):timed.append(step(m,o,data,u,1.,schedule_index=120+u))
 a55.save(OUT/'timed_resume.pt',checkpoint(m,o,20,{}));torch.cuda.synchronize();elapsed=time.time()-start;(OUT/'timed_resume.pt').unlink();drift={n:float((v.double()-initial[n].double()).norm()) for n,v in flat(m).items()};assert all(any(drift[n]>0 for n,p in m.named_parameters() if p is pp) for values in groups(m).values() for pp in values[:1]);zeros=[n for n,p in m.named_parameters() if not torch.count_nonzero(p.grad)];assert sha(a51.BASE)==manifest['Base_A08_sha256']
 rec={'passed':True,'input_manifest_sha256':sha(PUB/'input_manifest.json'),'initial_full_state_sha256':initial_sha,'A08_FIT250_parity_max_abs':parity,'independent_CE_formula_error':abs(float(ce.detach())-ref),'production_grouped_gradient_relative_error':ge,'coefficient0_eight_replay_backwards_exactzero_and_source_gradients_unchanged':True,'repeat_forcedRzero_disk3plus3_fullstate_optimizer_bitwise':True,'all_selected_tensor_count':len(list(m.parameters())),'all_selected_parameter_count':sum(p.numel() for p in m.parameters()),'zero_gradient_tensor_names':zeros,'all_three_group_live_and_drift':True,'timed_steps':20,'timed_seconds_per_step_including_checkpoint':elapsed/20,'formal_training_forecast_GPU_hours':2*U*elapsed/20*1.3/3600,'closing_GPU_reserve_hours':.35,'FIT_forwards':250+8+8+16+24*16+20*16,'FIT_backwards':8+16+24*16+20*16,'new_diagnostic_forwards':0,'GPU_peak_reserved_bytes':torch.cuda.max_memory_reserved(),'FIT_check_mutable_models':6};rec['source_FIT_independent_pixel_logit_checks']=source_pixel_checks;rec['FIT_forwards']=COUNTS['forwards'];rec['FIT_backwards']=COUNTS['backwards'];rec['FIT_check_mutable_models']=COUNTS['builds'];write(PUB/'checks.json',rec);print(json.dumps(rec),flush=True)


def train():
 verify();ch=j(PUB/'checks.json');assert ch['passed'];fr=j(OUT/'training_launch_freeze.json');assert fr['actual_git_objects_verified'] and fr['checks_sha256']==sha(PUB/'checks.json');assert fr['spent_GPU_hours']+ch['formal_training_forecast_GPU_hours']+.35<=3;data=Inputs();entries={}
 for arm in ARMS:
  rd=OUT/'models'/arm;rd.mkdir(parents=True,exist_ok=True);identity={'arm':arm,'coefficient':int(arm=='R'),'input_manifest_sha256':sha(PUB/'input_manifest.json'),'checks_sha256':sha(PUB/'checks.json'),'updates':U};assert not (rd/'claim.json').exists(),'no automatic formal retry';write(rd/'claim.json',identity);m=build();o=optimizer(m);initial=copy.deepcopy(flat(m));logs=[];panel=[];print(json.dumps({'formal_model_start':arm,'pid':os.getpid(),'device':torch.cuda.get_device_name(0),'total':U}),flush=True)
  with (rd/'updates.private.jsonl').open('x') as jf:
   def persist(v):jf.write(json.dumps(v)+'\n');jf.flush();os.fsync(jf.fileno());logs.append(v)
   for u in range(U):
    rec=dt.claimed_update(rd,u,lambda:step(m,o,data,u,int(arm=='R')),persist)
    if (u+1)%20==0:a55.save(rd/'resume.pt',checkpoint(m,o,u+1,identity))
    if (u+1)%25==0:write(OUT/'progress.json',{'stage':'real_source_training','arm':arm,'pid':os.getpid(),'device':rec['device'],'update':u+1,'total':U,'completed_models':list(entries),'epoch':time.time()});print(json.dumps({'arm':arm,'update':u+1,'loss':rec['loss']}),flush=True)
    if (u+1)%300==0:
     qq=fit_scores(m,data);panel.append({'update':u+1,'logits':qq.tolist()});write(rd/'fit_monitor.private.json',panel)
  a55.save(rd/'final.pt',{'model':flat(m),'identity':identity,'update':U});a55.save(rd/'optimizer_final.pt',checkpoint(m,o,U,identity));(rd/'resume.pt').unlink();source=fit_scores(m,data);replay=fit_scores(m,data,False);np.savez(rd/'fit_final.private.npz',source=source,replay=replay);current=flat(m);drift={n:float((current[n].double()-initial[n].double()).norm()) for n in current};write(rd/'drift.private.json',drift)
  entry={'identity':identity,'initial_full_state_sha256':state_sha(initial),'final_full_state_sha256':state_sha(current),'bundle_sha256':sha(rd/'final.pt'),'optimizer_final_sha256':sha(rd/'optimizer_final.pt'),'journal_sha256':sha(rd/'updates.private.jsonl'),'updates':len(logs),'train_forwards':sum(r['physical_forwards'] for r in logs),'train_backwards':sum(r['physical_backwards'] for r in logs),'FIT_monitor_final_forwards':4*32+32+250,'selected_tensors':len(list(m.parameters())),'changed_tensors':sum(drift[n]>0 for n,p in m.named_parameters()),'unchanged_selected_names':[n for n,p in m.named_parameters() if drift[n]==0],'all_groups_live':all(any(r['groups'][k]>0 for r in logs) for k in groups(m))};assert entry['updates']==U and entry['train_forwards']==U*16 and entry['all_groups_live'];write(rd/'manifest.json',entry);entries[arm]=entry;del m,o,initial;torch.cuda.empty_cache()
 write(PUB/'models_manifest.json',{'run_id':'A56','input_manifest_sha256':sha(PUB/'input_manifest.json'),'models':entries,'Base_unchanged':sha(a51.BASE)==j(PUB/'input_manifest.json')['Base_A08_sha256'],'diagnostic_forwards':0});print(json.dumps({'complete_models':list(entries)}),flush=True)


def audit_models():
 manifest=j(PUB/'models_manifest.json');data=Inputs();checks={};m=build('cpu');initial=copy.deepcopy(flat(m));names={id(p):n for n,p in m.named_parameters()};expected_groups=[(k,[names[id(p)] for p in pp]) for k,pp in groups(m).items()];expected_zero={'encoder.blocks.15.attn.pool_q.pool.weight','encoder.blocks.15.attn.rel_pos_h','encoder.blocks.15.attn.rel_pos_t','encoder.blocks.15.attn.rel_pos_w'}
 for arm,e in manifest['models'].items():
  rd=OUT/'models'/arm;assert sha(rd/'final.pt')==e['bundle_sha256'] and sha(rd/'optimizer_final.pt')==e['optimizer_final_sha256'] and sha(rd/'updates.private.jsonl')==e['journal_sha256'];st=torch.load(rd/'optimizer_final.pt',map_location='cpu',weights_only=True);logs=[json.loads(s) for s in (rd/'updates.private.jsonl').read_text().splitlines()];assert st['update']==len(logs)==U and state_sha(st['model'])==e['final_full_state_sha256'];assert st['identity']==e['identity']==j(rd/'claim.json');assert state_sha(initial)==e['initial_full_state_sha256'];assert all(int(v['step'])==U for v in st['optimizer']['state'].values());assert len(st['optimizer']['state'])==e['selected_tensors'];assert all(r['u']==i and r['schedule_index']==i and r['source_plan_sha256']==arsha(data.plan[i]) and r['replay_plan_sha256']==arsha(data.replay[i]) and r['coefficient']==int(arm=='R') and r['physical_forwards']==r['physical_backwards']==16 for i,r in enumerate(logs));dt.validate_recovery(rd,U,U);assert set(e['unchanged_selected_names'])==expected_zero
  seen=[]
  for actual,(group_name,param_names) in zip(st['optimizer']['param_groups'],expected_groups,strict=True):
   rate=1e-5 if group_name=='encoder' else 1e-4
   assert actual['name']==group_name and actual['base_lr']==rate and actual['lr']==rate*lr_factor(U-1,120,U) and actual['betas']==(.9,.999) and actual['eps']==1e-8 and actual['weight_decay']==1e-4 and len(actual['params'])==len(param_names)
   for index,name in zip(actual['params'],param_names,strict=True):
    seen.append(index);mom=st['optimizer']['state'][index];v=st['model'][name];assert v.shape==initial[name].shape and torch.isfinite(v).all()
    for key in ('exp_avg','exp_avg_sq'):assert mom[key].shape==v.shape and torch.isfinite(mom[key]).all()
    assert (mom['exp_avg_sq']>=0).all()
    if name in expected_zero:assert torch.equal(v,initial[name]) and torch.count_nonzero(mom['exp_avg'])==torch.count_nonzero(mom['exp_avg_sq'])==0
    else:assert not torch.equal(v,initial[name]) and torch.count_nonzero(mom['exp_avg_sq'])>0
  assert len(seen)==len(set(seen))==e['selected_tensors'];checks[arm]={'steps':U,'optimizer_tensors':len(seen),'all_plan_schedule_and_compute_rows_verified':True,'exact_optimizer_groups_binding_moments_shapes_finiteness_and_live_drift':True,'fullstate_sha_verified':True,'four_structural_zero_tensors':sorted(expected_zero)};del st
 assert sha(a51.BASE)==j(PUB/'input_manifest.json')['Base_A08_sha256'];write(PUB/'independent_models.json',{'passed':True,'models_manifest_sha256':sha(PUB/'models_manifest.json'),'models':checks,'new_model_forwards':0});print(json.dumps({'models_audit_passed':True}),flush=True)


def readout():
 verify();freeze=j(OUT/'models_freeze.json');assert freeze['actual_git_objects_verified'] and freeze['models_manifest_sha256']==sha(PUB/'models_manifest.json');assert j(PUB/'independent_models.json')['passed'];man=j(PUB/'models_manifest.json');data=Inputs();yy=np.array([r['true_label']=='A5C' for r in data.source.diag]);discord=np.array([r['true_label']!=r['directory_label'] for r in data.source.diag]);allq={}
 for arm in ARMS:
  e=man['models'][arm];rd=OUT/'models'/arm;assert sha(rd/'final.pt')==e['bundle_sha256'];m=build();m.load_state_dict(torch.load(rd/'final.pt',map_location='cpu',weights_only=True)['model']);cells=[dt.open_cell(OUT/'source_cells',arm+str(c),{'model_sha256':e['final_full_state_sha256'],'column':c,'source_manifest_sha256':sha(PUB/'input_manifest.json')},133) for c in range(5)]
  try:
   for i,r in enumerate(data.source.diag):
    indices=[dt.claim_row(cell) for cell in cells]
    if all(k is None for k in indices):break
    assert indices==[i]*5
    with torch.no_grad():q=logits(m,data.source.diagnostic(r,'C'),8).cpu().numpy()[0]
    for c,cell in enumerate(cells):dt.finish_row(cell,i,float(q[c]))
   for cell in cells:dt.seal_cell(cell)
   allq[arm]=np.stack([np.asarray(cell['scores']).copy() for cell in cells],axis=1)
  finally:
   for cell in cells:dt.close_cell(cell)
  del m;torch.cuda.empty_cache()
 np.savez(OUT/'source_scores.private.npz',**allq);write(OUT/'source_labels.private.json',{'y':yy.astype(int).tolist(),'discordant':discord.tolist(),'patient':[r['patient'] for r in data.source.diag],'video':[r['video'] for r in data.source.diag]});print(json.dumps({'once_source_readout_complete':True,'model_forwards':266,'score_columns':5}),flush=True)


if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','check','train','audit_models','readout']);args=parser.parse_args();OUT.mkdir(parents=True,exist_ok=True);PUB.mkdir(parents=True,exist_ok=True);lock=dt.acquire_lock(OUT/'owner.lock');start=time.time();ok=False
 try:globals()[args.stage]();ok=True
 finally:
  usage=resource.getrusage(resource.RUSAGE_SELF);write(OUT/f'{args.stage}_worker_process.json',{'pid':os.getpid(),'ok':ok,'CPU_seconds':usage.ru_utime+usage.ru_stime,'wall_seconds':time.time()-start,'GPU_job_wall_seconds':time.time()-start if args.stage in ('check','train','readout') else 0});lock.close()
