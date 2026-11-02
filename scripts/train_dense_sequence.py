#!/usr/bin/env python3
"""A57 single-owner, direct complete-target timeline training and inner evaluation."""
import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES','0');os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8');os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import copy,hashlib,json,resource,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from scripts.check_dense_sequence import build,opt,step,state,difference,read,write
from echo_routing.temporal import dense_runtime as data,dense_sequence as dense,jepa21_source as source
from echo_routing.temporal.dense_durable import atomic_torch_save,resume_accounting
RUN=ROOT/'runs/a57_dense_sequence_v1';PUB=ROOT/'docs/results/2026-11-02_dense_target_sequence';CFG=ROOT/'configs/experiments/dense_target_sequence_v1.yaml'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(16*1024*1024),b''):h.update(x)
 return h.hexdigest()
def configure():
 torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
 cfg=read(CFG);freeze=read(RUN/'pipeline_freeze.json')
 for p,h in freeze['objects'].items():assert sha(ROOT/p)==h,p
 src=read(PUB/'source.json')
 for p,h in src['loaded_model_subset_SHA'].items():assert sha(source.UPSTREAM/p)==h,p
 assert sha(ROOT/cfg['source_state'])==src['target_encoder_sha256']
 return cfg,freeze

def inputs(role):
 meta=read(PUB/'input_metadata.json');p=RUN/(role+'_sequences.private.npy');assert sha(p)==meta['roles'][role]['sequence_sha256'];seqs=np.load(p,allow_pickle=True).tolist();allow={q['id'] for q in seqs if q['kind']=='native'};data.verify_partition(seqs,allow);return seqs

def build_optimizer(m,cfg,arm):
 o=opt(m,cfg)
 if m.joint:
  for g in o.param_groups:
   if g['name']=='active_video':g['lr']=g['base_lr']=cfg['arms'][arm]['encoder_lr']
 return o

def checkpoint(m,o,cursor,cfg,arm):return {'model':state(m),'optimizer':o.state_dict(),'cpu_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'update':cursor,'seed':cfg['seed'],'arm':arm,'schedule_total':6000,'source_sha256':read(PUB/'source.json')['target_encoder_sha256']}

def load_model_weights(m,p):
 z=torch.load(p,map_location='cpu',weights_only=True)
 if 'head' in z:m.head.load_state_dict(z['head'],strict=True)
 else:m.load_state_dict(z['model'],strict=True)

def evaluate_inner(m,seqs,pixels,folder,arm,cursor,cfg):
 folder.mkdir(parents=True,exist_ok=True);result_path=folder/'metrics.json'
 if result_path.exists():return read(result_path)
 m.eval();preds={};start=time.time();probfile=folder/'probabilities.private.npy'
 for i,q in enumerate(seqs):
  p=data.predict_sequence(m,q,pixels);preds[q['id']]=p.argmax(1).astype(np.int8)
  if (i+1)%100==0:write(RUN/'runtimecoordination.json',{'phase':'inner_calibration','arm':arm,'seed':cfg['seed'],'step':cursor,'sequence_done':i+1,'sequences':len(seqs),'model_pid':os.getpid(),'epoch':time.time(),'next_action':'finish fixed inner readout then resume planned training'})
 np.save(folder/'predictions.private.npy',preds,allow_pickle=True);metrics,tables=data.metrics(seqs,preds)
 for k,t in tables.items():t.to_csv(folder/(k+'.private.csv'),index=False)
 row={**metrics,'arm':arm,'seed':cfg['seed'],'step':cursor,'backbone_lr':cfg['arms'][arm].get('encoder_lr',0),'input_sequence_sha256':read(PUB/'input_metadata.json')['roles']['inner_cal']['sequence_sha256'],'wall_seconds':time.time()-start,'selection_scope':'fixed protocoltrain session innercal only','model_pid':os.getpid()};write(result_path,row);return row

def pipeline_check():
 cfg,freeze=configure();seqs=inputs('fit');pixels=data.Pixels(seqs);plan=np.load(RUN/'plan.private.npz');assert sha(RUN/'plan.private.npz')==read(PUB/'input_metadata.json')['plan_sha256']
 before=time.time();raw=pixels.batch([seqs[int(i)] for i in plan['sequence'][0]],plan['start'][0],'cuda');cached=np.load(RUN/'FIT_smoke.private.npz');assert all(np.array_equal(x.cpu().numpy(),cached[k]) for x,k in zip(raw,['frames','labels','mask']));loader_parity=time.time()-before
 m=build('J-low',cfg).train();o=build_optimizer(m,cfg,'J-low');rows=[];torch.cuda.synchronize();start=time.time()
 for u in range(20):
  b=pixels.batch([seqs[int(i)] for i in plan['sequence'][u]],plan['start'][u],'cuda');rows.append(step(m,o,b,u))
 torch.cuda.synchronize();update_seconds=(time.time()-start)/20;cp=RUN/'pipeline_check_checkpoint.pt';start=time.time();atomic_torch_save(checkpoint(m,o,20,cfg,'J-low'),cp);cp_seconds=time.time()-start;cp_bytes=cp.stat().st_size;cp.unlink()
 # This diagnostic inference is FIT-only. Includes raw reads, normalization, model and probability stitching.
 chosen=[seqs[int(i)] for i in np.linspace(0,len(seqs)-1,12,dtype=int)];m.eval();start=time.time();nchunks=0;npositions=0
 for q in chosen:
  p=data.predict_sequence(m,q,pixels);assert len(p)==len(q['labels']) and np.allclose(p.sum(1),1,atol=1e-5);nchunks+=len(dense.chunk_starts(len(p)));npositions+=len(p)
 infer_seconds=time.time()-start;inner=inputs('inner_cal');inner_chunks=sum(len(dense.chunk_starts(len(q['labels']))) for q in inner)
 result={'passed':True,'pipeline_freeze_git_commit':freeze['git_commit'],'raw_vs_frozen_FIT_pixels_labels_masks_exact':True,'raw_initial_batch_seconds':loader_parity,'raw_20_J_seconds_per_update':update_seconds,'atomic_full_optimizer_checkpoint_seconds':cp_seconds,'atomic_checkpoint_bytes':cp_bytes,'FIT_complete_sequences':len(chosen),'FIT_inference_chunks':nchunks,'FIT_inference_positions':npositions,'raw_full_infer_seconds':infer_seconds,'raw_full_infer_seconds_per_chunk':infer_seconds/nchunks,'inner_sequences':len(inner),'inner_total_chunks':inner_chunks,'inner_original_positions':sum(len(q['labels']) for q in inner),'GPU_peak_reserved_bytes':torch.cuda.max_memory_reserved(),'formal_updates':0,'new_external_development_forwards':0,'pid':os.getpid()};write(PUB/'pipeline_checks.json',result);print(json.dumps(result),flush=True)

def train_one(arm,target,cfg,fit,cal,train_pixels,cal_pixels,plan):
 folder=RUN/f"seed{cfg['seed']}"/arm;folder.mkdir(parents=True,exist_ok=True);latest=folder/'latest.pt';trace=folder/'physical_updates.jsonl';rows=[json.loads(s) for s in trace.read_text().splitlines()] if trace.exists() else [];attempt=max((r['attempt'] for r in rows),default=0)+1;m=build(arm,cfg).train();o=build_optimizer(m,cfg,arm);cursor=0
 if latest.exists():
  z=torch.load(latest,map_location='cpu',weights_only=True);assert z['seed']==cfg['seed'] and z['arm']==arm and z['schedule_total']==6000;m.load_state_dict(z['model'],strict=True);o.load_state_dict(z['optimizer']);torch.set_rng_state(z['cpu_rng']);torch.cuda.set_rng_state_all(z['cuda_rng']);cursor=z['update'];write(folder/f'resume_attempt{attempt}.json',resume_accounting(rows,cursor));del z
 if cursor>=target:del m,o;torch.cuda.empty_cache();return
 for u in range(cursor,target):
  b=train_pixels.batch([fit[int(i)] for i in plan['sequence'][u]],plan['start'][u],'cuda');write(folder/'pending.json',{'attempt':attempt,'u':u,'epoch':time.time()});start=time.time();row=step(m,o,b,u);row.update({'attempt':attempt,'seed':cfg['seed'],'arm':arm,'wall_seconds':time.time()-start,'epoch':time.time()})
  with trace.open('a') as f:f.write(json.dumps(row)+'\n');f.flush();os.fsync(f.fileno())
  (folder/'pending.json').unlink()
  if (u+1)%100==0 or u+1==target:atomic_torch_save(checkpoint(m,o,u+1,cfg,arm),latest)
  if (u+1)%25==0:
   progress={'phase':'formal_training','arm':arm,'seed':cfg['seed'],'step':u+1,'target':target,'model_pid':os.getpid(),'epoch':time.time(),'last_loss':row['loss'],'log':str(trace.relative_to(ROOT)),'next_action':'continue declared matrix, inner evaluation every500'};write(RUN/'runtimecoordination.json',progress);write(RUN/'progress.json',progress);print(json.dumps(progress),flush=True)
  if (u+1)%500==0:
   milestone=folder/f'step{u+1}.pt';payload={'model':state(m)} if m.joint else {'head':{k:v.detach().cpu() for k,v in m.head.state_dict().items()}};atomic_torch_save(payload,milestone)
   modelsha=sha(milestone);write(folder/f'step{u+1}.identity.json',{'sha256':modelsha,'seed':cfg['seed'],'arm':arm,'step':u+1,'source':read(PUB/'source.json')['target_encoder_sha256'],'pipeline_git':read(RUN/'pipeline_freeze.json')['git_commit']})
   rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state_all();evaluate_inner(m,cal,cal_pixels,folder/f'inner_step{u+1}',arm,u+1,cfg);torch.set_rng_state(rng);torch.cuda.set_rng_state_all(cuda_rng);m.train()
 del m,o;torch.cuda.empty_cache()

def make_plan(fit,seed):
 rng=np.random.default_rng([seed,58]);native=[np.array([i for i,q in enumerate(fit) if q['kind']=='native' and q['labels'][0]==c]) for c in range(5)];cycles=[]
 for pool in native:
  stream=[]
  while len(stream)<1200:stream.extend(rng.permutation(pool).tolist())
  cycles.append(stream[:1200])
 mixed=np.array([i for i,q in enumerate(fit) if q['kind']=='mixed']);changing=np.array([i for i in mixed if np.any(np.diff(fit[i]['labels']))]);sequence=np.empty((6000,2),int);start=np.empty_like(sequence)
 for u in range(6000):
  sequence[u]=[cycles[u%5][u//5],int(rng.choice(mixed if u%2==0 else changing))]
  for j,i in enumerate(sequence[u]):
   q=fit[int(i)];hi=max(0,len(q['labels'])-64);start[u,j]=int(rng.integers(hi+1)) if j==0 or u%2==0 else int(np.clip(int(rng.choice(np.flatnonzero(np.diff(q['labels']))+1))-32,0,hi))
 return {'sequence':sequence,'start':start}

def formal():
 cfg,freeze=configure();assert read(PUB/'pipeline_checks.json')['passed'] and read(PUB/'forecast.json')['all_caps_fit'];fit,cal=inputs('fit'),inputs('inner_cal');tp,cp=data.Pixels(fit),data.Pixels(cal);plan=dict(np.load(RUN/'plan.private.npz'));assert sha(RUN/'plan.private.npz')==read(PUB/'input_metadata.json')['plan_sha256'];rebuild=make_plan(fit,cfg['seed']);assert all(np.array_equal(plan[k],rebuild[k]) for k in plan)
 for arm in ['L','F','J-low','J-high']:train_one(arm,2000,cfg,fit,cal,tp,cp,plan)
 # Only the predeclared inner2000 rows select LR. No external development is loaded here.
 rows=[read(RUN/f"seed{cfg['seed']}"/arm/'inner_step2000/metrics.json') for arm in ['J-low','J-high']];selected=data.choose_inner(rows);selection={'selected_joint_arm':selected['arm'],'rule':'inner2000 segmentF1;nativeMacroF1;fewer A4C/A5Cerrors;smallerlr','rows':rows,'new_external_development_forwards_at_selection':0};write(PUB/'inner_lr_selection.json',selection)
 for arm in ['F',selected['arm'],'L']:train_one(arm,6000,cfg,fit,cal,tp,cp,plan)
 second=copy.deepcopy(cfg);second['seed']=2026110258;plan2=make_plan(fit,second['seed']);np.savez(RUN/'second_seed_plan.private.npz',**plan2);write(PUB/'second_seed_plan.json',{'seed':second['seed'],'same_session_partition_and_constructed_banks':True,'plan_sha256':sha(RUN/'second_seed_plan.private.npz'),'steps':6000,'arms':['F',selected['arm']]})
 for arm in ['F',selected['arm']]:train_one(arm,6000,second,fit,cal,tp,cp,plan2)
 write(RUN/'runtimecoordination.json',{'phase':'formal_matrix_complete_pending_frozen_outer_readout','model_pid':None,'epoch':time.time(),'next_action':'audit all completed checkpoint/model identities, freeze external readout code and run declared2000/6000+inner-selected packages'});print('formal_matrix_complete',flush=True)

if __name__=='__main__':
 start=time.time();ok=False;mode=sys.argv[1]
 try:globals()[mode]();ok=True
 finally:
  r=resource.getrusage(resource.RUSAGE_SELF);write(RUN/(mode+'_latest_process.json'),{'pid':os.getpid(),'ok':ok,'CPU_core_seconds':r.ru_utime+r.ru_stime,'wall_seconds':time.time()-start,'GPU_job_wall_seconds':time.time()-start})
