#!/usr/bin/env python3
"""A57 actual2x64 FIT-only graph/liveness/resume/throughput preflight."""
import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES','0');os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8');os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import copy,hashlib,json,resource,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from echo_routing.temporal import dense_sequence as dense,jepa21_source as source
from echo_routing.temporal.dev_train import lr_factor
RUN=ROOT/'runs/a57_dense_sequence_v1';PUB=ROOT/'docs/results/2026-11-02_dense_target_sequence';CFG=ROOT/'configs/experiments/dense_target_sequence_v1.yaml'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2)+'\n')
def build(arm,cfg):
 torch.manual_seed(cfg['seed']);torch.cuda.manual_seed_all(cfg['seed']);enc=source.build_encoder();state=torch.load(ROOT/cfg['source_state'],map_location='cpu',weights_only=True,mmap=True);enc.load_state_dict(state,strict=True);m=dense.DenseSequence(enc,arm,seed=cfg['seed'],inactive_names=read(PUB/'source.json')['inactive_video_names']).cuda();return m
def opt(m,cfg):
 groups=[{'params':list(m.head.parameters()),'lr':.001,'base_lr':.001,'name':'head'}]
 if m.joint:groups.append({'params':[p for p in m.encoder.parameters() if p.requires_grad],'lr':1e-5,'base_lr':1e-5,'name':'active_video'})
 return torch.optim.AdamW(groups,betas=(.9,.999),eps=1e-8,weight_decay=.05)
def state(m):return {k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
def difference(a,b):return max(float((a[k]-b[k]).abs().max()) for k in a)
def step(m,o,b,u):
 for group in o.param_groups:group['lr']=group['base_lr']*lr_factor(u,200,6000)
 o.zero_grad(set_to_none=True);out=m(b[0],b[2]);loss=dense.dense_ce(out,b[1],b[2]);loss.backward();gr={g['name']:float(sum(p.grad.float().square().sum() for p in g['params'] if p.grad is not None).sqrt()) for g in o.param_groups};norm=float(torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad],1.));assert np.isfinite(float(loss)) and np.isfinite(norm);o.step();return {'u':u,'loss':float(loss.detach()),'gradient_norm':norm,'groups':gr,'physical_chunks':2,'encoder_clips':8,'valid_original_positions':int(b[2].sum()),'pid':os.getpid()}
def execute(cfg,b,steps=4,checkpoint=None):
 m=build('J-low',cfg).train();o=opt(m,cfg);start=0
 if checkpoint is not None:m.load_state_dict(checkpoint['model']);o.load_state_dict(checkpoint['optimizer']);torch.set_rng_state(checkpoint['cpu_rng']);torch.cuda.set_rng_state_all(checkpoint['cuda_rng']);start=checkpoint['update']
 rows=[step(m,o,b,u) for u in range(start,steps)];return m,o,rows
def main():
 start=time.time();cfg=read(CFG);freeze=read(RUN/'graph_freeze.json');assert freeze['actual_git_objects_verified']
 for p,h in freeze['objects'].items():assert sha(ROOT/p)==h,p
 src=read(PUB/'source.json');assert all(sha(source.UPSTREAM/p)==h for p,h in src['loaded_model_subset_SHA'].items());assert src['passed'] and sha(ROOT/cfg['source_state'])==src['target_encoder_sha256'];assert sha(RUN/'FIT_smoke.private.npz')==read(PUB/'smoke_input.json')['FIT_smoke_sha256'];torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
 z=np.load(RUN/'FIT_smoke.private.npz');b=tuple(torch.from_numpy(z[k]).cuda() for k in ['frames','labels','mask']);assert tuple(b[0].shape)==(2,64,224,224,3);print(json.dumps({'FIT_GPU_check_started':True,'pid':os.getpid(),'device':torch.cuda.get_device_name(0),'actual_shape':list(b[0].shape)}),flush=True)
 f=build('F',cfg).eval();j=build('J-low',cfg).eval()
 with torch.no_grad():fo,jo=f(b[0],b[2]),j(b[0],b[2])
 parity=max(float((x-y).abs().max()) for x,y in zip(fo,jo));assert parity<=1e-4 and len(fo)==len(jo)==2 and all(tuple(x.shape)==(2,5,32) for x in jo)
 # Independent CE on every valid original position, not an unweighted average of chunks.
 target=b[1].cpu().numpy();valid=b[2].cpu().numpy();manual=[]
 for stage in jo:
  x=dense.expand_logits(stage).cpu().numpy().transpose(0,2,1).astype(float);xx=x[valid];yy=target[valid];maximum=xx.max(1);manual.append(float(np.mean(maximum+np.log(np.exp(xx-maximum[:,None]).sum(1))-xx[np.arange(len(xx)),yy])))
 formula_error=abs(float(dense.dense_ce(jo,b[1],b[2]))-float(np.mean(manual)));assert formula_error<=1e-5
 f.train();o=opt(f,cfg);fr=step(f,o,b,0);assert all(p.grad is None for p in f.encoder.parameters());assert fr['groups']['head']>0;del f,j,o,fo,jo;torch.cuda.empty_cache()
 m,o,rows=execute(cfg,b);expected=state(m);base=read(PUB/'source.json');assert rows[-1]['groups']['active_video']>0 and rows[-1]['groups']['head']>0;del m,o;torch.cuda.empty_cache();m,o,again=execute(cfg,b);repeat=difference(expected,state(m));lossdiff=max(abs(x['loss']-y['loss']) for x,y in zip(rows,again));assert repeat<=1e-5 and lossdiff<=1e-4;del m,o;torch.cuda.empty_cache()
 m,o,_=execute(cfg,b,steps=2);path=RUN/'check_resume.pt';torch.save({'model':state(m),'optimizer':o.state_dict(),'cpu_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'update':2},path);del m,o;torch.cuda.empty_cache();checkpoint=torch.load(path,map_location='cpu',weights_only=True);m,o,_=execute(cfg,b,checkpoint=checkpoint);resume=difference(expected,state(m));assert resume<=1e-5;del m,o,checkpoint;path.unlink();torch.cuda.empty_cache()
 m=build('J-low',cfg).train();o=opt(m,cfg);initial=state(m);trace=[];torch.cuda.synchronize();timed_start=time.time()
 for u in range(40):
  row=step(m,o,b,u);trace.append(row)
  if (u+1)%5==0:write(RUN/'progress.json',{'phase':'FIT_graph_smoke','pid':os.getpid(),'device':torch.cuda.get_device_name(0),'update':u+1,'total':40,'loss':row['loss'],'epoch':time.time()});print(json.dumps({'FIT_update':u+1,'total':40,'loss':row['loss'],'pid':os.getpid()}),flush=True)
 torch.cuda.synchronize();elapsed=time.time()-timed_start;current=state(m);active=[n for n,p in m.encoder.named_parameters() if p.requires_grad];inactive=src['inactive_video_names'];assert all(torch.equal(current['encoder.'+n],initial['encoder.'+n]) for n in inactive);assert all(any(not torch.equal(current['encoder.'+n],initial['encoder.'+n]) for n in active if n.startswith(prefix)) for prefix in ['patch_embed.','blocks.0.','blocks.11.','norms_block.3.']);assert {id(p) for group in o.param_groups for p in group['params']}=={id(p) for p in m.parameters() if p.requires_grad};assert all(int(v['step'])==40 for v in o.state.values());write(RUN/'FIT_learning.private.json',trace)
 with torch.no_grad():torch.cuda.synchronize();fs=time.time();m.eval();out=m(b[0],b[2]);torch.cuda.synchronize();infer=time.time()-fs
 result={'passed':True,'graph_freeze_git_commit':freeze['git_commit'],'source_manifest_sha256':sha(PUB/'source.json'),'actual_FIT_shape':[2,64,224,224,3],'device':torch.cuda.get_device_name(0),'pid':os.getpid(),'initial_FJ_logits_max_abs':parity,'independent_original_position_CE_error':formula_error,'all_outputs_complete5class_32tubelets_64originalpositions':True,'frozen_encoder_gradients_absent':True,'joint_head_active_video_gradients_finite_nonzero':True,'active_video149optimizer_binding':True,'inactive9state_unchanged':True,'repeat_max_abs':repeat,'repeat_loss_max_abs':lossdiff,'disk2plus2_resume_max_abs':resume,'FIT_check_optimizer_updates':53,'J_timed_updates':40,'J_seconds_per_2x64_update':elapsed/40,'inference_seconds_per_2x64':infer,'initial_FIT_loss':trace[0]['loss'],'final_FIT_loss':trace[-1]['loss'],'GPU_peak_reserved_bytes':torch.cuda.max_memory_reserved(),'new_external_development_forwards':0,'formal_models_started':0,'wall_seconds':time.time()-start};write(PUB/'checks.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':
 start=time.time();ok=False
 try:main();ok=True
 finally:
  u=resource.getrusage(resource.RUSAGE_SELF);write(RUN/'graph_check_process.json',{'pid':os.getpid(),'ok':ok,'CPU_core_seconds':u.ru_utime+u.ru_stime,'wall_seconds':time.time()-start,'GPU_job_wall_seconds':time.time()-start})
