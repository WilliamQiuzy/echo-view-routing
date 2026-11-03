#!/usr/bin/env python3
"""FIT-only admission for new loss paths, lambda0, optimizer/RNG and raw timing."""
import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES','0');os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8');os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from scripts import train_dense_refit as run,train_dense_sequence as base
from echo_routing.temporal import dense_refit as loss
from echo_routing.temporal.dense_sequence import TwoStageTCN
from echo_routing.temporal.dense_durable import atomic_torch_save
from echo_routing.temporal.dense_readout import ReadoutPixels
from echo_routing.temporal.multidepth_audit import verify_optimizer
from echo_routing.temporal.dev_train import lr_factor

def execute(cfg,b,steps=4,saved=None):
    m,o=run.build(cfg);m.train();start=0
    if saved is not None:m.load_state_dict(saved['model'],strict=True);o.load_state_dict(saved['optimizer']);torch.set_rng_state(saved['cpu_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng']);start=saved['update']
    rows=[run.step(m,o,b,u,.15) for u in range(start,steps)];return m,o,rows

def main():
    cfg,freeze=run.context(False);assert not any((run.RUN/a/'physical_updates.jsonl').exists() for a in cfg['arms']);assert run.sha(run.RUN/'FIT_smoke.private.npz')==run.read(run.PUB/'smoke_input.json')['sha256'];z=np.load(run.RUN/'FIT_smoke.private.npz');b=tuple(torch.from_numpy(z[k]).cuda() for k in ['frames','labels','mask']);assert tuple(b[0].shape)==(2,64,224,224,3);print(json.dumps({'adaptive_FIT_check_started':True,'pid':os.getpid(),'formal_updates':0}),flush=True)
    a,ao=run.build(cfg);c,co=run.build(cfg);a.eval();c.eval()
    with torch.no_grad():x,y=a(b[0],b[2]),c(b[0],b[2])
    initial=max(float((u-v).abs().max()) for u,v in zip(x,y));assert initial<=1e-4;del a,c,ao,co,x,y;torch.cuda.empty_cache()
    p,po=run.build(cfg);p.train();parent=base.step(p,po,b,0);expected=base.state(p);mom=po.state_dict();del p,po;torch.cuda.empty_cache();m,o=run.build(cfg);m.train();zero=run.step(m,o,b,0,0.);zero_error=base.difference(expected,base.state(m));assert zero_error==0 and parent['loss']==zero['loss'];actual=o.state_dict();assert mom['param_groups']==actual['param_groups'] and set(mom['state'])==set(actual['state']) and all(torch.equal(mom['state'][k][n],v) for k,st in actual['state'].items() for n,v in st.items());del m,o,mom,actual;torch.cuda.empty_cache()
    m,o=run.build(cfg);m.train();stages=m(b[0],b[2]);diagnostic=loss.diagnostics(stages,b[1],b[2]);assert np.isfinite(diagnostic['CE']) and np.isfinite(diagnostic['S']);del stages,m,o;torch.cuda.empty_cache()
    m,o,rows=execute(cfg,b);expected=base.state(m);del m,o;torch.cuda.empty_cache();m,o,again=execute(cfg,b);repeat=base.difference(expected,base.state(m));lossdiff=max(abs(a['loss']-c['loss']) for a,c in zip(rows,again));assert repeat<=1e-5 and lossdiff<=1e-4;del m,o;torch.cuda.empty_cache()
    m,o,_=execute(cfg,b,2);path=run.RUN/'FIT_resume.pt';atomic_torch_save(run.payload(m,o,2,'R-ST','fixture'),path);del m,o;torch.cuda.empty_cache();saved=torch.load(path,map_location='cpu',weights_only=True);m,o,_=execute(cfg,b,4,saved);resume=base.difference(expected,base.state(m));assert resume<=1e-5;del m,o,saved;path.unlink();torch.cuda.empty_cache()
    seqs,plan=run.inputs();pixels=ReadoutPixels(seqs);raw=pixels.batch([seqs[int(i)] for i in plan['sequence'][0]],plan['start'][0],'cuda');assert all(torch.equal(x,y) for x,y in zip(raw,b));m,o=run.build(cfg);m.train();torch.cuda.synchronize();start=time.time()
    for u in range(20):batch=pixels.batch([seqs[int(i)] for i in plan['sequence'][u]],plan['start'][u],'cuda');run.step(m,o,batch,u,.15)
    torch.cuda.synchronize();timing=(time.time()-start)/20;source=torch.load(ROOT/cfg['source_state'],map_location='cpu',weights_only=True,mmap=True);assert len(source)==158;head_shapes={'head.'+n:tuple(v.shape) for n,v in TwoStageTCN(768,128).state_dict().items()};verified=verify_optimizer(base.state(m),source,o.state_dict(),20,True,run.read(base.PUB/'source.json')['inactive_video_names'],lr_factor(19,200,6000),head_shapes);assert verified['active_video_optimizer_tensors']==149 and verified['head_optimizer_tensors']==56 and all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for p in m.parameters() if p.requires_grad);assert {id(p) for g in o.param_groups for p in g['params']}=={id(p) for p in m.parameters() if p.requires_grad}
    path=run.RUN/'FIT_full_optimizer.pt';start=time.time();atomic_torch_save(run.payload(m,o,20,'R-ST','first20fullplans'),path);atomic_seconds=time.time()-start;size=path.stat().st_size;path.unlink();result={'passed':True,'stage_freeze_git':freeze['git_commit'],'source_keys':158,'actual_input_shape':[2,64,224,224,3],'initial_RCE_RST_logits_max_abs':initial,'lambda0_parent_optimizer_and_weights_exact':True,'lambda0_parameter_max_abs':zero_error,'new_loss_initial_diagnostic':diagnostic,'repeat_parameter_max_abs':repeat,'repeat_loss_max_abs':lossdiff,'disk2plus2_resume_max_abs':resume,'check_optimizer_updates':34,'check_neural_forwards':37,'diagnostic_logit_gradient_calls':2,**verified,'all205finite_nonzero_gradients':True,'raw20_RST_seconds_per_update':timing,'atomic_full_optimizer_seconds':atomic_seconds,'atomic_full_optimizer_bytes':size,'GPU_peak_reserved_bytes':torch.cuda.max_memory_reserved(),'formal_refit_updates':0,'new_outer_forwards':0,'protected_media_forwards':0,'pid':os.getpid()};run.write(run.PUB/'checks.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
