#!/usr/bin/env python3
"""Bounded A57-MD real FIT graph admission; cannot run before core confirmation."""
import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES','0');os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8');os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import json,resource,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from scripts import train_dense_multidepth as run
from scripts import train_dense_sequence as base
from echo_routing.temporal import dense_multidepth as md,jepa21_source as source,dense_runtime as data
from echo_routing.temporal.dense_readout import ReadoutPixels
from echo_routing.temporal.dense_durable import atomic_torch_save

def build(cfg,arm='J-MD'):
    model=md.build_model(cfg,arm,2026110257);return model,run.make_optimizer(model,cfg)

def execute(cfg,b,steps=4,checkpoint=None):
    model,optimizer=build(cfg);model.train();start=0
    if checkpoint is not None:
        model.load_state_dict(checkpoint['model'],strict=True);optimizer.load_state_dict(checkpoint['optimizer']);torch.set_rng_state(checkpoint['cpu_rng']);torch.cuda.set_rng_state_all(checkpoint['cuda_rng']);start=checkpoint['update']
    rows=[base.step(model,optimizer,b,u) for u in range(start,steps)];return model,optimizer,rows

def main():
    cfg,freeze=run.context(False);assert not any((run.RUN/f'seed{s}'/a/'physical_updates.jsonl').exists() for s in cfg['training_seeds'] for a in cfg['arms']);z=np.load(base.RUN/'FIT_smoke.private.npz');assert base.sha(base.RUN/'FIT_smoke.private.npz')==base.read(base.PUB/'smoke_input.json')['FIT_smoke_sha256'];b=tuple(torch.from_numpy(z[k]).cuda() for k in ['frames','labels','mask']);assert tuple(b[0].shape)==(2,64,224,224,3)
    print(json.dumps({'MD_real_FIT_graph_started':True,'pid':os.getpid(),'original_core_confirmation_complete':True}),flush=True)
    f,fo=build(cfg,'F-MD');j,jo=build(cfg);f.eval();j.eval()
    with torch.no_grad():fl,jl=f(b[0],b[2]),j(b[0],b[2])
    parity=max(float((a-bb).abs().max()) for a,bb in zip(fl,jl));assert parity<=1e-4 and all(tuple(x.shape)==(2,5,32) for x in fl+jl)
    # Same actual16-frame RGB tensors, independent official final-only instance.
    clips=b[0].reshape(8,16,224,224,3).float()/255.;clips=((clips-clips.new_tensor([.485,.456,.406]))/clips.new_tensor([.229,.224,.225])).permute(0,4,1,2,3).contiguous();plain=source.build_encoder();plain.load_state_dict(torch.load(ROOT/cfg['source_state'],map_location='cpu',weights_only=True,mmap=True),strict=True);plain=plain.cuda().eval()
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):hier=f.encoder(clips);final=plain(clips)
    assert tuple(hier.shape)==(8,1568,3072) and tuple(final.shape)==(8,1568,768);last_error=float((hier[:,:,-768:]-final).abs().max());assert last_error<=1e-5;del clips,plain,hier,final
    f.train();fr=base.step(f,fo,b,0);assert fr['groups']['head']>0 and all(p.grad is None for p in f.encoder.parameters());del f,j,fo,jo,fl,jl;torch.cuda.empty_cache()
    m,o,rows=execute(cfg,b);expected=base.state(m);names=[n for n,p in m.encoder.named_parameters() if p.requires_grad];assert len(names)==155 and all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for n,p in m.encoder.named_parameters() if p.requires_grad);assert all(int(v['step'])==4 and torch.isfinite(v['exp_avg']).all() and torch.isfinite(v['exp_avg_sq']).all() and v['exp_avg_sq'].sum()>0 for v in o.state.values()) and len(o.state)==211;del m,o;torch.cuda.empty_cache()
    m,o,again=execute(cfg,b);repeat=base.difference(expected,base.state(m));repeat_loss=max(abs(a['loss']-bb['loss']) for a,bb in zip(rows,again));assert repeat<=1e-5 and repeat_loss<=1e-4;del m,o;torch.cuda.empty_cache()
    m,o,_=execute(cfg,b,2);p=run.RUN/'FIT_resume.pt';atomic_torch_save(run.payload(m,o,2,'J-MD',2026110257,'fixedFITfixture'),p);del m,o;torch.cuda.empty_cache();saved=torch.load(p,map_location='cpu',weights_only=True);m,o,_=execute(cfg,b,4,saved);resume=base.difference(expected,base.state(m));assert resume<=1e-5;del m,o,saved;torch.cuda.empty_cache();p.unlink()
    fit=base.inputs('fit');pixels=ReadoutPixels(fit);plan=dict(np.load(base.RUN/'plan.private.npz'));raw=pixels.batch([fit[int(i)] for i in plan['sequence'][0]],plan['start'][0],'cuda');assert all(torch.equal(x,y) for x,y in zip(raw,b));model,optimizer=build(cfg);model.train();initial=base.state(model);torch.cuda.synchronize();start=time.time();trace=[]
    for u in range(20):
        batch=pixels.batch([fit[int(i)] for i in plan['sequence'][u]],plan['start'][u],'cuda');trace.append(base.step(model,optimizer,batch,u))
    torch.cuda.synchronize();seconds_per_update=(time.time()-start)/20;checkpoint=run.RUN/'FIT_full_optimizer.pt';start=time.time();atomic_torch_save(run.payload(model,optimizer,20,'J-MD',2026110257,'first20fixedplans'),checkpoint);checkpoint_seconds=time.time()-start;checkpoint_bytes=checkpoint.stat().st_size;checkpoint.unlink();current=base.state(model)
    assert all(torch.equal(current['encoder.'+n],initial['encoder.'+n]) for n in md.IMAGE_ONLY);assert all(not torch.equal(current['encoder.'+n],initial['encoder.'+n]) for n in names);assert {id(p) for g in optimizer.param_groups for p in g['params']}=={id(p) for p in model.parameters() if p.requires_grad};assert len(optimizer.state)==211 and all(int(v['step'])==20 for v in optimizer.state.values())
    chosen=[fit[int(i)] for i in np.linspace(0,len(fit)-1,12,dtype=int)];model.eval();start=time.time();chunks=0
    from echo_routing.temporal.dense_sequence import chunk_starts
    for q in chosen:
        prob=data.predict_sequence(model,q,pixels);assert prob.shape==(len(q['labels']),5) and np.allclose(prob.sum(1),1,atol=1e-5);chunks+=len(chunk_starts(len(q['labels'])))
    infer_seconds=time.time()-start;result={'passed':True,'stage_freeze_git':freeze['git_commit'],'strict_source_keys':158,'hierarchy':list(md.LAYERS),'feature_contract':md.FEATURE_CONTRACT,'actual_encoder_tokens':[8,1568,3072],'actual_supervised_axis':[2,64],'initial_FJ_logits_max_abs':parity,'hierarchical_last768_vs_official_final_grid_max_abs':last_error,'active_video_tensors':155,'head_tensors':56,'optimizer_tensors':211,'all155finite_nonzero_gradient_and_moment':True,'six_new_norms_gradient_paths_live':True,'image3_unchanged_unbound':True,'all155_video_tensors_changed_after20rawFITupdates':True,'repeat_parameter_max_abs':repeat,'repeat_loss_max_abs':repeat_loss,'disk2plus2_resume_max_abs':resume,'check_optimizer_updates':33,'raw_different_FIT_updates_timed':20,'raw_J_seconds_per2x64_update':seconds_per_update,'atomic_full_optimizer_checkpoint_seconds':checkpoint_seconds,'atomic_full_checkpoint_bytes':checkpoint_bytes,'FIT_full_sequences_infer':12,'FIT_infer_chunks':chunks,'raw_decode_infer_stitch_seconds_per_chunk':infer_seconds/chunks,'GPU_peak_reserved_bytes':torch.cuda.max_memory_reserved(),'new_MD_formal_updates':0,'new_MD_outer_forwards':0,'protected_media_forwards':0,'pid':os.getpid()};run.write(run.PUB/'checks.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':
    run.RUN.mkdir(parents=True,exist_ok=True);run.PUB.mkdir(parents=True,exist_ok=True);start=time.time();ok=False
    try:main();ok=True
    finally:
        r=resource.getrusage(resource.RUSAGE_SELF);run.write(run.RUN/'check_latest_process.json',{'pid':os.getpid(),'ok':ok,'CPU_core_seconds':r.ru_utime+r.ru_stime,'wall_seconds':time.time()-start,'GPU_job_wall_seconds':time.time()-start})
