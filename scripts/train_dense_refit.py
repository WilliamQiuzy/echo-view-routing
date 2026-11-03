#!/usr/bin/env python3
"""A57 adaptive full-data fixed6000 matched CE/stability pair, one global queue."""
import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES','0');os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8');os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import json,resource,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from scripts import train_dense_sequence as base
from echo_routing.temporal import dense_refit as loss
from echo_routing.temporal.dev_train import lr_factor
from echo_routing.temporal.dense_durable import atomic_torch_save,resume_accounting
from echo_routing.temporal.dense_readout import ReadoutPixels
from scripts.audit_dense_input_exposure import digest
RUN=base.RUN/'full_refit_v1';PUB=base.PUB/'full_refit';CFG=ROOT/'configs/experiments/dense_target_refit_v1.yaml';read,write,sha=base.read,base.write,base.sha
FEATURE={'kind':'official_final_layer768_dense_refit','source_video_tensors':149,'inactive_tensors':9,'head_tensors':56,'positions':64,'tubelet_positions':32,'TCN_channels':128,'stages':2}

def context(admitted=False):
    cfg=read(CFG);freeze=read(RUN/'stage_freeze.json');core=read(base.PUB/'summary.json');assert core['A08_historical_metrics_reproduced_all3seeds_bothhalves'] and core['raw_reader_metrics_independently_reproduced'] and read(base.PUB/'models_manifest.json')['formal_physical_updates']==32000
    for p,h in freeze['objects'].items():assert sha(ROOT/p)==h,p
    for p,h in read(base.RUN/'pipeline_freeze.json')['objects'].items():assert sha(ROOT/p)==h,p
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    if admitted:
        a=read(PUB/'admission.json');assert a['all_caps_fit'] and a['stage_freeze_git']==freeze['git_commit'] and a['formal_refit_updates_at_admission']==0 and a['checks_sha256']==sha(PUB/'checks.json') and a['MD_checks_sha256']==sha(base.PUB/'multidepth/checks.json') and a['preserved_MD_two_seed_obligations_included']
        admission_freeze=read(RUN/'formal_admission_freeze.json')
        for p,h in admission_freeze['objects'].items():assert sha(ROOT/p)==h,p
    return cfg,freeze

def build(cfg):
    parent=read(ROOT/cfg['base_config']);parent['seed']=cfg['seed'];assert sha(ROOT/cfg['source_state'])==read(ROOT/cfg['source_manifest'])['target_encoder_sha256'];model=base.build('J-high',parent);model.feature_contract=FEATURE;return model,base.build_optimizer(model,parent,'J-high')

def inputs():
    meta=read(PUB/'input_metadata.json');assert meta['passed'] and meta['native_clips']==3364 and meta['former628_inner_role'].startswith('TRAINING');assert sha(RUN/'train_sequences.private.npy')==meta['sequence_sha256'];q=np.load(RUN/'train_sequences.private.npy',allow_pickle=True).tolist();p=dict(np.load(RUN/'plan.private.npz'));assert digest(p)==meta['plan_canonical_sha256'];return q,p

def step(model,optimizer,batch,u,coefficient):
    for g in optimizer.param_groups:g['lr']=g['base_lr']*lr_factor(u,200,6000)
    optimizer.zero_grad(set_to_none=True);stages=model(batch[0],batch[2]);value,parts=loss.objective(stages,batch[1],batch[2],coefficient);value.backward();groups={g['name']:float(sum(p.grad.float().square().sum() for p in g['params'] if p.grad is not None).sqrt()) for g in optimizer.param_groups};norm=float(torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.));assert np.isfinite(float(value.detach())) and np.isfinite(norm);optimizer.step();edge=loss.eligible_edges(batch[1],batch[2]);clamp=[]
    with torch.no_grad():
        for z in stages:
            ell=torch.log_softmax(z.float(),1);clamp.append(int((((ell[:,:,1:]-ell[:,:,:-1]).square()>16)&edge[:,None]).sum()))
    return {'u':u,'loss':float(value.detach()),**{k:float(v.detach()) for k,v in parts.items()},'eligible_edges_per_chunk':edge.sum(1).tolist(),'clamped_class_edges_per_stage':clamp,'groups':groups,'gradient_norm':norm,'physical_chunks':2,'encoder_clips':8,'valid_original_positions':int(batch[2].sum()),'pid':os.getpid()}

def payload(m,o,cursor,arm,plan_sha):return {'model':base.state(m),'optimizer':o.state_dict(),'cpu_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'update':cursor,'arm':arm,'seed':2026110257,'schedule_total':6000,'feature_contract':FEATURE,'coefficient':read(CFG)['arms'][arm],'plan_canonical_sha256':plan_sha,'config_sha256':sha(CFG),'source_sha256':read(base.PUB/'source.json')['target_encoder_sha256']}

def milestone(m,folder,step,arm):
    path=folder/f'step{step}.pt'
    if path.exists():
        old=torch.load(path,map_location='cpu',weights_only=True,mmap=True);assert old['feature_contract']==FEATURE and read(folder/f'step{step}.identity.json')['sha256']==sha(path);current=base.state(m);assert set(old['model'])==set(current) and all(torch.equal(old['model'][k],v) for k,v in current.items());return
    atomic_torch_save({'model':base.state(m),'feature_contract':FEATURE,'arm':arm,'step':step,'seed':2026110257,'config_sha256':sha(CFG)},path);write(folder/f'step{step}.identity.json',{'sha256':sha(path),'feature_contract':FEATURE,'arm':arm,'step':step,'seed':2026110257,'source_sha256':read(base.PUB/'source.json')['target_encoder_sha256'],'plan_canonical_sha256':read(PUB/'input_metadata.json')['plan_canonical_sha256'],'config_sha256':sha(CFG)})

def train_one(cfg,arm,seqs,pixels,plan):
    folder=RUN/arm;folder.mkdir(exist_ok=True);latest=folder/'latest.pt';trace=folder/'physical_updates.jsonl';pending=folder/'pending.json';assert not pending.exists(),'ambiguous pending refit update; retain and stop';rows=[json.loads(s) for s in trace.read_text().splitlines()] if trace.exists() else [];attempt=max((r['attempt'] for r in rows),default=0)+1;m,o=build(cfg);m.train();cursor=0;plan_sha=digest(plan)
    if latest.exists():
        z=torch.load(latest,map_location='cpu',weights_only=True);assert z['arm']==arm and z['feature_contract']==FEATURE and z['plan_canonical_sha256']==plan_sha and z['config_sha256']==sha(CFG) and z['schedule_total']==6000;m.load_state_dict(z['model'],strict=True);o.load_state_dict(z['optimizer']);torch.set_rng_state(z['cpu_rng']);torch.cuda.set_rng_state_all(z['cuda_rng']);cursor=z['update'];write(folder/f'resume_attempt{attempt}.json',resume_accounting(rows,cursor));del z
        if cursor in [2000,6000]:milestone(m,folder,cursor,arm)
    elif rows:write(folder/f'resume_attempt{attempt}.json',resume_accounting(rows,0))
    for u in range(cursor,6000):
        b=pixels.batch([seqs[int(i)] for i in plan['sequence'][u]],plan['start'][u],'cuda');write(pending,{'u':u,'attempt':attempt,'epoch':time.time()});start=time.time();row=step(m,o,b,u,cfg['arms'][arm]);row.update({'attempt':attempt,'arm':arm,'epoch':time.time(),'model_step_wall_seconds':time.time()-start})
        with trace.open('a') as f:f.write(json.dumps(row)+'\n');f.flush();os.fsync(f.fileno())
        pending.unlink()
        if (u+1)%100==0:base.budget_guard();atomic_torch_save(payload(m,o,u+1,arm,plan_sha),latest)
        if u+1 in [2000,6000]:milestone(m,folder,u+1,arm)
        if (u+1)%25==0:
            progress={'phase':'adaptive_full_refit_training','arm':arm,'step':u+1,'target':6000,'model_pid':os.getpid(),'epoch':time.time(),'CE':row['CE'],'S':row['S'],'next_action':'finish bothfixed6000 arms;noinnerselection/noouterfeedback'};write(base.RUN/'runtimecoordination.json',progress);write(RUN/'progress.json',progress);print(json.dumps(progress),flush=True)
    del m,o;torch.cuda.empty_cache()

def formal():
    cfg,_=context(True);seqs,plan=inputs();pixels=ReadoutPixels(seqs)
    for arm in ['R-CE','R-ST']:train_one(cfg,arm,seqs,pixels,plan)
    write(base.RUN/'runtimecoordination.json',{'phase':'adaptive_refit_pair_complete_pending_audit','model_pid':None,'epoch':time.time(),'next_action':'audit bothfixed6000/models thenfrozen raw+transferredB3 readout;preservedMDnext'});print('full_refit_pair_complete',flush=True)

if __name__=='__main__':
    RUN.mkdir(exist_ok=True);PUB.mkdir(exist_ok=True);start=time.time();ok=False;mode=sys.argv[1]
    try:globals()[mode]();ok=True
    finally:
        r=resource.getrusage(resource.RUSAGE_SELF);write(RUN/(mode+'_latest_process.json'),{'pid':os.getpid(),'ok':ok,'CPU_core_seconds':r.ru_utime+r.ru_stime,'GPU_job_wall_seconds':time.time()-start})
