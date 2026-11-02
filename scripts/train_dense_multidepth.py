#!/usr/bin/env python3
"""A57-MD prospective same-axis full-video stage; definitions only until admitted."""
import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES','0');os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8');os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import json,resource,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from scripts import train_dense_sequence as base
from echo_routing.temporal import dense_multidepth as md,dense_runtime as data
from echo_routing.temporal.dense_durable import atomic_torch_save,resume_accounting
from echo_routing.temporal.dense_readout import ReadoutPixels,SequenceReadout
from scripts.audit_dense_input_exposure import digest
PHASE=base.RUN;RUN=PHASE/'multidepth_v1';PUB=base.PUB/'multidepth';CFG=ROOT/'configs/experiments/dense_target_multidepth_v1.yaml'
read,write,sha=base.read,base.write,base.sha

def context(require_admission=False):
    # Core confirmation has priority. These assertions run before any real MD forward.
    core=read(base.PUB/'models_manifest.json');summary=read(base.PUB/'summary.json');assert core['passed'] and core['formal_physical_updates']==32000 and summary['model_manifest_sha256']==sha(base.PUB/'models_manifest.json') and summary['A08_historical_metrics_reproduced_all3seeds_bothhalves']
    assert all(any(m['seed']==2026110258 and m['arm']==arm and m['updates']==6000 for m in core['models']) for arm in ['F','J-high'])
    cfg=read(CFG);freeze=read(RUN/'stage_freeze.json')
    for p,h in freeze['objects'].items():assert sha(ROOT/p)==h,p
    # Preserve every original producer object, including its original strategy copy.
    for p,h in read(PHASE/'pipeline_freeze.json')['objects'].items():assert sha(ROOT/p)==h,p
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    if require_admission:
        admission=read(PUB/'admission.json');assert admission['all_shared_caps_fit'] and admission['stage_freeze_git']==freeze['git_commit'] and admission['formal_MD_updates_at_admission']==0;assert admission['training_seeds'] in [[2026110257],[2026110257,2026110258]]
    return cfg,freeze

def make_optimizer(model,cfg):
    groups=[{'params':list(model.head.parameters()),'name':'head','base_lr':cfg['head_lr'],'lr':cfg['head_lr']}]
    if model.joint:groups.append({'params':[p for p in model.encoder.parameters() if p.requires_grad],'name':'active_video','base_lr':cfg['J_encoder_lr'],'lr':cfg['J_encoder_lr']})
    return torch.optim.AdamW(groups,betas=tuple(cfg['optimizer']['betas']),eps=cfg['optimizer']['eps'],weight_decay=cfg['optimizer']['weight_decay'])

def payload(model,optimizer,cursor,arm,seed,plan_digest):
    return {'model':base.state(model),'optimizer':optimizer.state_dict(),'cpu_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'update':cursor,'arm':arm,'seed':seed,'schedule_total':6000,'feature_contract':model.feature_contract,'source_sha256':read(base.PUB/'source.json')['target_encoder_sha256'],'plan_canonical_sha256':plan_digest}

def milestone(model,folder,cursor,arm,seed):
    path=folder/f'step{cursor}.pt'
    if path.exists():
        identity=read(folder/f'step{cursor}.identity.json');saved=torch.load(path,map_location='cpu',weights_only=True);expected=base.state(model) if model.joint else {k:v.detach().cpu() for k,v in model.head.state_dict().items()};stored=saved['model'] if model.joint else saved['head']
        if identity['sha256']!=sha(path) or saved['feature_contract']!=model.feature_contract or set(stored)!=set(expected) or any(not torch.equal(stored[k],v) for k,v in expected.items()):raise ValueError('existing immutable milestone differs from restored model')
        return identity
    value={'model':base.state(model)} if model.joint else {'head':{k:v.detach().cpu() for k,v in model.head.state_dict().items()}}
    value.update({'feature_contract':model.feature_contract,'arm':arm,'seed':seed,'step':cursor});atomic_torch_save(value,path);identity={'sha256':sha(path),'arm':arm,'seed':seed,'step':cursor,'feature_contract':model.feature_contract,'source_sha256':read(base.PUB/'source.json')['target_encoder_sha256'],'stage_freeze_git':read(RUN/'stage_freeze.json')['git_commit']};write(folder/f'step{cursor}.identity.json',identity);return identity

def inner(model,sequences,pixels,folder,cursor,arm,seed,identity):
    out=folder/f'inner_step{cursor}';out.mkdir(parents=True,exist_ok=True);result=out/'metrics.json'
    if result.exists():
        old=read(result);assert old['model_checkpoint_sha256']==identity['sha256'] and old['feature_contract']==md.FEATURE_CONTRACT;return old
    model.eval();start=time.time();input_sha=read(base.PUB/'input_metadata.json')['roles']['inner_cal']['sequence_sha256'];binding=identity['sha256']+'|'+input_sha+'|'+sha(CFG);cache=SequenceReadout(out/'sequence_cache',binding,len(sequences),lengths=[len(q['labels']) for q in sequences])
    for i in range(cache.cursor,len(sequences)):
        q=sequences[i];cache.run_one(i,lambda q=q:data.predict_sequence(model,q,pixels))
        if (i+1)%100==0:
            base.budget_guard();write(PHASE/'runtimecoordination.json',{'phase':'multidepth_inner','arm':arm,'seed':seed,'step':cursor,'sequence_done':i+1,'sequences':len(sequences),'model_pid':os.getpid(),'epoch':time.time(),'next_action':'finish fixed inner row then continue admitted6000 schedule'})
    probs={q['id']:cache.load(i) for i,q in enumerate(sequences)};preds={k:v.argmax(1).astype(np.int8) for k,v in probs.items()};np.save(out/'probabilities.private.npy',probs,allow_pickle=True);np.save(out/'predictions.private.npy',preds,allow_pickle=True);metrics,tables=data.metrics(sequences,preds)
    for name,t in tables.items():t.to_csv(out/(name+'.private.csv'),index=False)
    row={**metrics,'arm':arm,'seed':seed,'step':cursor,'backbone_lr':0 if arm=='F-MD' else 3e-5,'feature_contract':model.feature_contract,'model_checkpoint_sha256':identity['sha256'],'input_sequence_sha256':input_sha,'wall_seconds':time.time()-start,'selection_scope':'same fixed protocoltrain session innercal only'};write(result,row);return row

def train_one(cfg,arm,seed,fit,cal,tp,cp,plan):
    folder=RUN/f'seed{seed}'/arm;folder.mkdir(parents=True,exist_ok=True);latest=folder/'latest.pt';trace=folder/'physical_updates.jsonl';pending=folder/'pending.json'
    if pending.exists():raise RuntimeError('ambiguous pending MD update: retain evidence and stop, do not silently rerun')
    rows=[json.loads(x) for x in trace.read_text().splitlines()] if trace.exists() else [];attempt=max((r['attempt'] for r in rows),default=0)+1;model=md.build_model(cfg,arm,seed).train();optimizer=make_optimizer(model,cfg);cursor=0;plan_sha=digest(plan)
    if latest.exists():
        z=torch.load(latest,map_location='cpu',weights_only=True);assert z['arm']==arm and z['seed']==seed and z['feature_contract']==md.FEATURE_CONTRACT and z['plan_canonical_sha256']==plan_sha and z['schedule_total']==6000;model.load_state_dict(z['model'],strict=True);optimizer.load_state_dict(z['optimizer']);torch.set_rng_state(z['cpu_rng']);torch.cuda.set_rng_state_all(z['cuda_rng']);cursor=z['update'];write(folder/f'resume_attempt{attempt}.json',resume_accounting(rows,cursor));del z
        # Finish a committed checkpoint-boundary readout before advancing after a clean restart.
        if cursor and cursor%500==0 and not (folder/f'inner_step{cursor}/metrics.json').exists():
            identity=milestone(model,folder,cursor,arm,seed);rng,crng=torch.get_rng_state(),torch.cuda.get_rng_state_all();inner(model,cal,cp,folder,cursor,arm,seed,identity);torch.set_rng_state(rng);torch.cuda.set_rng_state_all(crng);model.train()
    if rows and not latest.exists():write(folder/f'resume_attempt{attempt}.json',resume_accounting(rows,0))
    if cursor>=6000:del model,optimizer;torch.cuda.empty_cache();return
    for u in range(cursor,6000):
        batch=tp.batch([fit[int(i)] for i in plan['sequence'][u]],plan['start'][u],'cuda');write(pending,{'attempt':attempt,'u':u,'epoch':time.time(),'plan_canonical_sha256':plan_sha});start=time.time();row=base.step(model,optimizer,batch,u);row.update({'attempt':attempt,'arm':arm,'seed':seed,'epoch':time.time(),'model_step_wall_seconds':time.time()-start,'feature_kind':'official_four_depth_concat'})
        with trace.open('a') as f:f.write(json.dumps(row)+'\n');f.flush();os.fsync(f.fileno())
        pending.unlink()
        if (u+1)%100==0:base.budget_guard();atomic_torch_save(payload(model,optimizer,u+1,arm,seed,plan_sha),latest)
        if (u+1)%25==0:
            progress={'phase':'multidepth_training','arm':arm,'seed':seed,'step':u+1,'target':6000,'model_pid':os.getpid(),'epoch':time.time(),'last_loss':row['loss'],'next_action':'continue admitted paired tier; inner every500'};write(PHASE/'runtimecoordination.json',progress);write(RUN/'progress.json',progress);print(json.dumps(progress),flush=True)
        if (u+1)%500==0:
            identity=milestone(model,folder,u+1,arm,seed);rng,crng=torch.get_rng_state(),torch.cuda.get_rng_state_all();inner(model,cal,cp,folder,u+1,arm,seed,identity);torch.set_rng_state(rng);torch.cuda.set_rng_state_all(crng);model.train()
    del model,optimizer;torch.cuda.empty_cache()

def formal():
    cfg,freeze=context(require_admission=True);admission=read(PUB/'admission.json');fit,cal=base.inputs('fit'),base.inputs('inner_cal');tp,cp=ReadoutPixels(fit),ReadoutPixels(cal);proof=read(base.PUB/'independent_input_exposure.json')
    for seed in admission['training_seeds']:
        path=PHASE/('plan.private.npz' if seed==2026110257 else 'second_seed_plan.private.npz');plan=dict(np.load(path));assert digest(plan)==proof['plans'][str(seed)]['canonical_little_endian_int64_plan_sha256']
        for arm in ['F-MD','J-MD']:train_one(cfg,arm,seed,fit,cal,tp,cp,plan)
    write(PHASE/'runtimecoordination.json',{'phase':'multidepth_tier_training_complete_pending_model_audit','model_pid':None,'epoch':time.time(),'training_seeds':admission['training_seeds'],'next_action':'freeze MD models, inner decoder, same-axis matched readout and shared resources'});print('multidepth_tier_complete',flush=True)

if __name__=='__main__':
    RUN.mkdir(parents=True,exist_ok=True);PUB.mkdir(parents=True,exist_ok=True);start=time.time();ok=False;mode=sys.argv[1]
    try:globals()[mode]();ok=True
    finally:
        r=resource.getrusage(resource.RUSAGE_SELF);write(RUN/(mode+'_latest_process.json'),{'pid':os.getpid(),'ok':ok,'CPU_core_seconds':r.ru_utime+r.ru_stime,'wall_seconds':time.time()-start,'GPU_job_wall_seconds':time.time()-start})
