#!/usr/bin/env python3
"""The fixed eight head-only6000-update A58 models; no outer data is opened."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import sys,time,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from scripts.structured_common import ROOT,RUN,PUB,CFG,read,write,sha,verify_freeze
from scripts.structured_head_runtime import load_bank,head_identity,evaluate_inner
from echo_routing.temporal.structured_evidence import StructuredEvidence
from echo_routing.temporal.structured_training import optimizer
from echo_routing.temporal.structured_loop import run_updates
from echo_routing.temporal.structured_lifecycle import state_digest

def main():
    freeze=verify_freeze('formal_freeze.json');verify_freeze('producer_freeze.json');cfg=read(CFG);admission=read(PUB/'admission.json');assert admission['all_caps_fit'];backend=admission['training_backend'];assert backend=='cpu';torch.set_num_threads(1);torch.use_deterministic_algorithms(True);fit=load_bank('fit',backend);inner=load_bank('inner_cal',backend);manifest=read(PUB/'input_manifest.json');completed=0;models=[]
    for seed in cfg['training']['seeds']:
        pin=manifest['plans'][str(seed)];assert sha(ROOT/pin['path'])==pin['sha256'];plan=np.load(ROOT/pin['path']);assert plan.shape==(6000,2)
        for arm in cfg['training']['arms']:
            model=StructuredEvidence(arm,seed).to(backend);initial=state_digest(model.state_dict());opt=optimizer(model);identity=head_identity(freeze,seed,arm,backend);folder=RUN/f'seed{seed}'/arm
            def progress(step,row):
                value={'phase':'formal_structured_head_training','model_pid':os.getpid(),'backend':backend,'seed':seed,'arm':arm,'updates':step,'Adam_steps':sorted({int(s['step']) for s in opt.state.values()}),'completed_formal_optimizer_updates':completed+step,'total_declared_formal_optimizer_updates':48000,'loss':row['loss'],'last_lr':row['lr'],'epoch':time.time(),'next_action':'finish all8models/24innerrows; freeze all weights and choices before outer'};write(RUN/'runtimecoordination.json',value);print(json.dumps(value),flush=True)
            start=time.time();record=run_updates(model,opt,fit.batch,plan,folder,identity,on_milestone=lambda step,path:evaluate_inner(model,inner,path,step,identity),on_progress=progress);completed+=record['physical_optimizer_updates'];models.append({'seed':seed,'arm':arm,'identity':identity,'initial_model_digest':initial,'final_model_digest':record['model_digest'],'physical_optimizer_updates':record['physical_optimizer_updates'],'wall_seconds_this_invocation':time.time()-start,'backend':backend,'fresh_seeded_initialization':True,'fixed_full6000_schedule':True});write(PUB/'formal_progress.json',{'completed_models':models,'completed_physical_optimizer_updates':completed,'declared_optimizer_updates':48000,'formal_freeze_git':freeze['git_commit']})
    assert completed==48000;rows=[read(PUB/'inner'/f'seed{seed}_{arm}_step{step}.json') for seed in cfg['training']['seeds'] for arm in cfg['training']['arms'] for step in cfg['training']['milestones']];write(PUB/'inner_results.json',{'rows':rows,'count':len(rows),'formal_optimizer_updates':completed,'inner_only':True});write(RUN/'runtimecoordination.json',{'phase':'formal_matrix_complete_pending_independent_model_audit','model_pid':None,'formal_optimizer_updates':completed,'inner_rows':len(rows),'epoch':time.time(),'next_action':'independently verify all model/Adam/plan/journal identities and freeze all8inner choices before fresh outer'});print(json.dumps({'formal_optimizer_updates':completed,'models':len(models),'inner_rows':len(rows),'backend':backend}),flush=True)
if __name__=='__main__':main()
