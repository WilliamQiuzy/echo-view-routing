#!/usr/bin/env python3
"""Zero-forward CPU binding of original evaluated final snapshots to optimizer models."""
import os,time,resource
START=time.time();os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from scripts.train_dense_sequence import RUN,PUB,read,write,sha
from echo_routing.temporal.dense_checkpoint_audit import compare_final_checkpoint

def main():
    torch.set_num_threads(1);freeze=read(RUN/'final_state_audit_freeze.json')
    for p,h in freeze['objects'].items():assert sha(ROOT/p)==h,p
    models=read(PUB/'models_manifest.json');assert models['passed'];out=[]
    for m in models['models']:
        folder=RUN/f"seed{m['seed']}"/m['arm'];latest=torch.load(folder/'latest.pt',map_location='cpu',weights_only=True,mmap=True);path=folder/f"step{m['updates']}.pt";saved=torch.load(path,map_location='cpu',weights_only=True,mmap=True);assert latest['update']==m['updates'] and latest['arm']==m['arm'] and latest['seed']==m['seed'] and sha(folder/'latest.pt')==m['latest_optimizer_sha256'] and sha(path)==read(folder/f"step{m['updates']}.identity.json")['sha256'];result=compare_final_checkpoint(saved,latest['model'],m['arm'].startswith('J'));out.append({'seed':m['seed'],'arm':m['arm'],'updates':m['updates'],'final_checkpoint_sha256':sha(path),**result});del latest,saved
    write(PUB/'final_checkpoint_binding.json',{'passed':True,'models':out,'formal_updates_bound':sum(m['updates'] for m in out),'new_model_forwards':0,'source_code_git':freeze['git_commit'],'models_manifest_sha256':sha(PUB/'models_manifest.json'),'epoch':time.time()});print(json.dumps({'all6_final_model_tensors_equal_audited_optimizer_models':True,'new_model_forwards':0}),flush=True)
if __name__=='__main__':
    ok=False
    try:main();ok=True
    finally:
        r=resource.getrusage(resource.RUSAGE_SELF);v={'stage':'core_final_tensor_binding_audit','model_pid':os.getpid(),'ok':ok,'process_CPU_seconds':r.ru_utime+r.ru_stime,'full_process_wall_seconds':time.time()-START,'GPU_job':False};write(RUN/'final_state_audit_process.json',v)
        with (RUN/'auxiliary_CPU_receipts.jsonl').open('a') as f:f.write(json.dumps(v)+'\n');f.flush();os.fsync(f.fileno())
