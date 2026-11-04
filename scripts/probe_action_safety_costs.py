#!/usr/bin/env python3
"""A61 cost-only structural head probe and label-free old-role geometry volumes."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from scripts.safety_common import *
from echo_routing.temporal.safety_budget import structural_cases
from echo_routing.temporal.safety_training import Utility,Risk
from echo_routing.temporal.safety_runtime import infer
from echo_routing.temporal.proposal_actions import propose,partition,apply


def main():
    freeze=verify_freeze('cost_probe_freeze.json')
    if (PUB/'inference_cost_bounds.json').exists():raise ValueError('cost probe already committed;no repeat')
    data=np.load(verify_pin(read(PUB/'training_inputs.json')['dataset']),allow_pickle=True).item()
    utility=Utility('S-L',2026110461);risk=Risk(2026110461)
    weightpins={}
    for name,model in [('S-L',utility),('risk',risk)]:
        path=RUN/'engineering'/name/'partial/latest.pt';value=torch.load(path,weights_only=True,map_location='cpu')
        if value['update']!=6 or value['seed']!=2026110461 or value['arm']!=name:raise ValueError('qualified engineering6 weights changed')
        model.load_state_dict(value['model'],strict=True);model.eval();weightpins[name]=array_pin(path)
    manifest=read(PUB/'input_manifest.json');T=max(640,max(v['max_length'] for role,v in manifest['roles'].items() if role!='fit'))
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);cases=[]
    for name,(a,j,b) in structural_cases(T).items():
        times=[];result=None
        for _ in range(3):
            start=time.perf_counter();result=infer(utility,risk,data['normalizers'],data['scale'],a,j,b);times.append(time.perf_counter()-start)
        actions=result['actions'];K=sum(np.count_nonzero(apply(b,x)!=b) for x in actions)
        cases.append({'name':name,'T':len(b),'C':len(actions),'K':int(K),'TC':len(b)*len(actions),'times_seconds':times,'max_seconds':max(times)})
    geometry={}
    for role in manifest['source_cache_roles']:
        if role=='fit':continue
        q,a,j,b,_=role_values(role);Cs=[];Ks=[];Ts=[]
        for pa,pj,base in zip(a,j,b):
            actions,_=partition(base,propose(base,pa.argmax(1),pj.argmax(1)));Cs.append(len(actions));Ks.append(sum(np.count_nonzero(apply(base,x)!=base) for x in actions));Ts.append(len(base))
        geometry[role]={'sequences':len(q),'T_sum':int(sum(Ts)),'C_sum':int(sum(Cs)),'K_sum':int(sum(Ks)),'TC_sum':int(sum(t*c for t,c in zip(Ts,Cs))),'max_T':max(Ts),'max_C':max(Cs),'max_K_multiset':int(max(Ks)),'nonempty_sequences':sum(c>0 for c in Cs)}
    checkpoints=list((RUN/'engineering').rglob('*.pt'));traces=list((RUN/'engineering').rglob('physical_updates.jsonl'))
    max_row=max(len(line.encode())+1 for p in traces for line in p.read_text().splitlines())
    if torch.cuda.is_initialized():raise ValueError('cost-only CPU initialized CUDA')
    write(PUB/'inference_cost_bounds.json',{'passed':True,'structural_cases':cases,'role_geometry':geometry,'largest_full_checkpoint_bytes':max(p.stat().st_size for p in checkpoints),'maximum_engineering_trace_row_bytes':max_row,'engineering_weight_pins':weightpins,'new_optimizer_updates':0,'new_source_neural_forwards':0,'synthetic_inference_calls':6,'MLP_forward_calls':18,'mutable_probe_models':2,'outer_A61_head_forwards':0,'raw_metadata_labels_transiently_parsed':True,'outer_labels_used_for_cost_training_metrics_or_selection':False,'outer_utility_or_risk_outcomes_evaluated':False,'cost_geometry_only_not_new_scientific_diagnostic_or_efficacy_result':True,'predeclared_structural_T_bound':T,'CUDA_initialized':False,'freeze_git':freeze['git_commit']})
    print({'cost_probe':True,'structural_cases':cases,'old_source_geometry_roles':len(geometry),'optimizer_updates':0},flush=True)
if __name__=='__main__':main()
