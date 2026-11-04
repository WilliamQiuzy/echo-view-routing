#!/usr/bin/env python3
"""Actual complete FIT data recount and three-output64-candidate CPU graph/IO."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import sys,time,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from scripts.proposal_common import *
from scripts.train_jepa_proposals import role_values,identity_for
from echo_routing.temporal.proposal_data import validate
from echo_routing.temporal.proposal_training import ProposalRegressor,optimizer
from echo_routing.temporal.proposal_loop import train_one
from echo_routing.temporal.proposal_features import normalize,physical_outputs
from echo_routing.temporal.proposal_runtime import infer


def equal(a,b):
    if torch.is_tensor(a):return torch.equal(a,b)
    if isinstance(a,dict):return set(a)==set(b) and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b


def main():
    freeze=verify_freeze('fullFIT_check_freeze.json');inputs=read(PUB/'training_inputs.json');data=np.load(verify_pin(inputs['dataset']),allow_pickle=True).item()
    qs,a,j,b,_=role_values('fit');norms={arm:read(verify_pin(pin)) for arm,pin in inputs['normalizers'].items()};scales=np.asarray(read(verify_pin(inputs['target_scales']))['scales'])
    plans={seed:np.load(verify_pin(pin),allow_pickle=True).item() for seed,pin in inputs['plans'].items()}
    write(RUN/'runtimecoordination.json',{'phase':'actual_wholeFIT_64candidate_CPU_graph_checks','model_pid':os.getpid(),'formal_optimizer_updates':0,'epoch':time.time(),'next_action':'actual graph/repeat/resume/timing then full matrix measuredadmission'})
    start=time.perf_counter();arithmetic=validate(data,qs,a,j,b,norms,scales,plans);arithmetic_wall=time.perf_counter()-start
    seed=2026110360;plan=plans[str(seed)];heads={};engineering_updates=0;mutable_models=0
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    for arm in ['A','J']:
        identity=identity_for(arm,seed,inputs,freeze);norm=norms[arm];folder=RUN/'engineering/fullFIT'/arm
        def run(name,n,existing=None):
            nonlocal engineering_updates,mutable_models
            model=ProposalRegressor(arm,seed);opt=optimizer(model);mutable_models+=1
            tick=time.perf_counter();result=train_one(model,opt,data,plan,norm,scales,identity,existing or folder/name,stop=n,latest_every=1,milestones=())
            seconds=time.perf_counter()-tick;engineering_updates+=result['new_updates_this_process']
            return model,opt,result,seconds
        indices=np.r_[*(data['kind_indices'][k][plan[k][0]] for k in ['native','constructed'])]
        initial=ProposalRegressor(arm,seed)
        with torch.no_grad():output=initial(torch.from_numpy(normalize(data['features'][arm][indices],norm))).numpy()
        if output.shape!=(len(indices),3) or np.any(physical_outputs(output,scales)):raise ValueError('zero-initial signed physical outputs changed')
        # Initial module above is a read-only forward-only checker, never optimizer-mutated.
        before=infer(initial,norm,scales,a[0],j[0],b[0],arm)
        if any(not np.array_equal(before[p]['prediction'],b[0]) for p in ['O','E','R']):raise ValueError('initial zero must be no-op')
        x,xo,xr,_=run('straight6',6);y,yo,yr,_=run('repeat6',6)
        if not equal(x.state_dict(),y.state_dict()) or not equal(xo.state_dict(),yo.state_dict()):raise ValueError('actual FIT deterministic repeat failed')
        _,_,_,_=run('resume',3)
        z,zo,zr,_=run('resume_newprocess_model',6,folder/'resume')
        if not equal(x.state_dict(),z.state_dict()) or not equal(xo.state_dict(),zo.state_dict()):raise ValueError('actual FIT disk-resume failed')
        timed,to,tr,seconds=run('timed20',20)
        trace=[json.loads(line) for line in (folder/'timed20/physical_updates.jsonl').read_text().splitlines()]
        live={name:any(row['nonzero_gradient_tensors'][name] for row in trace) for name,_ in timed.named_parameters()}
        if not all(live.values()) or any(p.grad is None or not torch.isfinite(p.grad).all() or p.is_cuda for p in timed.parameters()):raise ValueError('actual cached FIT graph lacks intended finite per-tensor gradient history')
        if any(not bool(v['exp_avg_sq'].count_nonzero()) for v in to.state.values()):raise ValueError('actual FIT second moment liveness failed')
        init=ProposalRegressor(arm,seed).state_dict()
        if any(torch.equal(v,timed.state_dict()[k]) for k,v in init.items()):raise ValueError('actual FIT tensor drift missing')
        selected=np.linspace(0,len(qs)-1,32,dtype=int);times=[];candidates=[]
        for index in selected:
            out=infer(timed,norm,scales,a[index],j[index],b[index],arm);times.append(out['cached_feature_head_policy_seconds']);candidates.append(len(out['actions']))
        heads[arm]={'parameters':sum(p.numel() for p in timed.parameters()),'actual_batch_candidates':len(indices),'zero_initial_physical_no_op':True,'repeat6_and_disk3plus3_exact':True,'optimizer_all_steps':20,'all_tensor_history_liveness':live,'first_update_hidden_zero_is_expected':not trace[0]['nonzero_gradient_tensors']['network.0.weight'],'actual_FIT_nonconstant_feature_columns':int(np.count_nonzero(data['features'][arm].std(0)>0)),'actual_targets_minmax':inputs['targets_minmax'],'full20_durable_updates_wall_seconds':seconds,'conservative_full_update_timing_seconds':max(2*seconds/20,2*max(r['optimizer_update_wall_seconds'] for r in trace)), 'cached_feature_head_policy_max_seconds_per_sequence':max(times),'sampled_sequences':len(times),'sampled_max_candidates':max(candidates),'formal_optimizer_updates':0}
    if torch.cuda.is_initialized():raise ValueError('CPU graph initialized CUDA context')
    write(PUB/'wholeFIT_checks.json',{'passed':True,'arithmetic':arithmetic,'complete_arithmetic_wall_seconds':arithmetic_wall,'heads':heads,'engineering_optimizer_updates':engineering_updates,'engineering_mutable_models':mutable_models,'forward_only_initial_check_models':2,'initial_state_reference_models':2,'formal_optimizer_updates':0,'CUDA_initialized':False,'CPU_tensors_only':True,'training_inputs_sha256':sha(PUB/'training_inputs.json'),'check_freeze_git':freeze['git_commit'],'actual_FIT_graph_not_synthetic_substitute':True,'epoch':time.time()})
    print({'wholeFIT_passed':True,'engineering_updates':engineering_updates,'mutable_models':mutable_models,'heads':heads},flush=True)

if __name__=='__main__':main()
