#!/usr/bin/env python3
"""A61 actual complete FIT graph repeat/resume, liveness, I/O and worst-size cost."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import sys,time,json,copy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from scripts.safety_common import *
from echo_routing.temporal.safety_data import validate,tensor_data,batch
from echo_routing.temporal.safety_training import Utility,Risk,optimizer,checkpoint,restore,atomic_save,validate_optimizer
from echo_routing.temporal.safety_loop import train_one
from echo_routing.temporal.safety_runtime import infer


def identity_for(role,seed,inputs,freeze):
    return {'code':digest(freeze['objects']),'config':sha(CFG),'sources':digest(inputs['source_cache_identities']),'input':inputs['dataset']['sha256'],'draw_plan':inputs['plans'][str(seed)]['sha256'],'normalizer':inputs['normalizers']['sha256'],'target_scales':digest({'scale':inputs['utility_scale']})}


def main():
    freeze=verify_freeze('fullFIT_check_freeze.json');inputs=read(PUB/'training_inputs.json')
    if (PUB/'FIT_checks.json').exists():raise ValueError('engineering check already frozen; do not repeat')
    data=np.load(verify_pin(inputs['dataset']),allow_pickle=True).item();frozen_plans={k:np.load(verify_pin(v),allow_pickle=True).item() for k,v in inputs['plans'].items()}
    q,a,j,b,_=role_values('fit');recount=validate(data,q,a,j,b,frozen_plans)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    seed=2026110461;identity=identity_for('utility',seed,inputs,freeze);summaries=[];updates=0;mutable=0;final={}
    for arm in ['risk','U-P','U-L','S-P','S-L']:
        role='risk' if arm=='risk' else 'utility';plan=frozen_plans[str(seed)][role];startpoint=Risk(seed) if role=='risk' else Utility(arm,seed);initial={k:v.clone() for k,v in startpoint.state_dict().items()}
        def make():return Risk(seed) if role=='risk' else Utility(arm,seed)
        expected=None
        for name,stop in [('full',6),('repeat',6),('partial',3)]:
            model=make();opt=optimizer(model);mutable+=1
            row=train_one(model,opt,data,plan,identity,RUN/'engineering'/arm/name,stop=stop,latest_every=1,milestones=())
            updates+=stop
            if name=='full':expected=checkpoint(model,opt,6,identity,data['normalizers'],data['scale']);liveness=row['all_tensor_gradient_liveness_over_history']
            if name=='repeat' and any(not torch.equal(v,model.state_dict()[k]) for k,v in expected['model'].items()):raise ValueError('actual FIT exact repeat failed')
        resumed=make();opt=optimizer(resumed);mutable+=1
        value=torch.load(RUN/'engineering'/arm/'partial/latest.pt',weights_only=True)
        if restore(resumed,opt,value,identity,data['normalizers'],data['scale'])!=3:raise ValueError('disk restore cursor changed')
        row=train_one(resumed,opt,data,plan,identity,RUN/'engineering'/arm/'partial',stop=6,latest_every=1,milestones=());updates+=3
        if any(not torch.equal(v,resumed.state_dict()[k]) for k,v in expected['model'].items()):raise ValueError('actual FIT disk3+3 differs from uninterrupted6')
        if not all(liveness.values()):raise ValueError('actual active gradient tensor never live over six-step history')
        if any(torch.equal(initial[k],v) for k,v in resumed.state_dict().items()):raise ValueError('active tensor never changed')
        if any(not bool(opt.state[p]['exp_avg_sq'].count_nonzero()) for p in resumed.parameters()):raise ValueError('active tensor has no nonzero Adam second moment')
        final[arm]=resumed;summaries.append({'arm':arm,'exact_repeat6':True,'exact_disk_resume3plus3':True,'gradient_liveness':liveness,'all_selected_tensors_drift_and_nonzero_Adam_second_moment':True,'parameters':sum(p.numel() for p in resumed.parameters())})
    timings={}
    for role,arm in [('utility','S-L'),('risk','risk')]:
        worst=copy.deepcopy(frozen_plans[str(seed)][role]);pools=data['kind_groups' if role=='utility' else 'kind_indices']
        maxima=[]
        for block,k in enumerate(worst['source_kinds']):
            kind=['native','constructed'][k];pool=pools[kind]
            weights=[len(data['groups'][i])**2 for i in pool] if role=='utility' else np.diff(data['position_offsets'])[pool]
            local=int(np.argmax(weights));worst['indices'][:20,block*32:(block+1)*32]=local;maxima.append(int(max(weights)))
        model=Utility(arm,seed) if role=='utility' else Risk(seed);opt=optimizer(model);mutable+=1
        began=time.perf_counter();report=train_one(model,opt,data,worst,identity,RUN/'engineering'/('timed_'+role),stop=20,latest_every=10,milestones=(10,20));elapsed=time.perf_counter()-began;updates+=20
        timings[role]={'twenty_updates_full_loop_IO_seconds':elapsed,'seconds_per_update_including_plan_batch_forward_backward_trace_and_full_checkpoint_IO':elapsed/20,'engineering_worst_block_C2_or_K':maxima,'report':report}
    # Probe all actual distinct worst T,C²,K-multiset and T*C source groups, not arbitrary32.
    c=np.asarray([len(g) for g in data['groups']]);K=np.diff(data['position_offsets']);tests=set()
    criteria={'max_T':np.asarray([len(x['labels']) for x in q]),'max_C2':c*c,'max_T_times_C':np.asarray([len(x['labels']) for x in q])*c,'max_changed_position_multiset':np.asarray([K[g].sum() for g in data['groups']])}
    for criterion,values in criteria.items():tests.add(int(np.argmax(values)))
    inference=[]
    for i in sorted(tests):
        start=time.perf_counter();result=infer(final['S-L'],final['risk'],data['normalizers'],data['scale'],a[i],j[i],b[i]);duration=time.perf_counter()-start
        inference.append({'criteria':[k for k,v in criteria.items() if i==int(np.argmax(v))],'T':len(b[i]),'C':len(result['actions']),'C2':len(result['actions'])**2,'K_multiset':int(K[data['groups'][i]].sum()),'feature_two_heads_policy_seconds':duration})
    if torch.cuda.is_initialized():raise ValueError('CPU checker initialized CUDA')
    write(PUB/'FIT_checks.json',{'passed':True,'full_FIT_recount':recount,'models':summaries,'actual_engineering_optimizer_updates':updates,'mutable_check_models':mutable,'timings':timings,'worst_actual_FIT_inference':inference,'fullFIT_worst_criteria_covered':list(criteria),'formal_updates':0,'CUDA_initialized':False,'new_source_neural_forwards':0,'freeze_git':freeze['git_commit']})
    print({'passed':True,'engineering_updates':updates,'mutable_models':mutable,'timings':{k:v['seconds_per_update_including_plan_batch_forward_backward_trace_and_full_checkpoint_IO'] for k,v in timings.items()}},flush=True)
if __name__=='__main__':main()
