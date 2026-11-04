#!/usr/bin/env python3
"""A62 cost-only synthetic graph or exactly72 new-FIT engineering updates."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import sys,time,copy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch
from scripts.event_risk_common import *
from echo_routing.temporal.event_risk_model import EventRisk,optimizer,step,checkpoint,restore,atomic_save
from echo_routing.temporal.event_risk_data import candidate_indices
from echo_routing.temporal.event_risk_features import context_features,event_matrix
from echo_routing.temporal.event_risk_audit import tensor_equal
from echo_routing.temporal.proposal_actions import propose,partition


def identity(inputs,seed,freeze):return {'code':digest(freeze['objects']),'config':sha(CFG),'sources':digest(read(CFG)['source']),'input':inputs['dataset']['sha256'],'draw_plan':inputs['plans'][str(seed)]['sha256'],'normalizer':inputs['normalizers']['sha256'],'target_scales':digest({'target':'unsafe_if_H_positive_or_dFP16_positive','unscaled_binary':True})}

def tensor_data(data):return {'x':{a:torch.from_numpy(v) for a,v in data['features'].items()},'y':torch.from_numpy(data['unsafe']),'kind_indices':data['kind_indices'],'normalizers':data['normalizers']}

def cost():
    freeze=verify_freeze('cost_freeze.json');inputs=read(PUB/'input_manifest.json');torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    T=max(r['max_length'] for r in inputs['roles'].values());C=2*T;h=EventRisk('R1',2026110461);g=torch.Generator().manual_seed(62);x=torch.randn(C,205,generator=g);times=[]
    with torch.no_grad():
        for _ in range(6):t=time.perf_counter();h(x);times.append(time.perf_counter()-t)
    b=np.zeros(T,np.int64);a=b.copy();a[1::2]=1;j=b.copy();j[::3]=2;pa=np.eye(5)[a]*.9+.02;pj=np.eye(5)[j]*.9+.02;actions,_=partition(b,propose(b,a,j));dur=[]
    for _ in range(2):t=time.perf_counter();[context_features(pa,pj,b,z) for z in actions];dur.append(time.perf_counter()-t)
    if torch.cuda.is_initialized():raise ValueError('synthetic CPU cost initialized CUDA')
    write(PUB/'synthetic_cost.json',{'passed':True,'T_upper':T,'candidate_upper':C,'physical_synthetic_MLP_forward_calls':6,'new_optimizer_updates':0,'teacher_forwards':0,'actual_outer_head_outputs_read':False,'synthetic_context_actions':len(actions),'max_forward_seconds':max(times),'max_context_seconds':max(dur),'forward_seconds_per_candidate':max(times)/C,'context_seconds_per_candidate':max(dur)/max(1,len(actions)),'synthetic_truth_labels_used':False,'real_metadata_refs_lengths_only_labels_transient_parse_not_used':True,'freeze_git':freeze['git_commit']});print({'cost_only':True,'synthetic_NN_forwards':6,'SGD':0,'T':T,'C':C},flush=True)


def fullfit():
    freeze=verify_freeze('FIT_check_freeze.json');inputs=read(PUB/'training_inputs.json');data=np.load(verify_pin(inputs['dataset']),allow_pickle=True).item();td=tensor_data(data);torch.set_num_threads(1);torch.use_deterministic_algorithms(True);records=[];updates=0;max_io=0;folder=RUN/'checks';folder.mkdir(exist_ok=False)
    for seed in [2026110461,2026110462]:
        plan=np.load(verify_pin(inputs['plans'][str(seed)]),allow_pickle=True).item();ident=identity(inputs,seed,freeze)
        for arm in ['R1','R2']:
            def run(model,opt,start,end):
                nonlocal updates
                rows=[]
                for u in range(start,end):
                    idx=candidate_indices(plan,u,data['kind_indices']['native'],data['kind_indices']['constructed']);t=time.perf_counter();r=step(model,opt,{'x':td['x'][arm][idx],'y':td['y'][idx]},u);r['complete_update_seconds']=time.perf_counter()-t;rows.append(r);updates+=1
                return rows
            model=EventRisk(arm,seed);initial=copy.deepcopy(model.state_dict());opt=optimizer(model);one=run(model,opt,0,6);expected=checkpoint(model,opt,6,ident,data['normalizers']);repeat=EventRisk(arm,seed);ro=optimizer(repeat);run(repeat,ro,0,6)
            if not tensor_equal(expected,checkpoint(repeat,ro,6,ident,data['normalizers'])):raise ValueError('actualFIT repeat6 fullstate differs')
            first=EventRisk(arm,seed);fo=optimizer(first);run(first,fo,0,3);p=folder/f'{seed}_{arm}_resume.pt';t=time.perf_counter();atomic_save(p,checkpoint(first,fo,3,ident,data['normalizers']));saved=torch.load(p,weights_only=True,map_location='cpu');max_io=max(max_io,time.perf_counter()-t);resumed=EventRisk(arm,seed);rr=optimizer(resumed)
            if restore(resumed,rr,saved,ident,data['normalizers'])!=3:raise ValueError('actual diskresume cursor changed')
            run(resumed,rr,3,6)
            if not tensor_equal(expected,checkpoint(resumed,rr,6,ident,data['normalizers'])):raise ValueError('actualFIT disk3+3 fullstate differs')
            live={n:any(r['nonzero_gradient_tensors'][n] for r in one) for n,_ in model.named_parameters()}
            if not all(live.values()) or any(not bool(opt.state[p]['exp_avg_sq'].count_nonzero()) for p in model.parameters()) or any(torch.equal(initial[k],v) for k,v in model.state_dict().items()):raise ValueError('active tensor never learned; decay does not substitute gradient/Adam liveness')
            records.append({'seed':seed,'arm':arm,'engineering_updates':18,'repeat6_fullstate_exact':True,'disk3plus3_fullstate_exact':True,'gradient_history':live,'max_complete_update_seconds':max(r['complete_update_seconds'] for r in one),'checkpoint_bytes':p.stat().st_size,'first_step_hidden_zero_is_expected':not one[0]['nonzero_gradient_tensors']['network.0.weight']})
    if updates!=72 or len(records)!=4 or torch.cuda.is_initialized():raise ValueError('exact72 newFIT engineering-only CPU updates required')
    write(PUB/'FIT_checks.json',{'passed':True,'records':records,'engineering_optimizer_updates':72,'mutable_check_models':16,'formal_optimizer_updates':0,'max_complete_update_seconds':max(r['max_complete_update_seconds'] for r in records),'max_disk_save_load_seconds':max_io,'normalizers_hash':inputs['normalizers']['sha256'],'dataset_hash':inputs['dataset']['sha256'],'no_engineering_state_warmstart_formal':True,'CUDA_initialized':False,'freeze_git':freeze['git_commit']});print({'actualFIT_engineering_updates':72,'formal':0},flush=True)


def main():
    p=execution_parser(__doc__);p.add_argument('--stage',choices=['synthetic-cost','full-FIT-new-graph'],default='synthetic-cost');args=p.parse_args()
    if not authorize(args):return 0
    (cost if args.stage=='synthetic-cost' else fullfit)();return 0
if __name__=='__main__':raise SystemExit(main())
