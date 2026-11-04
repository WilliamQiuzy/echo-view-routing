#!/usr/bin/env python3
"""Cost only: fixed teacher geometry, generated labels/scores, no NN/outcomes."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import sys,time,copy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from scripts.safety_common import *
from scripts.precache_action_safety_geometry import probability_reader,verify_source_manifest
from echo_routing.temporal.safety_precache import ROLES
from echo_routing.temporal.safety_runtime import policies
from echo_routing.temporal.safety_policy_cache import SafetyCache
from echo_routing.temporal.safety_metrics import action_summary
from echo_routing.temporal.safety_oracle import options,coverage,solve
from echo_routing.temporal.proposal_actions import propose,partition,apply
from echo_routing.temporal.baselines.registry import decode_with_prob


def synthetic(T,index):
    edges=np.linspace(0,T,min(8,T)+1,dtype=int)
    return np.concatenate([np.full(z-x,(i+index)%5) for i,(x,z) in enumerate(zip(edges,edges[1:]))]).astype(np.int64)


def operating_point(training_inputs):
    value=training_inputs['F_fit']
    if not isinstance(value,(int,float)) or value!=.8917446116307443:raise ValueError('qualified exact numeric FIT operating point required')
    return value


def truth_run_bound(q):
    refs=q['refs']
    if not refs:raise ValueError('complete source reference axis required')
    # Qualified native clips have one fixed family label. A reference can change
    # only when the underlying source clip changes; gamma preserves its class.
    count=1+sum(a[0]!=b[0] for a,b in zip(refs,refs[1:]))
    cap=1 if q['kind']=='native' else 8 if q['kind']=='multi' else 2
    if count>cap:raise ValueError('frozen constructor clip-piece ceiling violated')
    return count


def timed_role(role,qs,pa,pj,base,folder,F_fit):
    # Reference arrays are deliberately regenerated; only lengths/kinds are read.
    bounds=[truth_run_bound(q) for q in qs]
    generated=[{'id':f'cost_{i}','kind':'multi','labels':synthetic(len(b),i)} for i,b in enumerate(base)]
    raws_a=[p.argmax(1) for p in pa];raws_j=[p.argmax(1) for p in pj];values=[]
    start=time.perf_counter()
    for b,a,j in zip(base,raws_a,raws_j):
        all_actions=propose(b,a,j);actions,rejected=partition(b,all_actions);K=np.array([np.count_nonzero(apply(b,x)!=b) for x in actions]);C=len(actions)
        scores=np.column_stack([np.linspace(.1,1,C),np.zeros(C),K,K*.0,np.full(C,.1),np.full(C,.1)]) if C else np.empty((0,6))
        values.append({**policies(b,actions,scores,.9),'actions':actions,'all_actions':all_actions,'G_rejected':rejected,'score_matrix':scores,'tau':.9,'cached_feature_heads_policy_seconds':0.})
    geometry_policy_seconds=time.perf_counter()-start;descriptor=[]
    for arm in ['U-L','S-L']:
        for policy in ['E','N','C']:
            start=time.perf_counter();action_summary(generated,base,values,policy,arm,F_fit);descriptor.append(time.perf_counter()-start)
    cache=SafetyCache(folder/role,'synthetic_role_'+role,base,raws_a,raws_j,.9);start=time.perf_counter()
    for i,v in enumerate(values):cache.run_one(i,lambda v=v:copy.deepcopy(v))
    write_seconds=time.perf_counter()-start;start=time.perf_counter()
    for i in range(len(values)):cache.load(i)
    read_seconds=time.perf_counter()-start;oracle=[]
    for domain in ['old','expanded']:
        for safety in ['unrestricted','net-safe','gross-safe']:
            start=time.perf_counter();pools=[]
            for b,a,j,q in zip(base,raws_a,raws_j,generated):
                pools.append(options(b,a,j,q['labels'],domain,safety));coverage(b,a,j,q['labels'],domain,safety)
            solve(pools,domain,safety);oracle.append(time.perf_counter()-start)
    return {'N':len(base),'T':sum(map(len,base)),'descriptor_max_seconds':max(descriptor),'all6_descriptor_seconds':descriptor,'geometry_and_dummy_policy_seconds':geometry_policy_seconds,'cache_write_seconds':write_seconds,'cache_read_seconds':read_seconds,'oracle_max_seconds':max(oracle),'all6_oracle_seconds':oracle,'structural_true_run_ceiling_from_reference_source_clip_changes':max(bounds),'structural_true_run_bound_histogram':{str(k):bounds.count(k) for k in sorted(set(bounds))},'native_one_constant_family_per_qualified_clip_construction_invariant':True,'actual_GT_values_not_read_for_this_bound':True,'synthetic_truth_max_runs':8,'synthetic_score_matrix_no_heads':True}


def main():
    freeze=verify_freeze('role_cost_freeze.json');verify_source_manifest();folder=RUN/'role_cost_refinement';folder.mkdir(exist_ok=False);inputs=read(PUB/'input_manifest.json');cfg=read(CFG);result={}
    for role in [r for r in inputs['roles'] if r!='fit']:
        if role in ROLES:
            qs=sequences(role);ra=probability_reader('a08',role);rj=probability_reader('jepa',role);pa=[ra.load(i) for i in range(len(qs))];pj=[rj.load(i) for i in range(len(qs))];base=[decode_with_prob('b3',p,{'decode':{'b3':cfg['source']['a08']['original_B3']}})[0] for p in pa]
        else:qs,pa,pj,base,_=role_values(role)
        result[role]=timed_role(role,qs,pa,pj,base,folder,operating_point(read(PUB/'training_inputs.json')));result[role]['input_sha256']=inputs['roles'][role]['sha256']
        print({'cost_role':role,'sequences':len(qs),'complete':True},flush=True)
    if torch.cuda.is_initialized():raise ValueError('cost-only CPU initialized CUDA')
    write(PUB/'role_cost_measurements.json',{'passed':True,'roles':result,'new_optimizer_updates':0,'new_NN_forwards':0,'A61_outer_head_outcomes_used':False,'actual_GT_values_used':False,'synthetic_max8_run_reference_arrays_replace_original_labels':True,'original_metadata_label_arrays_transiently_parsed_not_used':True,'RLE_geometry_sha256':sha(PUB/'rle_cost_measurements.json'),'producer_manifest_sha256':sha(PUB/'producer_precache_manifest.json'),'source_Git_commit':freeze['git_commit'],'epoch':time.time()})
if __name__=='__main__':main()
