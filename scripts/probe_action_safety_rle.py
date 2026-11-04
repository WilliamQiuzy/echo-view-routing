#!/usr/bin/env python3
"""Cost-only low/high RLE separation and complete frozen observable geometry."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import sys,time,copy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from scripts.safety_common import *
from scripts.precache_action_safety_geometry import probability_reader,verify_source_manifest
from echo_routing.temporal.safety_rle_cost import geometry,pair_kernel
from echo_routing.temporal.safety_runtime import policies
from echo_routing.temporal.safety_policy_cache import SafetyCache
from echo_routing.temporal.safety_metrics import action_summary
from echo_routing.temporal.safety_oracle import options,coverage,solve
from echo_routing.temporal.proposal_actions import propose,partition,apply
from echo_routing.temporal.safety_precache import ROLES
from echo_routing.temporal.baselines.registry import decode_with_prob


def main():
    freeze=verify_freeze('rle_cost_freeze.json');verify_source_manifest();folder=RUN/'rle_cost_refinement';folder.mkdir(exist_ok=False);T=701;b=np.zeros(T,np.int64);a=b.copy();a[1::2]=1;j=2+np.arange(T)%2;all_actions=propose(b,a,j);acts,rejected=partition(b,all_actions);K=np.array([np.count_nonzero(apply(b,x)!=b) for x in acts]);scores=np.column_stack([np.sin(np.arange(len(acts)))+.1,.1*K,.2*K,.7*K,np.full(len(acts),.2),np.full(len(acts),.2)]);v={**policies(b,acts,scores,.8),'actions':acts,'all_actions':all_actions,'G_rejected':rejected,'score_matrix':scores,'tau':.8,'cached_feature_heads_policy_seconds':0.};cuts=np.linspace(0,T,9,dtype=int);truth=np.concatenate([np.full(z-x,(k%2)*4) for k,(x,z) in enumerate(zip(cuts,cuts[1:]))]).astype(np.int64);q={'id':'synthetic_cost_low_RLE','kind':'multi','labels':truth};descriptor=[]
    for arm in ['U-L','S-L']:
        for policy in ['E','N','C']:
            start=time.perf_counter();action_summary([q],[b],[v],policy,arm,.8917446116307443);descriptor.append(time.perf_counter()-start)
    cache=SafetyCache(folder/'synthetic_low_cache','synthetic_low_RLE',[b],[a],[j],.8);start=time.perf_counter();cache.run_one(0,lambda:copy.deepcopy(v));write_seconds=time.perf_counter()-start;start=time.perf_counter();cache.load(0);read_seconds=time.perf_counter()-start
    oracle=[];truth=np.concatenate([np.full(z-x,3+k%2) for k,(x,z) in enumerate(zip(cuts,cuts[1:]))]).astype(np.int64)
    for domain in ['old','expanded']:
        for safety in ['unrestricted','net-safe','gross-safe']:
            start=time.perf_counter();pool=options(b,a,j,truth,domain,safety);coverage(b,a,j,truth,domain,safety);solve([pool],domain,safety);oracle.append(time.perf_counter()-start)
    low={**geometry(b,a,j),'descriptor_max_seconds':max(descriptor),'oracle_max_seconds':max(oracle),'cache_write_seconds':write_seconds,'cache_read_seconds':read_seconds};high_b=np.arange(T,dtype=np.int64)%2;high_a=high_b.copy();high_a[1::2]=2;high_j=3+np.arange(T)%2;high=geometry(high_b,high_a,high_j);old=read(PUB/'component_cost_measurements.json');high.update(old['timings']['dense_701']);kernels=[pair_kernel(len(acts)) for _ in range(3)];observable={};cfg=read(CFG);manifest=read(PUB/'input_manifest.json')
    for role in [r for r in manifest['roles'] if r not in ['fit','inner_cal']]:
        if role in ROLES:
            qs=sequences(role);ra=probability_reader('a08',role);rj=probability_reader('jepa',role);pa=[ra.load(i) for i in range(len(qs))];pj=[rj.load(i) for i in range(len(qs))];base=[decode_with_prob('b3',x,{'decode':{'b3':cfg['source']['a08']['original_B3']}})[0] for x in pa]
        else:qs,pa,pj,base,_=role_values(role)
        total={k:0 for k in ['N','T','C','C2','TC','all_C','all_TC','K','CR','native_N','native_T','native_T2']};max_C=max_T=0
        for q,x,y,z in zip(qs,pa,pj,base):
            g=geometry(z,x.argmax(1),y.argmax(1));max_C=max(max_C,g['max_C']);max_T=max(max_T,g['max_T'])
            for k in total:
                if k in g:total[k]+=g[k]
            if q['kind']=='native':total['native_N']+=1;total['native_T']+=len(z);total['native_T2']+=len(z)**2
        observable[role]={**total,'max_C':max_C,'max_T':max_T,'input_sha256':manifest['roles'][role]['sha256']}
    if torch.cuda.is_initialized():raise ValueError('RLE cost-only CPU initialized CUDA')
    write(PUB/'rle_cost_measurements.json',{'passed':True,'low_RLE_dense':low,'retained_high_RLE_dense':high,'all_unequal_pair_kernel':max(kernels,key=lambda k:k['seconds']),'observable_geometry_all10roles':observable,'producer_precache_manifest_sha256':sha(PUB/'producer_precache_manifest.json'),'producer_freeze_sha256':sha(RUN/'producer_precache_freeze.json'),'component_measurements_sha256':sha(PUB/'component_cost_measurements.json'),'new_optimizer_updates':0,'new_neural_forwards_in_this_cost_stage':0,'A61_outer_head_outcomes_used':False,'outer_GT_values_used':False,'label_arrays_transiently_parsed_but_values_not_used_for_geometry':True,'all240_real_teacher_streams_already_billed_as_production_not_zero_source_claim':True,'source_Git_commit':freeze['git_commit'],'epoch':time.time()});print({'RLE_cost_and_observable_geometry_complete':True,'action_head_forwards':0,'new_source_forwards_in_this_CPU_stage':0},flush=True)
if __name__=='__main__':main()
