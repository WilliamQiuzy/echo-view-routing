#!/usr/bin/env python3
"""Independent CPU-only matched-kind cost timing and future trace archive IO."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['PYTHONDONTWRITEBYTECODE']='1'
import sys,time,resource,math,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch
from scripts.selected_risk_shared_contracts import *
from echo_routing.temporal.selected_risk_budget import totals,forecast
from echo_routing.temporal.safety_training import Utility,Risk,optimizer,step
from echo_routing.temporal.selected_risk_storage_engineering import archive_trace,trace_bytes
from scripts.probe_selected_risk import storage_bytes


def guard(initial):
    a=resource.getrusage(resource.RUSAGE_SELF);b=resource.getrusage(resource.RUSAGE_CHILDREN)
    if a.ru_utime+a.ru_stime+b.ru_utime+b.ru_stime>=280 or storage_bytes([RUN,PUB])>initial+64000000:raise ValueError('shape probe280CPU/64MB budget reached; keep actual prefix')


def shapes(profile,manifest):
    maxima=profile['old_FIT_maxima'];T=max(manifest['roles'][r]['max_length'] for r in ['F0_mixed','F1_mixed'])
    v={'native_C':maxima['native']['C'],'mixed_C':profile['max_training_C'],'native_K':maxima['native']['K'],'mixed_K':T,'native_axis_T_retained':maxima['native']['T']}
    if any(isinstance(x,bool) or not isinstance(x,int) or x<1 for x in v.values()) or v['native_K']>v['native_axis_T_retained']:raise ValueError('canonical positive integer observable shape bounds required')
    return v


def synthetic_training_data(bounds):
    from echo_routing.temporal.safety_training import draw_plan
    from echo_routing.temporal.safety_data import tensor_data
    rng=np.random.default_rng(630021);groups=[];kinds={'native':[],'constructed':[]};kg={'native':[],'constructed':[]};lengths=[];target=[]
    for qi in range(64):
        native=qi<32;kind='native' if native else 'constructed';C=bounds['native_C' if native else 'mixed_C'];K=bounds['native_K' if native else 'mixed_K'];first=len(lengths);ix=np.arange(first,first+C,dtype=np.int64);groups.append(ix);kinds[kind].extend(ix.tolist());kg[kind].append(qi);lengths.extend([K]*C);target.extend(np.linspace(-1,1,C))
    C=len(lengths);K=sum(lengths);t=np.asarray(target,np.float64);data={'groups':groups,'kind_indices':{k:np.asarray(v,np.int64) for k,v in kinds.items()},'kind_groups':{k:np.asarray(v,np.int64) for k,v in kg.items()},'features':rng.normal(size=(C,109)),'position_features':rng.normal(size=(K,47)),'position_labels':np.arange(K,dtype=np.int64)%3,'position_offsets':np.r_[0,np.cumsum(lengths)].astype(np.int64),'targets':np.column_stack([t,t]),'effects':[{'dFP':int(i%2)} for i in range(C)],'normalizers':{n:{'mean':np.zeros(d),'std':np.ones(d)} for n,d in [('candidate',109),('position',47)]},'scale':1.}
    plans={role:draw_plan(*(len(data['kind_groups' if role=='utility' else 'kind_indices'][k]) for k in ['native','constructed']),2026110461,role) for role in ['utility','risk']}
    return data,tensor_data(data),plans


def time_training(bounds,folder,initial):
    from echo_routing.temporal.safety_data import batch
    t=time.perf_counter();data,tensors,plans=synthetic_training_data(bounds);result={'tensor_preparation_once_seconds':time.perf_counter()-t};scope=[]
    for role,model in [('utility',Utility('S-L',2026110461)),('risk',Risk(2026110461))]:
        opt=optimizer(model);times=[]
        for u in range(5):
            t=time.perf_counter();value,exposure=batch(data,tensors,plans[role],u,role,model.arm);row=step(model,opt,value,u,role);times.append(time.perf_counter()-t);guard(initial)
        result[role+'_update_seconds']=max(times);scope.append({'role':role,'updated_synthetic_models':1,'actual_optimizer_updates':5,'MLP_module_forward_calls':5 if role=='utility' else 10,'lengths_native_mixed':[value['lengths'][0],value['lengths'][-1]],'physical_candidates_per_update':row['physical_candidates'],'full_unscaled_synthetic_targets':True,'actual_qualified_full_batch_gather':True,'cache_generation_is_once_not_per_update':True})
    return result,scope


def trace_profile(bounds,folder,initial):
    rng=np.random.default_rng(630022);identity={k:'a'*64 for k in ['code','config','sources','input','draw_plan','normalizer','target_scales']};rows=[]
    for role in ['utility','risk','event']:
        d=folder/role;d.mkdir();p=d/'physical_updates.jsonl';length=32*(bounds['native_C']+bounds['mixed_C']) if role=='utility' else 64;maxline=0
        with p.open('x') as f:
            for u in range(6000):
                row={'u':u,'identity':identity,'loss':float(rng.random()),'lr':float(rng.random()/1000),'gradient_norm':float(rng.random()),'finite_gradients':True,'nonzero_gradient_tensors':{str(k):bool(k%2) for k in range(8)},'candidate_indices':rng.integers(0,32768,size=length).tolist(),'query_sequence_indices':rng.integers(0,4096,size=64).tolist() if role=='utility' else [],'source_kind_blocks':[0,1],'changed_position_exposures':32*(bounds['native_K']+bounds['mixed_K']),'physical_candidates':length,'optimizer_update_wall_seconds':float(rng.random())}
                if role=='utility':row.update(pointwise_MSE=float(rng.random()),pairwise_logistic=float(rng.random()),physical_groups=64,physical_unequal_pairs=99999)
                elif role=='risk':row.update(position_CE=float(rng.random()),candidate_BCE=float(rng.random()),physical_changed_positions=32*(bounds['native_K']+bounds['mixed_K']))
                else:row.update(candidate_BCE=float(rng.random()),unsafe_targets=32)
                line=json.dumps(row,sort_keys=True)+'\n';maxline=max(maxline,len(line.encode()));f.write(line)
                if u%100==0:f.flush();guard(initial)
        t=time.perf_counter();v=archive_trace(p,identity);trace_bytes(d/v['archive_name'],v);seconds=time.perf_counter()-t;guard(initial)
        rows.append({'role':role,'rows':6000,'uncompressed_bytes':v['uncompressed_bytes'],'compressed_bytes':v['archive_bytes'],'ratio':v['archive_bytes']/v['uncompressed_bytes'],'archive_and_exact_roundtrip_seconds':seconds,'max_line_bytes':maxline,'synthetic_bytes_only':True})
    return rows


def main():
    a=execution_parser(__doc__).parse_args()
    if not authorize(a):return 0
    freeze=verify_shared_freeze('cost_shape_probe');cpu_only();old=read(PUB/'cost_profile.json');m=read(PUB/'input_manifest.json')
    if old.get('passed') is not True or old['input_manifest_sha256']!=sha(PUB/'input_manifest.json'):raise ValueError('same completed old cost profile and frozen metadata required')
    initial=storage_bytes([RUN,PUB]);now=time.time();admit=forecast(totals(actual_receipts()),0,3900,initial+300000000+64000000+128000000,now,7500)
    if (PUB/'cost_shape_stage_admission.json').exists() or (PUB/'cost_shape_profile.json').exists():raise ValueError('one bounded shape attempt, preserve old admission/profile')
    write(PUB/'cost_shape_stage_admission.json',{**admit,'scope':'independent300CPU/64MB cost-only plus original closing; NOT source admission'})
    if not admit['all_caps_fit']:return 1
    folder=RUN/'cost_shape_probe_v1';folder.mkdir(exist_ok=False);bounds=shapes(old,m);guard(initial);training,scope=time_training(bounds,folder,initial);traces=trace_profile(bounds,folder,initial)
    write(PUB/'cost_shape_profile.json',{'passed':True,'input_manifest_sha256':sha(PUB/'input_manifest.json'),'old_cost_profile_sha256':sha(PUB/'cost_profile.json'),'failed_source_admission_v1_sha256':sha(PUB/'sources_admission_v1.json'),'shape_bounds':bounds,'training':training,'synthetic_training_scope':scope,'synthetic_optimizer_updates':10,'synthetic_MLP_module_forward_calls':15,'teacher_forwards':0,'production_C0_C1_instances':0,'trace_profiles':traces,'trace_ratio_forecast':max(.5,max(r['ratio'] for r in traces)),'new_Gt_or_head_outcomes_used':False,'freeze_git':freeze['git_commit'],'all_original_backend_T_C_C2_TC_K_KT_CR_terms_unchanged':True})
    if torch.cuda.is_initialized():raise ValueError('unexpected CUDA')
    print({'shape_cost_complete':True,'engineering_SGD':10,'MLP_forwards':15,'teacher':0},flush=True);return 0
if __name__=='__main__':raise SystemExit(main())
