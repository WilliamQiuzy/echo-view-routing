#!/usr/bin/env python3
"""Timing-only synthetic/FIT component probe; no NN/SGD/outer head outcome."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import sys,time,json,copy,math
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from scripts.safety_common import *
from echo_routing.temporal.dense_runtime import metric_stream
from echo_routing.temporal.safety_metrics import evaluate,prepare_system,paired,action_summary
from echo_routing.temporal.safety_runtime import policies
from echo_routing.temporal.safety_policy_cache import SafetyCache
from echo_routing.temporal.safety_oracle import options,coverage,solve
from echo_routing.temporal.anchor_oracle import fractional_oracle
from echo_routing.temporal.proposal_actions import propose,partition,apply
from echo_routing.temporal.safety_audit import validate_trace
from echo_routing.temporal.safety_training import lr_factor


def timed(fn):
    began=time.perf_counter();value=fn();return time.perf_counter()-began,value


def task(T):
    qs=[];base={};pred={}
    for kind,n in [('native',320),('multi',160),('pair',160)]:
        for i in range(n):
            joins=[] if kind=='native' else np.linspace(0,T,9 if kind=='multi' else 3,dtype=int)[1:-1].tolist();edges=[0,*joins,T];y=np.full(T,i%5,np.int64) if kind=='native' else np.concatenate([np.full(b-a,k%5) for k,(a,b) in enumerate(zip(edges,edges[1:]))]).astype(np.int64);name=f'cost_{kind}_{i:04d}_0';q={'id':name,'kind':kind,'labels':y,'joins':np.asarray(joins),'refs':[('synthetic_only',j,1.) for j in range(T)]};qs.append(q);base[name]=np.arange(T,dtype=np.int64)%2;pred[name]=(np.arange(T,dtype=np.int64)+i)%3+2
    ctx={'sequences':qs,'eval_set':SimpleNamespace(natives=[SimpleNamespace(seq_id=q['id'],labels=q['labels']) for q in qs if q['kind']=='native'],multi=[metric_stream(q) for q in qs if q['kind']=='multi'],pairs=[metric_stream(q) for q in qs if q['kind']=='pair'])}
    return qs,base,pred,ctx


def case(T,long=False):
    b=np.arange(T,dtype=np.int64)%2;a=b.copy();a[1::2]=2;j=3+np.arange(T)%2
    if long:
        b[:T//2]=0;a=b.copy();a[:T//4]=1+np.arange(T//4)%2;j=b.copy()
    all_actions=propose(b,a,j);acts,rejected=partition(b,all_actions);K=np.asarray([np.count_nonzero(apply(b,x)!=b) for x in acts]);scores=np.column_stack([np.sin(np.arange(len(acts)))+.1,.1*K,.2*K,.7*K,np.full(len(acts),.2),np.full(len(acts),.2)]) if acts else np.empty((0,6));value={**policies(b,acts,scores,.8),'actions':acts,'all_actions':all_actions,'G_rejected':rejected,'score_matrix':scores,'tau':.8,'cached_feature_heads_policy_seconds':0.}
    y=np.concatenate([np.full(z-x,k%2*4) for k,(x,z) in enumerate(zip(np.linspace(0,T,9,dtype=int)[:-1],np.linspace(0,T,9,dtype=int)[1:]))]).astype(np.int64);q={'id':'synthetic_cost','kind':'multi','labels':y}
    return q,b,a,j,value,{'T':T,'C':len(acts),'all_C':len(all_actions),'C2':len(acts)**2,'TC':T*len(acts),'all_TC':T*len(all_actions),'K':int(K.sum())}


def trace_rows(data,plan,role,n_updates=6000):
    K=np.diff(data['position_offsets']);pairs=[]
    for g in data['groups']:
        t=np.r_[data['targets'][g,1],0.];pairs.append(int(np.count_nonzero(np.triu(t[:,None]!=t,1))))
    rows=[]
    if not 0<n_updates<=6000:raise ValueError('bounded synthetic metadata trace length required')
    for u in range(n_updates):
        cs=[];qs=[]
        for block,k in enumerate(plan['source_kinds']):
            pool=data['kind_groups' if role=='utility' else 'kind_indices'][['native','constructed'][k]];draw=pool[plan['indices'][u,block*32:(block+1)*32]].tolist()
            if role=='utility':
                qs.extend(draw)
                for q in draw:cs.extend(data['groups'][q].tolist())
            else:cs.extend(draw)
        positions=int(K[cs].sum());row={'u':u,'finite_gradients':True,'loss':1.,'gradient_norm':1.,'lr':.001*lr_factor(u),'candidate_indices':cs,'query_sequence_indices':qs,'source_kind_blocks':plan['source_kinds'],'physical_candidates':len(cs),'changed_position_exposures':positions,'physical_groups':len(qs),'physical_unequal_pairs':sum(pairs[q] for q in qs),'nonzero_gradient_tensors':{'synthetic_metadata_not_actual_gradient':True},'synthetic_cost_trace_no_optimizer_executed':True}
        if role=='risk':row['physical_changed_positions']=positions
        rows.append(row)
    return rows


def main():
    freeze=verify_freeze('component_cost_freeze.json');folder=RUN/'cost_refinement';folder.mkdir(parents=True,exist_ok=False);result={};torch.set_num_threads(1)
    for T in [16,701]:
        qs,b,p,ctx=task(T);times=[];prepared=[]
        for pred in [b,p]:
            elapsed,(metric,tables)=timed(lambda:evaluate(pred,ctx,b));times.append(elapsed);elapsed2,ready=timed(lambda:prepare_system(pred,b,qs,tables));prepared.append(ready);times.append(elapsed2)
        boot,_=timed(lambda:paired(*prepared,5000,2026110461));result['task_'+str(T)]={'T':T,'N':len(qs),'positions':len(qs)*T,'evaluation_prepare_max_seconds':max(times),'paired5000_seconds':boot,'native_sessions':320,'multi':160,'pairs':160}
    for T,long in [(16,False),(701,False),(701,True)]:
        q,b,a,j,v,g=case(T,long);name=('long_' if long else 'dense_')+str(T);ds=[]
        for policy in ['E','N','C']:
            elapsed,_=timed(lambda:action_summary([q],[b],[v],policy,'S-L',.8917446116307443));ds.append(elapsed)
        cache=SafetyCache(folder/name,sha(CFG)+name,[b],[a],[j],.8);write_seconds,_=timed(lambda:cache.run_one(0,lambda:copy.deepcopy(v)));read_seconds,_=timed(lambda:cache.load(0));oracle_times=[]
        # Oracle truth never makes Base correct: all candidate harms/metrics remain synthetic.
        truth=np.concatenate([np.full(z-x,3+k%2) for k,(x,z) in enumerate(zip(np.linspace(0,T,9,dtype=int)[:-1],np.linspace(0,T,9,dtype=int)[1:]))]).astype(np.int64)
        for domain in ['old','expanded']:
            for safety in ['unrestricted','net-safe','gross-safe']:
                def oracle_work():
                    pool=options(b,a,j,truth,domain,safety);coverage(b,a,j,truth,domain,safety);solve([pool],domain,safety)
                elapsed,_=timed(oracle_work);oracle_times.append(elapsed)
        result[name]={**g,'descriptor_max_seconds':max(ds),'cache_write_seconds':write_seconds,'cache_read_seconds':read_seconds,'oracle_max_seconds':max(oracle_times)}
    rng=np.random.default_rng(2026110461);pools=[]
    for k in range(120):
        pool=[{'M':0,'P':8,'T':8,'span':None}]
        for i in range(1051):pool.append({'M':int(rng.integers(0,9)),'P':int(rng.integers(8,702)),'T':8,'span':(i,i+1)})
        pools.append(pool)
    elapsed,answer=timed(lambda:fractional_oracle(pools));result['integer_solver']={'streams':120,'options_each':1052,'seconds':elapsed,'iterations':answer['iterations'],'no_F1_or_count_outcome_exported':True}
    inputs=read(PUB/'training_inputs.json');data=np.load(verify_pin(inputs['dataset']),allow_pickle=True).item();plans=np.load(verify_pin(inputs['plans']['2026110461']),allow_pickle=True).item()
    for role in ['risk','utility']:
        start=time.perf_counter();rows=trace_rows(data,plans[role],role);path=folder/('synthetic_'+role+'_trace.private.jsonl')
        with path.open('x') as f:
            for row in rows:f.write(json.dumps(row)+'\n')
            f.flush();os.fsync(f.fileno())
        loaded=[json.loads(x) for x in path.read_text().splitlines()];validate_trace(loaded,data,plans[role],role,'risk' if role=='risk' else 'S-L');result['trace_'+role]={'rows':6000,'generation_IO_and_audit_seconds':time.perf_counter()-start,'actual_optimizer_updates':0}
    start=time.perf_counter();checkpoint=verify_pin(read(PUB/'inference_cost_bounds.json')['engineering_weight_pins']['risk'])
    for i in range(40):
        path=folder/f'state_copy_{i}.private.pt';raw=checkpoint.read_bytes()
        with path.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
        sha(path);torch.load(path,weights_only=True,map_location='cpu')
    result['state40']={'seconds':time.perf_counter()-start,'loads_hashes_writes':40,'no_new_model_or_optimizer':True}
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    start=time.perf_counter();plt.rcParams.update({'pdf.fonttype':42,'savefig.dpi':180})
    for i,(rows,cols,size) in enumerate([(4,2,(17,18)),(5,2,(15,17)),(1,2,(14,7)),(1,1,(14,9)),(2,2,(14,10)),(2,2,(14,8))]):
        fig,axes=plt.subplots(rows,cols,figsize=size)
        for ax in np.asarray(axes).reshape(-1):
            for k in range(8):ax.plot(np.arange(60),np.sin(np.arange(60)+k),label='synthetic '+str(k))
            ax.legend(fontsize=7);ax.set_title('synthetic timing only;not A61 result')
        for ext in ['png','pdf']:fig.savefig(folder/f'synthetic_figure_{i}.{ext}')
        plt.close(fig)
    result['six_figures']={'seconds':time.perf_counter()-start,'PNG':6,'PDF':6,'scientific_result':False}
    # Observable-only old geometry, including all proposals; no label values used.
    geometry={}
    for role in read(PUB/'input_manifest.json')['source_cache_roles']:
        if role in ['fit','inner_cal']:continue
        qs,pa,pj,bases,_=role_values(role);g={k:0 for k in ['N','T','C','C2','TC','all_C','all_TC','K','native_N','native_T','native_T2']}
        for q,a,j,b in zip(qs,pa,pj,bases):
            acts=propose(b,a.argmax(1),j.argmax(1));eligible,_=partition(b,acts);T=len(b);C=len(eligible);g['N']+=1;g['T']+=T;g['C']+=C;g['C2']+=C*C;g['TC']+=T*C;g['all_C']+=len(acts);g['all_TC']+=T*len(acts);g['K']+=sum(np.count_nonzero(apply(b,x)!=b) for x in eligible)
            if q['kind']=='native':g['native_N']+=1;g['native_T']+=T;g['native_T2']+=T*T
        geometry[role]={k:int(v) for k,v in g.items()}
    if torch.cuda.is_initialized():raise ValueError('cost-only CPU initialized CUDA')
    write(PUB/'component_cost_measurements.json',{'passed':True,'timings':result,'old_observable_geometry':geometry,'freeze_git':freeze['git_commit'],'new_optimizer_updates':0,'new_neural_forwards':0,'A61_outer_head_outcomes_used':False,'outer_label_values_used':False,'synthetic_or_FIT_only_computational_inputs':True,'synthetic_trace_rows_not_formal_updates':12000,'private_artifacts_retained':str(folder.relative_to(ROOT)),'scope':'engineering timings only;not scientific F1/CI/oracle evidence','epoch':time.time()});print({'component_costs_passed':True,'new_optimizer_updates':0,'new_neural_forwards':0},flush=True)
if __name__=='__main__':main()
