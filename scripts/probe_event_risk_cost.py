#!/usr/bin/env python3
"""A62 cost-only observable geometry and synthetic full analysis/cache/CI backend."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import sys,time,copy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch
from scripts.event_risk_common import *
from echo_routing.temporal.proposal_actions import propose,partition,apply
from echo_routing.temporal.baselines.registry import decode_with_prob
from echo_routing.temporal.event_risk_features import context_features
from echo_routing.temporal.event_risk_model import EventRisk
from echo_routing.temporal.event_risk_analysis import summaries
from echo_routing.temporal.event_risk_policy import decode
from echo_routing.temporal.event_risk_metrics import paired
from echo_routing.temporal.event_risk_cache import EventCache
from echo_routing.temporal.event_risk_runtime import C6_sha


def benchmark(T,pattern,folder):
    b=np.zeros(T,np.int64) if pattern in ['constant','empty'] else np.arange(T,dtype=np.int64)%2;ra=b.copy();ra[1::2]=2;rj=2+np.arange(T,dtype=np.int64)%2;
    if pattern=='empty':ra=b.copy();rj=b.copy()
    pa=np.eye(5)[ra]*.9+.02;pj=np.eye(5)[rj]*.9+.02;actions,rejected=partition(b,propose(b,ra,rj));K=np.asarray([np.count_nonzero(apply(b,a)!=b) for a in actions]);scores=np.column_stack([np.linspace(1.,2.,len(actions)),.1*K,.9*K,np.zeros(len(actions)),np.full(len(actions),.01),np.full(len(actions),.1)]);q=np.tile([.1,.2],(len(actions),1));taus={k:.6 for k in ['L0','M0','R1','R2']};v={**decode(b,actions,scores,q[:,0],q[:,1],taus),'old_scores':scores,'q':q,'actions':actions,'all_actions':propose(b,ra,rj),'G_rejected':rejected};truth=np.minimum(np.arange(T)*8//T,7)%5;qs=[{'id':'synthetic_cost_only','kind':'multi','labels':truth}]
    t=time.perf_counter();[context_features(pa,pj,b,a) for a in actions];context=time.perf_counter()-t
    from echo_routing.temporal.event_risk_runtime import infer_events
    norms={'candidate':{'mean':np.zeros(109),'std':np.ones(109)},'context':{'mean':np.zeros(96),'std':np.ones(96)}};heads={a:EventRisk(a,2026110461) for a in ['R1','R2']};t=time.perf_counter();infer_events(heads,norms,pa,pj,b,scores,{'kind':'synthetic_cost'});full_infer=time.perf_counter()-t
    t=time.perf_counter();summaries(qs,[b],[v],.8917446116307443);analysis=time.perf_counter()-t
    ref={'kind':'synthetic_cost_not_teacher','scores_sha256':C6_sha(scores)};cache=EventCache(folder,pattern,[b],[ra],[rj],None,[ref]);t=time.perf_counter();cache.run_one(0,lambda:{'old_scores':scores,'q':q,'old_reference':ref,'seconds':0.});cache.load(0);io=time.perf_counter()-t
    return {'T':T,'C':len(actions),'T_times_C':T*len(actions),'K_total':int(K.sum()),'context_seconds':context,'full_q_inference_seconds':full_infer,'synthetic_event_MLP_calls':2*bool(actions),'six_gate_summary_seconds':analysis,'raw_cache_init_write_read_seconds':io,'payload_bytes':cache.file(0).stat().st_size,'GT_run_upper_synthetic':8,'synthetic_only_no_effect_sizes_exported':True}


def bundle(n_sessions,n_multi,n_pair):
    rng=np.random.default_rng(62);sessions=[f'cost_session{i}' for i in range(n_sessions)];conf=np.zeros((n_sessions,5,5),np.int64)
    for i in range(n_sessions):conf[i,i%5,i%5]=3
    streams={};gross={};matched={};lengths={}
    for kind,n in [('multi',n_multi),('pair',n_pair)]:
        streams[kind]=[];gross[kind]=[];matched[kind]={}
        for i in range(n):
            name=f'cost_{kind}_{i}';row=np.array([2,3,3,100,90,1,2,2,1,2,2,1,2,1,2,1,2,1],np.int64);row[4]+=int(rng.integers(0,8));streams[kind].append((name,row));gross[kind].append((name,np.array([0,int(rng.integers(0,5))],np.int64)));matched[kind][name]={0,1};lengths[name]=[10,20,30]
    return {'session_order':sessions,'session_confusion':conf,'session_extra':np.tile([0,300,270,3],(n_sessions,1)),'session_gross':np.zeros((n_sessions,2),np.int64),'streams':streams,'stream_gross':gross,'native':{f'cost_native{i}':(i%5,i%5) for i in range(n_sessions*3)},'matched':matched,'true_lengths':lengths}


def benchmark_metrics(T):
    from types import SimpleNamespace
    from echo_routing.temporal.dense_runtime import metric_stream
    from echo_routing.temporal.event_risk_metrics import evaluate
    qs=[];natives=[];streams={'multi':[],'pair':[]};pred={}
    for kind,N,pieces in [('native',300,1),('multi',150,8),('pair',300,2)]:
        for i in range(N):
            name=f'cost_{kind}_session{i//3}_{i%3}';cuts=np.linspace(0,T,pieces+1,dtype=int);labels=np.concatenate([np.full(z-x,(k+i)%5,np.int64) for k,(x,z) in enumerate(zip(cuts,cuts[1:]))]);refs=[(f'cost_source{k}',n,1.) for k,(a,b) in enumerate(zip(cuts,cuts[1:])) for n in range(b-a)];q={'id':name,'kind':kind,'labels':labels,'refs':refs,'joins':cuts[1:-1]};qs.append(q);pred[name]=np.arange(T,dtype=np.int64)%5
            if kind=='native':natives.append(SimpleNamespace(seq_id=name,labels=labels))
            else:streams[kind].append(metric_stream(q))
    ctx={'sequences':qs,'eval_set':SimpleNamespace(natives=natives,multi=streams['multi'],pairs=streams['pair'])};t=time.perf_counter();evaluate(pred,ctx,pred);return time.perf_counter()-t


def main():
    p=execution_parser(__doc__);args=p.parse_args()
    if not authorize(args):return 0
    freeze=verify_freeze('cost_freeze.json');torch.set_num_threads(1);torch.use_deterministic_algorithms(True);m=read(PUB/'input_manifest.json');folder=RUN/'cost_only';folder.mkdir(exist_ok=False);roles=['fit','inner_cal','selection','calibration','old_stress_selection','old_stress_calibration','confirmation_selection','confirmation_calibration','a60_confirmation_selection','a60_confirmation_calibration','a61_confirmation_selection','a61_confirmation_calibration'];geometry={};maxT=max(r['max_length'] for r in m['roles'].values());params=read(CFG)['source']['a08']['original_B3']
    for role in roles:
        qs=sequences(role);a=probability_reader('a08',role,purpose='geometry');j=probability_reader('jepa',role,purpose='geometry');N=totT=C=TC=K=0;maxC=pieces=0
        for i,q in enumerate(qs):
            if role=='inner_cal' and q['kind']!='native':continue
            T=len(q['refs']);source_runs=1+sum(q['refs'][z][0]!=q['refs'][z-1][0] for z in range(1,T));limit=1 if q['kind']=='native' else 8 if q['kind']=='multi' else 2
            if source_runs>limit:raise ValueError('structural GT-run ceiling not supported by source-reference grammar; no label-values inspected')
            pa,pj=a.load(i),j.load(i);base=decode_with_prob('b3',pa,{'decode':{'b3':params}})[0];actions,_=partition(base,propose(base,pa.argmax(1),pj.argmax(1)));N+=1;totT+=T;C+=len(actions);TC+=T*len(actions);K+=sum(np.count_nonzero(apply(base,z)!=base) for z in actions);maxC=max(maxC,len(actions));pieces=max(pieces,source_runs)
        geometry[role]={'N':N,'T':totT,'C':C,'TC':TC,'K':int(K),'max_C':maxC,'max_source_runs':pieces,'input_sha256':m['roles'][role]['sha256']}
    profiles=[benchmark(maxT,p,folder/p) for p in ['empty','constant','dense']];head=EventRisk('R1',2026110461);x=torch.ones((64,205),dtype=torch.float32);forward=[]
    with torch.no_grad():
        for _ in range(6):t=time.perf_counter();head(x);forward.append(time.perf_counter()-t)
    ci=bundle(100,150,300);other=copy.deepcopy(ci)
    for i in range(100):other['session_confusion'][i,i%5,i%5]=2;other['session_confusion'][i,i%5,(i+1)%5]=1;other['session_extra'][i,2]-=i%11
    for kind in ['multi','pair']:
        for i,(name,row) in enumerate(other['streams'][kind]):row[0]=1+i%2;row[4]-=i%7;other['matched'][kind][name]={0} if i%2==0 else {0,1}
    t=time.perf_counter();paired(ci,other,primary=True);ci_seconds=time.perf_counter()-t
    metric_seconds=benchmark_metrics(maxT)
    if torch.cuda.is_initialized():raise ValueError('CPU cost stage initialized CUDA')
    write(PUB/'cost_profile.json',{'passed':True,'geometry':geometry,'profiles':profiles,'max_64candidate_forward_seconds':max(forward),'synthetic_MLP_forward_calls':6+sum(p['synthetic_event_MLP_calls'] for p in profiles),'new_optimizer_updates':0,'teacher_forwards':0,'actual_A62_head_outputs_or_actual_GT_values_used':False,'metadata_label_bytes_transient_parse_not_used':True,'synthetic_fulltask_metric_seconds':metric_seconds,'synthetic_CI5000_seconds':ci_seconds,'synthetic_CI_unit_upper':{'native_sessions':100,'multi_streams':150,'pair_streams':300},'input_manifest_sha256':sha(PUB/'input_manifest.json'),'GT_run_ceiling_proved_by_source_ref_blocks':True,'GT_run_upper_by_kind':{'native':1,'multi':8,'pair':2},'all_profile_payloads_synthetic':True,'freeze_git':freeze['git_commit']});print({'cost_only':True,'newMLPforwards':6+sum(p['synthetic_event_MLP_calls'] for p in profiles),'newSGD':0,'teacherforwards':0,'profiles':len(profiles),'old_geometry_roles':len(geometry)},flush=True);return 0
if __name__=='__main__':raise SystemExit(main())
