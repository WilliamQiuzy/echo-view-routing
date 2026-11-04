#!/usr/bin/env python3
"""Bounded cost-only legal-RLE/long-envelope/long-K measurement, no teacher/SGD."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch
from scripts.event_risk_common import *
from echo_routing.temporal.event_risk_engineering import verify
from echo_routing.temporal.event_risk_budget import totals,forecast
from echo_routing.temporal.structured_guard import storage_bytes
from echo_routing.temporal.proposal_actions import propose,partition,apply
from echo_routing.temporal.event_risk_policy import decode
from echo_routing.temporal.event_risk_analysis import summaries
from echo_routing.temporal.event_risk_runtime import infer_events,C6_sha
from echo_routing.temporal.event_risk_model import EventRisk
from echo_routing.temporal.event_risk_cache import EventCache
from echo_routing.temporal.safety_training import Utility,Risk
from echo_routing.temporal.safety_runtime import infer as fixed_infer
from echo_routing.temporal.safety_oracle import options,coverage,solve
from echo_routing.temporal.structured_inputs import windows
from echo_routing.evaluate.window_target import nearest_centre


def synthetic_axis(T,name):
    if T<1 or name not in ['empty','envelope','legal_RLE','long_K']:raise ValueError('registered nonempty synthetic axis required')
    b=(np.arange(T)//6%2).astype(np.int64) if name=='legal_RLE' else np.zeros(T,np.int64)
    centres=windows(T,5)['centre'];ra=(2+np.arange(len(centres))%2)[nearest_centre(T,centres)]
    rj=2+np.arange(T,dtype=np.int64)%2
    if name=='empty':ra=b.copy();rj=b.copy()
    if name=='long_K':
        ra=b.copy();rj=b.copy();rj[:max(1,T//2)]=2
    return b,ra,rj


def profile_value(b,ra,rj):
    actions,rejected=partition(b,propose(b,ra,rj));K=np.asarray([np.count_nonzero(apply(b,a)!=b) for a in actions]);n=len(actions)
    old=np.column_stack([np.linspace(1.,2.,n),.1*K,.9*K,np.zeros(n),np.full(n,.01),np.full(n,.1)]);q=np.tile([.1,.2],(n,1));taus={p:.6 for p in ['L0','M0','R1','R2']}
    truth=np.minimum(np.arange(len(b))*8//len(b),7)%5
    return {'actions':actions,'old':old,'q':q,'value':{**decode(b,actions,old,q[:,0],q[:,1],taus),'old_scores':old,'q':q,'actions':actions,'all_actions':propose(b,ra,rj),'G_rejected':rejected},'sequences':[{'id':'cost_only_synthetic_axis','kind':'multi','labels':truth}]}


def solver_pools(streams,options_each):
    # All integer counts are synthetic, with fixed eight-run truth per stream.
    # No private prediction/truth or trained score enters this cost backend.
    return [[{'M':(i+k)%9,'P':8+(k*7+i)%max(1,options_each),'T':8,'span':None if k==0 else (k,k+1)} for k in range(options_each+1)] for i in range(streams)]


def benchmark(T,name,folder):
    from echo_routing.temporal.event_risk_cost_bounds import observed_geometry
    b,ra,rj=synthetic_axis(T,name);v=profile_value(b,ra,rj);pa=np.eye(5)[ra]*.9+.02;pj=np.eye(5)[rj]*.9+.02
    g=observed_geometry(b,ra,rj);norm={'candidate':{'mean':np.zeros(109),'std':np.ones(109)},'context':{'mean':np.zeros(96),'std':np.ones(96)},'position':{'mean':np.zeros(47),'std':np.ones(47)}}
    heads={a:EventRisk(a,2026110461) for a in ['R1','R2']};utility=Utility('S-L',2026110461);risk=Risk(2026110461)
    t=time.perf_counter();infer_events(heads,norm,pa,pj,b,v['old'],{'scope':'synthetic_cost_only'});event=time.perf_counter()-t
    t=time.perf_counter();fixed_infer(utility,risk,norm,1.,pa,pj,b);fixed=time.perf_counter()-t
    t=time.perf_counter();summaries(v['sequences'],[b],[v['value']],.8917446116307443);summary=time.perf_counter()-t
    t=time.perf_counter();cache=EventCache(folder,name,[b],[ra],[rj],None,[ref:={'scope':'synthetic_only','scores_sha256':C6_sha(v['old'])}]);cache.run_one(0,lambda:{'old_scores':v['old'],'q':v['q'],'old_reference':ref,'seconds':0.});cache.load(0);io=time.perf_counter()-t
    y=v['sequences'][0]['labels'];t=time.perf_counter();pools=[]
    for safety in ['unrestricted','net-safe','gross-safe']:
        pools.append(options(b,ra,rj,y,'expanded',safety));coverage(b,ra,rj,y,'expanded',safety);solve([pools[-1]],'expanded',safety)
    oracle=time.perf_counter()-t
    return {'name':name,'geometry':g,'event_inference_seconds':event,'fixed_inference_seconds':fixed,'six_gate_summary_seconds':summary,'three_oracle_options_coverage_solve_seconds':oracle,'raw_cache_init_write_read_seconds':io,'payload_bytes':cache.file(0).stat().st_size,'synthetic_MLP_module_forward_calls':5*bool(v['actions']),'synthetic_GT_run_ceiling':8,'effect_sizes_exported':False}


def main():
    args=execution_parser(__doc__).parse_args()
    if not authorize(args):return 0
    freeze=verify('refinement_probe_freeze.json');m=read(PUB/'input_manifest.json');spent=totals(actual_receipts());upper=storage_bytes([RUN,PUB])+64000000;now=time.time();admit=forecast(spent,0.,4500.,upper,now,1500.)
    path=PUB/'refinement_probe_admission.json'
    if path.exists():raise ValueError('one prospective refinement attempt only; retain earlier evidence')
    write(path,{**admit,'scope':'independent CPU-only cost stage900CPU/64MB plus existing closure; NOT complete-source/formal admission','stage_CPU_upper_seconds':900,'additional_storage_upper_bytes':64000000,'actual_source_forwards':0,'input_manifest_sha256':sha(PUB/'input_manifest.json'),'engineering_freeze_git':freeze['git_commit']})
    if not admit['all_caps_fit']:return 1
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    from echo_routing.temporal.event_risk_cost_bounds import observed_geometry,aggregate_bounds
    from echo_routing.temporal.baselines.registry import decode_with_prob
    roles=list(read(PUB/'cost_profile.json')['geometry']);geometry={};params=read(CFG)['source']['a08']['original_B3']
    for role in roles:
        a=probability_reader('a08',role,purpose='geometry');j=probability_reader('jepa',role,purpose='geometry');records=[]
        for i,q in enumerate(sequences(role)):
            if role=='inner_cal' and q['kind']!='native':continue
            pa,pj=a.load(i),j.load(i);base=decode_with_prob('b3',pa,{'decode':{'b3':params}})[0];records.append(observed_geometry(base,pa.argmax(1),pj.argmax(1)))
        geometry[role]={k:sum(v[k] for v in records) for k in records[0] if isinstance(records[0][k],(int,float))};geometry[role]['input_sha256']=m['roles'][role]['sha256']
    future={role:aggregate_bounds([len(q['refs']) for q in sequences(role)]) for role in ['CAL_mixed','AUDIT_mixed','a62_confirmation_selection','a62_confirmation_calibration']}
    T=max(r['max_length'] for r in m['roles'].values());folder=RUN/'cost_refinement_v1';folder.mkdir(exist_ok=False);profiles=[]
    for name in ['empty','envelope','legal_RLE','long_K']:
        profiles.append(benchmark(T,name,folder/name));print({'cost_profile_completed':name,'profiles':len(profiles)},flush=True)
    from echo_routing.temporal.anchor_oracle import fractional_oracle
    from echo_routing.temporal.event_risk_cost_bounds import axis_bounds
    options_each=axis_bounds(T)['C'];pools=solver_pools(150,options_each);t=time.perf_counter();fractional_oracle(pools);solver=time.perf_counter()-t
    if torch.cuda.is_initialized():raise ValueError('CPU-only probe initialized CUDA')
    write(PUB/'cost_refinement_profile.json',{'passed':True,'geometry':geometry,'future_structural_bounds':future,'profiles':profiles,'integer_solver':{'streams':150,'options_each_excluding_noop':options_each,'seconds':solver},'input_manifest_sha256':sha(PUB/'input_manifest.json'),'original_cost_profile_sha256':sha(PUB/'cost_profile.json'),'engineering_freeze_git':freeze['git_commit'],'synthetic_MLP_module_forward_calls':sum(p['synthetic_MLP_module_forward_calls'] for p in profiles),'optimizer_updates':0,'teacher_forwards':0,'actual_trained_head_outcomes_or_GT_values_used':False,'metadata_labels_transient_parse_not_benchmark_truth':True,'prospective_only_not_admission':True});return 0


if __name__=='__main__':raise SystemExit(main())
