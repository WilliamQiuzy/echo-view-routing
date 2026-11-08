#!/usr/bin/env python3
"""A63 independent bounded CPU engineering probe, no new teachers or C0/C1.

Entry: sole existing supervisor --seconds890 cost_probe. Entry/startup/IO are
charged by its actual whole-process receipt, not this perf-counter profile.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1';os.environ['PYTHONDONTWRITEBYTECODE']='1'
import sys,time,resource,math
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch
from scripts.selected_risk_common import *
from echo_routing.temporal.selected_risk_forecast import TERMS,verify_cost_inputs
from echo_routing.temporal.selected_risk_budget import totals,forecast
from echo_routing.temporal.event_risk_cost_bounds import observed_geometry,aggregate_bounds,_centres
from echo_routing.temporal.proposal_actions import propose,partition
from echo_routing.evaluate.window_target import nearest_centre
from echo_routing.temporal.baselines.registry import decode_with_prob


def storage_bytes(paths):
    total=0
    for root in paths:
        for p in Path(root).rglob('*') if Path(root).exists() else []:
            if p.is_symlink():raise ValueError('tracked cost namespace contains symlink')
            if p.is_file():total+=p.stat().st_size
    return total


def stage_storage_upper(initial):
    return initial+300000000+64000000+128000000

def synthetic_axis(T,name):
    if T<1 or name not in ['empty','envelope','legal_RLE','long_K']:raise ValueError('fixed synthetic geometry domains required')
    b=(np.arange(T)//6%2).astype(np.int64) if name=='legal_RLE' else np.zeros(T,np.int64)
    centres=_centres(T);a=(2+np.arange(len(centres))%2)[nearest_centre(T,centres)];j=2+np.arange(T,dtype=np.int64)%2
    if name=='empty':a=b.copy();j=b.copy()
    if name=='long_K':a=b.copy();j=b.copy();j[:max(1,T//2)]=2
    return b,a,j


def usage(initial):
    a=resource.getrusage(resource.RUSAGE_SELF);b=resource.getrusage(resource.RUSAGE_CHILDREN)
    if a.ru_utime+a.ru_stime+b.ru_utime+b.ru_stime>=880 or storage_bytes([RUN,PUB])>initial+64000000:raise ValueError('probe880CPU/64MB boundary reached; preserve charged prefix')


def observe_old_fit(initial):
    m=read(PUB/'input_manifest.json');qs=sequences('fit');rows={k:[] for k in ['native','mixed']};maxima={k:{'C':0,'K':0,'T':0} for k in rows}
    caches={}
    for name in ['a08','jepa']:
        r=m['source_cache_roles']['fit'][name];c=ProbabilityCache(ROOT/r['path'],r['identity'],[len(q['refs']) for q in qs],readonly=True)
        if sha(c.journal)!=r['journal_sha256'] or r['input_sha256']!=m['roles']['fit']['sha256']:raise ValueError('old complete teacher journal/input identity drift')
        caches[name]=c
    cfg=read(CFG);params=cfg['source']['a08']['original_B3']
    for i,q in enumerate(qs):
        a,j=caches['a08'].load(i),caches['jepa'].load(i);b=decode_with_prob('b3',a,{'decode':{'b3':params}})[0];g=observed_geometry(b,a.argmax(1),j.argmax(1));g['nonempty_query_axes']=int(g['C']>0);kind='native' if q['kind']=='native' else 'mixed';rows[kind].append(g)
        for k in maxima[kind]:maxima[kind][k]=max(maxima[kind][k],g[k])
        if i%100==0:usage(initial)
    out={k:{x:sum(r[x] for r in rr) for x in rr[0]} for k,rr in rows.items()}
    if (out['native']['C'],out['mixed']['C'])!=(265,3918):raise ValueError('original G candidate parity265+3918 required')
    return out,maxima


def benchmark_backend(T,name,folder):
    from echo_routing.temporal.selected_risk_fit import prepare_aux_training
    from echo_routing.temporal.selected_risk_effects import record
    from echo_routing.temporal.selected_risk_cache import SupplyCache
    from echo_routing.temporal.event_risk_features import context_features
    from echo_routing.temporal.safety_features import utility_features
    from echo_routing.temporal.safety_runtime import infer
    from echo_routing.temporal.safety_training import Utility,Risk
    b,a,j=synthetic_axis(T,name);pa=.02+.9*np.eye(5)[a];pj=.02+.9*np.eye(5)[j];acts,_=partition(b,propose(b,a,j));g=observed_geometry(b,a,j)
    truth=(np.arange(T)*8//T%5).astype(np.int64);q={'id':'synthetic_0','kind':'multi','labels':truth,'refs':[('synthetic_0',i,1.) for i in range(T)]}
    norm={'candidate':{'mean':np.zeros(109),'std':np.ones(109)},'position':{'mean':np.zeros(47),'std':np.ones(47)}}
    u,r=Utility('S-L',2026110461),Risk(2026110461);t=time.perf_counter()
    if acts:
        prepare_aux_training([q],[pa],[pj],[b],{'synthetic'},{'not_training'})
    value=infer(u,r,norm,1.,pa,pj,b);matrix=value['score_matrix'];x=np.stack([utility_features(pa,pj,b,v) for v in acts]) if acts else np.empty((0,109));ctx=np.stack([context_features(pa,pj,b,v) for v in acts]) if acts else np.empty((0,96))
    for action,c in zip(acts,ctx):record(q,b,action,c,D=False,X=False)
    # Timer includes durable constructor, atomic write and validated read.
    cache=SupplyCache(folder,name,[b],[a],[j]);cache.run_one(0,lambda:{'D_C6':matrix,'X_C6':matrix.copy(),'x109':x,'context96':ctx});cache.load(0)
    elapsed=time.perf_counter()-t
    return {'name':name,'geometry':g,'backend_seconds':elapsed,'MLP_module_forwards':3*bool(acts),'payload_bytes':cache.file(0).stat().st_size,'truth_scope':'synthetic at most8 runs; no actual GT values','exports_simulated_effect_sizes':False}


def benchmark_training(C,K,folder):
    from echo_routing.temporal.safety_training import Utility,Risk,optimizer,step,checkpoint
    from echo_routing.temporal.event_risk_model import EventRisk,optimizer as event_opt,step as event_step
    from echo_routing.temporal.proposal_training import atomic_save
    identity={k:'a'*64 for k in ['code','config','sources','input','draw_plan','normalizer','target_scales']};norms={n:{'mean':np.zeros(d),'std':np.ones(d)} for n,d in [('candidate',109),('position',47)]}
    rng=np.random.default_rng(63001);times={};models=[];calls=0
    for role,model in [('utility',Utility('S-L',2026110461)),('risk',Risk(2026110461)),('event',EventRisk('R2',2026110461))]:
        opt=event_opt(model) if role=='event' else optimizer(model);steps=[]
        for update in range(5):
            t=time.perf_counter()
            if role=='utility':batch={'x':torch.from_numpy(rng.normal(size=(64*C,109)).astype(np.float32)),'target':torch.from_numpy(np.tile(np.linspace(-1,1,C),64).astype(np.float64)),'lengths':[C]*64}
            elif role=='risk':batch={'x':torch.from_numpy(rng.normal(size=(64,109)).astype(np.float32)),'positions':torch.from_numpy(rng.normal(size=(64*K,47)).astype(np.float32)),'labels':torch.from_numpy(np.arange(64*K,dtype=np.int64)%3),'fp':torch.from_numpy((np.arange(64)%2).astype(np.float32)),'lengths':[K]*64}
            else:batch={'x':torch.from_numpy(rng.normal(size=(64,205)).astype(np.float32)),'y':torch.from_numpy((np.arange(64)%2).astype(np.float32))}
            (event_step(model,opt,batch,update) if role=='event' else step(model,opt,batch,update,role));steps.append(time.perf_counter()-t);calls+=1 if role!='risk' else 2
        times[role+'_update_seconds']=max(steps);models.append((model,opt))
    # Complete largest safety state, RNG and populated Adam moments included.
    model,opt=models[1];value=checkpoint(model,opt,5,identity,norms,1.);path=folder/'synthetic_fullstate.pt';durable_directory(folder);tim=[]
    for i in range(5):
        t=time.perf_counter();atomic_save(value,path);torch.load(path,weights_only=True,map_location='cpu');tim.append(time.perf_counter()-t)
    times['fullstate_write_read_seconds']=max(tim)
    return times,{'optimizer_updates':15,'logical_synthetic_updated_models':3,'MLP_module_forward_calls':calls,'scope':'fresh old Utility/Risk/R2 only, no C0/C1; not reused for actual models','utility_complete_queries_per_update':64,'utility_candidates_per_query':C,'risk_candidates':64,'risk_all_changed_positions_per_candidate':K,'event_candidates':64,'state_bytes':path.stat().st_size}


def main():
    args=execution_parser(__doc__).parse_args()
    if not authorize(args):return 0
    freeze=verify_freeze('cost_probe');costproof=verify_cost_inputs(ROOT);cpu_only();initial=storage_bytes([RUN,PUB]);spent=totals(actual_receipts());admit=forecast(spent,0,4500,stage_storage_upper(initial),time.time(),8100)
    if (PUB/'cost_profile.json').exists() or (PUB/'cost_stage_admission.json').exists():raise ValueError('one immutable bounded probe attempt; no implicit rerun')
    write(PUB/'cost_stage_admission.json',{**admit,'stage_CPU_upper_seconds':900,'stage_storage_increment_upper_bytes':64000000,'scope':'cost-only independent stage plus original closing, NOT full source admission'})
    if not admit['all_caps_fit']:return 1
    folder=RUN/'cost_probe_v1';folder.mkdir(exist_ok=False);os.environ['TMPDIR']=str(folder);os.environ['MPLCONFIGDIR']=str(folder/'mpl');usage(initial)
    geometry,maxima=observe_old_fit(initial);m=read(PUB/'input_manifest.json');T=max([v['max_length'] for v in m['roles'].values()]);profiles=[]
    for name in ['empty','envelope','legal_RLE','long_K']:
        profiles.append(benchmark_backend(T,name,folder/name));usage(initial);print({'profile':name,'completed':len(profiles)},flush=True)
    mapping={'N':'empty','T':'empty','T2':'empty','C':'envelope','C2':'envelope','TC':'envelope','K':'long_K','KT':'long_K','CR':'legal_RLE'};pr={p['name']:p for p in profiles}
    coef={k:pr[n]['backend_seconds']/max(1.,pr[n]['geometry'].get('all_'+k,pr[n]['geometry'][k])) for k,n in mapping.items()}
    for p in profiles:
        if sum(p['geometry'].get('all_'+k,p['geometry'][k])*coef[k] for k in TERMS)+1e-12<p['backend_seconds']:raise ValueError('nonnegative backend profile domination failed')
    C=max(1,math.ceil(2*maxima['mixed']['C']*max(m['roles'][r]['max_length'] for r in ['F0_mixed','F1_mixed'])/maxima['mixed']['T']));K=max(maxima['native']['T'],m['roles']['F0_mixed']['max_length'],m['roles']['F1_mixed']['max_length'])
    training,scope=benchmark_training(C,K,folder);usage(initial)
    old=read(prior_path('docs/results/2026-11-04_anchor_context_risk/input_manifest.json'))
    # Existing qualified 5000-draw backend cost is reused with explicit pin;
    # paired140 counts and factor2 are in the new estimator, not recomputed CI.
    op=ROOT/'docs/results/2026-11-04_anchor_context_risk/cost_profile.json';prior_profile=read(op)
    grammar={r:aggregate_bounds([len(q['refs']) for q in sequences(r)]) for r in ['F0_mixed','F1_mixed','a63_confirmation_selection','a63_confirmation_calibration']}
    grammar_feature_bytes=sum(8*(109*g['C']+47*g['K']+96*g['C']) for g in grammar.values())
    write(PUB/'cost_profile.json',{'passed':True,'geometry':geometry,'old_FIT_maxima':maxima,'profiles':profiles,'backend_coefficients':coef,'training':training,'synthetic_training_scope':scope,'synthetic_MLP_module_forward_calls':scope['MLP_module_forward_calls']+sum(p['MLP_module_forwards'] for p in profiles),'synthetic_optimizer_updates':15,'teacher_forwards':0,'production_C0_C1_instances':0,'max_training_C':C,'max_training_K':K,'CI5000_seconds':prior_profile['synthetic_CI5000_seconds'],'readonly_CI_measurement_pin':pin(op),'readonly_old_cost_actual_Git_proof':costproof,'grammar_bounds':grammar,'original_grammar_feature_bytes':grammar_feature_bytes,'grammar_upper_fit_is_not_assumed':True,'input_manifest_sha256':sha(PUB/'input_manifest.json'),'freeze_git':freeze['git_commit'],'metadata_label_bytes_parsed_but_not_read_as_geometry_or_benchmark_truth':True,'new_actual_GT_or_head_results_used':False,'engineered_cost_not_model_result':True})
    if torch.cuda.is_initialized():raise ValueError('CPU probe initialized CUDA')
    print({'cost_probe_complete':True,'toy_updates':15,'teacher':0,'C0_C1_instances':0},flush=True);return 0
if __name__=='__main__':raise SystemExit(main())
