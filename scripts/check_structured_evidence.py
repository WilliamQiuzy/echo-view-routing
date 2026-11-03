#!/usr/bin/env python3
"""Bounded real FIT producer parity and complete-sequence new-head cost/graph checks."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import sys,time,copy,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from scripts.structured_common import ROOT,RUN,PUB,CFG,read,write,sha,verify_freeze,sequences
from scripts.cache_structured_evidence import determinism,build_a08,build_jepa,a08_windows,a08_sequence,jepa_sequence
from echo_routing.temporal.dense_readout import ReadoutPixels
from echo_routing.temporal import dense_runtime
from echo_routing.temporal.structured_inputs import check_probabilities,windows
from echo_routing.temporal.structured_evidence import StructuredEvidence
from echo_routing.temporal import structured_training as train
from echo_routing.temporal.dense_durable import atomic_torch_save

def equal(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b

def main():
    freeze=verify_freeze('engineering_freeze.json');assert read(PUB/'source_qualification.json')['passed'];assert read(PUB/'complementarity.json')['model_forwards']==0;determinism();qs=sequences('fit');manifest=read(PUB/'input_manifest.json');assert sha(RUN/'a08_FIT_parity.private.json')==manifest['parity_private_sha256'];pixels=ReadoutPixels(qs);counts={'a08_windows':0,'jepa_chunks':0};models={'a08':build_a08('cuda'),'jepa':build_jepa('cuda')};handles=[]
    for name,model in models.items():
        def count(_model,args,name=name):counts['a08_windows' if name=='a08' else 'jepa_chunks']+=len(args[0])
        handles.append(model.register_forward_pre_hook(count))
    write(RUN/'runtimecoordination.json',{'phase':'real_FIT_source_and_DP_checks','model_pid':os.getpid(),'epoch':time.time(),'next_action':'finish bounded producer parity/fullsequence graph/resume/timing; no formal model result'});print(json.dumps({'real_GPU_FIT_check_started':True,'model_pid':os.getpid(),'GPU':torch.cuda.get_device_name(0)}),flush=True)
    panel=read(RUN/'a08_FIT_parity.private.json');actual=[]
    for r in panel:actual.append(a08_windows(models['a08'],qs[r['sequence']],pixels,[r['centre']])[0])
    parity=float(np.max(np.abs(np.stack(actual)-np.array([r['expected_logits'] for r in panel]))));assert parity<=1e-4
    pools={kind:[i for i,q in enumerate(qs) if (q['kind']=='native')==(kind=='native')] for kind in ['native','mixed']};profiles={}
    for label,select in [('shortest',min),('longest',max)]:profiles[label]=[select(pool,key=lambda i:len(qs[i]['refs'])) for pool in pools.values()]
    profiles['median']=[sorted(pool,key=lambda i:len(qs[i]['refs']))[len(pool)//2] for pool in pools.values()];chosen=sorted(set(i for batch in profiles.values() for i in batch));prob={name:{} for name in models};costs={name:[] for name in models};jepa_parity=[]
    for name,model in models.items():
        function=a08_sequence if name=='a08' else jepa_sequence
        for i in chosen:
            q=qs[i];pixels.frame.cache_clear();torch.cuda.synchronize();start=time.time();value=function(model,q,pixels);torch.cuda.synchronize();elapsed=time.time()-start;check_probabilities(value,len(q['refs']));prob[name][str(i)]=value;units=len(windows(len(q['refs']),2)['centre']) if name=='a08' else (len(q['refs'])+31)//32;costs[name].append({'FIT_kind':q['kind'],'original_positions':len(q['refs']),'physical_units':units,'raw_decode_forward_stitch_wall_seconds':elapsed,'seconds_per_unit':elapsed/units})
            if name=='jepa' and i in profiles['median']:
                old=dense_runtime.predict_sequence(model,q,pixels);err=float(np.max(np.abs(old-value)));assert err<=1e-6;jepa_parity.append(err)
    out=RUN/'engineering';out.mkdir(exist_ok=True);np.save(out/'FIT_posteriors.private.npy',prob,allow_pickle=True);write(out/'profiles.private.json',profiles)
    for handle in handles:handle.remove()
    assert all(not p.requires_grad and p.grad is None for model in models.values() for p in model.parameters());del models;torch.cuda.empty_cache();timing={};graph=[];physical=0
    for device in ['cpu','cuda']:
        timing[device]={}
        for name,indices in profiles.items():
            batch=[tuple(torch.from_numpy(v).to(device) for v in [prob['a08'][str(i)],prob['jepa'][str(i)],np.asarray(qs[i]['labels'],np.int64)]) for i in indices];head=StructuredEvidence('D2',2026110358).to(device);opt=train.optimizer(head)
            for a,b,_ in batch:assert torch.equal(head.decode(a,b),a.argmax(1))
            rows=[]
            for u in range(2):rows.append(train.step(head,opt,batch,u));physical+=1
            if device=='cuda':torch.cuda.synchronize()
            start=time.time()
            for u in range(2,6):rows.append(train.step(head,opt,batch,u));physical+=1
            if device=='cuda':torch.cuda.synchronize()
            seconds=(time.time()-start)/4;start=time.time()
            for a,b,_ in batch:head.decode(a,b)
            if device=='cuda':torch.cuda.synchronize()
            map_seconds=time.time()-start;timing[device][name]={'sequence_lengths':[len(y) for _,_,y in batch],'seconds_per_update':seconds,'per_valid_position_seconds':seconds/sum(len(y) for _,_,y in batch),'MAP_two_sequences_seconds':map_seconds,'MAP_per_position_seconds':map_seconds/sum(len(y) for _,_,y in batch),'gradients_finite':all(r['finite_gradients'] for r in rows),'first_layer_live_after_updates':rows[-1]['nonzero_gradient_tensors']['transition.0.weight'],'Adam_steps':sorted({int(state['step']) for state in opt.state.values()})};assert timing[device][name]['first_layer_live_after_updates'] and timing[device][name]['Adam_steps']==[6];del head,opt
        batch=[tuple(torch.from_numpy(v).to(device) for v in [prob['a08'][str(i)],prob['jepa'][str(i)],np.asarray(qs[i]['labels'],np.int64)]) for i in profiles['median']]
        def run(n,payload=None):
            nonlocal physical
            h=StructuredEvidence('D2',2026110358).to(device);o=train.optimizer(h);start=0 if payload is None else train.restore(h,o,payload,'FIT-check')
            for u in range(start,n):train.step(h,o,batch,u);physical+=1
            return train.checkpoint(h,o,n,'FIT-check')
        reference=run(4);repeat=run(4);assert equal(reference['model'],repeat['model']) and equal(reference['optimizer'],repeat['optimizer']);path=out/f'{device}_resume.pt';start=time.time();atomic_torch_save(run(2),path);atomic_seconds=time.time()-start;resumed=run(4,torch.load(path,map_location='cpu',weights_only=True));assert equal(reference['model'],resumed['model']) and equal(reference['optimizer'],resumed['optimizer']);graph.append({'backend':device,'repeat_and_disk2plus2_exact_weights_and_moments':True,'atomic_checkpoint_bytes':path.stat().st_size,'atomic_save_plus_two_updates_seconds':atomic_seconds});del reference,repeat,resumed;torch.cuda.empty_cache()
    record={'passed':True,'source_freeze_git':freeze['git_commit'],'input_manifest_sha256':sha(PUB/'input_manifest.json'),'A08_16nativeFIT_original5logit_cache_parity_max_abs':parity,'JEPA_sameFIT_original_loader_stitch_max_abs':jepa_parity,'source_physical_forward_counts':counts,'source_costs':costs,'head_complete_sequence_timing':timing,'real_FIT_graph_repeat_resume':graph,'engineering_optimizer_updates':physical,'formal_optimizer_updates':0,'producer_parameter_gradients':False,'backend_not_selected_by_performance':True,'GPU_peak_reserved_bytes':torch.cuda.max_memory_reserved(),'FIT_engineering_probability_sha256':sha(out/'FIT_posteriors.private.npy'),'FIT_engineering_profile_sha256':sha(out/'profiles.private.json'),'normalization':'nativeandconstructedcompleteNLLpervalidlengththenmean','epoch':time.time()};write(PUB/'checks.json',record);print(json.dumps({'A58_real_FIT_checks_passed':True,'physical_updates':physical,'source_counts':counts,'timing':timing}),flush=True)
if __name__=='__main__':main()
