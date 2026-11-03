#!/usr/bin/env python3
"""A58 immutable producer/cache job; full stream context, stride-specific identities."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import sys,time,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from scripts.structured_common import ROOT,RUN,PUB,CFG,read,write,sha,verify_freeze,sequences
from scripts.cache_structured_evidence import build_a08,build_jepa,a08_windows,a08_sequence,jepa_sequence,determinism
from echo_routing.temporal.structured_inputs import windows,check_probabilities
from echo_routing.temporal.dense_readout import SequenceReadout,ReadoutPixels
from echo_routing.evaluate.window_target import nearest_centre

def cache_identity(source,role,input_sha,producer):
    return hashlib.sha256(json.dumps({'source':source,'role':role,'input':input_sha,'producer':producer},sort_keys=True).encode()).hexdigest()

@torch.no_grad()
def a08_original_sequence(model,q,pixels,device='cuda'):
    n=len(q['refs']);w=windows(n,5);z=a08_windows(model,q,pixels,w['centre'],device);p=torch.softmax(torch.from_numpy(z),1).numpy()[nearest_centre(n,w['centre'])];check_probabilities(p,n);return p

def load_readonly_prefix(folder,identity,seqs,journal_sha,identity_sha):
    folder=Path(folder)
    if sha(folder/'committed.jsonl')!=journal_sha or sha(folder/'identity.json')!=identity_sha or read(folder/'identity.json')!={'identity':identity,'total':len(seqs)} or (folder/'pending.json').exists():raise ValueError('old frozen producer prefix identity changed')
    rows=[json.loads(s) for s in (folder/'committed.jsonl').read_text().splitlines()]
    if len(rows)!=len(seqs):raise ValueError('old producer prefix is incomplete')
    values={}
    for i,(row,q) in enumerate(zip(rows,seqs)):
        p=folder/f'sequence{i:06d}.private.npz'
        if row['index']!=i or row['identity']!=identity or row['positions']!=len(q['refs']) or sha(p)!=row['sha256']:raise ValueError('old prefix ordering/axis/output checksum changed')
        with np.load(p) as z:value=z['prob'].copy()
        check_probabilities(value,len(q['refs']));values[q['id']]=value
    return values

def old_jepa(role,seqs,cfg,manifest):
    if role=='inner_cal':
        pin=cfg['source']['jepa'];assert sha(ROOT/pin['model_manifest'])==pin['model_manifest_sha256'];model=read(ROOT/pin['model_manifest']);row=next(r for r in model['outer_rows'] if r['sha256']==pin['sha256'] and r['step']==6000 and r['kind']=='fixed_step' and r['seed']==2026110257 and r['arm']=='J-high');path=ROOT/'runs/a57_dense_sequence_v1/seed2026110257/J-high/inner_step6000/probabilities.private.npy';assert row['inner_sequence_sha256']==cfg['data']['roles']['inner_cal']['sha256'];assert sha(path)==row['inner_probabilities_sha256'];values=np.load(path,allow_pickle=True).item()
        assert set(values)=={q['id'] for q in seqs}
        for q in seqs:check_probabilities(values[q['id']],len(q['refs']))
        return values
    if role in ['selection','calibration']:
        pin=manifest['diagnostic_inputs']['jepa'][role];identity=cfg['source']['jepa']['sha256']+'|'+cfg['data']['old_outer_metadata_sha256']+'|'+role;return load_readonly_prefix(ROOT/pin['path'],identity,seqs,pin['journal_sha256'],pin['identity_sha256'])
    return None

def descriptor(source,freeze,cfg):
    base='jepa' if source=='jepa' else 'a08';return {'weight_sha256':cfg['source'][base]['sha256'],'frozen_objects':freeze['objects'],'functional_contract':cfg['source'][base],'stride_override':5 if source=='a08_original' else 2 if source=='a08_dense' else 32,'posterior':'FP32full10Hz5class','no_truth_or_join_feature':True}

def reader(source,role,freeze=None):
    freeze=verify_freeze('producer_freeze.json') if freeze is None else freeze;cfg=read(CFG);m=read(PUB/'input_manifest.json');qs=sequences(role);identity=cache_identity(source,role,m['roles'][role]['sha256'],descriptor(source,freeze,cfg));return SequenceReadout(RUN/'cache'/source/role,identity,len(qs),lengths=[len(q['refs']) for q in qs])

def main(mode):
    freeze=verify_freeze('producer_freeze.json');cfg=read(CFG);manifest=read(PUB/'input_manifest.json');admission=read(PUB/'cache_admission.json');assert admission['all_caps_fit'] and admission['formal_updates_at_cache_admission']==0;determinism()
    if mode=='outer':
        models=read(PUB/'models_manifest.json');assert models['passed'] and models['formal_optimizer_updates']==48000 and models['all_head_weights_and_inner_choices_frozen'];roles=['selection','calibration','stress_selection','stress_calibration']
    else:assert mode=='fit';roles=['fit','inner_cal']
    records=[]
    for source in ['a08_dense','jepa',*(['a08_original'] if mode=='outer' else [])]:
        model=None
        for role in roles:
            if source=='a08_original' and not role.startswith('stress_'):continue
            qs=sequences(role);cache=reader(source,role,freeze);initial=cache.cursor;reused=old_jepa(role,qs,cfg,manifest) if source=='jepa' else None;pixels=None;start=time.time()
            if initial<len(qs) and reused is None:
                if model is None:model=(build_jepa if source=='jepa' else build_a08)('cuda')
                pixels=ReadoutPixels(qs)
            new_units=0
            for i in range(initial,len(qs)):
                q=qs[i]
                if reused is not None:cache.run_one(i,lambda q=q:reused[q['id']])
                else:
                    fn=jepa_sequence if source=='jepa' else a08_sequence if source=='a08_dense' else a08_original_sequence;cache.run_one(i,lambda q=q:fn(model,q,pixels));new_units+=(len(q['refs'])+31)//32 if source=='jepa' else len(windows(len(q['refs']),2 if source=='a08_dense' else 5)['centre'])
                if (i+1)%25==0:
                    progress={'phase':'frozen_producer_cache','producer':source,'role':role,'sequences_done':i+1,'sequences':len(qs),'model_pid':os.getpid(),'new_neural_units_this_role_process':new_units,'old_JEPA_cache_reused':reused is not None,'epoch':time.time(),'next_action':'finish admitted complete cache prefixes; thenCPUhead formaladmission'};write(RUN/'runtimecoordination.json',progress);print(json.dumps(progress),flush=True)
            records.append({'producer':source,'role':role,'identity':cache.identity,'committed_sequences':cache.cursor,'input_sequence_sha256':manifest['roles'][role]['sha256'],'journal_sha256':sha(cache.journal),'new_neural_units_this_process':new_units,'reused_old_JEPA_sequences_this_process':len(qs)-initial if reused is not None else 0,'prior_committed_sequences':initial,'wall_seconds_this_process':time.time()-start,'complete':cache.cursor==len(qs)})
        if model is not None:del model;torch.cuda.empty_cache()
    write(PUB/f'cache_{mode}_manifest.json',{'passed':True,'records':records,'producer_freeze_git':freeze['git_commit'],'input_manifest_sha256':sha(PUB/'input_manifest.json'),'source_code_contracts_frozen':True,'formal_optimizer_updates_during_cache':0,'new_protected_media_forwards':0,'epoch':time.time()});write(RUN/'runtimecoordination.json',{'phase':'cache_'+mode+'_complete','model_pid':None,'epoch':time.time(),'next_action':'freezecompletecachemanifestandformalheadadmission' if mode=='fit' else 'readoutallfixedheadssamemain/stressaxes'});print(json.dumps({'cache_complete':mode,'records':len(records)}),flush=True)
if __name__=='__main__':main(sys.argv[1])
