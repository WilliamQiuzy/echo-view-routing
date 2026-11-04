#!/usr/bin/env python3
"""Opt-in frozen teacher source cache for cost geometry; no A61 action heads."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import sys,time,argparse,inspect,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from scripts.safety_common import *
from scripts.cache_action_safety import build_source,a08_sequence,jepa_sequence,a08_windows,determinism,ReadoutPixels
from echo_routing.temporal.safety_precache import ROLES,identity,folder,validate
from echo_routing.temporal.anchor_cache import ProbabilityCache
from echo_routing.temporal.structured_inputs import windows
from echo_routing.temporal.dense_sequence import chunk_starts


def semantic_functions():
    return {f.__name__:hashlib.sha256(inspect.getsource(f).encode()).hexdigest() for f in [build_source,determinism,a08_windows,a08_sequence,jepa_sequence]}


def verify_source_manifest():
    freeze=verify_freeze('producer_precache_freeze.json');cfg=read(CFG);inputs=read(PUB/'input_manifest.json');m=read(PUB/'producer_precache_manifest.json')
    if m['producer_semantic_function_sha256']!=semantic_functions() or m['producer_freeze_sha256']!=sha(RUN/'producer_precache_freeze.json'):raise ValueError('semantic source/preprocessing or original producer freeze changed')
    return validate(m,cfg,inputs,freeze)


def probability_reader(source,role):
    records=verify_source_manifest();row=records[(source,role)];qs=sequences(role);path=ROOT/row['path']
    if sha(path/'committed.jsonl')!=row['journal_sha256']:raise ValueError('early producer source journal changed')
    return ProbabilityCache(path,row['identity'],[len(q['refs']) for q in qs],readonly=True)


def execute_precache():
    freeze=verify_freeze('producer_precache_freeze.json');admit=read(PUB/'producer_precache_admission.json');cfg=read(CFG);inputs=read(PUB/'input_manifest.json')
    if not admit['all_caps_fit'] or (RUN/'models').exists() or (PUB/'formal_completion.json').exists() or (PUB/'producer_precache_manifest.json').exists():raise ValueError('one prospective admitted precache, before any formal heads, required')
    determinism();records=[]
    for source in ['a08','jepa']:
        model=build_source(source,cfg,'cuda')
        for role in ROLES:
            qs=sequences(role);target=ProbabilityCache(ROOT/folder(source,role),identity(cfg,inputs,freeze,source,role),[len(q['refs']) for q in qs]);pixels=ReadoutPixels(qs);new=0;start=time.time()
            for i in range(target.cursor,len(qs)):
                target.run_one(i,lambda i=i:(a08_sequence if source=='a08' else jepa_sequence)(model,qs[i],pixels));new+=len(windows(len(qs[i]['refs']),5)['centre']) if source=='a08' else len(chunk_starts(len(qs[i]['refs'])))
                if (i+1)%25==0:write(RUN/'runtimecoordination.json',{'phase':'producer_geometry_precache_before_formal','source':source,'role':role,'sequences_done':i+1,'sequences':120,'model_pid':os.getpid(),'formal_optimizer_updates':0,'actual_new_source_neural_units_this_invocation':new,'epoch':time.time(),'next_action':'finishfixed240teachers;CPUobservablegeometry/RLEcost;fullmeasuredadmissionbefore60000'})
            records.append({'source':source,'role':role,'path':folder(source,role),'identity':target.identity,'input_sha256':inputs['roles'][role]['sha256'],'journal_sha256':sha(target.journal),'committed_sequences':target.cursor,'complete':target.cursor==120,'new_neural_units':new,'committed_total_neural_units':inputs['roles'][role]['a08_stride5_windows' if source=='a08' else 'jepa_chunks'],'wall_seconds':time.time()-start})
        del model;torch.cuda.empty_cache()
    manifest={'passed':True,'records':records,'formal_updates_at_production':0,'no_A61_action_heads_instantiated_or_forwarded':True,'no_outer_GT_metrics_or_utility_evaluated':True,'class_order':cfg['channel_mapping']['classes'],'producer_semantic_function_sha256':semantic_functions(),'producer_freeze_sha256':sha(RUN/'producer_precache_freeze.json'),'producer_Git_commit':freeze['git_commit'],'source_production_stage':'producer_precache','real_new240_genuine_joined_streams_per_producer':True,'source_NN_forwards_are_real_not_zero':True,'constructor_label_metadata_exception_retained_no_GT_values_for_geometry':True,'epoch':time.time()};validate(manifest,cfg,inputs,freeze);write(PUB/'producer_precache_manifest.json',manifest);print({'producer_precache_complete':True,'formal_updates':0,'real_source_units':{s:sum(r['new_neural_units'] for r in records if r['source']==s) for s in ['a08','jepa']}},flush=True)


def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');a=p.parse_args(argv)
    if not a.execute:print({'precache_enabled':False,'source_forwards_launched':0});return 0
    execute_precache();return 0
if __name__=='__main__':raise SystemExit(main())
