#!/usr/bin/env python3
"""A62 genuine540 dual-teacher joined inputs; later strict read-only adoption."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.event_risk_common import *
from echo_routing.temporal.event_risk_data import ROLES
from echo_routing.temporal.anchor_cache import ProbabilityCache
from echo_routing.temporal.anchor_inputs import workload


def produce():
    freeze=verify_freeze('producer_freeze.json');cfg=read(CFG);admit=read(PUB/'sources_admission.json')
    if not admit['all_caps_fit'] or admit['input_manifest_sha256']!=sha(PUB/'input_manifest.json') or (PUB/'producer_manifest.json').exists() or (RUN/'models').exists():raise ValueError('fresh admitted wholematrix teacher stage at0 eventupdates required')
    from echo_routing.temporal.event_risk_source import determinism,build_source,a08_sequence,jepa_sequence
    from echo_routing.temporal.dense_readout import ReadoutPixels
    import torch
    determinism();records=[];inputs=read(PUB/'input_manifest.json')
    for name in ['a08','jepa']:
        model=build_source(name,cfg,'cuda')
        for role in ROLES:
            qs=sequences(role);identity=digest({'source':name,'source_contract':cfg['source'][name],'input':inputs['roles'][role]['sha256'],'producer_freeze':freeze['objects'],'new_event_updates':0});folder=RUN/'producer_v1'/name/role;target=ProbabilityCache(folder,identity,[len(q['refs']) for q in qs]);pixels=ReadoutPixels(qs);start=time.time();units=0;cursor0=target.cursor
            for i in range(target.cursor,len(qs)):
                target.run_one(i,lambda i=i:(a08_sequence if name=='a08' else jepa_sequence)(model,qs[i],pixels));w=workload([qs[i]]);units+=w['a08_stride5_windows' if name=='a08' else 'jepa_chunks']
                if (i+1)%25==0:write(RUN/'runtimecoordination.json',{'phase':'headless_540_joined_dual_teacher_cache','source':name,'role':role,'sequences_done':i+1,'sequences':len(qs),'new_neural_units_this_attempt':units,'model_pid':os.getpid(),'formal_event_updates':0,'epoch':time.time(),'next_action':'all1080payloads_thennewFIT72_actualformaladmission_andfour6000'})
            logical=workload(qs);records.append({'source':name,'role':role,'path':str(folder.relative_to(ROOT)),'identity':identity,'journal_sha256':sha(target.journal),'committed_sequences':target.cursor,'input_sha256':inputs['roles'][role]['sha256'],'new_neural_units':units,'committed_logical_neural_units':logical['a08_stride5_windows' if name=='a08' else 'jepa_chunks'],'resumed_committed_prefix_sequences':cursor0,'wall_seconds':time.time()-start,'complete':target.cursor==len(qs)})
        del model;torch.cuda.empty_cache()
    if len(records)!=8 or sum(r['committed_sequences'] for r in records)!=1080:raise ValueError('all540 stream dual producer contexts required')
    write(PUB/'producer_manifest.json',{'complete':True,'records':records,'new_event_updates_at_production':0,'new_event_head_instances_or_forwards':0,'source_production_stage':'sources','source_freeze_git':freeze['git_commit'],'input_manifest_sha256':sha(PUB/'input_manifest.json'),'genuine_joined_context_forward':True,'protected_media_forwards':0,'global_manifest_nonpath_fields_transiently_parsed':True,'protected_labels_used_for_training_or_selection':False,'old_namespaces_written':False});print({'payloads':1080,'genuine_teacher_streams':540,'new_event_updates':0},flush=True)


def adopt():
    verify_freeze('outer_freeze.json')
    from echo_routing.temporal.event_risk_audit import validate_barrier
    from echo_routing.temporal.event_risk_cache import adopt_sources
    models=read(PUB/'models_manifest.json');seal=read(PUB/'outer_identity.json');validate_barrier(models,seal,sha(PUB/'models_manifest.json'),sha(PUB/'calibration.json'),sha(PUB/'readout_plan.json'),read(verify_pin(seal['actual_Git_object_proof'])))
    if not read(PUB/'outer_admission.json')['all_caps_fit'] or (PUB/'source_adoption.json').exists():raise ValueError('fresh complete outer-and-closure admission / one readonly binding required')
    expected={r:(read(PUB/'input_manifest.json')['roles'][r]['sha256'],[len(q['refs']) for q in sequences(r)]) for r in ROLES};v=adopt_sources(read(PUB/'producer_manifest.json'),expected,ROOT,24000);v.update(models_manifest_sha256=sha(PUB/'models_manifest.json'),actual_head_Git_seal_sha256=sha(PUB/'outer_identity.json'),producer_manifest_sha256=sha(PUB/'producer_manifest.json'));write(PUB/'source_adoption.json',v);print({'readonly_payloads':1080,'new_teacher_forwards':0,'binding_updates':24000},flush=True)


def main():
    p=execution_parser(__doc__);p.add_argument('--stage',choices=['headless-sources','adopt'],default='headless-sources');args=p.parse_args()
    if not authorize(args):return 0
    (produce if args.stage=='headless-sources' else adopt)();return 0
if __name__=='__main__':raise SystemExit(main())
