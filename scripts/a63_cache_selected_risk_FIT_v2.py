#!/usr/bin/env python3
"""A63 FIT-only versioned producer; original prepared source_FIT is retained."""
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.selected_risk_common import *
from scripts.a63_selected_risk_FIT_v2_contract import verify_shared_freeze,STAGE
from echo_routing.temporal.selected_risk_policy import production_phase
from echo_routing.temporal.anchor_inputs import workload


def produce(phase):
    
    if phase!='FIT':raise ValueError('one literal FIT-only recovery contract')
    freeze=verify_shared_freeze(STAGE);cfg=read(CFG);inputs=read(PUB/'input_manifest.json')
    cov=read(PUB/'coverage.json') if phase=='CONFIRM' else None;audit=read(PUB/'aux_model_audit.json') if phase=='CONFIRM' else None
    if phase=='CONFIRM':verify_aux_seal()
    rule=production_phase(phase,cov,audit);admit=read(PUB/('sources_admission.json' if phase=='FIT' else 'confirm_admission.json'))
    if admit['all_caps_fit'] is not True or admit['input_manifest_sha256']!=sha(PUB/'input_manifest.json') or admit['source_phase']!=phase or (PUB/('producer_'+phase+'_manifest.json')).exists():raise ValueError('fresh fullremaining admitted phase plus exact source input required')
    if (RUN/'models').exists() or phase=='FIT' and (RUN/'aux_models').exists():raise ValueError('production counters/phase cannot claim no updates after models exist')
    from echo_routing.temporal.selected_risk_source import determinism,build_source,a08_sequence,jepa_sequence
    from echo_routing.temporal.dense_readout import ReadoutPixels
    import torch
    determinism();records=[]
    for name in ['a08','jepa']:
        model=build_source(name,cfg,'cuda')
        for role in rule['roles']:
            qs=sequences(role);identity=digest({'source':name,'source_contract':cfg['source'][name],'input':inputs['roles'][role]['sha256'],'producer_freeze':freeze['objects'],'counter_axes_at_production':rule,'phase':phase});folder=RUN/('producer_'+phase+'_v1')/name/role;target=ProbabilityCache(folder,identity,[len(q['refs']) for q in qs]);pixels=ReadoutPixels(qs);start=time.time();units=0;cursor0=target.cursor
            for i in range(cursor0,len(qs)):
                target.run_one(i,lambda i=i:(a08_sequence if name=='a08' else jepa_sequence)(model,qs[i],pixels));w=workload([qs[i]]);units+=w['a08_stride5_windows' if name=='a08' else 'jepa_chunks']
                if (i+1)%25==0:write(RUN/'runtimecoordination.json',{'phase':phase+'_dual_teacher','source':name,'role':role,'sequences_done':i+1,'sequences':len(qs),'new_units_this_attempt':units,'PID':os.getpid(),'aux_updates_at_production':rule['aux_updates_at_production'],'formal_updates_at_production':0,'epoch':time.time()})
            logical=workload(qs);records.append({'source':name,'role':role,'path':relative(folder),'identity':identity,'journal_sha256':sha(target.journal),'committed_sequences':target.cursor,'input_sha256':inputs['roles'][role]['sha256'],'actual_new_neural_units_this_invocation':units,'committed_logical_neural_units':logical['a08_stride5_windows' if name=='a08' else 'jepa_chunks'],'resumed_committed_prefix_sequences':cursor0,'wall_seconds':time.time()-start,'complete':target.cursor==len(qs)})
        del model;torch.cuda.empty_cache()
    if len(records)!=4 or sum(r['committed_sequences'] for r in records)!=rule['payloads']:raise ValueError('exact phase dual teacher coverage required')
    out={'complete':True,'phase':phase,'records':records,'aux_updates_at_production':rule['aux_updates_at_production'],'formal_updates_at_production':0,'new_event_head_instances_or_forwards':0,'source_production_stage':STAGE,'source_freeze_git':freeze['git_commit'],'input_manifest_sha256':sha(PUB/'input_manifest.json'),'genuine_joined_context_forward':True,'teacher_streams':rule['payloads']//2,'committed_payloads':rule['payloads'],'protected_media_forwards':0,'constructor_metadata_label_parse_disclosed':True,'labels_used_for_new_teacher_predictions':False,'old_namespaces_written':False};write(PUB/('producer_'+phase+'_manifest.json'),out);print({'phase':phase,'payloads':rule['payloads'],'teacher_units':sum(r['actual_new_neural_units_this_invocation'] for r in records),'production_aux':rule['aux_updates_at_production'],'formal':0},flush=True)

def main():
    p=execution_parser(__doc__);p.add_argument('--phase',choices=['FIT'],required=True);a=p.parse_args()
    if not authorize(a):return 0
    produce(a.phase);return 0
if __name__=='__main__':raise SystemExit(main())
