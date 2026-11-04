#!/usr/bin/env python3
"""Explicit A62 metadata partition/recipes and new FIT context only."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scripts.event_risk_common import *
from echo_routing.temporal.event_risk_data import partition_inner,construct,ROLES,unsafe_targets,validate_draw_plan
from echo_routing.temporal.event_risk_features import context_features,fit_context_normalizer,event_matrix
from echo_routing.temporal.anchor_inputs import verify_native_groups,workload,native_reuse_index
from echo_routing.temporal.proposal_actions import Action,propose,partition
from echo_routing.temporal.proposal_features import candidate_features
from echo_routing.temporal.safety_targets import effects
from echo_routing.temporal.baselines.registry import decode_with_prob


def prior():return {k:read(verify_pin(v)) for k,v in read(CFG)['prior_A61'].items()}

def metadata():
    freeze=verify_freeze('metadata_freeze.json');verify_prior_science();cfg=read(CFG);old=prior()
    if (PUB/'input_manifest.json').exists():raise ValueError('metadata immutable; do not redraw/repartition')
    for path,h in cfg['readonly_code_pins'].items():
        if sha(ROOT/path)!=h:raise ValueError('unchanged qualified scientific source code required')
    if old['collector_OS_exit_receipt.json']['observed_collector_OS_exit_code']!=0 or not old['resources.json']['server_scientific_and_resource_collection_complete']:raise ValueError('completed prior A61 receipt required, not stale running flag')
    previous=old['input_manifest.json']
    if previous['class_order']!=['PLAX','PSAX','A4C','A5C','SC4C'] or not previous['A08_J_Base_truth_channel_semantics_unchanged']:raise ValueError('five-channel teacher/Base/label class semantics qualification required for JEPA replacement use')
    roles=dict(previous['roles']);source={k:dict(v) for k,v in previous['source_cache_roles'].items()}
    for r in old['cache_outer_manifest.json']['records']:source.setdefault(r['role'],{})[r['source']]=dict(r)
    qs={r:np.load(verify_pin(p),allow_pickle=True).tolist() for r,p in roles.items()};natives={r:[q for q in qs[r] if q['kind']=='native'] for r in ['fit','inner_cal','selection','calibration']};isolation=verify_native_groups(natives)
    cal,audit,part=partition_inner(natives['inner_cal']);groups={'CAL_native':cal,'AUDIT_native':audit};recipes={};summaries={};sources={'CAL_mixed':cal,'AUDIT_mixed':audit,'a62_confirmation_selection':natives['selection'],'a62_confirmation_calibration':natives['calibration']}
    for role in ROLES:
        group,recipe,summary=construct(sources[role],role);groups[role]=group;summaries[role]=summary;p=RUN/'metadata'/f'{role}_recipes.private.json';write(p,recipe);recipes[role]=pin(p)
    groups['CAL']=cal+groups['CAL_mixed'];groups['AUDIT']=audit+groups['AUDIT_mixed']
    for role,group in groups.items():
        p=RUN/'metadata'/f'{role}.private.npy';save_array(p,np.asarray(group,dtype=object));roles[role]={**pin(p),**workload(group),'native':sum(q['kind']=='native' for q in group)}
    for role,rows in source.items():
        for name,r in rows.items():
            if sha(ROOT/r['path']/'committed.jsonl')!=r['journal_sha256'] or r['input_sha256']!=roles[role]['sha256']:raise ValueError('old context/input/journal contract changed')
    mapping={population:{'native_original_inner_indices':native_reuse_index(qs['inner_cal'],groups[population]),'new_mixed_role':population+'_mixed'} for population in ['CAL','AUDIT']}
    partition_sources={g:{q['id'] for q in group} for g,group in [('CAL',cal),('AUDIT',audit)]}
    if partition_sources['CAL']&partition_sources['AUDIT']:raise ValueError('partition source overlap')
    for role,group in groups.items():
        if role in ROLES:
            allowed={q['id'] for q in sources[role]}
            if {r[0] for q in group for r in q['refs']}-allowed:raise ValueError('construction footprint crosses declared source group')
    p=RUN/'metadata/partition_mapping.private.json';write(p,mapping)
    write(PUB/'input_manifest.json',{'passed':True,'roles':roles,'source_cache_roles':source,'recipes':recipes,'construction_summaries':summaries,'partition':part,'partition_mapping':pin(p),'source_session_isolation':isolation,'new_mixed_roles':list(ROLES),'new_streams':540,'new_teacher_payloads':1080,'new_event_updates':0,'class_order':previous['class_order'],'A08_J_Base_truth_channel_semantics_unchanged':previous['A08_J_Base_truth_channel_semantics_unchanged'],'old_inner_mixed_for_calibration_or_AUDIT':False,'old_namespaces_written':False,'protected_media_forwards':0,'protected_labels_used_for_training_or_selection':False,'constructor_sourceclass_labels_and_new_references_created':True,'metadata_global_nonpath_fields_transient_parse_disclosed':True,'observed_old_source_kinds':{r:sorted({q['kind'] for q in qs[r]}) for r in ['selection','calibration']},'freeze_git':freeze['git_commit']})
    from echo_routing.temporal.event_risk_metrics import plan
    write(PUB/'readout_plan.json',plan());write(RUN/'runtimecoordination.json',{'phase':'metadata_frozen_before_q_outputs','model_pid':os.getpid(),'epoch':time.time(),'next_action':'synthetic205_cost/fullremaining_sources_admission_thenunique540teachers'})
    print({'metadata':True,'CAL_native':len(cal),'AUDIT_native':len(audit),'newstreams':540,'source_payloads':1080},flush=True)


def fit():
    freeze=verify_freeze('FIT_prepare_freeze.json');old=prior();training=old['training_inputs.json'];data=np.load(verify_pin(training['dataset']),allow_pickle=True).item();original_norm=read(verify_pin(training['normalizers']))['candidate']
    if sha(verify_pin(training['dataset']))!='b7ba5812e4ade5eb87339cef51603d3a710774d8a93b36072a8c6a168bec76ff' or len(data['candidate_rows'])!=4183 or [len(data['kind_indices'][k]) for k in ['native','constructed']]!=[265,3918]:raise ValueError('exact previously audited FIT candidate domain required')
    qs=sequences('fit');a=probability_reader('a08','fit');j=probability_reader('jepa','fit');c96=[];cursor=0;params=read(CFG)['source']['a08']['original_B3']
    for qi,q in enumerate(qs):
        pa,pj=a.load(qi),j.load(qi);b=decode_with_prob('b3',pa,{'decode':{'b3':params}})[0];actions,rejected=partition(b,propose(b,pa.argmax(1),pj.argmax(1)))
        for action in actions:
            row=data['candidate_rows'][cursor]
            if (row['sequence_index'],row['start'],row['end'],tuple(row['replacement']),row['old'],row['jepa'])!=(qi,action.start,action.end,action.replacement,action.old,action.jepa) or not np.array_equal(candidate_features(pa,pj,b,action,'J'),data['features'][cursor]) or effects(b,action,q['labels'],training['F_fit'])!=data['effects'][cursor]:raise ValueError('original candidate/order/x109/full-axis effects changed')
            c96.append(context_features(pa,pj,b,action));cursor+=1
    if cursor!=4183:raise ValueError('complete4183 candidate recount required')
    context=np.stack(c96);contextnorm=fit_context_normalizer(context);normalizers={'candidate':original_norm,'context':{k:v.tolist() for k,v in contextnorm.items()}};targets=unsafe_targets(data['effects']);plans={}
    for seed,p in training['plans'].items():
        plan=np.load(verify_pin(p),allow_pickle=True).item()['risk'];validate_draw_plan(plan,265,3918,int(seed));plans[seed]=plan
    output={'features':{'R1':event_matrix(data['features'],None,'R1',original_norm),'R2':event_matrix(data['features'],context,'R2',original_norm,contextnorm)},'unsafe':targets,'context96':context,'kind_indices':data['kind_indices'],'normalizers':normalizers,'old_candidate_rows':data['candidate_rows'],'old_effects':data['effects'],'old_x109':data['features'],'old_dataset_sha256':training['dataset']['sha256']}
    p=RUN/'training_dataset.private.npy';save_array(p,output);dataset=pin(p);p=RUN/'normalizers.private.json';write(p,normalizers);normpin=pin(p);drawpins={}
    for seed,plan in plans.items():
        p=RUN/f'draw_plan_seed{seed}.private.npy';save_array(p,plan);drawpins[seed]=pin(p)
    write(PUB/'training_inputs.json',{'passed':True,'dataset':dataset,'normalizers':normpin,'plans':drawpins,'candidates':4183,'kind_candidate_counts':{'native':265,'constructed':3918},'positive_unsafe_events':int(targets.sum()),'kind_unsafe_events':{k:int(targets[idx].sum()) for k,idx in data['kind_indices'].items()},'context_nonconstant_columns':int((context.std(0)>0).sum()),'features_dims':[205,205],'old_x109_byte_exact':True,'old_candidate_order_and_effects_exact':True,'old_dataset_sha256':training['dataset']['sha256'],'F_fit':training['F_fit'],'actual_event_optimizer_updates':0,'freeze_git':freeze['git_commit']})
    print({'FIT_candidates':4183,'new_context':96,'unsafe_events':int(targets.sum()),'new_optimizer_updates':0},flush=True)


def main():
    p=execution_parser(__doc__);p.add_argument('--stage',choices=['metadata','FIT'],default='metadata');p.add_argument('--metadata-only',action='store_true');args=p.parse_args()
    if not authorize(args):return 0
    (metadata if args.stage=='metadata' else fit)();return 0
if __name__=='__main__':raise SystemExit(main())
