#!/usr/bin/env python3
"""A61 new-only metadata and cached FIT full-query/local-risk arrays."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,yaml
from scripts.safety_common import *
from echo_routing.temporal.safety_data import construct_confirmation,prepare_training,plans
from echo_routing.temporal.safety_readout_plan import plan
from echo_routing.temporal.anchor_cache import ProbabilityCache
from echo_routing.temporal.anchor_inputs import verify_native_groups,workload
from echo_routing.temporal.windowed import FAMILY5


def metadata():
    freeze=verify_freeze('preparation_freeze.json');cfg=read(CFG);verify_pin(cfg['card'])
    if (PUB/'input_manifest.json').exists():raise ValueError('metadata already frozen; never regenerate')
    old={name:read(verify_pin(pin)) for name,pin in cfg['prior_A60'].items()}
    previous=old['input_manifest.json']
    if not previous['passed'] or not old['resources.json']['all_authorized_A60_stages_complete']:raise ValueError('priorA60 not closed/qualified')
    mapping=cfg['channel_mapping'];ontology=yaml.safe_load(verify_pin(mapping['ontology']).read_text())
    if ontology['mapping_version']!='family5_v1' or ontology['family5_classes']!=FAMILY5 or FAMILY5!=mapping['classes']:raise ValueError('sourcechannel/Base/GT family order differs')
    feature=ROOT/mapping['training_index_mapping']
    if sha(feature)!=mapping['training_index_mapping_sha256'] or read(feature)['label_mapping']!='family5_v1 (configs/labels.yaml)':raise ValueError('qualified originaltraining channel semantics changed')
    roles=dict(previous['roles']);source_roles={k:dict(v) for k,v in previous['source_cache_roles'].items()};groups={r:np.load(verify_pin(pin),allow_pickle=True).tolist() for r,pin in roles.items()}
    for row in old['cache_outer_manifest.json']['records']:
        role=row['role'];folder=ROOT/'runs/a60_jepa_proposals_v1/cache'/row['source']/role
        source_roles.setdefault(role,{})[row['source']]={**row,'path':str(folder.relative_to(ROOT))}
    records=[]
    for role,qs in groups.items():
        if len({q['id'] for q in qs})!=len(qs) or any(len(q['refs'])!=len(q['labels']) for q in qs):raise ValueError('originalsequence order/axis invalid')
        for source in ['a08','jepa']:
            row=source_roles[role][source];folder=ROOT/row['path']
            if sha(folder/'committed.jsonl')!=row['journal_sha256'] or row['input_sha256']!=roles[role]['sha256']:raise ValueError('prior source journal or complete input identity changed')
            cache=ProbabilityCache(folder,row['identity'],[len(q['refs']) for q in qs],readonly=True)
            records.append({'source':source,'role':role,'sequences':cache.cursor,'posterior_positions':sum(len(q['refs']) for q in qs),'journal_sha256':row['journal_sha256'],'model_forwards':0})
    natives={r:[q for q in groups[r] if q['kind']=='native'] for r in ['fit','inner_cal','selection','calibration']};isolation=verify_native_groups(natives)
    if [len(natives[r]) for r in ['fit','inner_cal']]!=[2736,628]:raise ValueError('originalnativepool split differs')
    recipes={};summaries={}
    for half in ['selection','calibration']:
        role='a61_confirmation_'+half;qs,recipe,summary=construct_confirmation(natives[half],120,half)
        path=RUN/'metadata'/f'{role}.private.npy';save_array(path,np.asarray(qs,dtype=object));roles[role]={**array_pin(path),**workload(qs),'native':0}
        path=RUN/'metadata'/f'{role}_recipes.private.json';write(path,recipe);recipes[role]=array_pin(path);summaries[role]=summary
        if {ref[0] for q in qs for ref in q['refs']}-{q['id'] for q in natives[half]}:raise ValueError('construction source outside authorized half')
    write(PUB/'input_manifest.json',{'passed':True,'roles':roles,'source_cache_roles':source_roles,'recipes':recipes,'construction_summaries':summaries,'native_split_isolation':isolation,'class_order':FAMILY5,'A08_J_Base_truth_channel_semantics_unchanged':True,'old_namespaces_written':False,'protected_media_forwards':0,'protected_labels_used_for_training_or_selection':False,'new_model_forwards':0,'fresh_confirmation_metrics_before_head_freeze':False,'construction_only_sourceclass_metadata_and_fresh_labels_created':True,'inner_role':'all928 diagnostic/calibration;not independent safety validation','freeze_git':freeze['git_commit']})
    write(PUB/'source_cache_qualification.json',{'passed':True,'source_roles':len(records),'records':records,'all_original_axes_N5_payload_checksums_normalization_verified':True,'old_source_weight_or_neural_requalification_forwards':0,'model_forwards':0,'A60resources_sha256':cfg['prior_A60']['resources.json']['sha256'],'A60collector_directly_waited_exit':old['collector_OS_exit_receipt.json']['observed_worker_exit_code'],'class_order':FAMILY5})
    write(PUB/'readout_plan.json',plan());print({'metadata':True,'old_source_roles':len(records),'newstreams':240},flush=True)


def fit():
    freeze=verify_freeze('training_prepare_freeze.json')
    if (PUB/'training_inputs.json').exists() or not read(PUB/'source_cache_qualification.json')['passed']:raise ValueError('fresh actualFIT preparation required')
    start=time.perf_counter();qs,a,j,b,sources=role_values('fit');data=prepare_training(qs,a,j,b)
    if data['FIT_operating_counts']!={'M':5482,'P':5916,'T':6379} or data['F_fit']!=.8917446116307443:raise ValueError('originalFIT baseline operating point changed')
    path=RUN/'training_dataset.private.npy';save_array(path,data);dataset=array_pin(path)
    path=RUN/'normalizers.private.json';write(path,{name:{k:v.tolist() for k,v in norm.items()} for name,norm in data['normalizers'].items()});norms=array_pin(path)
    frozen_plans=plans(data);pins={}
    for seed,value in frozen_plans.items():
        path=RUN/f'draw_plans_seed{seed}.private.npy';save_array(path,value);pins[seed]=array_pin(path)
    c=np.asarray([len(g) for g in data['groups']]);K=np.diff(data['position_offsets']);Ts=np.asarray([row['T'] for row in data['candidate_rows']]);physicalposition=len(data['position_labels'])
    write(PUB/'training_inputs.json',{'passed':True,'dataset':dataset,'normalizers':norms,'plans':pins,'utility_scale':data['scale'],'source_cache_identities':sources,'F_fit':data['F_fit'],'FIT_operating_counts':data['FIT_operating_counts'],'candidates':len(data['targets']),'all_candidates':data['all_candidates'],'G_rejected_candidates':data['G_rejected_candidates'],'kind_candidate_counts':{k:len(v) for k,v in data['kind_indices'].items()},'kind_nonempty_query_counts':{k:len(v) for k,v in data['kind_groups'].items()},'all_source_sequences':len(qs),'empty_queries':int((c==0).sum()),'changed_position_training_multiset':physicalposition,'feature_dims':[109,47],'utility_features_nonconstant':int(np.count_nonzero(data['features'].std(0))),'position_features_nonconstant':int(np.count_nonzero(data['position_features'].std(0))),'max_C':int(c.max()),'max_C2':int((c*c).max()),'max_K':int(K.max()),'max_T_times_C':int(max(len(q['labels'])*len(g) for q,g in zip(qs,data['groups']))),'total_T_times_C':int(Ts.sum()),'target_ranges':{name:[min(row[name] for row in data['effects']),max(row[name] for row in data['effects'])] for name in ['u','v','H','R','dFP']},'full_FIT_prepare_seconds':time.perf_counter()-start,'inner_or_outer_used_for_targets_or_normalizers_or_plans':False,'formal_optimizer_updates':0,'freeze_git':freeze['git_commit']})
    print({'FIT_candidates':len(data['targets']),'position_multiset':physicalposition,'max_C':int(c.max()),'max_K':int(K.max())},flush=True)

if __name__=='__main__':
    if sys.argv[1:]==['metadata']:metadata()
    elif sys.argv[1:]==['fit']:fit()
    else:raise ValueError('explicit metadata/fit stage required')
