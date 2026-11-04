#!/usr/bin/env python3
"""A60 qualified read-only old caches and construction-only fresh metadata."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,yaml
from scripts.proposal_common import *
from echo_routing.temporal.proposal_data import construct_confirmation
from echo_routing.temporal.anchor_inputs import verify_native_groups,workload
from echo_routing.temporal.anchor_cache import ProbabilityCache
from echo_routing.temporal.proposal_readout_plan import plan
from echo_routing.temporal.windowed import FAMILY5


def main():
    freeze=verify_freeze('preparation_freeze.json');cfg=read(CFG)
    if (PUB/'input_manifest.json').exists():raise ValueError('prospective metadata already sealed; never regenerate')
    verify_pin(cfg['card']);prior={k:read(verify_pin(p)) for k,p in cfg['prior_A59'].items()}
    if not prior['input_manifest.json']['passed']:raise ValueError('prior source/input qualification incomplete')
    labels=yaml.safe_load(verify_pin(cfg['channel_mapping']['ontology']).read_text())
    feature_path=ROOT/cfg['channel_mapping']['training_index_mapping']
    if sha(feature_path)!=cfg['channel_mapping']['training_index_mapping_sha256']:raise ValueError('qualified family5 training index manifest changed')
    feature=read(feature_path)
    if labels['mapping_version']!='family5_v1' or labels['family5_classes']!=FAMILY5 or FAMILY5!=cfg['channel_mapping']['classes'] or feature['label_mapping']!='family5_v1 (configs/labels.yaml)':raise ValueError('A08/J direct replacement channel family mapping differs from Base labels')
    roles={};source_roles={};groups={};records=[]
    old=prior['input_manifest.json'];cached=prior['cache_fit_manifest.json']['records']+prior['cache_outer_manifest.json']['records']
    for role,pin in old['roles'].items():
        qs=np.load(verify_pin(pin),allow_pickle=True).tolist();groups[role]=qs;roles[role]=pin;source_roles[role]={}
        if len({q['id'] for q in qs})!=len(qs) or any(len(q['refs'])!=len(q['labels']) for q in qs):raise ValueError('complete unique original axes required')
        for source in ['a08','jepa']:
            row=next(r for r in cached if r['source']==source and r['role']==role)
            folder=ROOT/'runs/a59_anchor_edits_v1/cache'/source/role
            if row['input_sha256']!=pin['sha256'] or sha(folder/'committed.jsonl')!=row['journal_sha256'] or not row['complete']:raise ValueError('prior complete source/ref/gamma/order identity changed')
            cache=ProbabilityCache(folder,row['identity'],[len(q['refs']) for q in qs],readonly=True)
            source_roles[role][source]={'path':str(folder.relative_to(ROOT)),'identity':cache.identity,'journal_sha256':sha(cache.journal),'input_sha256':pin['sha256'],'complete':True,'committed_sequences':cache.cursor}
            records.append({'role':role,'source':source,'complete':True,'sequences':cache.cursor,'posterior_positions':sum(len(q['refs']) for q in qs),'model_forwards':0})
    natives={r:[q for q in groups[r] if q['kind']=='native'] for r in ['fit','inner_cal','selection','calibration']}
    isolation=verify_native_groups(natives)
    if [len(natives[r]) for r in ['fit','inner_cal']]!=[2736,628]:raise ValueError('declared native split changed')
    recipes={};summaries={}
    for half in ['selection','calibration']:
        name='a60_confirmation_'+half;qs,recipe,summary=construct_confirmation(natives[half],120,half)
        path=RUN/'metadata'/f'{name}.private.npy';save_array(path,np.asarray(qs,dtype=object));roles[name]={**array_pin(path),**workload(qs),'native':0}
        path=RUN/'metadata'/f'{name}_recipes.private.json';write(path,recipe);recipes[name]=array_pin(path);summaries[name]=summary
        if {r[0] for q in qs for r in q['refs']} - {q['id'] for q in natives[half]}:raise ValueError('new bank source outside declared outer half')
    write(PUB/'input_manifest.json',{'passed':True,'roles':roles,'source_cache_roles':source_roles,'recipes':recipes,'construction_summaries':summaries,'native_split_isolation':isolation,'prior_A59_input_manifest_sha256':cfg['prior_A59']['input_manifest.json']['sha256'],'old_namespace_writes':0,'model_forwards':0,'formal_optimizer_updates':0,'construction_only_source_class_metadata_read_and_fresh_labels_created':True,'fresh_confirmation_metrics_or_targets_before_head_freeze':False,'protected_media_forwards':0,'protected_labels_used_for_training_or_selection':False,'global_manifest_nonpath_fields_transiently_parsed':False,'A08_historical_inner_exposure_not_OOF':True})
    write(PUB/'source_cache_qualification.json',{'passed':True,'records':records,'exact16_prior_source_roles':len(records)==16,'all_probabilities_full_N5_finite_normalized_and_checksum_verified':True,'class_order':FAMILY5,'A08_J_Base_label_channel_semantics_equal':True,'evidence':cfg['channel_mapping'],'old400_214_state_checks_reused_not_rerun':True,'new_model_forwards':0,'prior_resources_sha256':cfg['prior_A59']['resources_final.json']['sha256'],'prior_terminal_sha256':cfg['prior_A59']['terminal_exit_receipt_final.json']['sha256'],'preparation_freeze_git':freeze['git_commit'],'epoch':time.time()})
    write(PUB/'readout_plan.json',plan());print({'prepared':True,'source_roles':len(records),'new_confirmation_sequences':240,'new_model_forwards':0},flush=True)

if __name__=='__main__':main()
