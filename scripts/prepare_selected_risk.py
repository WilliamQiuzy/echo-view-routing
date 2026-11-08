#!/usr/bin/env python3
"""A63 fixed source split/recipes before auxiliary outputs; no NN."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['PYTHONDONTWRITEBYTECODE']='1'
import sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.selected_risk_common import *
from echo_routing.temporal.selected_risk_supply import partition_sources,session_of
from echo_routing.temporal.selected_risk_data import construct
from echo_routing.temporal.anchor_inputs import verify_native_groups,workload


def metadata():
    freeze=verify_freeze('metadata');cpu_only()
    if (PUB/'input_manifest.json').exists():raise ValueError('immutable metadata already exists; no redraw')
    basis=read(verify_pin(read(CFG)['basis']))
    for original in basis['predecessor_public_and_source_pins']:prior_path(original)
    prior=read(prior_path('docs/results/2026-11-04_anchor_context_risk/input_manifest.json'))
    if prior['class_order']!=['PLAX','PSAX','A4C','A5C','SC4C'] or prior['A08_J_Base_truth_channel_semantics_unchanged'] is not True:raise ValueError('fixed five-channel semantics binding required')
    roles=dict(prior['roles']);source={k:dict(v) for k,v in prior['source_cache_roles'].items()}
    producer=read(prior_path('docs/results/2026-11-04_anchor_context_risk/producer_manifest.json'))
    for r in producer['records']:source.setdefault(r['role'],{})[r['source']]=dict(r)
    native={}
    for role in ['fit','inner_cal','selection','calibration']:
        # Existing constructor metadata includes labels; only original native
        # members are used. No old mixed reference enters fold targets/training.
        native[role]=[q for q in load_array(roles[role]) if q['kind']=='native']
    if len(native['fit'])!=2736:raise ValueError('exact original2736 native FIT required')
    isolation=verify_native_groups(native);partition=partition_sources(native['fit']);groups={};recipes={};summaries={}
    private_partition={'sessions':partition['sessions'],'native_original_FIT_indices':partition['native_indices'],'train_fold_for_held':[1,0]}
    for fold in [0,1]:groups[f'F{fold}_native']=[native['fit'][i] for i in partition['native_indices'][fold]]
    sources=[groups['F0_native'],groups['F1_native'],native['selection'],native['calibration']]
    names=['F0_mixed','F1_mixed','a63_confirmation_selection','a63_confirmation_calibration']
    for domain,(role,bank) in enumerate(zip(names,sources)):
        sequences,rr,info=construct(bank,domain);allowed={q['id'] for q in bank}
        if any({r[0] for r in q['refs']}-allowed for q in sequences):raise ValueError('new recipe reference crosses its frozen source fold/half')
        groups[role]=sequences;rp=RUN/'metadata'/(role+'_recipes.private.json');write(rp,rr);recipes[role]=pin(rp);summaries[role]=info
    groups['FIT_native']=native['fit']
    for role,qs in groups.items():
        p=RUN/'metadata'/(role+'.private.npy');save_array(p,np.asarray(qs,dtype=object));roles[role]={**pin(p),**workload(qs),'native':sum(q['kind']=='native' for q in qs)}
    pp=RUN/'metadata/partition_mapping.private.json';write(pp,private_partition)
    for role,rr in source.items():
        for source_name,row in rr.items():
            if sha(ROOT/row['path']/'committed.jsonl')!=row['journal_sha256'] or row['input_sha256']!=roles[role]['sha256']:raise ValueError('readonly predecessor teacher journal/input changed')
    publicpart={k:v for k,v in partition.items() if k not in ['sessions','native_indices']};publicpart.update({'sessions':[201,201],'native_counts':[len(x) for x in partition['native_indices']],'source_overlap':0,'generation_retries':0})
    write(PUB/'input_manifest.json',{'passed':True,'roles':roles,'source_cache_roles':source,'recipes':recipes,'construction_summaries':summaries,'partition_mapping':pin(pp),'partition':publicpart,'source_session_isolation':isolation,'FIT_new_streams':3000,'FIT_new_payloads':6000,'CONFIRM_new_streams_if_coverage_pass':240,'CONFIRM_new_payloads_if_coverage_pass':480,'source_updates_at_production':{'FIT':{'aux':0,'formal':0},'CONFIRM':{'aux':48000,'formal':0}},'class_order':prior['class_order'],'A08_J_Base_truth_channel_semantics_unchanged':True,'old1500_mixed_not_used_in_aux_training':True,'old_namespaces_written':False,'constructor_reference_label_bytes_parsed_not_outer_outcomes':True,'new_source_forwards':0,'new_optimizer_updates':0,'freeze_git':freeze['git_commit']})
    write(RUN/'runtimecoordination.json',{'phase':'all_recipes_frozen_before_aux_outputs','PID':os.getpid(),'epoch':time.time(),'next':'cost/fullremaining sources admission, then unique FIT dualproducer'})
    print({'metadata_complete':True,'fold_native_counts':publicpart['native_counts'],'fold_sessions':[201,201],'FIT_mixed':3000,'confirm_recipes_frozen_not_forwarded':240},flush=True)

def main():
    p=execution_parser(__doc__);a=p.parse_args()
    if not authorize(a):return 0
    metadata();return 0
if __name__=='__main__':raise SystemExit(main())
