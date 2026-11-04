#!/usr/bin/env python3
"""A61 independent314 original-label recounts and all240 fixed paired comparisons."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd
from scripts.safety_common import *
from scripts.readout_action_safety import contexts,role_for,bank,source_banks,current_cache,historical_cache,full_values,legacy_full_values
from scripts.cache_action_safety import original_a08
from scripts.audit_action_safety import equivalent
from echo_routing.temporal.safety_metrics import evaluate,prepare_system,paired,action_summary
from echo_routing.temporal.proposal_metrics import action_summary as old_action_summary
from echo_routing.temporal.safety_readout_plan import plan as fixed_plan,TABLES,HALVES,BASE,SEEDS
from echo_routing.temporal.structured_metrics import compose_stress_predictions,verify_reference
from echo_routing.temporal.baselines.registry import decode_with_prob
from echo_routing.temporal.anchor_edits import majority


def main():
    freeze=verify_freeze('aggregate_freeze.json');verify_freeze('readout_freeze.json');models=read(PUB/'models_manifest.json');result=read(PUB/'outer_metrics.json');plan=read(PUB/'readout_plan.json');cfg=read(CFG);inputs=read(PUB/'training_inputs.json')
    if plan!=fixed_plan() or (result['metric_rows'],result['policy_metric_rows'],result['reference_metric_rows'])!=(314,240,74):raise ValueError('complete frozen314/240/60/56 graph required')
    for field,name in [('models_manifest_sha256','models_manifest.json'),('calibration_sha256','calibration.json'),('readout_plan_sha256','readout_plan.json'),('source_outer_cache_manifest_sha256','cache_outer_manifest.json')]:
        if result[field]!=sha(PUB/name):raise ValueError('readout model/threshold/plan/source pin changed')
    predictions=np.load(verify_pin(result['private_predictions']),allow_pickle=True).item();ctx=contexts(cfg);banks={role_for(t,h):bank(role_for(t,h)) for t in TABLES for h in HALVES};source_banks(read(PUB/'cache_outer_manifest.json'),banks)
    expected={(r['table'],r['half'],r['system'],r['role']) for r in plan['rows']};actual=[(r['table'],r['half'],r['system'],r['role']) for r in result['rows']]
    if len(actual)!=314 or set(actual)!=expected:raise ValueError('missing/duplicate/unplanned metric row')
    newmodels={(r['seed'],r['arm']):r for r in models['models']};riskrows={r['seed']:r for r in models['models'] if r['role']=='risk'};taus={(r['seed'],r['arm']):r['tau'] for r in models['thresholds']}
    oldmodels=read(verify_pin(cfg['prior_A60']['models_manifest.json']));legacy={r['seed']:r for r in oldmodels['models'] if r['arm']=='J'}
    cached={};tables={};prepared={};systems={t:{h:{} for h in HALVES} for t in TABLES};newcells=set();oldcells=set();original=read(ROOT/'docs/results/2026-11-02_dense_target_sequence/A08_reference_metrics.json')['systems']
    def values(row,role,historical=False):
        key=(historical,row['final_sha256'] if historical else row['sha256'],role)
        if key not in cached:
            c=historical_cache(row,role,banks[role],readonly=True) if historical else current_cache(row,riskrows[row['seed']],taus[(row['seed'],row['arm'])],role,banks[role],readonly=True)
            cached[key]=(c,[c.load(i) for i in range(len(banks[role][0]))])
        return cached[key]
    for row in result['rows']:
        table,half,name=row['table'],row['half'],row['system'];pred=predictions[table][half][name];basepred=predictions[table][half][BASE];role=role_for(table,half);qs,a,j,b,_=banks[role]
        if row['role']=='reference':
            if name.startswith('A08_s') and name.endswith('_original_B3'):
                seed=int(name.split('_')[1][1:])
                if table=='main':
                    p,params=original_a08(half,qs,seed,cfg);expected_pred={q['id']:decode_with_prob('b3',z,{'decode':{'b3':params}})[0] for q,z in zip(qs,p)}
                else:
                    if seed!=0:raise ValueError('unplanned stress source seed')
                    expected_pred={q['id']:z for q,z in zip(qs,b)}
            else:
                p=a if name=='A08_s0_original_raw' else j if name=='JEPA_core_fixed6000_raw' else None
                if p is None:raise ValueError('unplanned raw reference')
                expected_pred={q['id']:z.argmax(1) for q,z in zip(qs,p)}
            if table!='main':expected_pred=compose_stress_predictions(predictions['main'][half][name],expected_pred,banks[half][0],qs)
            if set(pred)!=set(expected_pred) or any(not np.array_equal(pred[k],v) for k,v in expected_pred.items()):raise ValueError('reference labels differ from original posterior/decoder')
        elif row['role'] in ['policy','A60_historical']:
            old=row['role']=='A60_historical';model=legacy[row['old_seed']] if old else newmodels[(row['seed'],row['arm'])];weight=model['final_sha256'] if old else model['sha256'];path=model['final_path'] if old else model['path']
            if row['weight_sha256']!=weight or sha(ROOT/path)!=weight:raise ValueError('fixedmodel fileSHA changed')
            c,v=values(model,role,old)
            if row['policy_cache']['journal_sha256']!=sha(c.journal) or row['policy_cache']['identity']!=c.identity:raise ValueError('score journal/source identity changed')
            for q,base,z in zip(qs,b,v):
                if not np.array_equal(pred[q['id']],z[row['policy']]['prediction']) or majority(z[row['policy']]['prediction'])!=majority(base):raise ValueError('private labels/score/gate/guard differ')
            if old:
                source_stats=old_action_summary(qs,a,j,b,v,row['policy'],inputs['F_fit']);main_v=values(model,half,True)[1];group=legacy_full_values(table,half,v,main_v,banks,ctx[table][half]);fullstats=old_action_summary(*group,row['policy'],inputs['F_fit']);oldcells.add((weight,role))
                if row['old_seed']-2026110360!=row['replicate_seed']-2026110461 or not row['original_A60_selection_semantics']:raise ValueError('historical replicate/selection semantics changed')
            else:
                if row['risk_weight_sha256']!=riskrows[row['seed']]['sha256'] or row['tau']!=taus[(row['seed'],row['arm'])]:raise ValueError('sharedrisk/fixedthreshold changed')
                source_stats=action_summary(qs,b,v,row['policy'],row['arm'],inputs['F_fit']);main_v=values(model,half)[1];group=full_values(table,half,v,main_v,banks,ctx[table][half]);fullstats=action_summary(*group,row['policy'],row['arm'],inputs['F_fit']);newcells.add((weight,role))
                if row['candidate_count']!=fullstats['candidate_count'] or row['accepted_actions']!=fullstats['accepted_actions']:raise ValueError('source/fulltask candidate/accepted counts mislabeled')
            if not equivalent(source_stats,row['action_statistics_source_role']) or not equivalent(fullstats,row['action_statistics_full_task']):raise ValueError('original-axis target/risk/rank/length/harm descriptors differ')
        else:raise ValueError('unplanned endpoint role')
        if table!='main':
            for q in ctx['main'][half]['sequences']:
                if q['kind']!='multi' and not np.array_equal(pred[q['id']],predictions['main'][half][name][q['id']]):raise ValueError('stress native/pairs changed')
        metric,tab=evaluate(pred,ctx[table][half],basepred)
        if not equivalent(metric,{k:row[k] for k in metric}):raise ValueError('full original-label metric/CM/MPT/gross recount differs')
        if row['role']=='reference' and table=='main' and name.startswith('A08_s') and name.endswith('_original_B3'):
            reference=original[half][f"A08_original_B3_s{int(name.split('_')[1][1:])}"]
            verify_reference(metric,reference)
        systems[table][half][name]=metric;tables[(table,half,name)]=tab
        folder=RUN/'aggregate'/table/half/name;folder.mkdir(parents=True,exist_ok=True)
        for k,frame in tab.items():frame.to_csv(folder/(k+'.private.csv'),index=False)
    if len(newcells)!=80 or len(oldcells)!=20:raise ValueError('complete80new/20historical source score cells required')
    def prepared_system(t,h,n):
        key=(t,h,n)
        if key not in prepared:prepared[key]=prepare_system(predictions[t][h][n],predictions[t][h][BASE],ctx[t][h]['sequences'],tables[key])
        return prepared[key]
    comparisons=[]
    for item in plan['comparisons']:
        table,half,left,right=item['table'],item['half'],item['left'],item['right'];a,b=systems[table][half][left],systems[table][half][right]
        keys=set(a)&set(b);delta={k:a[k]-b[k] if a[k] is not None and b[k] is not None else None for k in keys if (a[k] is None or type(a[k]) in [int,float]) and (b[k] is None or type(b[k]) in [int,float])}
        stats=paired(prepared_system(table,half,left),prepared_system(table,half,right),plan['bootstrap_replicates'],plan['bootstrap_seed'])
        comparisons.append({**item,'delta':delta,**stats,'native_confusion_delta':(np.asarray(a['native_confusion'])-np.asarray(b['native_confusion'])).tolist(),'native_per_class_f1_delta':(np.asarray(a['native_per_class_f1'])-np.asarray(b['native_per_class_f1'])).tolist()})
        if len(comparisons)%4==0:
            write(PUB/'paired_progress.json',{'completed':len(comparisons),'declared':240,'model_pid':os.getpid(),'epoch':time.time()});write(RUN/'runtimecoordination.json',{'phase':'all240paired5000_originalaxis_aggregation','comparisons_done':len(comparisons),'comparisons':240,'model_pid':os.getpid(),'epoch':time.time(),'next_action':'finish240 then60safeoracles16oldG/56native/figures/actualclosure'})
    if len(comparisons)!=240:raise ValueError('all240 fixed comparisons required')
    write(PUB/'summary.json',{'status':'completed exploratory action safety/ranking matrix','systems':systems,'paired_comparisons':comparisons,'metric_rows':314,'policy_metric_rows':240,'reference_metric_rows':74,'paired_comparison_count':240,'formal_models':10,'formal_optimizer_updates':60000,'primary_system':'S-L-C','all314_metrics_independently_recounted':True,'all80_new_joint_score_caches_and20_historicalbindings_verified':True,'all240_policy_source_fulltask_effects_risk_rank_length_descriptors_recounted':True,'all40_historical_A60_controls_original_risk_filter_then_best_not_reinterpreted':True,'inner_calibration_not_independent_validation':True,'all10finals24inner8taus_frozen_before_outer':True,'no_checkpoint_seed_or_outer_threshold_selection':True,'stress_native_pairs_identical_to_main':True,'models_manifest_sha256':sha(PUB/'models_manifest.json'),'calibration_sha256':sha(PUB/'calibration.json'),'outer_metrics_sha256':sha(PUB/'outer_metrics.json'),'readout_plan_sha256':sha(PUB/'readout_plan.json'),'aggregate_freeze_git':freeze['git_commit'],'OPEN_reused_development':True,'old_A08_inner_exposure_not_OOF':True,'sessions_not_confirmed_patients':True,'conditional_shared_source_constructed_bank_CIs_not_population_validation':True,'no_manual_annotation_or_clinical_validation':True,'protected_media_forwards':0,'protected_labels_used_for_training_or_selection':False,'test_Pediatric_Unity_quarantine_used':False,'posterior_features_not_latent_anatomy_phase_evidence':True,'risk_evidence_not_probability_of_any_harm_or_finite_sample_guarantee':True,'historical_seed_mapping_is_replicate_index_not_equal_training_seed':True,'epoch':time.time()})
    pd.json_normalize(result['rows']).to_csv(PUB/'metrics.csv',index=False)
    write(RUN/'runtimecoordination.json',{'phase':'all314_240_complete_pending60oracle56native_package_closure','model_pid':None,'epoch':time.time(),'next_action':'freeze314/240 then60oracle16reproductions/56native/plots/resources'})
if __name__=='__main__':main()
