#!/usr/bin/env python3
"""A60 independent124cached recounds and all144 predeclared paired comparisons."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import sys,time,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd
from scripts.proposal_common import RUN,PUB,CFG,read,write,sha,verify_pin,verify_freeze
from scripts.readout_jepa_proposals import contexts,role_for,pair_cache,validate_source_banks
from scripts.train_jepa_proposals import role_values
from scripts.cache_jepa_proposals import original_a08
from echo_routing.temporal.baselines.registry import decode_with_prob
from echo_routing.temporal.structured_metrics import compose_stress_predictions
from echo_routing.temporal.proposal_metrics import evaluate,paired,action_summary
from echo_routing.temporal.proposal_readout_plan import plan as fixed_plan,TABLES,HALVES,BASE
from echo_routing.temporal.structured_metrics import verify_reference
from echo_routing.temporal.structured_uncertainty import prepare_system
from echo_routing.temporal.anchor_edits import majority


def main():
    freeze=verify_freeze('aggregate_freeze.json');verify_freeze('readout_freeze.json');cfg=read(CFG)
    manifest=read(PUB/'models_manifest.json');result=read(PUB/'outer_metrics.json');plan=read(PUB/'readout_plan.json')
    if plan!=fixed_plan() or result['metric_rows']!=124 or result['policy_metric_rows']!=96 or result['reference_metric_rows']!=28:
        raise ValueError('exact fixed124/144 plan required')
    for field,path in [('models_manifest_sha256','models_manifest.json'),('readout_plan_sha256','readout_plan.json'),
                       ('source_outer_cache_manifest_sha256','cache_outer_manifest.json')]:
        if result[field]!=sha(PUB/path):raise ValueError('readout source/model/plan pin changed')
    predictions=np.load(verify_pin(result['private_predictions']),allow_pickle=True).item();ctx=contexts(cfg)
    banks={role_for(t,h):role_values(role_for(t,h)) for t in TABLES for h in HALVES}
    validate_source_banks(read(PUB/'cache_outer_manifest.json'),banks)
    modelrows={(r['seed'],r['arm']):r for r in manifest['models']}
    expected={(r['table'],r['half'],r['system'],r['role']) for r in plan['rows']}
    actual=[(r['table'],r['half'],r['system'],r['role']) for r in result['rows']]
    if len(actual)!=124 or set(actual)!=expected:raise ValueError('missing/duplicate/unplanned metric row')
    systems={t:{h:{} for h in HALVES} for t in TABLES};tables={};prepared={};verified_cells=set();cached_values={};inputs=read(PUB/'training_inputs.json')
    for row in result['rows']:
        table,half,name=row['table'],row['half'],row['system'];pred=predictions[table][half][name]
        role=role_for(table,half);qs,a,j,b,sources=banks[role]
        if row['role']=='reference':
            if name.startswith('A08_s') and name.endswith('_original_B3'):
                seed=int(name.split('_')[1][1:])
                if table=='main':
                    source,params=original_a08(half,qs,seed,cfg)
                    expected_pred={q['id']:decode_with_prob('b3',p,{'decode':{'b3':params}})[0] for q,p in zip(qs,source)}
                else:
                    if seed!=0:raise ValueError('unplanned stress reference seed')
                    expected_pred={q['id']:p for q,p in zip(qs,b)}
            else:
                values=a if name=='A08_s0_original_raw' else j if name=='JEPA_core_fixed6000_raw' else None
                if values is None:raise ValueError('unplanned reference producer')
                expected_pred={q['id']:p.argmax(1) for q,p in zip(qs,values)}
            if table!='main':
                expected_pred=compose_stress_predictions(predictions['main'][half][name],expected_pred,banks[half][0],qs)
            if set(pred)!=set(expected_pred) or any(not np.array_equal(pred[k],v) for k,v in expected_pred.items()):
                raise ValueError('reference labels disagree with pinned original source posterior/decoder')
        if row['role']=='policy':
            modelrow=modelrows[(row['seed'],row['arm'])]
            if row['weight_sha256']!=modelrow['final_sha256'] or sha(ROOT/modelrow['final_path'])!=row['weight_sha256']:
                raise ValueError('final checkpoint binding changed')
            cell=(row['weight_sha256'],role)
            if cell not in cached_values:
                cache=pair_cache(modelrow,role,qs,a,j,b,sources,readonly=True)
                cached_values[cell]=(cache,[cache.load(i) for i in range(len(qs))])
            cache,values=cached_values[cell]
            if sha(cache.journal)!=row['policy_cache']['journal_sha256'] or cache.identity!=row['policy_cache']['identity']:
                raise ValueError('shared policy cache journal changed')
            for i,q in enumerate(qs):
                value=values[i]
                if not np.array_equal(pred[q['id']],value[row['policy']]['prediction']):raise ValueError('private labels/shared scores mismatch')
                if majority(pred[q['id']])!=majority(b[i]):raise ValueError('whole-task guard failed')
            actual_stats=action_summary(qs,a,j,b,values,row['policy'],inputs['F_fit'])
            verify_reference(actual_stats,row['action_statistics_source_role'])
            if table=='main':fullstats=actual_stats
            else:
                mq,ma,mj,mb,msources=banks[half];maincell=(row['weight_sha256'],half)
                if maincell not in cached_values:
                    mc=pair_cache(modelrow,half,mq,ma,mj,mb,msources,readonly=True);cached_values[maincell]=(mc,[mc.load(i) for i in range(len(mq))])
                mv=cached_values[maincell][1];lookup={q['id']:(q,x,y,z,v) for q,x,y,z,v in zip(mq,ma,mj,mb,mv) if q['kind']!='multi'}
                lookup.update({q['id']:(q,x,y,z,v) for q,x,y,z,v in zip(qs,a,j,b,values)})
                group=[lookup[q['id']] for q in ctx[table][half]['sequences']]
                fullstats=action_summary(*[list(v) for v in zip(*group)],row['policy'],inputs['F_fit'])
            verify_reference(fullstats,row['action_statistics_full_task'])
            if row['candidate_count']!=fullstats['candidate_count'] or row['accepted_actions']!=fullstats['accepted_actions']:
                raise ValueError('full-task vs source-role candidate/acceptance counts mislabeled')
            # Separate count algebra for interval vs changed-position reporting.
            accepted=[(base,v[row['policy']]['accepted'],v[row['policy']]['prediction']) for base,v in zip(b,values) if v[row['policy']]['accepted'] is not None]
            if (sum(x.end-x.start for _,x,_ in accepted)!=actual_stats['accepted_lengths']['interval_length']['sum']
                or sum(int(np.count_nonzero(prediction!=base)) for base,_,prediction in accepted)!=actual_stats['accepted_lengths']['changed_positions']['sum']
                or sum(x.start==0 and x.end==len(base) for base,x,_ in accepted)!=actual_stats['accepted_lengths']['whole_axis_count']):
                raise ValueError('independent accepted interval/changed/whole-axis algebra failed')
            verified_cells.add(cell)
        if table!='main':
            for q in ctx['main'][half]['sequences']:
                if q['kind']!='multi' and not np.array_equal(pred[q['id']],predictions['main'][half][name][q['id']]):
                    raise ValueError('stress native/pair predictions were changed')
        metric,tab=evaluate(pred,ctx[table][half]);verify_reference(metric,{k:row[k] for k in metric})
        if metric['counts']['native_sessions']!=36 or metric['counts']['multi_streams']!=120:raise ValueError('fixed group denominators changed')
        systems[table][half][name]=metric;tables[(table,half,name)]=tab
        folder=RUN/'aggregate'/table/half/name;folder.mkdir(parents=True,exist_ok=True)
        for key,frame in tab.items():frame.to_csv(folder/(key+'.private.csv'),index=False)
    if len(verified_cells)!=32:raise ValueError('complete32physical pair-score cache cells required')
    comparisons=[]
    def prepare(t,h,n):
        key=(t,h,n)
        if key not in prepared:prepared[key]=prepare_system(predictions[t][h][n],ctx[t][h]['sequences'],tables[key])
        return prepared[key]
    for item in plan['comparisons']:
        table,half,left,right=item['table'],item['half'],item['left'],item['right']
        a,b=systems[table][half][left],systems[table][half][right]
        keyset=set(a)&set(b)
        delta={k:a[k]-b[k] if a[k] is not None and b[k] is not None else None for k in keyset
               if (a[k] is None or type(a[k]) in [int,float]) and (b[k] is None or type(b[k]) in [int,float])}
        stats=paired(prepare(table,half,left),prepare(table,half,right),plan['bootstrap_replicates'],plan['bootstrap_seed'])
        comparisons.append({**item,'delta':delta,**stats,
            'native_confusion_delta':(np.asarray(a['native_confusion'])-np.asarray(b['native_confusion'])).tolist(),
            'native_per_class_f1_delta':(np.asarray(a['native_per_class_f1'])-np.asarray(b['native_per_class_f1'])).tolist()})
        if len(comparisons)%4==0:
            write(PUB/'paired_progress.json',{'completed':len(comparisons),'declared':144,'model_pid':os.getpid(),'epoch':time.time()})
            write(RUN/'runtimecoordination.json',{'phase':'cached_A60_144paired_aggregation','comparisons_done':len(comparisons),
                  'comparisons':144,'model_pid':os.getpid(),'epoch':time.time(),'next_action':'all144 then exact action-space oracle and resource closure'})
    if len(comparisons)!=144:raise ValueError('all144 fixed paired comparisons required')
    write(PUB/'summary.json',{'status':'completed exploratory anchored-edit matrix','systems':systems,
        'paired_comparisons':comparisons,'metric_rows':124,'policy_metric_rows':96,'reference_metric_rows':28,
        'paired_comparison_count':144,'formal_models':4,'formal_optimizer_updates':24000,
        'primary_contrast':'J minus A separately for O/E/R byseed/half/table;Jproposalsetcommon;ERdomainandREveto controls;allnativeguardmechanical',
        'all124_metrics_independently_recounted':True,'all32_shared_policy_score_journals_and_fullaxes_verified':True,'all96_policy_source_and_fulltask_length_fraction_changed_wholeaxis_harm_scores_recounted':True,
        'no_inner_checkpoint_or_outer_model_selection':True,'stress_native_pair_predictions_exactly_reuse_main':True,
        'models_manifest_sha256':sha(PUB/'models_manifest.json'),'outer_metrics_sha256':sha(PUB/'outer_metrics.json'),
        'readout_plan_sha256':sha(PUB/'readout_plan.json'),'aggregate_freeze_git':freeze['git_commit'],
        'OPEN_reused_development':True,'old_A08_inner_exposure_not_OOF':True,'sessions_not_confirmed_patients':True,
        'conditional_shared_source_constructed_bank_CIs_not_population_validation':True,
        'no_manual_annotation_resources_or_clinical_validation':True,'protected_media_forwards':0,
        'protected_labels_used_for_training_or_selection':False,'test_Pediatric_Unity_quarantine_used':False,
        'posterior_evidence_is_not_latent_anatomy_phase_evidence':True,
        'full_system_runtime_requires_A08_and_JEPA_for_both_A_and_J_proposal_controls':True,'epoch':time.time()})
    pd.json_normalize(result['rows']).to_csv(PUB/'metrics.csv',index=False)
    write(RUN/'runtimecoordination.json',{'phase':'all124_144_scientific_matrix_complete_pending_oracle_package_closure',
          'model_pid':None,'epoch':time.time(),'next_action':'frozen action-domain oracle/figures then actualresource/exit/publicclosure'})

if __name__=='__main__':main()
