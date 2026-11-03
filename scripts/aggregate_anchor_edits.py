#!/usr/bin/env python3
"""A59 independent70cached recounds and all96 predeclared paired comparisons."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import sys,time,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd
from scripts.anchor_common import RUN,PUB,CFG,read,write,sha,verify_pin,verify_freeze
from scripts.readout_anchor_edits import contexts,role_for,pair_cache
from scripts.train_anchor_edits import role_values
from echo_routing.temporal.anchor_metrics import evaluate,paired
from echo_routing.temporal.anchor_readout_plan import plan as fixed_plan,TABLES,HALVES,BASE
from echo_routing.temporal.structured_metrics import verify_reference
from echo_routing.temporal.structured_uncertainty import prepare_system
from echo_routing.temporal.anchor_edits import majority


def main():
    freeze=verify_freeze('aggregate_freeze.json');verify_freeze('readout_freeze.json');cfg=read(CFG)
    manifest=read(PUB/'models_manifest.json');result=read(PUB/'outer_metrics.json');plan=read(PUB/'readout_plan.json')
    if plan!=fixed_plan() or result['metric_rows']!=70 or result['policy_metric_rows']!=48 or result['reference_metric_rows']!=22:
        raise ValueError('exact fixed70/96 plan required')
    for field,path in [('models_manifest_sha256','models_manifest.json'),('readout_plan_sha256','readout_plan.json'),
                       ('source_outer_cache_manifest_sha256','cache_outer_manifest.json')]:
        if result[field]!=sha(PUB/path):raise ValueError('readout source/model/plan pin changed')
    predictions=np.load(verify_pin(result['private_predictions']),allow_pickle=True).item();ctx=contexts(cfg)
    banks={role_for(t,h):role_values(role_for(t,h)) for t in TABLES for h in HALVES}
    modelrows={(r['seed'],r['arm']):r for r in manifest['models']}
    expected={(r['table'],r['half'],r['system'],r['role']) for r in plan['rows']}
    actual=[(r['table'],r['half'],r['system'],r['role']) for r in result['rows']]
    if len(actual)!=70 or set(actual)!=expected:raise ValueError('missing/duplicate/unplanned metric row')
    systems={t:{h:{} for h in HALVES} for t in TABLES};tables={};prepared={};verified_cells=set()
    for row in result['rows']:
        table,half,name=row['table'],row['half'],row['system'];pred=predictions[table][half][name]
        role=role_for(table,half);qs,a,j,b,sources=banks[role]
        if row['role']=='policy':
            modelrow=modelrows[(row['seed'],row['arm'])]
            if row['weight_sha256']!=modelrow['final_sha256'] or sha(ROOT/modelrow['final_path'])!=row['weight_sha256']:
                raise ValueError('final checkpoint binding changed')
            cache=pair_cache(modelrow,role,qs,a,b,sources,readonly=True)
            if sha(cache.journal)!=row['policy_cache']['journal_sha256'] or cache.identity!=row['policy_cache']['identity']:
                raise ValueError('shared policy cache journal changed')
            for i,q in enumerate(qs):
                value=cache.load(i)
                if not np.array_equal(pred[q['id']],value[row['policy']]['prediction']):raise ValueError('private labels/shared scores mismatch')
                if row['policy']=='G' and majority(pred[q['id']])!=majority(b[i]):raise ValueError('whole-task guard failed')
            verified_cells.add((row['weight_sha256'],role))
        if table!='main':
            for q in ctx['main'][half]['sequences']:
                if q['kind']!='multi' and not np.array_equal(pred[q['id']],predictions['main'][half][name][q['id']]):
                    raise ValueError('stress native/pair predictions were changed')
        metric,tab=evaluate(pred,ctx[table][half]);verify_reference(metric,{k:row[k] for k in metric})
        if metric['counts']['native_sessions']!=36 or metric['counts']['multi_streams']!=120:raise ValueError('fixed group denominators changed')
        systems[table][half][name]=metric;tables[(table,half,name)]=tab
        folder=RUN/'aggregate'/table/half/name;folder.mkdir(parents=True,exist_ok=True)
        for key,frame in tab.items():frame.to_csv(folder/(key+'.private.csv'),index=False)
    if len(verified_cells)!=24:raise ValueError('complete24physical pair-score cache cells required')
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
            write(PUB/'paired_progress.json',{'completed':len(comparisons),'declared':96,'model_pid':os.getpid(),'epoch':time.time()})
            write(RUN/'runtimecoordination.json',{'phase':'cached_anchor96paired_aggregation','comparisons_done':len(comparisons),
                  'comparisons':96,'model_pid':os.getpid(),'epoch':time.time(),'next_action':'all96 then exact action-space oracle and resource closure'})
    if len(comparisons)!=96:raise ValueError('all96 fixed paired comparisons required')
    write(PUB/'summary.json',{'status':'completed exploratory anchored-edit matrix','systems':systems,
        'paired_comparisons':comparisons,'metric_rows':70,'policy_metric_rows':48,'reference_metric_rows':22,
        'paired_comparison_count':96,'formal_models':4,'formal_optimizer_updates':24000,
        'primary_contrast':'J-G minus A-G separately byseed/half; raw-supported single edit; native guard is mechanical invariance',
        'all70_metrics_independently_recounted':True,'all24_shared_policy_score_journals_and_fullaxes_verified':True,
        'no_inner_checkpoint_or_outer_model_selection':True,'stress_native_pair_predictions_exactly_reuse_main':True,
        'models_manifest_sha256':sha(PUB/'models_manifest.json'),'outer_metrics_sha256':sha(PUB/'outer_metrics.json'),
        'readout_plan_sha256':sha(PUB/'readout_plan.json'),'aggregate_freeze_git':freeze['git_commit'],
        'OPEN_reused_development':True,'old_A08_inner_exposure_not_OOF':True,'sessions_not_confirmed_patients':True,
        'conditional_shared_source_constructed_bank_CIs_not_population_validation':True,
        'no_manual_annotation_resources_or_clinical_validation':True,'protected_media_forwards':0,
        'protected_labels_used_for_training_or_selection':False,'test_Pediatric_Unity_quarantine_used':False,
        'posterior_evidence_is_not_latent_anatomy_phase_evidence':True,
        'full_system_runtime_requires_A08_for_A_and_A08_plus_JEPA_for_J':True,'epoch':time.time()})
    pd.json_normalize(result['rows']).to_csv(PUB/'metrics.csv',index=False)
    write(RUN/'runtimecoordination.json',{'phase':'all70_96_scientific_matrix_complete_pending_oracle_package_closure',
          'model_pid':None,'epoch':time.time(),'next_action':'frozen action-domain oracle/figures then actualresource/exit/publicclosure'})

if __name__=='__main__':main()
