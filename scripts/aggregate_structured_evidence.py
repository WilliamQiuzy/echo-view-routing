#!/usr/bin/env python3
"""Independent cached-only full metric recount and all132predeclared paired contrasts."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import sys,time,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd
from scripts.structured_common import ROOT,RUN,PUB,CFG,read,write,sha,sequences,verify_freeze
from echo_routing.temporal.structured_metrics import evaluate,stress_context,verify_reference
from echo_routing.temporal.structured_predictions import LabelReadout
from echo_routing.temporal.structured_uncertainty import prepare_system,paired_bootstrap

def main():
    freeze=verify_freeze('aggregate_freeze.json');verify_freeze('readout_freeze.json');cfg=read(CFG);models=read(PUB/'models_manifest.json');result=read(PUB/'outer_metrics.json');assert result['metric_rows']==72 and result['head_metric_rows']==48 and result['reference_metric_rows']==24;assert result['models_manifest_sha256']==sha(PUB/'models_manifest.json');assert result['source_outer_cache_manifest_sha256']==sha(PUB/'cache_outer_manifest.json');path=RUN/'readout_predictions.private.npy';assert sha(path)==result['private_prediction_bundle_sha256'];predictions=np.load(path,allow_pickle=True).item();metadata=np.load(ROOT/cfg['data']['old_outer_metadata'],allow_pickle=True).item();assert sha(ROOT/cfg['data']['old_outer_metadata'])==cfg['data']['old_outer_metadata_sha256'];plan=read(PUB/'readout_plan.json');contexts={};systems={};tables={};prepared={};rows=[];halves=['selection','calibration'];table_names=['fixed6000','inner_selected','duration_stress_fixed6000']
    for table in table_names:
        contexts[table]={};systems[table]={};tables[table]={}
        for half in halves:
            contexts[table][half]=stress_context(metadata[half],sequences('stress_'+half)) if table.startswith('duration_stress') else metadata[half];systems[table][half]={};tables[table][half]={}
    for old in result['rows']:
        table,half,name=old['table'],old['half'],old['system'];context=contexts[table][half];pred=predictions[table][half][name]
        if old['role']=='head':
            qs=sequences('stress_'+half) if table.startswith('duration_stress') else context['sequences'];role='stress_'+half if table.startswith('duration_stress') else half;cache=LabelReadout(RUN/'head_readout'/old['sha256']/role,old['head_label_cache']['identity'],[len(q['labels']) for q in qs]);assert cache.cursor==len(qs) and sha(cache.journal)==old['head_label_cache']['journal_sha256']
            for i,q in enumerate(qs):assert np.array_equal(cache.load(i),pred[q['id']])
            if table.startswith('duration_stress'):
                main=predictions['fixed6000'][half][name]
                for q in metadata[half]['sequences']:
                    if q['kind']!='multi':assert np.array_equal(main[q['id']],pred[q['id']])
        metric,t=evaluate(pred,context);verify_reference(metric,{k:old[k] for k in metric});assert metric['counts']['native_sessions']==36 and metric['counts']['multi_streams']==120;systems[table][half][name]=metric;tables[table][half][name]=t;rows.append(old);folder=RUN/'aggregate'/table/half/name;folder.mkdir(parents=True,exist_ok=True)
        for label,frame in t.items():frame.to_csv(folder/(label+'.private.csv'),index=False)
    assert len(rows)==72
    for half in halves:
        for row in [r for r in rows if r['table']=='fixed6000' and r['half']==half and r['role']=='reference']:
            name=row['system'];systems['inner_selected'][half][name]=systems['fixed6000'][half][name];tables['inner_selected'][half][name]=tables['fixed6000'][half][name]
    comparisons=[];delta_keys=['native_macro_f1','native_accuracy','native_frame_accuracy','segment_f1_50_multi','segment_f1_50_pairs','boundary_recall_multi','boundary_precision_multi','boundary_f1_multi','boundary_recall_pairs','same_class_cut_rate_multi','native_internal_cuts_per_clip','native_A4C_to_A5C','native_A5C_to_A4C','short_true_unmatched_10samples','short_spurious_segments_10samples','short_true_unmatched_16samples','short_spurious_segments_16samples']
    for table in table_names:
        for half in halves:
            def prepared_system(name):
                key=(table,half,name)
                if key not in prepared:prepared[key]=prepare_system(predictions[table][half][name],contexts[table][half]['sequences'],tables[table][half][name])
                return prepared[key]
            for seed in cfg['training']['seeds']:
                suffix='inner_selected' if table=='inner_selected' else 'fixed6000'
                for left_arm,right_arm in plan['paired_contrasts_each_seed_half_table']:
                    left=f'seed{seed}_{left_arm}_{suffix}';right=f'seed{seed}_{right_arm}_{suffix}' if right_arm.startswith('D') else right_arm;a,b=systems[table][half][left],systems[table][half][right];delta={k:None if a[k] is None or b[k] is None else a[k]-b[k] for k in delta_keys};paired=paired_bootstrap(prepared_system(left),prepared_system(right),n_boot=plan['bootstrap_replicates'],seed=plan['bootstrap_seed']);comparisons.append({'table':table,'half':half,'seed':seed,'left':left,'right':right,'contrast':left_arm+'-'+right_arm,'delta':delta,'native_confusion_delta':(np.asarray(a['native_confusion'])-np.asarray(b['native_confusion'])).tolist(),'native_per_class_f1_delta':(np.asarray(a['native_per_class_f1'])-np.asarray(b['native_per_class_f1'])).tolist(),**paired})
                    if len(comparisons)%4==0:write(PUB/'paired_progress.json',{'completed_paired_comparisons':len(comparisons),'declared':132,'bootstrap_replicates_each':5000,'model_pid':os.getpid(),'epoch':time.time()});write(RUN/'runtimecoordination.json',{'phase':'cached_independent_paired_aggregation','model_pid':os.getpid(),'comparisons_done':len(comparisons),'comparisons':132,'epoch':time.time(),'next_action':'finish all predeclared comparisons then model/resource/exit closure'})
    assert len(comparisons)==132;out={'status':'completed exploratory full-task structured evidence matrix','systems':systems,'paired_comparisons':comparisons,'metric_rows':72,'head_metric_rows':48,'reference_metric_rows':24,'paired_comparison_count':len(comparisons),'formal_models':8,'formal_optimizer_updates':48000,'primary_contrast':'fixed6000matchedD2-D1separatelybybothseedsandbothhalves;innerselectedanddurationstressdeclaredsupplements','model_manifest_sha256':sha(PUB/'models_manifest.json'),'inner_choices_sha256':sha(PUB/'inner_choices.json'),'outer_metrics_sha256':sha(PUB/'outer_metrics.json'),'readout_plan_sha256':sha(PUB/'readout_plan.json'),'aggregate_freeze_git':freeze['git_commit'],'all72reader_metrics_independently_recounted_exactly':True,'head_label_journal_hashes_and_full_original_axes_verified':True,'duration_stress_native_and_pairs_exactly_reuse_main_predictions':True,'feature_scope':'frozenfiveclassposteriortrajectories;notlatent/anatomy/phaseevidence','OPEN_reused_development':True,'old_A08_inner_exposure_not_OOF':True,'sessions_not_confirmed_patients':True,'conditional_shared_source_constructed_bank_uncertainty_not_population_CI':True,'intervals_crossing_zero_not_equivalence':True,'human_annotation_or_clinical_validation_performed':False,'independent_clinical_validation':False,'test_Pediatric_Unity_quarantine_evaluated_or_used_for_training':False,'global_manifest_ID_path_columns_transiently_parsed':True,'protected_label_columns_parsed_by_source_loader':False,'protected_media_forwards':0,'protected_labels_used_for_training_or_selection':False,'automatic_next_architecture_authorized':False,'epoch':time.time()};write(PUB/'summary.json',out);pd.json_normalize(rows).to_csv(PUB/'metrics.csv',index=False);write(RUN/'runtimecoordination.json',{'phase':'scientific_matrix_complete_pending_terminal_resource_and_public_report_closure','model_pid':None,'metric_rows':72,'paired_comparisons':132,'epoch':time.time(),'next_action':'freeze completed summary; verify all process exits dual resource caps and immutable pins; close round'});print(json.dumps({'summary_written':True,'metric_rows':72,'paired_comparisons':132,'formal_optimizer_updates':48000}),flush=True)
if __name__=='__main__':main()
