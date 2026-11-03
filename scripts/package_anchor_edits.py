#!/usr/bin/env python3
"""Aggregate-only A59 scientific figures/tables from the sealed complete matrix."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ.setdefault('OMP_NUM_THREADS','1')
import sys,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.anchor_common import RUN,PUB,read,write,sha,verify_freeze,sequences
from echo_routing.temporal.anchor_readout_plan import TABLES,HALVES,SEEDS,BASE,system


def main():
    freeze=verify_freeze('package_freeze.json');summary=read(PUB/'summary.json');oracle=read(PUB/'action_space_diagnostic.json')
    if summary['metric_rows']!=70 or summary['paired_comparison_count']!=96 or oracle['summary_sha256']!=sha(PUB/'summary.json'):
        raise ValueError('sealed complete70/96 and12domain oracle required')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,
                        'pdf.fonttype':42,'ps.fonttype':42,'savefig.dpi':180})
    fig,axes=plt.subplots(3,2,figsize=(14,11));labels={'main':'Main','old_stress':'Reused stress','fresh_confirmation':'Fresh construction'}
    metrics=[('segment_f1_50_multi','Multi segment F1 difference (pp)',100),
             ('short_true_unmatched_16samples','True short segments missed: difference',1),
             ('short_spurious_segments_16samples','False short segments: difference',1)]
    colors=['#be5c2e','#286c9c'];contrasts=[('J-G','A-G','JEPA guard − matched A08 guard'),('J-G',BASE,'JEPA guard − original A08')]
    comparison={(r['table'],r['half'],r['seed'],r['left'],r['right']):r for r in summary['paired_comparisons']}
    for col,half in enumerate(HALVES):
        for ri,(key,xlabel,scale) in enumerate(metrics):
            ax=axes[ri,col];ax.axvline(0,color='#777777',lw=.8);yt=[];yl=[]
            for ti,table in enumerate(TABLES):
                for si,seed in enumerate(SEEDS):
                    y=ti*2+si;yt.append(y);yl.append(f'{labels[table]} · seed {si+1}')
                    for ci,(left,right,title) in enumerate(contrasts):
                        row=comparison[(table,half,seed,system(seed,left),right if right==BASE else system(seed,right))]
                        value=row['delta'][key];interval=row['delta95'][key]
                        if value is None or interval is None:continue
                        lo,hi=np.asarray(interval)*scale;value*=scale
                        # A percentile interval need not contain the original point.
                        ax.plot([lo,hi],[y+(ci-.5)*.22]*2,color=colors[ci],lw=1.8)
                        ax.scatter(value,y+(ci-.5)*.22,color=colors[ci],s=24,marker='o' if ci==0 else 's',zorder=3)
            ax.set_yticks(yt,yl);ax.invert_yaxis();ax.set_xlabel(xlabel);ax.grid(axis='x',alpha=.2)
            ax.set_title(f'{half.capitalize()} · '+('higher is better' if ri==0 else 'lower is better'),loc='left',fontsize=11)
    fig.suptitle('A59: fixed A08 anchor, learned single edits',x=.07,ha='left',fontsize=17,y=.98)
    fig.text(.07,.945,'Fixed 6000 updates in both seeds; paired conditional 95% intervals. Short = at most 16 positions (1.6 s at 10 Hz).',fontsize=10)
    handles=[plt.Line2D([0],[0],color=c,marker='o' if i==0 else 's',label=row[2],lw=1.8) for i,(c,row) in enumerate(zip(colors,contrasts))]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.125),ncol=2,frameon=False,fontsize=10)
    fig.text(.07,.09,'G mechanically preserves the original occupancy majority for every sequence; this is not learned recognition improvement.',fontsize=9)
    fig.text(.07,.065,'Intervals resample complete sessions / fixed streams / fixed pairs. Banks share sources; these are not independent patient or clinical intervals.',fontsize=9)
    fig.text(.07,.04,f'OPEN reused development; fresh bank is construction confirmation. Full 70 rows and 96 contrasts retained. Summary SHA256 {sha(PUB/"summary.json")[:16]}.',fontsize=9)
    fig.subplots_adjust(left=.17,right=.98,top=.90,bottom=.22,hspace=.48,wspace=.56)
    figures=[]
    for suffix in ['png','pdf']:
        path=PUB/f'paired_effects.{suffix}';fig.savefig(path);figures.append(path)
    plt.close(fig)
    curves=read(PUB/'training_curves.json')['bins'];frame=pd.DataFrame(curves);frame.to_csv(PUB/'training_curves.csv',index=False)
    fig,axes=plt.subplots(2,2,figsize=(12,7),sharex=True)
    for col,seed in enumerate(SEEDS):
        for arm,color in [('A','#286c9c'),('J','#be5c2e')]:
            sub=frame[(frame.seed==seed)&(frame.arm==arm)]
            axes[0,col].plot(sub.end_update,sub.loss,color=color,label=arm+' evidence')
            axes[1,col].plot(sub.end_update,sub.native_MSE,color=color,label=arm+' native')
            axes[1,col].plot(sub.end_update,sub.constructed_MSE,color=color,ls='--',label=arm+' constructed')
        axes[0,col].set_title(f'Seed {col+1} ({seed})');axes[0,col].set_ylabel('Balanced utility MSE')
        axes[1,col].set_ylabel('Per kind utility MSE');axes[1,col].set_xlabel('Actual Adam updates')
        for ax in axes[:,col]:ax.grid(alpha=.2);ax.legend(frameon=False);ax.set_xticks([0,2000,4000,6000])
    fig.suptitle('A59 training diagnostics: all fixed updates, no checkpoint selection',fontsize=14)
    fig.tight_layout(rect=(0,.04,1,.95));fig.text(.03,.015,'Each point averages 100 physical updates. Frozen producers; 3201 trainable utility parameters per head. G and U share weights.',fontsize=9)
    for suffix in ['png','pdf']:
        path=PUB/f'training_curves.{suffix}';fig.savefig(path);figures.append(path)
    plt.close(fig)
    keys=['native_macro_f1','native_accuracy','native_frame_accuracy','native_internal_cuts_per_clip','native_A4C_to_A5C','native_A5C_to_A4C',
          'segment_f1_50_multi','segment_f1_50_pairs','boundary_precision_multi','boundary_recall_multi','boundary_precision_pairs','boundary_recall_pairs',
          'short_true_unmatched_10samples','short_spurious_segments_10samples','short_true_unmatched_16samples','short_spurious_segments_16samples',
          'pairs_short_true_unmatched_16samples','pairs_short_spurious_segments_16samples']
    rows=[{'table':table,'half':half,'system':name,**{k:m[k] for k in keys}} for table in TABLES for half in HALVES for name,m in summary['systems'][table][half].items()]
    pd.DataFrame(rows).to_csv(PUB/'task_tradeoffs.csv',index=False)
    inner=[]
    for item in read(PUB/'models_manifest.json')['inner_records']:
        path=ROOT/item['inner_record_path']
        if sha(path)!=item['inner_record_sha256']:raise ValueError('frozen inner diagnostic record changed')
        record=read(path)
        for policy,metric in record['policies'].items():
            inner.append({'seed':item['seed'],'arm':item['arm'],'step':item['step'],'policy':policy,
                          **{k:metric[k] for k in ['native_macro_f1','segment_f1_50_multi','short_true_unmatched_16samples','short_spurious_segments_16samples']}})
    if len(inner)!=24:raise ValueError('complete fixed24 inner policy metric sets required')
    pd.DataFrame(inner).to_csv(PUB/'inner_curves.csv',index=False)
    pd.DataFrame(oracle['oracle_domains']).to_csv(PUB/'action_space_oracle.csv',index=False)
    caches={mode:read(PUB/f'cache_{mode}_manifest.json') for mode in ['fit','outer']}
    from echo_routing.temporal.structured_inputs import windows
    from echo_routing.temporal.dense_sequence import chunk_starts
    forward_counts={'a08':0,'jepa':0}
    for mode,cache in caches.items():
        for record in cache['records']:
            source,role=record['source'],record['role'];qs=sequences(role)
            fresh=[q for q in qs if (mode=='fit' and (source=='a08' or q['kind']!='native')) or role.startswith('confirmation_')]
            units=[len(windows(len(q['refs']),5)['centre']) if source=='a08' else len(chunk_starts(len(q['refs']))) for q in fresh]
            if sum(units)!=record['new_neural_units'] or record['prior_committed_sequences']!=0:
                raise ValueError('sealed source physical workload failed independent geometry recount')
            batch=32 if source=='a08' else 2
            forward_counts[source]+=sum((n+batch-1)//batch for n in units)
    write(PUB/'source_physical_costs.json',{'source_cache_records':caches,
        'source_cache_GPU_job_wall_seconds':{mode:read(PUB/f'process_cache_{mode}_1.json')['full_process_wall_seconds'] for mode in caches},
        'new_A08_windows':sum(r['new_neural_units'] for x in caches.values() for r in x['records'] if r['source']=='a08'),
        'new_JEPA_chunks':sum(r['new_neural_units'] for x in caches.values() for r in x['records'] if r['source']=='jepa'),
        'study_cache_producer_Python_forward_calls':forward_counts,
        'forward_call_count_basis':'derived from exact sealed original axes and frozen batch loops:A08batch32/JEPA2;not GPUkernel counts;engineering separate',
        'engineering_profile_charged_separately':True,'engineering_GPU_checks_sha256':sha(PUB/'FIT_GPU_checks.json'),
        'operational_A_requires':['full original A08 producer','original fixed B3 decoder','feature extraction and utility policy'],
        'operational_J_requires':['full original A08 producer','full frozen JEPA producer','original fixed B3 decoder','feature extraction and utility policy'],
        'cached_head_latency_is_not_full_system_inference_latency':True,
        'frozen_producers_pixel_preprocessing_and_original_B3_excluded_from_cached_head_latency':True,'standalone_end_to_end_inference_latency_measured':False,
        'matrix_shared_cache_charged_physically_once_not_per_logical_policy':True,
        'global_manifest_ID_path_columns_transiently_parsed':True,
        'protected_label_columns_parsed_by_source_loader':False,'protected_media_forwards':0,
        'protected_labels_used_for_training_or_selection':False,
        'fresh_confirmation_classmetadata_labels_created_before_heads_only_for_prospective_bank_construction':True})
    write(PUB/'figure_receipt.json',{'passed':True,'summary_sha256':sha(PUB/'summary.json'),'action_space_diagnostic_sha256':sha(PUB/'action_space_diagnostic.json'),
          'figures':{str(p.relative_to(ROOT)):sha(p) for p in figures},'package_freeze_git':freeze['git_commit'],
          'all_plots_aggregate_only_no_model_metric_or_threshold_changes':True,'epoch':time.time()})

if __name__=='__main__':main()
