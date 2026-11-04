#!/usr/bin/env python3
"""Fixed aggregate-only A60 scientific tables/figures, never model selection."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ.setdefault('OMP_NUM_THREADS','1')
import sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.proposal_common import *
from echo_routing.temporal.proposal_readout_plan import TABLES,HALVES,SEEDS,BASE,system
from echo_routing.temporal.structured_inputs import windows
from echo_routing.temporal.dense_sequence import chunk_starts


def main():
    freeze=verify_freeze('package_freeze.json');s=read(PUB/'summary.json');o=read(PUB/'action_space_diagnostic.json');rows=read(PUB/'outer_metrics.json')['rows']
    if s['metric_rows']!=124 or s['paired_comparison_count']!=144 or o['domain_count']!=32 or o['summary_sha256']!=sha(PUB/'summary.json'):raise ValueError('sealed124/144/32 package required')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,'savefig.dpi':180})
    figures=[]
    def save(fig,name):
        for ext in ['png','pdf']:
            p=PUB/f'{name}.{ext}';fig.savefig(p);figures.append(p)
        plt.close(fig)
    labels={'main':'Main','a58_stress':'A58 reused stress','a59_confirmation':'A59 reused bank','a60_confirmation':'A60 fresh construction'}
    contrast=[('J-R','A-R','Posterior features: J-R − A-R','#286c9c','o'),('J-E','J-O','Proposal domain: J-E − J-O','#9c6b2d','^'),('J-R','J-E','Risk veto: J-R − J-E','#327b66','D'),('J-R',BASE,'Complete system: J-R − A08','#be5c2e','s')]
    comp={(r['table'],r['half'],r['seed'],r['left'],r['right']):r for r in s['paired_comparisons']}
    keys=[('segment_f1_50_multi','Multi segment F1 difference (pp)',100),('short_true_unmatched_16samples','True short units missed: difference',1),('short_spurious_segments_16samples','False short units: difference',1),('native_frame_errors','Native wrong positions: difference',1)]
    fig,axes=plt.subplots(4,2,figsize=(16,16))
    for col,h in enumerate(HALVES):
        for ri,(key,label,scale) in enumerate(keys):
            ax=axes[ri,col];ax.axvline(0,color='.5',lw=.8);yt=[];yl=[]
            for ti,t in enumerate(TABLES):
                for si,seed in enumerate(SEEDS):
                    y=ti*2+si;yt.append(y);yl.append(f'{labels[t]} · seed {si+1}')
                    for ci,(left,right,_,color,marker) in enumerate(contrast):
                        r=comp[(t,h,seed,system(seed,left),right if right==BASE else system(seed,right))];value=r['delta'].get(key);ci95=r['delta95'].get(key)
                        if value is None or ci95 is None:continue
                        yy=y+(ci-1.5)*.14;lo,hi=np.asarray(ci95)*scale;ax.plot([lo,hi],[yy,yy],color=color,lw=1.5);ax.scatter(value*scale,yy,c=color,s=19,marker=marker,zorder=3)
            ax.set_yticks(yt,yl);ax.invert_yaxis();ax.grid(axis='x',alpha=.18);ax.set_xlabel(label);ax.set_title(h.capitalize()+(' · higher is better' if ri==0 else ' · lower is better'),loc='left',fontsize=10)
    fig.suptitle('A60: JEPA proposals, A08 anchor and learned harm veto',x=.06,ha='left',y=.985,fontsize=17)
    fig.text(.06,.955,'Fixed 6000 updates and both seeds; conditional paired 95% intervals. Short = at most 16 original positions (1.6 s at 10 Hz).',fontsize=10)
    handles=[plt.Line2D([0],[0],color=c,marker=m,label=l,lw=1.5) for _,_,l,c,m in contrast];fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.135),ncol=2,frameon=False,fontsize=9)
    fig.text(.06,.105,'Both feature controls require JEPA proposal geometry. O/E/R share weights; the veto predicts harms and supplies no calibrated risk guarantee.',fontsize=9)
    fig.text(.06,.080,'G preserves the baseline occupancy majority mechanically; native frame/cut harms remain measurable. All 124 rows and 144 contrasts are retained.',fontsize=9)
    fig.text(.06,.055,'Bootstrap units: whole sessions / fixed streams / fixed pairs. Reused sources and propagated clip labels do not provide independent clinical validation.',fontsize=9)
    fig.text(.06,.030,'Summary SHA256 '+sha(PUB/'summary.json')[:16]+'. Points and percentile intervals are drawn separately, including when the point lies outside the interval.',fontsize=9)
    fig.subplots_adjust(left=.19,right=.98,top=.925,bottom=.22,hspace=.48,wspace=.58);save(fig,'paired_effects')
    curves=pd.DataFrame(read(PUB/'training_curves.json')['bins']);curves.to_csv(PUB/'training_curves.csv',index=False)
    fig,axes=plt.subplots(2,2,figsize=(12,7),sharex=True)
    for col,seed in enumerate(SEEDS):
        for arm,color in [('A','#286c9c'),('J','#be5c2e')]:
            f=curves[(curves.seed==seed)&(curves.arm==arm)];axes[0,col].plot(f.end_update,f.loss,color=color,label=arm+' feature head');axes[1,col].plot(f.end_update,f.native_MSE,color=color,label=arm+' native');axes[1,col].plot(f.end_update,f.constructed_MSE,color=color,ls='--',label=arm+' constructed')
        axes[0,col].set_title(f'Seed {col+1}: {seed}');axes[0,col].set_ylabel('Balanced three-output MSE');axes[1,col].set_ylabel('Three-output MSE by kind');axes[1,col].set_xlabel('Actual Adam updates')
        for ax in axes[:,col]:ax.grid(alpha=.2);ax.legend(frameon=False);ax.set_xticks([0,2000,4000,6000])
    fig.suptitle('A60 fixed training: shared FIT target RMS, no checkpoint selection',fontsize=14);fig.tight_layout(rect=(0,.07,1,.94));fig.text(.03,.02,'100 physical updates per bin; 3619 parameters per head. Three signed targets have uncentered FIT RMS scales; both controls use the same JEPA proposals.',fontsize=8);save(fig,'training_curves')
    fig,axes=plt.subplots(4,2,figsize=(14,14));colors={'A':'#286c9c','J':'#be5c2e'};markers={'O':'^','E':'o','R':'s'}
    for ri,t in enumerate(TABLES):
        for col,h in enumerate(HALVES):
            ax=axes[ri,col];base=s['systems'][t][h][BASE];ax.scatter(base['short_spurious_segments_16samples'],base['short_true_unmatched_16samples'],s=100,c='black',marker='*',label='A08 anchor',zorder=4)
            for si,seed in enumerate(SEEDS):
                for arm in ['A','J']:
                    for p in ['O','E','R']:
                        r=s['systems'][t][h][system(seed,arm+'-'+p)];ax.scatter(r['short_spurious_segments_16samples'],r['short_true_unmatched_16samples'],s=34,marker=markers[p],edgecolors=colors[arm],facecolors=colors[arm] if si==0 else 'none')
            ax.set_title(f'{labels[t]} · {h}',loc='left',fontsize=10);ax.set_xlabel('False short segments (lower is better)');ax.set_ylabel('True short segments missed');ax.grid(alpha=.2)
    handles=[plt.Line2D([0],[0],ls='',marker=markers[p],color=colors[a],label=f'{a}-{p}') for a in ['A','J'] for p in ['O','E','R']]+[plt.Line2D([0],[0],ls='',marker='*',color='black',label='A08 anchor')]
    fig.suptitle('A60 short-segment recovery versus false alarms',x=.08,ha='left',y=.985,fontsize=16);fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.105),ncol=7,frameon=False)
    fig.text(.08,.075,'Filled = seed 1; hollow = seed 2. Short ≤16 positions; class-aware IoU50 matching. All 12 fixed policies and the same A08 anchor appear per panel.',fontsize=9)
    fig.text(.08,.045,'Lower left improves both counts. A60 fresh construction reuses OPEN sources; none of these panels establish independent patient or clinical evidence.',fontsize=9)
    fig.subplots_adjust(left=.09,right=.98,top=.94,bottom=.18,hspace=.50,wspace=.28);save(fig,'short_recovery_false_alarms')
    fig,ax=plt.subplots(figsize=(13,7));oracle={(r['table'],r['half'],r['domain'],r['guard']):r for r in o['oracle_domains']};yt=[];yl=[]
    for ti,t in enumerate(TABLES):
        for hi,h in enumerate(HALVES):
            y=2*ti+hi;yt.append(y);yl.append(labels[t]+' · '+h);base=s['systems'][t][h][BASE]['segment_f1_50_multi']
            for di,d in enumerate(['old','expanded']):
                for gi,g in enumerate(['U','G']):
                    r=oracle[(t,h,d,g)];ax.scatter(100*(r['F1']-base),y+(di*2+gi-1.5)*.13,color=['#286c9c','#be5c2e'][di],marker=['o','s'][gi],s=38)
    ax.axvline(0,c='.5',lw=.8);ax.set_yticks(yt,yl);ax.invert_yaxis();ax.grid(axis='x',alpha=.2);ax.set_xlabel('Exact oracle multi F1 − A08 anchor (percentage points)');ax.set_title('Action-domain capability ceiling, label assisted and not deployable',loc='left',fontsize=15)
    handles=[plt.Line2D([0],[0],ls='',marker=['o','s'][gi],color=['#286c9c','#be5c2e'][di],label=d+' / '+g) for di,d in enumerate(['old','expanded']) for gi,g in enumerate(['U','G'])];fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.6,.13),ncol=4,frameon=False)
    fig.text(.06,.075,'Max ONE canonical action + no-op per stream. Objective includes multi streams only; native/pair metrics are excluded. U/G vary the observable majority guard.',fontsize=9)
    fig.text(.06,.035,'Union recovery is not jointly attainable. These are domain-limited ceilings, not upper bounds on all JEPA information. All 12 overlapping old domains reproduce A59.',fontsize=9)
    fig.subplots_adjust(left=.30,right=.98,top=.90,bottom=.24);save(fig,'action_domain_ceiling')
    pd.json_normalize(rows).to_csv(PUB/'metrics.csv',index=False)
    pd.DataFrame(o['oracle_domains']).to_csv(PUB/'action_space_oracle.csv',index=False)
    pd.DataFrame(o['action_diagnostics']).to_csv(PUB/'action_domain_diagnostics.csv',index=False)
    inner=[]
    for item in read(PUB/'models_manifest.json')['inner_records']:
        path=ROOT/item['inner_record_path']
        if sha(path)!=item['inner_record_sha256']:raise ValueError('inner aggregate changed')
        r=read(path)
        for p,m in r['policies'].items():inner.append({'seed':r['seed'],'arm':r['arm'],'step':r['step'],'policy':p,**{k:m[k] for k in ['native_macro_f1','segment_f1_50_multi','short_true_unmatched_16samples','short_spurious_segments_16samples']}})
    if len(inner)!=36:raise ValueError('complete36 inner diagnosticsets required')
    pd.DataFrame(inner).to_csv(PUB/'inner_curves.csv',index=False)
    cache=read(PUB/'cache_outer_manifest.json');forward={'a08':0,'jepa':0};units={'a08':0,'jepa':0}
    for r in cache['records']:
        qs=sequences(r['role']);counts=[len(windows(len(q['refs']),5)['centre']) if r['source']=='a08' else len(chunk_starts(len(q['refs']))) for q in qs]
        if sum(counts)!=r['new_neural_units']:raise ValueError('source forward geometry failed independent recount')
        source=r['source'];batch=32 if source=='a08' else 2;forward[source]+=sum((n+batch-1)//batch for n in counts);units[source]+=sum(counts)
    write(PUB/'source_physical_costs.json',{'new_source_neural_units':units,'source_python_model_forward_calls':forward,'source_forward_call_basis':'exact original axes and frozen per-sequenceA08batch32/JEPA2;notGPUkernelcounts','source_cache_GPU_jobwall_seconds':read(PUB/'process_cache_outer_1.json')['full_process_wall_seconds'],'source_cache_CPU_seconds':read(PUB/'process_cache_outer_1.json')['measured_CPU_core_seconds'],'old16source_role_caches_reused_with_zero_neural_forward_this_round':True,'physical_formal_models':4,'formal_CPU_optimizer_updates':24000,'formal_head_forward_calls':sum(r['result']['physical_candidate_exposures']//32 for r in read(PUB/'models_manifest.json')['models']),'formal_backward_calls':24000,'engineering_optimizer_updates':76,'operational_required_producers_for_both_A_and_J':['full original A08','full frozen JEPA','fixed B3','proposal/features/threeoutput scoring/gate'],'cached_head_latency_is_not_fullsystem_inference_latency':True,'standalone_end_to_end_inference_latency_measured':False,'physical_shared_caches_and_scores_charged_once_not_perpolicy':True,'global_manifest_ID_path_columns_transiently_parsed':True,'protected_label_columns_parsed':False,'protected_media_forwards':0,'protected_labels_used_for_training_or_selection':False,'fresh_labels_created_from_allowlisted_source_class_metadata_only_for_prespecified_construction':True})
    write(PUB/'figure_receipt.json',{'passed':True,'summary_sha256':sha(PUB/'summary.json'),'action_space_diagnostic_sha256':sha(PUB/'action_space_diagnostic.json'),'figures':{str(p.relative_to(ROOT)):sha(p) for p in figures},'package_freeze_git':freeze['git_commit'],'aggregate_only_no_threshold_model_or_metric_changes':True,'epoch':time.time()})
if __name__=='__main__':main()
