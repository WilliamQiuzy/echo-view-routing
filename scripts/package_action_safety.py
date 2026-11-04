#!/usr/bin/env python3
"""A61 fixed aggregate-only tables/figures; no model/threshold selection."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ.setdefault('OMP_NUM_THREADS','1')
import sys,time,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.safety_common import *
from echo_routing.temporal.safety_readout_plan import TABLES,HALVES,SEEDS,ARMS,BASE,system,historical
from echo_routing.temporal.structured_inputs import windows
from echo_routing.temporal.dense_sequence import chunk_starts
from echo_routing.temporal.safety_package import source_costs


def main():
    freeze=verify_freeze('package_freeze.json');s=read(PUB/'summary.json');o=read(PUB/'action_space_diagnostic.json');native=read(PUB/'native_attribution.json');models=read(PUB/'models_manifest.json');rows=read(PUB/'outer_metrics.json')['rows']
    if (s['metric_rows'],s['paired_comparison_count'],o['domain_count'],native['cell_count'])!=(314,240,60,56) or not s['all314_metrics_independently_recounted'] or o['summary_sha256']!=sha(PUB/'summary.json'):raise ValueError('complete frozen314/240/60/56 package required')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,'savefig.dpi':180});figures=[]
    labels={'main':'Main','a58_stress':'A58 reused stress','a59_confirmation':'A59 reused bank','a60_confirmation':'A60 reused bank','a61_confirmation':'A61 fresh construction'};colors={'U-P':'#476d95','U-L':'#78a9ba','S-P':'#bb7a45','S-L':'#a14848'};marks={'E':'o','N':'^','C':'s'}
    def save(fig,name):
        for ext in ['png','pdf']:
            path=PUB/f'{name}.{ext}';fig.savefig(path);figures.append(path)
        plt.close(fig)
    def footer(fig,lines,y=.10,spacing=.025):
        for i,line in enumerate(lines):fig.text(.04,y-i*spacing,line,fontsize=8.5)
    contrasts=[('S-L-C',BASE,'Complete system: S-L-C − A08','#a14848','s'),('S-L-C','A60-J-R','Prior policy: S-L-C − A60 J-R','#476d95','o'),('S-L-C','S-P-C','Ranking + calibration response: S-L-C − S-P-C','#39806f','^'),('S-L-C','S-L-N','Calibration: S-L-C − S-L-N','#aa8640','D')]
    ci={(r['table'],r['half'],r['seed'],r['left'],r['right']):r for r in s['paired_comparisons']}
    keys=[('segment_f1_50_multi','Multi segment F1 difference (pp)',100),('short_true_unmatched_16samples','True short misses: difference',1),('short_spurious_segments_16samples','False short units: difference',1),('native_new_wrong_reference_positions','New wrong native positions: difference',1)]
    fig,axes=plt.subplots(4,2,figsize=(17,18))
    for col,h in enumerate(HALVES):
        for ri,(key,title,scale) in enumerate(keys):
            ax=axes[ri,col];ax.axvline(0,color='.5',lw=.8);yt=[];yl=[]
            for ti,t in enumerate(TABLES):
                for si,seed in enumerate(SEEDS):
                    y=ti*2+si;yt.append(y);yl.append(f'{labels[t]} · rep {si+1}')
                    for c,(left,right,_,color,marker) in enumerate(contrasts):
                        right=BASE if right==BASE else historical(seed,'R') if right=='A60-J-R' else system(seed,right);r=ci[(t,h,seed,system(seed,left),right)];value=r['delta'].get(key);interval=r['delta95'].get(key)
                        if value is None or interval is None:continue
                        yy=y+(c-1.5)*.14;ax.plot(np.asarray(interval)*scale,[yy,yy],color=color,lw=1.3);ax.scatter(value*scale,yy,color=color,marker=marker,s=18,zorder=3)
            ax.set_yticks(yt,yl);ax.invert_yaxis();ax.grid(axis='x',alpha=.18);ax.set_xlabel(title);ax.set_title(h.capitalize()+(' · higher is better' if ri==0 else ' · lower is better'),loc='left')
    fig.suptitle('A61: safety target, within-sequence ranking and same-top veto',x=.05,ha='left',y=.986,fontsize=17)
    fig.text(.05,.962,'Fixed6000 / both seeds; conditional paired95% intervals. Primary S-L-C; all240 prespecified contrasts retained in the tables.',fontsize=10)
    fig.legend(handles=[plt.Line2D([0],[0],color=c,marker=m,label=l) for _,_,l,c,m in contrasts],loc='lower center',bbox_to_anchor=(.5,.115),ncol=2,frameon=False)
    footer(fig,['Points and intervals are separate; short ≤16 original positions (1.6s at10Hz). Hist. A60 seeds are replicate-mapped, not equal training seeds.','Session / fixed-stream / fixed-pair bootstrap; shared sources and propagated clip labels do not establish independent clinical evidence.','G preserves the occupancy majority only. C uses reused inner calibration, an empirical evidence cutoff and no safety guarantee.'],y=.09,spacing=.023)
    fig.subplots_adjust(left=.19,right=.98,top=.938,bottom=.19,hspace=.48,wspace=.62);save(fig,'paired_effects')
    fig,axes=plt.subplots(5,2,figsize=(15,17))
    for ti,t in enumerate(TABLES):
        for hi,h in enumerate(HALVES):
            ax=axes[ti,hi];base=s['systems'][t][h][BASE];ax.scatter(base['short_spurious_segments_16samples'],base['short_true_unmatched_16samples'],color='black',marker='*',s=100,zorder=5)
            for si,seed in enumerate(SEEDS):
                for arm in ARMS:
                    for policy in ['E','N','C']:
                        row=s['systems'][t][h][system(seed,arm+'-'+policy)];ax.scatter(row['short_spurious_segments_16samples'],row['short_true_unmatched_16samples'],marker=marks[policy],s=34,edgecolors=colors[arm],facecolors=colors[arm] if si==0 else 'none')
            ax.set_title(labels[t]+' · '+h,loc='left');ax.set_xlabel('False short units');ax.set_ylabel('True short misses');ax.grid(alpha=.2)
    handles=[plt.Line2D([0],[0],ls='',marker='o',color=colors[a],label=a) for a in ARMS]+[plt.Line2D([0],[0],ls='',marker=marks[p],color='.3',label=p) for p in ['E','N','C']]
    fig.suptitle('A61 short recovery versus false alarms: every fixed policy',x=.07,ha='left',y=.985,fontsize=16);fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.075),ncol=7,frameon=False)
    footer(fig,['Black star=A08; filled/hollow=seed1/2. Lower left is better. All24 A61 policies appear; coincident no-op policies can overlap.','Short≤16, full class-aware IoU50 matching. New construction reuses OPEN recordings, not new patients or clinician-annotated transitions.'],y=.052,spacing=.02);fig.subplots_adjust(left=.10,right=.98,top=.95,bottom=.13,hspace=.54,wspace=.30);save(fig,'short_recovery_false_alarms')
    fig,axes=plt.subplots(1,2,figsize=(14,7))
    for hi,h in enumerate(HALVES):
        ax=axes[hi];ax.scatter(0,0,c='black',marker='*',s=100,zorder=5)
        for si,seed in enumerate(SEEDS):
            for arm in ARMS:
                for policy in ['E','N','C']:
                    r=s['systems']['main'][h][system(seed,arm+'-'+policy)];ax.scatter(r['native_new_wrong_reference_positions'],r['native_repaired_reference_positions'],marker=marks[policy],s=44,edgecolors=colors[arm],facecolors=colors[arm] if si==0 else 'none')
            for p in ['E','R']:
                r=s['systems']['main'][h][historical(seed,p)];ax.scatter(r['native_new_wrong_reference_positions'],r['native_repaired_reference_positions'],marker='x' if p=='R' else '+',c='.3',s=70)
        ax.set_title(h.capitalize());ax.set_xlabel('Gross newly-wrong positions H (lower is better)');ax.set_ylabel('Gross repaired positions R (higher is better)');ax.grid(alpha=.2)
    fig.suptitle('Stable-majority recognition can still gain or lose reference frames',x=.06,ha='left',fontsize=15);fig.legend(handles=handles+[plt.Line2D([0],[0],ls='',marker='x',color='.3',label='A60 R'),plt.Line2D([0],[0],ls='',marker='+',color='.3',label='A60 E')],loc='lower center',bbox_to_anchor=(.5,.14),ncol=5,frameon=False)
    footer(fig,['G mechanically keeps every native majority class equal to A08; H and R expose changes hidden by that equality or their net difference.','Propagated clip-family references are not expert frame visibility. Native predictions are reused across all five task tables.'],y=.08,spacing=.027);fig.subplots_adjust(left=.08,right=.98,top=.88,bottom=.29,wspace=.3);save(fig,'native_gross_harm_repair')
    oracle={(r['table'],r['half'],r['domain'],r['safety']):r for r in o['oracle_domains']};fig,ax=plt.subplots(figsize=(14,9));yt=[];yl=[];safe_names=['unrestricted','net-safe','gross-safe'];safe_marks=['o','^','s']
    for ti,t in enumerate(TABLES):
        for hi,h in enumerate(HALVES):
            y=2*ti+hi;yt.append(y);yl.append(labels[t]+' · '+h);base=s['systems'][t][h][BASE]['segment_f1_50_multi']
            for di,d in enumerate(['old','expanded']):
                for ki,k in enumerate(safe_names):
                    r=oracle[(t,h,d,k)];ax.scatter(100*(r['F1']-base),y+(di*3+ki-2.5)*.105,color=['#476d95','#a14848'][di],marker=safe_marks[ki],s=35)
    ax.axvline(0,color='.5',lw=.8);ax.set_yticks(yt,yl);ax.invert_yaxis();ax.grid(axis='x',alpha=.2);ax.set_xlabel('Exact oracle multi F1 − A08 (percentage points)');ax.set_title('Observable action domain and label-assisted true-safety ceilings',loc='left',fontsize=15)
    fig.legend(handles=[plt.Line2D([0],[0],ls='',marker=safe_marks[k],color=['#476d95','#a14848'][d],label=domain+' / '+safety) for d,domain in enumerate(['old','expanded']) for k,safety in enumerate(safe_names)],loc='lower center',bbox_to_anchor=(.60,.14),ncol=3,frameon=False)
    footer(fig,['MaxONE action+no-op per multi stream; every domain uses the observable G majority guard. Native and pair metrics are outside the oracle objective.','True net-safe: H−R≤0 and ΔFP16≤0. True gross-safe: H=0 and ΔFP16≤0. These constraints use labels and are not deployable.','Only unrestricted ceilings bound policies that actually violate safety. A safe-oracle gap is not solely learnable ranking loss or guaranteed attainable gain.'],y=.105,spacing=.027);fig.subplots_adjust(left=.31,right=.98,top=.92,bottom=.25);save(fig,'action_domain_safe_ceilings')
    # Risk calibration views use just S-L to avoid counting the same shared-risk
    # predictions once for each utility arm. E/N/C still share one score matrix.
    fig,axes=plt.subplots(2,2,figsize=(14,10));tau_rows=[]
    for hi,h in enumerate(HALVES):
        for si,seed in enumerate(SEEDS):
            for ti,t in enumerate(TABLES):
                row=next(r for r in rows if r['table']==t and r['half']==h and r['system']==system(seed,'S-L-C'));stats=row['action_statistics_source_role'];y=stats['accepted_actions']/stats['role_sequences'];axes[0,hi].scatter(ti+si*.10,y,color=['#476d95','#a14848'][si],marker=['o','s'][si])
            bins=[]
            for t in TABLES:
                row=next(r for r in rows if r['table']==t and r['half']==h and r['system']==system(seed,'S-L-C'));bins.append(row['action_statistics_source_role']['risk_bins'])
            for k in range(5):
                n=sum(b[k]['candidates'] for b in bins);bad=sum(b[k]['unsafe_candidates'] for b in bins)
                if n:axes[1,hi].scatter((k+.5)/5,bad/n,color=['#476d95','#a14848'][si],marker=['o','s'][si])
        axes[0,hi].set_title(h.capitalize()+' · primary S-L-C acceptance');axes[0,hi].set_xticks(range(5),[labels[t].replace(' ','\n') for t in TABLES]);axes[0,hi].set_ylabel('Accepted actions / source-role sequences');axes[1,hi].set_xlabel('Fixed evidence bin midpoint r (uncalibrated)');axes[1,hi].set_ylabel('Observed H>0 or ΔFP16>0 fraction');axes[1,hi].set_ylim(-.02,1.02)
        for ax in axes[:,hi]:ax.grid(alpha=.2)
    for pin in models['thresholds']:
        record=read(verify_pin(pin));tau_rows.append({k:record[k] for k in ['seed','arm','tau','C_accepted','C_unsafe']})
    pd.DataFrame(tau_rows).to_csv(PUB/'calibration_summary.csv',index=False);fig.suptitle('Empirical calibrated veto: coverage and remaining outer false-safe evidence',fontsize=15)
    footer(fig,['Blue/red=seed1/2. All G candidates, not accepted-only, enter the five fixed risk bins; shared-risk candidate repeats across utility arms are excluded.','C calibration uses all928 inner diagnostic/calibration axes; zero unsafe inner accepts can mean no-op. No finite-sample or clinical guarantee.','Source-role panels use main native/multi/pair once and four separate constructed banks; shared source recordings are not independent observations.'],y=.095,spacing=.027);fig.subplots_adjust(left=.09,right=.98,top=.92,bottom=.21,hspace=.52,wspace=.30);save(fig,'risk_evidence_and_coverage')
    curves=[]
    for model in models['models']:
        path=ROOT/model['path'];trace=path.parent/'physical_updates.jsonl'
        if sha(trace)!=model['physical_trace_sha256']:raise ValueError('audited physical trace changed')
        data=[json.loads(x) for x in trace.read_text().splitlines()]
        if len(data)!=6000 or [r['u'] for r in data]!=list(range(6000)):raise ValueError('all6000 physical updates required for each curve')
        for start in range(0,6000,100):
            block=data[start:start+100];keys=['loss','pointwise_MSE','pairwise_logistic'] if model['role']=='utility' else ['loss','position_CE','candidate_BCE'];curves.append({'seed':model['seed'],'arm':model['arm'],'role':model['role'],'end_update':start+100,**{k:float(np.mean([r[k] for r in block])) for k in keys}})
    curves=pd.DataFrame(curves);curves.to_csv(PUB/'training_curves.csv',index=False);fig,axes=plt.subplots(2,2,figsize=(14,8))
    for si,seed in enumerate(SEEDS):
        for arm in ARMS:
            f=curves[(curves.seed==seed)&(curves.arm==arm)];axes[0,si].plot(f.end_update,f.loss,color=colors[arm],label=arm)
        f=curves[(curves.seed==seed)&(curves.role=='risk')]
        for key,color in [('position_CE','#476d95'),('candidate_BCE','#a14848')]:axes[1,si].plot(f.end_update,f[key],label=key,color=color)
        axes[0,si].set_title(f'Seed{si+1}: {seed}');axes[0,si].set_ylabel('Point MSE + optional pair logistic');axes[1,si].set_ylabel('Risk loss');axes[1,si].set_xlabel('Committed Adam updates')
        for ax in axes[:,si]:ax.grid(alpha=.2);ax.legend(frameon=False);ax.set_xticks([0,2000,4000,6000])
    fig.suptitle('A61 full fixed training trajectories; loss is not outer performance',fontsize=15);footer(fig,['100-update bins, all60000 updates. Utility uses64 complete sequence groups; risk uses64 candidates with all changed positions.','The shared FIT utility scale fixes units, but U/S target semantics and P/L objectives differ. No loss-based checkpoint or seed selection.'],y=.065,spacing=.026);fig.subplots_adjust(left=.08,right=.98,top=.91,bottom=.19,hspace=.32,wspace=.25);save(fig,'training_curves')
    pd.json_normalize(rows).to_csv(PUB/'metrics.csv',index=False);pd.DataFrame(o['oracle_domains']).to_csv(PUB/'action_space_oracle.csv',index=False);pd.json_normalize(s['paired_comparisons']).to_csv(PUB/'paired_comparisons.csv',index=False)
    inner=[]
    for item in models['inner_records']:
        record=read(verify_pin(item))
        for p,m in record['policies'].items():inner.append({'seed':record['seed'],'arm':record['arm'],'step':record['step'],'policy':p,**m})
    for pin in models['thresholds']:
        record=read(verify_pin(pin));inner.append({'seed':record['seed'],'arm':record['arm'],'step':6000,'policy':'C',**record['C_metrics']})
    if len(inner)!=56:raise ValueError('complete56 inner diagnostic/calibration sets required')
    pd.json_normalize(inner).to_csv(PUB/'inner_diagnostic_metrics.csv',index=False)
    first=read(verify_pin(models['inner_records'][0]));inner_scores=np.load(verify_pin(first['private_outputs']),allow_pickle=True).item()['scores']
    if len(inner_scores)!=928:raise ValueError('complete inner928 score axes required for physical forward counts')
    inner_nonempty=sum(len(v)>0 for v in inner_scores)
    write(PUB/'inner_physical_costs.json',{'fixed_source_sequences':928,'nonempty_G_sequences':inner_nonempty,'milestone_inner_readouts':24,'inner_utility_module_forwards':24*inner_nonempty,'inner_risk_module_forwards':48*inner_nonempty,'eight_final_calibrations_reuse_cached_scores_with_zero_new_neural_forward':True,'same_shapes_and_complete_coverage_verified_by_independent_model_audit':True,'new_source_producer_forwards':0,'inner_calibration_is_not_independent_validation':True})
    cache=read(PUB/'cache_outer_manifest.json');geometry={}
    for r in cache['records']:
        qs=sequences(r['role']);counts=[len(windows(len(q['refs']),5)['centre']) if r['source']=='a08' else len(chunk_starts(len(q['refs']))) for q in qs]
        geometry[(r['source'],r['role'])]=counts
    if set(geometry)!={(source,'a61_confirmation_'+half) for source in ['a08','jepa'] for half in HALVES}:raise ValueError('exactfour new source-role cells required')
    source_report=source_costs(cache['records'],geometry,[json.loads(line) for line in (RUN/'processes.jsonl').read_text().splitlines()],production_stage=cache.get('source_production_stage','cache_outer'));source_report['formal_optimizer_updates_at_source_production']=cache.get('formal_updates_at_production',60000);source_report['formal_optimizer_updates_at_head_binding']=cache.get('formal_updates_at_readonly_binding',60000)
    write(PUB/'source_physical_costs.json',{**source_report,'new240_genuine_joined_streams_only':True,'old20source_role_caches_reused_with_zero_neural_forward_this_round':True,'formal_physical_models':10,'formal_utility_optimizer_updates':48000,'formal_risk_optimizer_updates':12000,'formal_backward_calls':60000,'formal_utility_module_forward_calls':48000,'formal_risk_module_forward_calls':24000,'engineering_optimizer_updates':130,'engineering_mutable_models':22,'cost_only_structural_module_forwards':18,'cost_only_optimizer_updates':0,'outer_utility_module_forwards':read(PUB/'outer_metrics.json')['actual_utility_MLP_forwards'],'outer_localrisk_module_forwards':read(PUB/'outer_metrics.json')['actual_localrisk_MLP_forwards'],'outer_historical_newrole_module_forwards':read(PUB/'outer_metrics.json')['actual_historical_newrole_MLP_forwards'],'all_policies_operationally_require_full_A08_and_JEPA_producers':True,'cached_head_runtime_is_not_standalone_system_inference_latency':True,'standalone_end_to_end_latency_measured':False,'shared_caches_and_scores_charged_physically_once_not_perlogicalpolicy':True,'local_synthetic_software_tests_can_execute_additional_MLP_steps_separate_from130_engineering_and60000_formal':True,'global_manifest_ID_path_columns_transiently_parsed':True,'protected_media_forwards':0,'protected_labels_used_for_training_or_selection':False,'fresh_labels_created_for_prespecified_construction_only':True})
    write(PUB/'figure_receipt.json',{'passed':True,'summary_sha256':sha(PUB/'summary.json'),'action_space_diagnostic_sha256':sha(PUB/'action_space_diagnostic.json'),'native_attribution_sha256':sha(PUB/'native_attribution.json'),'figures':{str(p.relative_to(ROOT)):sha(p) for p in figures},'package_freeze_git':freeze['git_commit'],'aggregate_only_no_threshold_model_metric_or_endpoint_changes':True,'PNG_visual_QA_still_required':True,'epoch':time.time()})
    print({'figures':len(figures),'metric_rows':314,'inner_sets':56},flush=True)
if __name__=='__main__':main()
