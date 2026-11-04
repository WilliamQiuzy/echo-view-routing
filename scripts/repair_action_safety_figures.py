#!/usr/bin/env python3
"""Four-artifact plot-only clarification, retaining sealed scientific inputs."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ.setdefault('OMP_NUM_THREADS','1')
import sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.safety_common import *
from echo_routing.temporal.safety_readout_plan import TABLES,HALVES,SEEDS,ARMS,system,historical


def main():
    freeze=verify_freeze('figure_repair_freeze.json');old=read(PUB/'figure_receipt_v1.json');s=read(PUB/'summary.json');rows=read(PUB/'outer_metrics.json')['rows']
    if (PUB/'figure_repair_receipt.json').exists():raise ValueError('figure repair already complete')
    for key,name in [('summary_sha256','summary.json'),('action_space_diagnostic_sha256','action_space_diagnostic.json'),('native_attribution_sha256','native_attribution.json')]:
        if old[key]!=sha(PUB/name):raise ValueError('scientific input changed')
    for path,h in old['figures'].items():
        if sha(ROOT/path)!=h:raise ValueError('first-version artifact changed')
    cal=[next(r for r in rows if r['table']=='a61_confirmation' and r['half']=='calibration' and r['system']==system(seed,'S-L-C'))['action_statistics_source_role'] for seed in SEEDS]
    if [(r['accepted_unsafe_actions'],r['accepted_actions']) for r in cal]!=[(1,10),(1,6)]:raise ValueError('frozen risk annotation differs')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,'savefig.dpi':180});changed=[]
    labels={'main':'Main','a58_stress':'A58 reused stress','a59_confirmation':'A59 reused bank','a60_confirmation':'A60 reused bank','a61_confirmation':'A61 fresh construction'};colors={'U-P':'#476d95','U-L':'#78a9ba','S-P':'#bb7a45','S-L':'#a14848'};marks={'E':'o','N':'^','C':'s'}
    handles=[plt.Line2D([0],[0],ls='',marker='o',color=colors[a],label=a) for a in ARMS]+[plt.Line2D([0],[0],ls='',marker=marks[p],color='.3',label=p) for p in ['E','N','C']]
    def footer(fig,lines,y=.10,spacing=.025):
        for i,line in enumerate(lines):fig.text(.04,y-i*spacing,line,fontsize=8.5)
    def save(fig,name):
        for ext in ['png','pdf']:
            path=PUB/f'{name}.{ext}';fig.savefig(path);changed.append(str(path.relative_to(ROOT)))
        plt.close(fig)
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
    footer(fig,['G mechanically keeps every native majority class equal to A08; H and R expose changes hidden by that equality or their net difference.','Black star=A08; filled/hollow=replicate1/2. Overlapping C/no-op points can coincide.',
        'Propagated clip-family references are not expert frame visibility. Native predictions are reused across all five task tables.'],y=.09,spacing=.027);fig.subplots_adjust(left=.08,right=.98,top=.88,bottom=.29,wspace=.3);save(fig,'native_gross_harm_repair')
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
        axes[0,hi].set_title(h.capitalize()+' · primary S-L-C acceptance'+('\nA61 outer calibration bank: unsafe C 1/10 (rep1), 1/6 (rep2)' if h=='calibration' else ''));axes[0,hi].set_xticks(range(5),[labels[t].replace(' ','\n') for t in TABLES]);axes[0,hi].set_ylabel('Accepted actions / source-role sequences');axes[1,hi].set_xlabel('Fixed evidence bin midpoint r (uncalibrated)');axes[1,hi].set_ylabel('Observed H>0 or ΔFP16>0 fraction');axes[1,hi].set_ylim(-.02,1.02)
        for ax in axes[:,hi]:ax.grid(alpha=.2)
    fig.suptitle('Empirical calibrated veto: coverage and remaining outer false-safe evidence',fontsize=15)
    footer(fig,['Blue/red=seed1/2. All G candidates, not accepted-only, enter the five fixed risk bins; shared-risk candidate repeats across utility arms are excluded.','C calibration uses all928 inner diagnostic/calibration axes; zero unsafe inner accepts can mean no-op. No finite-sample or clinical guarantee.','Source-role panels use main native/multi/pair once and four separate constructed banks; shared source recordings are not independent observations.'],y=.095,spacing=.027);fig.subplots_adjust(left=.09,right=.98,top=.92,bottom=.21,hspace=.52,wspace=.30);save(fig,'risk_evidence_and_coverage')
    pins=dict(old['figures'])
    for path in changed:pins[path]=sha(ROOT/path)
    if len(changed)!=4 or len(pins)!=12:raise ValueError('exact4 repaired/12total required')
    for path,h in old['figures'].items():
        if path not in changed and sha(ROOT/path)!=h:raise ValueError('unrelated figure changed')
    write(PUB/'figure_receipt.json',{**old,'figures':pins,'supersedes_first_figure_receipt_sha256':sha(PUB/'figure_receipt_v1.json'),'figure_only_revision':True,'figure_repair_freeze_git':freeze['git_commit'],'PNG_visual_QA_still_required':True,'epoch':time.time()})
    write(PUB/'figure_repair_receipt.json',{'passed':True,'changed_artifacts':changed,'unchanged_artifacts':sorted(set(pins)-set(changed)),'old_figure_receipt_sha256':sha(PUB/'figure_receipt_v1.json'),'new_figure_receipt_sha256':sha(PUB/'figure_receipt.json'),'summary_sha256':sha(PUB/'summary.json'),'outer_metrics_sha256':sha(PUB/'outer_metrics.json'),'oracle_sha256':sha(PUB/'action_space_diagnostic.json'),'native_sha256':sha(PUB/'native_attribution.json'),'no_model_metric_threshold_or_endpoint_changes':True,'new_NN_forwards':0,'optimizer_updates':0,'source_Git':freeze['git_commit'],'epoch':time.time()});print({'figure_only_repair_complete':True,'changed':len(changed)},flush=True)
if __name__=='__main__':main()
