#!/usr/bin/env python3
"""Fixed6000 complete-arm comparison figure from the sealed paired aggregate only."""
import os
os.environ.setdefault('MPLBACKEND','Agg');os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scripts.structured_common import PUB,read,write,sha,verify_freeze

def main():
    freeze=verify_freeze('figure_layout_freeze.json');summary=read(PUB/'summary.json');assert summary['formal_optimizer_updates']==48000 and summary['paired_comparison_count']==132;arms=['D0','D1','D2','D3'];seeds=[2026110358,2026110359];halves=['selection','calibration'];colors=['#66717e','#2176ae','#df6b21','#8052a0'];keys=[('native_macro_f1','Native macro-F1 delta\n(higher preferred)'),('segment_f1_50_multi','Multi segment-F1@50 delta\n(higher preferred)'),('short_true_unmatched_16samples','Missed true short<=16 delta\n(lower preferred)'),('short_spurious_segments_16samples','False short<=16 delta\n(lower preferred)')];fig,axes=plt.subplots(4,2,figsize=(12,14));fig.subplots_adjust(left=.1,right=.98,bottom=.18,top=.9,hspace=.35,wspace=.25)
    for col,half in enumerate(halves):
        for row,(metric,label) in enumerate(keys):
            ax=axes[row,col];ax.axhline(0,color='#243746',lw=.9);ax.grid(axis='y',color='#e6ebef',lw=.6);ax.set_axisbelow(True)
            for i,arm in enumerate(arms):
                for j,seed in enumerate(seeds):
                    for shift,table in enumerate(['fixed6000','duration_stress_fixed6000']):
                        if row==0 and shift:continue
                        matches=[r for r in summary['paired_comparisons'] if r['table']==table and r['half']==half and r['seed']==seed and r['contrast']==arm+'-A08_s0_original_B3'];assert len(matches)==1;r=matches[0];value=r['delta'][metric];interval=r['delta95'][metric];x=i+(j-.5)*.24+(shift-.5)*.07;color=colors[i];marker='o' if j==0 else 's'
                        if value is None:continue
                        if interval is not None:ax.vlines(x,interval[0],interval[1],color=color,lw=1.1,alpha=.75);ax.hlines([interval[0],interval[1]],x-.025,x+.025,color=color,lw=1.1,alpha=.75)
                        ax.plot(x,value,marker=marker,ms=5.5,mec=color,mfc=color if shift==0 else 'white',linestyle='none',zorder=3)
            ax.set_xticks(range(4),arms);ax.set_xlim(-.5,3.5);ax.set_ylabel(label,fontsize=9)
            if row==0:ax.set_title(half.capitalize()+' (36 source sessions)',fontweight='bold')
    handles=[Line2D([],[],marker='o',mec='#243746',mfc='#243746',ls='none',label='head seed2026110358'),Line2D([],[],marker='s',mec='#243746',mfc='#243746',ls='none',label='head seed2026110359'),Line2D([],[],marker='o',mec='#243746',mfc='white',ls='none',label='hollow: duration-shift multi bank')]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.54,.105),ncol=3,frameon=False,fontsize=9)
    fig.suptitle('A58 fixed6000 vs original A08-s0 + B3\nOPEN reused development; all arms and both seeds retained',fontsize=15,fontweight='bold',y=.985)
    fig.text(.54,.088,'D0: g=0  |  D1: A08 trajectories + learned g  |  D2: A08 + JEPA trajectories + g  |  D3: JEPA-only + g',ha='center',fontsize=8)
    fig.text(.54,.065,'95% paired descriptive intervals: whole native sessions / conditional fixed streams.\nSessions are not confirmed patients; shared-source constructed banks are not independent clinical evidence.',ha='center',fontsize=8)
    fig.text(.54,.037,'Stress reuses identical native/pair predictions; its120multi streams are newly drawn under the4..80fragment rule.\nInner-selected tables and all native, boundary and short-segment metrics remain in CSV/JSON.',ha='center',fontsize=8)
    from scripts.structured_common import RUN
    import shutil
    history=RUN/'figure_layout_history_v1';history.mkdir(exist_ok=True);old={}
    for name in ['fixed6000_paired_overview.png','fixed6000_paired_overview.pdf','figure_receipt.json']:
        source=PUB/name
        if source.exists():old[name]=sha(source);shutil.copy2(source,history/name)
    png=PUB/'fixed6000_paired_overview.png';pdf=PUB/'fixed6000_paired_overview.pdf';fig.savefig(png,dpi=180);fig.savefig(pdf);plt.close(fig)
    receipt={'figure_layout_freeze_git':freeze['git_commit'],'summary_sha256':sha(PUB/'summary.json'),'files':{p.name:sha(p) for p in [png,pdf]},'layout_revision_only':True,'superseded_v1_artifact_SHA256':old,'all_data_points_intervals_arm_seed_half_and_reference_inputs_unchanged':True,'no_private_identifiers_or_media':True,'clinical_or_patient_population_CI':False}
    write(PUB/'figure_receipt.json',receipt);write(PUB/'figure_layout_repair.json',{'reason':'separatelegendandfootersandwraplongylabels','before':old,'after':receipt['files'],'sealed_summary_sha256':sha(PUB/'summary.json'),'model_metric_or_statistical_change':False,'original_layout_retained_server_only':True});print(json.dumps({'figure_layout_repaired':True,'files':[png.name,pdf.name]}),flush=True)
if __name__=='__main__':main()
