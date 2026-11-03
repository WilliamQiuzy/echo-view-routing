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
    freeze=verify_freeze('plot_freeze.json');summary=read(PUB/'summary.json');assert summary['formal_optimizer_updates']==48000 and summary['paired_comparison_count']==132;arms=['D0','D1','D2','D3'];seeds=[2026110358,2026110359];halves=['selection','calibration'];colors=['#66717e','#2176ae','#df6b21','#8052a0'];keys=[('native_macro_f1','Native macro-F1 delta (higher preferred)'),('segment_f1_50_multi','Multi segment-F1@50 delta (higher preferred)'),('short_true_unmatched_16samples','Missed true short<=16 segments delta (lower preferred)'),('short_spurious_segments_16samples','False short<=16 segments delta (lower preferred)')];fig,axes=plt.subplots(4,2,figsize=(11,13),layout='constrained')
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
    handles=[Line2D([],[],marker='o',mec='#243746',mfc='#243746',ls='none',label='head seed2026110358'),Line2D([],[],marker='s',mec='#243746',mfc='#243746',ls='none',label='head seed2026110359'),Line2D([],[],marker='o',mec='#243746',mfc='white',ls='none',label='hollow: duration-shift multi bank')];fig.legend(handles=handles,loc='outside lower center',ncol=3,frameon=False,fontsize=9);fig.suptitle('A58 fixed6000 vs original A08-s0 + B3\nOPEN reused development; all arms and both seeds retained',fontsize=14,fontweight='bold');fig.supxlabel('D0: g=0  |  D1: A08 trajectories + learned g  |  D2: A08 + JEPA trajectories + g  |  D3: JEPA-only + g\n95% paired descriptive intervals: whole native sessions / conditional fixed streams. Sessions are not confirmed patients.\nStress reuses identical native/pair predictions; its120multi streams are newly drawn under the4..80fragment rule. Inner-selected tables and all metrics remain in CSV/JSON.',fontsize=8);png=PUB/'fixed6000_paired_overview.png';pdf=PUB/'fixed6000_paired_overview.pdf';fig.savefig(png,dpi=180);fig.savefig(pdf);plt.close(fig);write(PUB/'figure_receipt.json',{'plot_freeze_git':freeze['git_commit'],'summary_sha256':sha(PUB/'summary.json'),'files':{p.name:sha(p) for p in [png,pdf]},'data':'all4arms/bothseeds/bothhalves/fixed6000;predeclaredoriginalA08s0B3pairedcontrasts','no_private_identifiers_or_media':True,'clinical_or_patient_population_CI':False});print(json.dumps({'figure_written':True,'files':[png.name,pdf.name]}),flush=True)
if __name__=='__main__':main()
