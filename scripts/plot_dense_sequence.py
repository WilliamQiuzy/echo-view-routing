#!/usr/bin/env python3
"""A57 aggregate-only standalone scientific figures; no cases, weights or neural calls."""
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
PUB=ROOT/'docs/results/2026-11-02_dense_target_sequence';RUN=ROOT/'runs/a57_dense_sequence_v1'
def read(p):return json.loads(Path(p).read_text())
def main():
 summary=read(PUB/'summary.json');models=read(PUB/'models_manifest.json');selected=read(PUB/'inner_lr_selection.json')['selected_joint_arm'];figures=PUB/'figures';figures.mkdir(exist_ok=True);plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
 fig,axes=plt.subplots(1,3,figsize=(12,3.8));keys=[('native_macro_f1','Native five-class macro-F1'),('segment_f1_50_multi','Multi-stream segment F1@50'),('boundary_recall_multi','True-change recall (0.5 s)')];colors={'L':'#727272','F':'#2166ac','J-low':'#ef8a62','J-high':'#b2182b'}
 for model in models['models']:
  seed,arm,count=model['seed'],model['arm'],model['updates'];rows=[read(RUN/f'seed{seed}'/arm/f'inner_step{u}/metrics.json') for u in range(500,count+1,500)];style='-' if seed==2026110257 else '--';label=f'{arm}, seed {1 if seed==2026110257 else 2}'
  for ax,(key,title) in zip(axes,keys):ax.plot([r['step'] for r in rows],[r[key] for r in rows],style,color=colors[arm],label=label,linewidth=1.5);ax.set(xlabel='Optimizer updates',ylabel=title);ax.set_ylim(0,1.02);ax.grid(alpha=.2)
 axes[0].legend(fontsize=7,loc='lower right');fig.suptitle('A57 training-only inner calibration (repeated selection data)');fig.tight_layout();fig.savefig(figures/'inner_learning_curves.png',dpi=180);fig.savefig(figures/'inner_learning_curves.svg');plt.close(fig)
 fig,axes=plt.subplots(1,2,figsize=(14,4.8))
 for half,ax in zip(['selection','calibration'],axes):
  systems=summary['systems'][half]
  for seed,index in [(2026110257,1),(2026110258,2)]:
   for arm,color in [('F','#2166ac'),(selected,'#b2182b')]:
    for mode,marker in [('raw','o'),('inner_B3','s')]:
     m=systems[f'seed{seed}_{arm}_fixed6000|{mode}'];ax.scatter(m['native_macro_f1'],m['segment_f1_50_multi'],facecolors=color if index==1 else 'none',edgecolors=color,marker=marker,s=55,linewidths=1.5,label=f'{"F" if arm=="F" else "J"}{index} {"raw" if mode=="raw" else "B3"}')
  for seed in [0,1,2]:
   m=systems[f'A08_original_B3_s{seed}'];ax.scatter(m['native_macro_f1'],m['segment_f1_50_multi'],color=['#238b45','#74c476','#005a32'][seed],marker='*',s=85,label=f'A08 s{seed}')
  ax.set(xlabel='Native five-class macro-F1',ylabel='Multi-stream segment F1@50',title=f'Reused development: {half}');ax.grid(alpha=.2);ax.legend(loc='center left',bbox_to_anchor=(1.01,.5),fontsize=7)
 fig.suptitle('Fixed 6000-update models; original A08+B3 references');fig.tight_layout();fig.savefig(figures/'development_pareto.png',dpi=180);fig.savefig(figures/'development_pareto.svg');plt.close(fig)
 fig,axes=plt.subplots(2,2,figsize=(10,7))
 for half,column in [('selection',0),('calibration',1)]:
  names=[f'seed2026110257_F_fixed6000|raw',f'seed2026110257_{selected}_fixed6000|raw',f'seed2026110258_F_fixed6000|raw',f'seed2026110258_{selected}_fixed6000|raw','A08_original_B3_s0'];labels=['F1 raw','J1 raw','F2 raw','J2 raw','A08 s0 B3'];records=[summary['systems'][half][n] for n in names];x=np.arange(len(names));axes[0,column].bar(x-.18,[r['short_true_unmatched_16samples'] for r in records],.36,label='Missed true short segments');axes[0,column].bar(x+.18,[r['short_spurious_segments_16samples'] for r in records],.36,label='Spurious short segments');axes[0,column].set(title=half,ylabel='Segments (short <=16 samples / 1.6 s)',xticks=x,xticklabels=labels);axes[1,column].bar(x,[r['native_A4C_A5C_errors'] for r in records],color=['#2166ac','#b2182b','#2166ac','#b2182b','#238b45']);axes[1,column].set(ylabel='Native A4C/A5C bidirectional errors',xticks=x,xticklabels=labels)
 axes[0,0].legend(fontsize=7);fig.suptitle('Full task diagnostics: fewer cuts alone do not establish improvement');fig.tight_layout();fig.savefig(figures/'short_segments_and_native_errors.png',dpi=180);fig.savefig(figures/'short_segments_and_native_errors.svg');plt.close(fig)
 print(json.dumps({'aggregate_figures_written':6}),flush=True)
if __name__=='__main__':main()
