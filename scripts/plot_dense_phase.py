#!/usr/bin/env python3
"""Completed-phase aggregate figures; no case identifiers or model calls."""
from pathlib import Path
import hashlib,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
PUB=ROOT/'docs/results/2026-11-02_dense_target_sequence'
def read(p):return json.loads(Path(p).read_text())
def main():
    paths=[PUB/'summary.json',PUB/'full_refit/summary.json',PUB/'multidepth/summary.json'];core,refit,md=map(read,paths)
    assert core['raw_reader_metrics_independently_reproduced'] and refit['raw_metrics_independently_reproduced'] and md['raw_reader_metrics_independently_reproduced']
    out=PUB/'phase_comparison';out.mkdir(exist_ok=True)
    specs=[('A08 s0 B3',core,'A08_original_B3_s0','#238b45'),('J1 raw',core,'seed2026110257_J-high_fixed6000|raw','#2166ac'),('J2 raw',core,'seed2026110258_J-high_fixed6000|raw','#67a9cf'),('R-CE raw',refit,'R-CE_fixed6000|raw','#ef8a62'),('R-ST raw',refit,'R-ST_fixed6000|raw','#b2182b'),('MD-J1 raw',md,'seed2026110257_J-MD_fixed6000|raw','#762a83'),('MD-J2 raw',md,'seed2026110258_J-MD_fixed6000|raw','#af8dc3')]
    rows=[]
    for half in ['selection','calibration']:
        assert core['systems'][half]['A08_original_B3_s0']==refit['systems'][half]['A08_original_B3_s0']==md['systems'][half]['A08_original_B3_s0']
        for label,summary,key,color in specs:rows.append({'half':half,'label':label,'system_key':key,**summary['systems'][half][key]})
    pd.json_normalize(rows).to_csv(out/'fixed6000_raw_comparison.csv',index=False)
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    fig,axes=plt.subplots(1,2,figsize=(12,4.6))
    for half,ax in zip(['selection','calibration'],axes):
        for label,summary,key,color in specs:
            m=summary['systems'][half][key];ax.scatter(m['native_macro_f1'],m['segment_f1_50_multi'],s=85 if label.startswith('A08') else 45,marker='*' if label.startswith('A08') else 'o',color=color,label=label)
        ax.set(title=half,xlabel='Native five-class macro-F1',ylabel='Multi-stream segment F1@50');ax.grid(alpha=.2)
    axes[1].legend(loc='center left',bbox_to_anchor=(1.01,.5),fontsize=8);fig.suptitle('Fixed 6000-update attempts on reused development; A08 uses original B3')
    fig.tight_layout();fig.savefig(out/'fixed6000_task_tradeoff.png',dpi=180);fig.savefig(out/'fixed6000_task_tradeoff.svg');plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(13,7));x=np.arange(len(specs))
    for col,half in enumerate(['selection','calibration']):
        m=[summary['systems'][half][key] for _,summary,key,_ in specs];labels=[s[0] for s in specs]
        for row,(key,title) in enumerate([('short_true_unmatched_16samples','Missed true short segments'),('short_spurious_segments_16samples','Spurious predicted short segments')]):
            ax=axes[row,col];ax.bar(x,[a[key] for a in m],color=[s[3] for s in specs]);ax.set(title=half if row==0 else '',ylabel=title,xticks=x,xticklabels=labels);ax.tick_params(axis='x',labelrotation=25);ax.grid(axis='y',alpha=.2)
        axes[0,col].text(.02,.97,f'True short denominator: {m[0]["short_true_segments_16samples"]}',transform=axes[0,col].transAxes,va='top',fontsize=8)
    fig.suptitle('Short segments <=16 samples / 1.6 seconds; FN and FP shown separately')
    fig.tight_layout();fig.savefig(out/'fixed6000_short_segments.png',dpi=180);fig.savefig(out/'fixed6000_short_segments.svg');plt.close(fig)
    (out/'figure_inputs.json').write_text(json.dumps({'input_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},'fixed_rows':[{k:v for k,v in r.items() if k in ['half','label','system_key']} for r in rows],'new_model_forwards':0,'fixed6000_raw_only_primary_display':True,'all_modes_and_controls_retained_in_component_reports':True,'A08_is_operational_reference_not_matched_architecture_control':True,'reused_development_not_independent_validation':True},indent=2)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(10,3.8));curves=[]
    for seed,style in [(2026110257,'-'),(2026110258,'--')]:
        for arm,color in [('F-MD','#2166ac'),('J-MD','#762a83')]:
            records=[]
            for step in range(500,6001,500):
                p=ROOT/f'runs/a57_dense_sequence_v1/multidepth_v1/seed{seed}/{arm}/inner_step{step}/metrics.json';r=read(p);assert r['seed']==seed and r['arm']==arm and r['step']==step
                record={k:r[k] for k in ['seed','arm','step','native_macro_f1','segment_f1_50_multi','model_checkpoint_sha256','input_sequence_sha256']};record['record_sha256']=hashlib.sha256(p.read_bytes()).hexdigest();curves.append(record);records.append(record)
            for ax,(key,title) in zip(axes,[('native_macro_f1','Native five-class macro-F1'),('segment_f1_50_multi','Multi-stream segment F1@50')]):ax.plot([r['step'] for r in records],[r[key] for r in records],style,color=color,label=f'{arm}, seed {seed-2026110256}');ax.set(xlabel='Optimizer updates',ylabel=title,ylim=(0,1.02));ax.grid(alpha=.2)
    axes[0].legend(fontsize=8);fig.suptitle('MD repeated training-only inner calibration; not independent validation');fig.tight_layout();fig.savefig(out/'multidepth_inner_curves.png',dpi=180);fig.savefig(out/'multidepth_inner_curves.svg');plt.close(fig)
    (out/'multidepth_inner_curves.json').write_text(json.dumps({'rows':curves,'source_records_bound_in_model_manifest':True,'new_model_forwards':0},indent=2)+'\n')
    print(json.dumps({'phase_figures':6,'fixed_rows':len(rows),'new_model_forwards':0}),flush=True)
if __name__=='__main__':main()
