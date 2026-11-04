"""Six A62 aggregate figure families; no private IDs or model/data loading.

Rendered artifacts remain a separate runtime/visual-QA stage. Software-fixture
rendering is visibly watermarked and cannot be presented as A62 model evidence.
"""
import csv,hashlib,json,math
from pathlib import Path
import numpy as np
from echo_routing.temporal.event_risk_metrics import TABLES,HALVES,SEEDS,BASE,system

FAMILIES=('01_R0_calibration','02_R1_R2_matched','03_gross_short','04_risk_coverage','05_context_regression','06_training_resources')
COLORS={2026110461:'#176b87',2026110462:'#b24c66'}


def _number(v):
    if isinstance(v,(bool,np.bool_)) or not isinstance(v,(int,float,np.number)) or not math.isfinite(v):raise ValueError('finite actual aggregate required')
    return float(v)


def render(pub,rows,comparisons,inner_rows,calibration,regression,curves,resources=None,*,synthetic_fixture=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    pub=Path(pub);folder=pub/'figures';folder.mkdir(parents=True,exist_ok=True)
    if any((folder/(name+'.'+ext)).exists() for name in FAMILIES for ext in ['png','pdf','csv']):raise ValueError('immutable figure attempt exists; preserve prior artifacts, use distinct attempt folder')
    index={(r['table'],r['half'],r['system']):r for r in rows}
    ci={(r['table'],r['half'],r['seed'],r['contrast']):r for r in comparisons}
    if len(rows)!=156 or len(index)!=156 or len(comparisons)!=240 or len(ci)!=240 or len(inner_rows)!=26:raise ValueError('complete frozen156/240/26 aggregate inputs required')
    cells=[(t,h) for t in TABLES for h in HALVES];labels=[t.replace('_confirmation','').replace('a58_stress','a58 stress')+' / '+h[:3] for t,h in cells]
    artifacts=[];all_records={}
    style={'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':False,'pdf.fonttype':42,'ps.fonttype':42,'savefig.facecolor':'white'}
    scope='Conditional OPEN constructed banks; shared sources and two fixed seeds. Not new-patient or clinical safety evidence.'
    def finish(name,fig,title,records,subtitle=''):
        fig.suptitle(title,fontsize=14,fontweight='bold',y=.985)
        if subtitle:fig.text(.5,.947,subtitle,ha='center',fontsize=8,color='#475569')
        fig.text(.015,.018,scope,fontsize=7,color='#475569')
        if synthetic_fixture:fig.text(.5,.045,'SYNTHETIC SOFTWARE FIXTURE - NOT A62 RESULTS',ha='center',fontsize=11,color='#a31621',fontweight='bold')
        fig.tight_layout(rect=[0,.075,1,.92])
        for ext in ['png','pdf']:
            p=folder/(name+'.'+ext);fig.savefig(p,dpi=160,metadata={'Title':title});artifacts.append({'family':name,'path':str(p.relative_to(pub)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size})
        plt.close(fig)
        p=folder/(name+'.csv');fields=list(dict.fromkeys(k for row in records for k in row))
        with p.open('x',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
            for record in records:writer.writerow({k:json.dumps(v,sort_keys=True) if isinstance(v,(dict,list,tuple)) else v for k,v in record.items()})
        artifacts.append({'family':name,'path':str(p.relative_to(pub)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size});all_records[name]=len(records)
    def interval(ax,y,delta,bounds,color,marker='o',primary_bounds=None):
        ax.scatter([delta],[y],color=color,s=23,marker=marker,zorder=4)
        if primary_bounds is not None:ax.plot(primary_bounds,[y,y],color='#303846',lw=1,zorder=2)
        if bounds is not None:ax.plot(bounds,[y,y],color=color,lw=3,zorder=3)
    with plt.rc_context(style):
        fig,axes=plt.subplots(1,2,figsize=(13,8),sharey=True);records=[]
        taus={(r['seed'],r['policy']):r['tau'] for r in calibration['thresholds']+calibration['legacy_thresholds']}
        for ax,seed in zip(axes,SEEDS):
            for i,(table,half) in enumerate(cells):
                a=index[(table,half,system(seed,'M0'))];b=index[(table,half,system(seed,'L0'))];delta=_number(a['segment_f1_50_multi'])-_number(b['segment_f1_50_multi']);bounds=ci[(table,half,seed,'M0-L0')]['delta95']['segment_f1_50_multi']
                interval(ax,i,delta,bounds,COLORS[seed]);records.append({'table':table,'half':half,'seed':seed,'contrast':'M0-L0','F1_delta':delta,'delta95':bounds,'legacy_tau':taus[(seed,'L0')],'matched_CAL_tau':taus[(seed,'M0')]})
            ax.axvline(0,color='#94a3b8',lw=1);ax.set_title(f'seed {seed} | old r score, changed CAL only');ax.set_xlabel('Pooled multi F1 difference (M0 - L0)');ax.set_yticks(range(12));ax.set_yticklabels(labels)
        axes[0].invert_yaxis()
        finish(FAMILIES[0],fig,'R0: calibration-response control',records,'Same frozen S-L top / original N / old r. Bars: conditional 95% intervals; zero-crossing is not equivalence.')

        fig,axes=plt.subplots(1,2,figsize=(13,8),sharey=True);records=[]
        for ax,(left,right,title) in zip(axes,[('R1','M0','H1: action-event evidence package'),('R2','R1','H2: new anchor context')]):
            for seed in SEEDS:
                offset=-.13 if seed==SEEDS[0] else .13
                for i,(table,half) in enumerate(cells):
                    a=index[(table,half,system(seed,left))];b=index[(table,half,system(seed,right))];delta=_number(a['segment_f1_50_multi'])-_number(b['segment_f1_50_multi']);c=ci[(table,half,seed,left+'-'+right)];bounds=c['delta95']['segment_f1_50_multi'];primary=c.get('delta99_375',{}).get('segment_f1_50_multi') if table=='a62_confirmation' else None
                    interval(ax,i+offset,delta,bounds,COLORS[seed],primary_bounds=primary);records.append({'table':table,'half':half,'seed':seed,'contrast':left+'-'+right,'F1_delta':delta,'delta95':bounds,'delta99_375':primary,'primary_family':table=='a62_confirmation'})
                ax.scatter([],[],color=COLORS[seed],label=str(seed))
            ax.axvline(0,color='#94a3b8',lw=1);ax.set_title(title);ax.set_xlabel('Pooled multi F1 difference');ax.set_yticks(range(12));ax.set_yticklabels(labels);ax.legend(fontsize=8)
        axes[0].invert_yaxis()
        finish(FAMILIES[1],fig,'Prespecified matched R1/R2 contrasts',records,'Thick: 95%; thin dark for new A62: 99.375% on the SAME 5000 draws. Both seeds and every table retained.')

        fig,axes=plt.subplots(2,2,figsize=(14,9),sharex=True);records=[]
        policies=['L0','M0','R1','R2'];palette={'L0':'#64748b','M0':'#947d1b','R1':'#b24c66','R2':'#176b87'}
        for row_index,seed in enumerate(SEEDS):
            for policy in policies:
                hs=[];recover=[];lost=[]
                for table,half in cells:
                    r=index[(table,half,system(seed,policy))];g=r['gross_reference_position_effects'];H=sum(_number(g[k]['H']) for k in ['native','multi','pair']);R=sum(_number(g[k]['R']) for k in ['native','multi','pair']);s=r['action_statistics_source_role'];f=r['action_statistics_full_task'];rec=_number(s['accepted_new_JEPA_short16_true_units_recovered']);lo=_number(f['accepted_short16_true_units_lost']);hs.append(H);recover.append(rec);lost.append(lo)
                    records.append({'table':table,'half':half,'seed':seed,'policy':policy,'H_full_task':H,'R_full_task':R,'JEPA_exclusive_short16_recovered_source':rec,'gross_short16_lost_full_task':lo,'dFP16_full_task':f['accepted_effect_totals']['dFP']})
                axes[row_index,0].plot(range(12),hs,'o-',ms=3,color=palette[policy],label=policy)
                if policy in ['R1','R2']:
                    axes[row_index,1].plot(range(12),recover,'o-',ms=3,color=palette[policy],label=policy+' recovered (source)')
                    axes[row_index,1].plot(range(12),-np.asarray(lost),'x--',ms=4,color=palette[policy],label=policy+' lost (full task, negative axis)')
            for j in [0,1]:axes[row_index,j].axhline(0,color='#94a3b8',lw=1);axes[row_index,j].set_xticks(range(12));axes[row_index,j].set_xticklabels(labels,rotation=55,ha='right',fontsize=7);axes[row_index,j].legend(fontsize=7)
            axes[row_index,0].set_ylim(bottom=0,top=max(1,axes[row_index,0].get_ylim()[1]));axes[row_index,0].set_title(f'seed {seed}: gross new wrong positions H');axes[row_index,1].set_title(f'seed {seed}: short16 gross counts, kept separate')
        finish(FAMILIES[2],fig,'Gross harm and real short-segment preservation',records,'No cancellation by H-R or recovered-lost. Native/pair predictions reused across tables; cells are not independent cohorts.')

        fig,axes=plt.subplots(2,2,figsize=(13,8));records=[]
        inner={(r['population'],r['system']):r for r in inner_rows}
        for col,population in enumerate(['CAL','AUDIT']):
            for seed in SEEDS:
                ref=inner[(population,system(seed,'R2'))]
                for arm,linestyle in [('R1','--'),('R2','-')]:
                    diag=ref['risk_diagnostics'][arm]['N_top'];x=[];y=[]
                    for b in diag['bins']:
                        x.append((b['lower']+b['upper'])/2);y.append(np.nan if b['observed_unsafe_fraction'] is None else _number(b['observed_unsafe_fraction']));records.append({'population':population,'seed':seed,'arm':arm,'kind':'N_top_event_risk_bin',**{k:b[k] for k in ['lower','upper','upper_inclusive','count','unsafe','mean_score','observed_unsafe_fraction']},'BCE':diag['BCE'],'Brier':diag['Brier']})
                    axes[0,col].plot(x,y,'o'+linestyle,color=COLORS[seed],label=f'{seed} {arm}',ms=4)
                coverage=[]
                for policy in policies:
                    s=inner[(population,system(seed,policy))]['action_statistics_full_task'];a=_number(s['accepted_actions']);n=_number(s['N_admitted_tops']);value=a/n if n else np.nan;coverage.append(value);records.append({'population':population,'seed':seed,'policy':policy,'kind':'coverage_given_same_original_N','accepted':a,'N_tops':n,'fraction':None if n==0 else value})
                axes[1,col].plot(range(4),coverage,'o-',color=COLORS[seed],label=str(seed))
            axes[0,col].set_title(population+': observed unsafe fraction on original N-top');axes[0,col].set_xlabel('Uncalibrated q score, fixed 5 bins');axes[0,col].set_ylim(-.03,1.03);axes[0,col].legend(fontsize=7)
            axes[1,col].set_title(population+': coverage conditional on original N');axes[1,col].set_xticks(range(4));axes[1,col].set_xticklabels(policies);axes[1,col].set_ylim(-.03,1.03);axes[1,col].legend(fontsize=8)
        finish(FAMILIES[3],fig,'Calibration, event evidence and refusal coverage',records,'CAL is used for tau; source-separated AUDIT is diagnostic only. Missing bins/zero denominators remain gaps, not zeros.')

        fig=plt.figure(figsize=(14,9));grid=GridSpec(2,2,figure=fig,width_ratios=[1.5,1]);gate=fig.add_subplot(grid[:,0]);harm=fig.add_subplot(grid[0,1]);context=fig.add_subplot(grid[1,1]);records=[]
        ordered=sorted(regression,key=lambda r:(r['axis'],r['seed'],policies.index(r['policy'])))
        if len(ordered)!=16 or {(r['axis'],r['seed'],r['policy']) for r in ordered}!={(a,s,p) for a in ['Case_A','Case_B'] for s in SEEDS for p in policies}:raise ValueError('exact16 anonymous originaltwo-axis regression records required')
        for i,r in enumerate(ordered):
            if not isinstance(r['accepted'],bool):raise ValueError('actual anonymous regression gate decision required')
            score=r['gate_score'];tau=_number(r['tau']);margin=np.nan if score is None else _number(score)-tau
            gate.scatter([margin],[i],color='#176b87' if r['accepted'] else '#a31621',marker='o' if r['accepted'] else 'x',s=35)
            records.append({k:r[k] for k in ['axis','seed','policy','accepted','gate_score','tau','H','R','short16_lost']})
        gate.set_yticks(range(16));gate.set_yticklabels([r['axis'].replace('Case_','')+' / '+str(r['seed'])[-3:]+' / '+r['policy'] for r in ordered]);gate.invert_yaxis();gate.axvline(0,color='#64748b',lw=1);gate.set_xlabel('Gate score - tau (r and q have different meanings)');gate.set_title('Accepted: circle; refused: cross; None: no score')
        picked=[r for r in ordered if r['policy']=='R2'];x=np.arange(4);harm.bar(x-.16,[_number(r['H']) for r in picked],.32,label='candidate H',color='#a31621');harm.bar(x+.16,[_number(r['R']) for r in picked],.32,label='candidate R',color='#176b87');harm.set_title('Same frozen top candidate, before event veto');harm.set_xticks(x);harm.set_xticklabels([r['axis'].replace('Case_','')+'/'+str(r['seed'])[-3:] for r in picked]);harm.legend(fontsize=8)
        fields=['minimum_overlap_over_full_run','minimum_full_Base_run_A08_margin']
        for i,key in enumerate(fields):
            values=[_number(r['context_summary'][key]) for r in picked];context.plot(x,values,'o-',label=key.replace('minimum_','min ').replace('_',' '));
            for r,value in zip(picked,values):records.append({'axis':r['axis'],'seed':r['seed'],'policy':'R2','context_observable':key,'value':value})
        context.set_xticks(x);context.set_xticklabels([r['axis'].replace('Case_','')+'/'+str(r['seed'])[-3:] for r in picked]);context.axhline(0,color='#94a3b8',lw=1);context.set_title('Label-free full-anchor observables');context.legend(fontsize=7)
        finish(FAMILIES[4],fig,'Anchor context and anonymous development regressions',records,'Exactly two previously inspected axes, not 16 independent cases. Candidate harm and actual acceptance remain distinct.')

        fig,axes=plt.subplots(1,2,figsize=(13,7));records=[]
        for seed in SEEDS:
            for arm,styleline in [('R1','--'),('R2','-')]:
                selected=sorted([r for r in curves if r['seed']==seed and r['arm']==arm],key=lambda r:r['update'])
                if not selected or selected[-1]['update']!=6000 or len({r['update'] for r in selected})!=len(selected):raise ValueError('allfour complete6000 actual training curves required')
                axes[0].plot([_number(r['update']) for r in selected],[_number(r['loss']) for r in selected],styleline,color=COLORS[seed],label=f'{seed} {arm}')
                records.extend({'kind':'training','seed':seed,'arm':arm,'update':r['update'],'loss':r['loss']} for r in selected)
        axes[0].set_title('Actual FIT BCE, fixed 6000-step heads');axes[0].set_xlabel('Optimizer updates per head');axes[0].set_ylabel('Unweighted candidate-event BCE');axes[0].legend(fontsize=8)
        if resources is None:
            axes[1].axis('off');axes[1].text(.05,.5,'Resource snapshot not yet supplied.\nFinal ledger remains authoritative.',fontsize=11)
        else:
            specs=[('incremental_GPU_seconds',1800,'GPU occupied seconds'),('incremental_CPU_seconds',21600,'Measured CPU seconds'),('incremental_tracked_storage_upper_bytes',1250000000,'Tracked storage upper bytes')];fraction=[]
            for key,cap,label in specs:
                actual=_number(resources[key])
                if actual<0:raise ValueError('nonnegative actual resource snapshot required')
                fraction.append(actual/cap);records.append({'kind':'actual_resource_snapshot','resource':key,'actual':actual,'incremental_cap':cap,'fraction':actual/cap})
            axes[1].barh(range(3),fraction,color='#176b87');axes[1].axvline(1,color='#a31621',ls='--');axes[1].set_yticks(range(3));axes[1].set_yticklabels([s[2] for s in specs]);axes[1].set_xlabel('Actual snapshot / approved incremental cap');axes[1].set_title('Inherited cumulative accounting is not reset')
            for i,(key,cap,label) in enumerate(specs):axes[1].text(fraction[i]+.02,i,f'{resources[key]:,.0f} / {cap:,}',va='center',fontsize=8)
            axes[1].set_xlim(0,max(1.4,max(fraction)*1.4))
        finish(FAMILIES[5],fig,'Training and actual resource accounting',records,'No CAL/outer checkpoint selection. Reserves and forecasts are not spend; plotting/closing tails may postdate this snapshot.')
    return {'families':6,'PNG':6,'PDF':6,'CSV':6,'artifacts':artifacts,'plotted_record_counts':all_records,'synthetic_fixture':synthetic_fixture,'actual_visual_QA_complete':False,'scope':'aggregate rendering only; actual scientific/resource/OS audits and PNG plus renderedPDF visual QA remain separate'}
