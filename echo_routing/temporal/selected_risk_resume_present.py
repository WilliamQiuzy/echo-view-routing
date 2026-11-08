"""A63 continuation aggregate presentation, never a model or safety audit.

Only anonymous fixed-scope projections reach CSV/plots. Actual sealed input,
private recomputation and budget checks belong to the caller. Rendering never
sets visual QA, overall completion, patient independence or clinical validity.
"""
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import textwrap
import tempfile

import numpy as np

from echo_routing.temporal import selected_risk_metrics as ep
from echo_routing.temporal import selected_risk_figures as stageS
from echo_routing.temporal.selected_risk_resume_storage import _parent

CLASSES = ('PLAX', 'PSAX', 'A4C', 'A5C', 'SC4C')
EFFECTS = ('H','R','dE','dFP','u','true_recovered','true_lost','short10_recovered',
           'short10_lost','short16_recovered','short16_lost','JEPA_exclusive_short16_recovered')
SUPPORT_COUNTS = ('axes','N_admitted_tops','available_tops','positive_u_available_tops',
                  'union_recovered_true_units','union_recovered_short10_true_units','union_recovered_short16_true_units')


def _real(v):
    if isinstance(v,(bool,np.bool_)) or not isinstance(v,(int,float,np.integer,np.floating)) or not math.isfinite(v):
        raise ValueError('finite aggregate scalar required')
    return float(v)


def _table(rows, specs, keys):
    return ep._matrix(rows, specs, keys)


def _effects(value):
    out={}
    for k in EFFECTS:
        out[k]=_real(value[k]) if k=='u' else ep._count(value[k],signed=k in ('dE','dFP'))
    if out['dE']!=out['H']-out['R'] or out['JEPA_exclusive_short16_recovered']>out['short16_recovered']:
        raise ValueError('complete reconciled anonymous effect counts required')
    return out


def _support_kind(value,kind,safety):
    """Explicit anonymous projection; never dump arbitrary result/private keys."""
    out={k:ep._count(value[k]) for k in SUPPORT_COUNTS}
    if not out['positive_u_available_tops']<=out['available_tops']<=out['N_admitted_tops']<=out['axes']:
        raise ValueError('support availability denominators differ')
    for k in ('all_N_top_effects','all_available_top_effects'):out[k]=_effects(value[k])
    safe=out['all_available_top_effects']
    if safe['H'] or safe['dFP']>0 or safety=='acceptance-safe' and safe['short16_lost']:
        raise ValueError('available support must retain its declared safety filter')
    if kind=='multi':
        oracle=value['oracle'];o={k:ep._count(oracle[k]) for k in ('M','P','T','numerator','denominator','iterations')}
        f=_real(oracle['F1'])
        if (not o['denominator'] or o['numerator']!=2*o['M'] or o['denominator']!=o['P']+o['T'] or
                o['M']>min(o['P'],o['T']) or f!=o['numerator']/o['denominator']):
            raise ValueError('support pooled F1 must match exact anonymous counts')
        out['oracle']=dict(o,F1=f)
        out['joint_chosen_tops']=ep._count(value['joint_chosen_tops']);out['joint_chosen_effects']=_effects(value['joint_chosen_effects'])
        if out['joint_chosen_tops']>out['available_tops']:
            raise ValueError('joint choice exceeds available original tops')
    else:
        if value['pooled_F1_objective_member'] is not False:
            raise ValueError('native/pair are not the pooled multi objective')
        out['pooled_F1_objective_member']=False
    return out


def _native_public(value):
    fields=('positions','native_sequences','Base_correct_to_wrong','Base_wrong_to_correct','Base_frame_errors',
        'new_frame_errors','frame_error_delta','changed_positions','changed_native_sequences','accepted_native_actions',
        'majority_changed_native_sequences','both_correct','both_wrong_changed_class','both_wrong_unchanged_class',
        'new_source_forwards','new_optimizer_updates','all_native_majorities_mechanically_equal_Base',
        'majority_equality_is_frame_safety_claim','native_and_pair_reuse_across_tables_is_independent_evidence',
        'true_Base_prediction_joint5cube','true_Base_frame_confusion5','true_prediction_frame_confusion5',
        'Base_prediction_frame_transition5')
    flags={'all_native_majorities_mechanically_equal_Base','majority_equality_is_frame_safety_claim',
        'native_and_pair_reuse_across_tables_is_independent_evidence'}
    matrices={'true_Base_prediction_joint5cube':(5,5,5),'true_Base_frame_confusion5':(5,5),
        'true_prediction_frame_confusion5':(5,5),'Base_prediction_frame_transition5':(5,5)}
    out={}
    for k in fields:
        if k not in value:continue
        v=value[k]
        if k in flags:
            if type(v) is not bool:raise ValueError('explicit anonymous boolean aggregate required')
            out[k]=v
        elif k in matrices:
            a=np.asarray(v)
            if a.shape!=matrices[k] or a.dtype.kind not in 'iu' or (a<0).any():
                raise ValueError('anonymous fixed-shape nonnegative integer cube/confusion required')
            out[k]=a.tolist()
        else:out[k]=ep._count(v,signed=k=='frame_error_delta')
    out['classes']=list(CLASSES)
    allowed_group_numbers={'actions','actual_H','actual_R','actual_dE','actual_dFP','actual_delta_M','actual_delta_P',
        'actual_u','actual_v','added_cuts','changed_positions_sum','interval_length_sum','old_pFP_sum','old_predicted_H_sum',
        'old_predicted_R_sum','old_r_sum','predicted_utility_sum','q_R1_sum','q_R2_sum','q_C0_sum','q_C1_sum',
        'new_wrong_positions_touch_long_wrong_segment','newly_wrong_positions_extending_old_wrong_run',
        'newly_wrong_positions_in_long_wrong_segments','newly_wrong_positions_new_wrong_segment','wrong_run_crossed_short16_to_long'}
    groups=[]
    for group in value.get('accepted_groups',[]):
        row={k:_real(v) for k,v in group.items() if k in allowed_group_numbers}
        for k in ('Base_changed_classes','prediction_changed_classes'):
            if k in group:
                if not isinstance(group[k],list) or any(v not in CLASSES for v in group[k]):raise ValueError('anonymous fixed classes required')
                row[k]=group[k]
        if 'true_class' in group:
            if group['true_class'] not in CLASSES:raise ValueError('anonymous fixed true class required')
            row['true_class']=group['true_class']
        if 'provenance' in group:
            if group['provenance'] not in ('JEPA_only','old_only','both'):raise ValueError('anonymous candidate provenance required')
            row['provenance']=group['provenance']
        groups.append(row)
    out['accepted_groups']=groups
    return out


def _validate(rows, comparisons, inner, support, native, regression, curves, resources):
    p = ep.readout_plan(); verdict = ep.acceptance(rows, comparisons)
    r = _table(rows,p['rows'],('table','half','seed','policy'))
    ci = _table(comparisons,p['comparisons'],('table','half','seed','left_policy','right_policy'))
    inn = _table(inner,p['inner_rows'],('population','seed','policy'))
    for row in inn.values():
        if (not 0 <= _real(row['segment_f1_50_multi']) <= 1 or row['inner_pair_bank_absent'] is not True or
                row['segment_f1_50_pairs'] is not None or ep._count(row['pair_reference_positions']) != 0):
            raise ValueError('complete pair-absent CAL/AUDIT metrics required')
    support_specs=[{k:s[k] for k in ('table','half','seed','safety')} for s in p['support_cells']]
    sup = _table(support,support_specs,('table','half','seed','safety'))
    for row in sup.values():
        result=row['result']
        if (type(row['readonly_A62_reference']) is not bool or row['readonly_A62_reference'] != (row['table']!='a63_confirmation') or
                result['safety']!=row['safety'] or not {'native','multi','pair'}<=set(result) or
                not 0 <= _real(result['multi']['oracle']['F1']) <= 1):
            raise ValueError('all56 exact old/new N-top/noop support records required')
        ep._count(result['multi']['joint_chosen_effects']['JEPA_exclusive_short16_recovered'])
        for kind in ep.KINDS:_support_kind(result[kind],kind,row['safety'])
    n_specs=[dict(population=s['half'],seed=s['seed'],policy=s['policy']) for s in p['base_native_references']+p['native_cells']]
    nat = _table(native,n_specs,('population','seed','policy'))
    for row in nat.values():
        result=row['result'];cube=np.asarray(result['true_Base_prediction_joint5cube'])
        if cube.shape!=(5,5,5) or cube.dtype.kind not in 'iu' or (cube<0).any():
            raise ValueError('complete nonnegative native truth/Base/prediction cube required')
        t,b,n=np.indices((5,5,5));h=int(cube[(t==b)&(n!=t)].sum());repair=int(cube[(t!=b)&(n==t)].sum())
        point=r[('main',row['population'],row['seed'],row['policy'])]
        if (ep._count(result['Base_correct_to_wrong'])!=h or ep._count(result['Base_wrong_to_correct'])!=repair or
                point['H']['native']!=h or point['R']['native']!=repair):
            raise ValueError('native cube gross effects must match exact main readout')
        _native_public(result)
    reg_specs=[{k:s[k] for k in ('axis','seed','policy')} for s in p['regression_records']]
    reg = _table(regression,reg_specs,('axis','seed','policy'))
    for row in reg.values():
        if any(type(row[k]) is not bool for k in ('top_exists','N_admitted','accepted')):
            raise ValueError('explicit regression top/N/accept flags required')
        if not row['top_exists'] and (row['N_admitted'] or row.get('gate_score') is not None or
                any(row[k] is not None for k in ('H','R','dFP16','short16_lost')) or
                row.get('candidate_key_sha256') is not None or row.get('context_summary') is not None):
            raise ValueError('absent top potential/score/key/context must remain null')
        if row['top_exists']:
            key=row.get('candidate_key_sha256')
            if (any(row[k] is None for k in ('H','R','dFP16','short16_lost')) or not isinstance(key,str) or
                    len(key)!=64 or any(c not in '0123456789abcdef' for c in key)):
                raise ValueError('existing top requires complete potential effects and anonymous candidate SHA')
        for k in ('H','R','dFP16','short16_lost'):
            if row[k] is not None: ep._count(row[k],signed=k=='dFP16')
            ep._count(row['actual_'+k],signed=k=='dFP16')
            if (not row['accepted'] and row['actual_'+k]!=0 or
                    row['accepted'] and row['actual_'+k]!=row[k]):
                raise ValueError('case potential vs actual accepted effects must reconcile')
        if row['accepted'] and (not row['top_exists'] or row['policy']!='E' and not row['N_admitted']):
            raise ValueError('regression same-top/no-fallback gate differs')
        for k in ('gate_score','tau'):
            if row[k] is not None and not 0 <= _real(row[k]) <= 1:
                raise ValueError('nullable unit-range original score/threshold required')
        context=row.get('context_summary')
        if context is not None:
            if not isinstance(context,dict) or set(context)!={'minimum_overlap_over_full_run','minimum_full_Base_run_A08_margin'}:
                raise ValueError('only declared anonymous context aggregates allowed')
            for value in context.values():_real(value)
    series={(s,a):[] for s in ep.SEEDS for a in ('C0','C1')}
    for row in curves:
        key=(row['seed'],row['arm'])
        if (type(row['seed']) is not int or key not in series or type(row['update']) is not int or
                not 1<=row['update']<=6000 or _real(row['loss'])<0):
            raise ValueError('four complete nonnegative fixed6000 risk loss series required')
        series[key].append(row)
    if any([x['update'] for x in v]!=list(range(1,6001)) for v in series.values()):
        raise ValueError('exact ordered6000 updates per C0/C1 seed required')
    from echo_routing.temporal.selected_risk_resume_contract import STORAGE_AMENDMENT_SHA,STORAGE_APPROVAL
    cap=1000000000
    if resources.get('storage_cap')==1300000000 and resources.get('storage_amendment_sha256')==STORAGE_AMENDMENT_SHA and resources.get('storage_approval_receipt_sha256')==STORAGE_APPROVAL['sha256']:cap=1300000000
    caps={'CPU_cap':28800,'GPU_cap':5400,'storage_cap':cap}
    if any(not ep._same(resources.get(k),v) for k,v in caps.items()):
        raise ValueError('original A63 incremental caps must remain fixed')
    for k in ('known_CPU_seconds','CPU_accounting_seconds','incremental_GPU_seconds','incremental_tracked_storage_upper_bytes'):
        if resources.get(k) is not None and _real(resources[k])<0: raise ValueError('nonnegative known resource values required')
    if resources.get('actual_caps_fit') is not None and type(resources['actual_caps_fit']) is not bool:
        raise ValueError('unknown actual resource fit remains null')
    return r,ci,inn,sup,nat,reg,series,verdict


def _write(path, raw):
    path,parent=_parent(path,True)
    try:
        fd=os.open(path.name,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600,dir_fd=parent)
        with os.fdopen(fd,'wb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
        os.fsync(parent)
    finally:os.close(parent)


def _csv(rows):
    fields=list(dict.fromkeys(k for row in rows for k in row))
    out=io.StringIO();writer=csv.DictWriter(out,fieldnames=fields);writer.writeheader()
    for row in rows:
        writer.writerow({k:json.dumps(v,sort_keys=True,separators=(',',':')) if isinstance(v,(dict,list,tuple)) else v for k,v in row.items()})
    return out.getvalue().encode()


def _project(rows, fields):
    return [{k:r.get(k) for k in fields} for r in rows]


def _setup(pub,subfolder,synthetic,names):
    if type(synthetic) is not bool:raise ValueError('explicit synthetic marker required')
    pub=Path(pub)
    root=Path(__file__).resolve().parents[2]
    actual=root/'docs/results/2026-11-05_a63_resume_execution'
    prep=root/'docs/results/2026-11-05_a63_resume_preparation'
    if not synthetic and pub.absolute()!=actual:
        raise ValueError('actual continuation figures restricted to their new public namespace')
    if synthetic:
        # Check the lexical destination before any mkdir; _parent separately
        # rejects symlink ancestors. Only the new preparation namespace or an
        # explicit child of a real system temporary root may receive fixtures.
        temporary={Path(tempfile.gettempdir()).resolve(),Path('/tmp').resolve()}
        destination=pub.absolute()
        in_temporary=any(destination!=t and destination.is_relative_to(t) for t in temporary)
        if not destination.is_relative_to(prep) and not in_temporary:
            raise ValueError('synthetic renders restricted to new preparation or system temporary children')
        if destination.is_relative_to(root/'docs/results') and not destination.is_relative_to(prep):
            raise ValueError('synthetic renders cannot write into existing experiment namespaces')
    folder=pub/subfolder
    if any(p.exists() or p.is_symlink() for p in [folder/n for n in names]):
        raise FileExistsError('preserve existing figures; explicit new reviewed version required')
    # Validate every existing parent with descriptor-based NOFOLLOW before any output.
    _,fd=_parent(folder/'not-created',True);os.close(fd)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return pub,folder,plt


def _finish(fig,name,folder,pub,files,synthetic,note):
    fig.suptitle('SYNTHETIC FIXTURE - NO EXPERIMENT RESULT' if synthetic else 'A63 fixed-bank continuation',fontsize=13)
    fig.text(.02,.015,textwrap.fill(note,145)+'\nShared-source constructed banks; sessions are not patients. Visual QA pending.',fontsize=8)
    if synthetic:fig.text(.5,.5,'SYNTHETIC / NOT ACTUAL',ha='center',rotation=25,alpha=.15,fontsize=30)
    fig.tight_layout(rect=(0,.085,1,.94))
    try:
        for ext in ('png','pdf'):
            out=io.BytesIO();fig.savefig(out,format=ext,dpi=150,facecolor='white');raw=out.getvalue();path=folder/(name+'.'+ext)
            _write(path,raw);files.append(dict(path=str(path.relative_to(pub)),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw)))
    finally:
        import matplotlib.pyplot as plt
        plt.close(fig)


def render_full(pub, rows238, comparisons140, inner34, support56, native34, regression32, curves4, resources, *, synthetic=False):
    """Render six full-matrix families; no selection or metric recomputation."""
    try:r,ci,inn,sup,nat,reg,series,verdict=_validate(rows238,comparisons140,inner34,support56,native34,regression32,curves4,resources)
    except (KeyError,TypeError,IndexError) as exc:raise ValueError('complete fixed anonymous presentation inputs required') from exc
    families=('01_paired_gain','02_gross_short_effects','03_inner_diagnostics','04_N_top_support','05_native_regression','06_training_resources')
    csv_names=('metrics.csv','paired.csv','inner.csv','support.csv','native.csv','regression.csv','training_curves.csv','resources.csv')
    names=[n+'.'+ext for n in families for ext in ('png','pdf')]+list(csv_names)+['README.md','render_receipt.json']
    pub,folder,plt=_setup(pub,'synthetic_full_figures' if synthetic else 'full_figures',synthetic,names);files=[]
    cells=[(t,h,s) for t in ep.TABLES for h in ep.HALVES for s in ep.SEEDS]
    labels=[t.replace('_confirmation','').capitalize()+'/'+h[0]+'/'+str(s)[-2:] for t,h,s in cells]
    colors={'C0':'#31688e','C1':'#c66d25'};x=np.arange(len(cells))
    fig,ax=plt.subplots(figsize=(14,6))
    for off,a in [(-.12,'C0'),(.12,'C1')]:
        delta=[100*(r[(t,h,s,a)]['multi_f1']-r[(t,h,None,'Base')]['multi_f1']) for t,h,s in cells]
        ax.plot(x+off,delta,'o',ms=4,label=a,color=colors[a])
        for i,(t,h,s) in enumerate(cells):
            interval=ci[(t,h,s,a,'Base')]['delta95'][ep.METRIC]
            if interval is not None:ax.vlines(i+off,100*interval[0],100*interval[1],color=colors[a],lw=1)
    ax.axhline(0,color='gray',lw=.7);ax.set_xticks(x,labels,rotation=70,ha='right',fontsize=8);ax.set_ylabel('Pooled multi F1 difference (percentage points)');ax.legend()
    _finish(fig,families[0],folder,pub,files,synthetic,'Points: C0/C1 − Base; bars: fixed paired95% CI. All140 comparisons and8 primary99.375% intervals remain in CSV.')
    fig,axes=plt.subplots(3,4,figsize=(16,9))
    for ri,kind in enumerate(ep.KINDS):
        for ax,field,label in zip(axes[ri],['H','R','dFP16','short16_lost'],['Gross H','Repairs R','Short16 FP change','True short16 gross loss']):
            for a in ('C0','C1'):ax.plot(x,[r[(t,h,s,a)][field][kind] for t,h,s in cells],'.-',label=a,color=colors[a],lw=.8)
            ax.set_xticks(x,labels if ri==2 else [],rotation=80,fontsize=5);ax.set_title(kind+': '+label,fontsize=9);ax.legend(fontsize=6)
    _finish(fig,families[1],folder,pub,files,synthetic,'All3 kinds remain separate: FP increases cannot cancel across kinds; repairs elsewhere never cancel gross H or true-unit loss.')
    fig,axes=plt.subplots(1,2,figsize=(11,5))
    for ax,pop in zip(axes,('CAL','AUDIT')):
        for a in ('C0','C1'):
            vals=[100*(inn[(pop,s,a)]['segment_f1_50_multi']-inn[(pop,None,'Base')]['segment_f1_50_multi']) for s in ep.SEEDS]
            ax.plot(['61','62'],vals,'o-',label=a,color=colors[a])
        ax.axhline(0,color='gray',lw=.7);ax.set_title(pop);ax.set_ylabel('Multi F1 − Base (pp)');ax.legend()
    _finish(fig,families[2],folder,pub,files,synthetic,'CAL is the fixed threshold population; AUDIT is diagnostic only. Absent pair ratios stay null. No checkpoint/seed selection.')
    fig,axes=plt.subplots(2,1,figsize=(14,8))
    for safe,style in [('event-safe','--'),('acceptance-safe','-')]:
        axes[0].plot(x,[100*sup[(t,h,s,safe)]['result']['multi']['oracle']['F1'] for t,h,s in cells],style,label=safe)
        axes[1].plot(x,[sup[(t,h,s,safe)]['result']['multi']['joint_chosen_effects']['JEPA_exclusive_short16_recovered'] for t,h,s in cells],style,label=safe)
    for ax in axes:ax.set_xticks(x,labels,rotation=75,fontsize=7);ax.legend(fontsize=8)
    axes[0].set_ylabel('Label-assisted pooled multi F1 (%)');axes[1].set_ylabel('Joint J-exclusive short16 recovered')
    _finish(fig,families[3],folder,pub,files,synthetic,'Only {noop, original N-admitted S-L top}.48 old records reused/8 new solves. Not a q-threshold guarantee or bound on an unsafe learned policy.')
    fig,axes=plt.subplots(1,3,figsize=(15,5));policies=list(ep.POLICIES)
    for ax,a in zip(axes[:2],('C0','C1')):
        delta=np.zeros((5,5),dtype=np.int64)
        for h in ep.HALVES:
            for s in ep.SEEDS:
                cube=np.asarray(nat[(h,s,a)]['result']['true_Base_prediction_joint5cube']);delta+=cube.sum(1)-cube.sum(2)
        # Summing these reused cells is a display of fixed-cell effects, never a patient/sample count.
        image=ax.imshow(delta,cmap='RdBu_r');ax.set_title(a+' native frame-count delta')
        ax.set_xticks(range(5),CLASSES,rotation=45);ax.set_yticks(range(5),CLASSES);fig.colorbar(image,ax=ax,shrink=.6)
    for case,style in [('Case_A','-'),('Case_B','--')]:
        axes[2].plot(range(8),[sum(reg[(case,s,p)]['actual_H'] for s in ep.SEEDS) for p in policies],style+'o',label=case)
    axes[2].set_xticks(range(8),policies);axes[2].set_ylabel('Actual gross H (two fixed seed cells)');axes[2].legend()
    _finish(fig,families[4],folder,pub,files,synthetic,'Heatmaps sum4 fixed reused half/seed cells per arm, not new samples. All34 cubes/32 exposed case records are exported; native recognition and frame safety differ.')
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for (s,a),points in series.items():axes[0].plot([v['update'] for v in points],[v['loss'] for v in points],label=f'{a}/{str(s)[-2:]}',lw=.7)
    axes[0].set_xlabel('Physical optimizer update');axes[0].set_ylabel('BCE loss');axes[0].legend(fontsize=8)
    axes[1].axis('off');axes[1].text(0,1,'Provided resource accounting\n'+'\n'.join(k+': '+('unknown' if v is None else str(v)) for k,v in resources.items() if k in ('known_CPU_seconds','CPU_accounting_seconds','incremental_GPU_seconds','incremental_tracked_storage_upper_bytes','actual_caps_fit')),va='top',fontsize=9)
    _finish(fig,families[5],folder,pub,files,synthetic,'All4×6000 formal loss points exported. Auxiliary48000 and prior/administrative costs belong to the separate full ledger; reserves are not actual spend.')
    row_fields=('table','half','seed','policy','system','role','step','input_identity','native_confusion','native_vote_changes','H','R','dFP16','short16_lost','class_H','JEPA_short16_recovered','multi_f1')
    exports={
        'metrics.csv':_project(rows238,row_fields),
        'paired.csv':_project(comparisons140,('table','half','seed','left_policy','right_policy','contrast','primary_family','input_identity','observed_delta','delta95','delta99_375','undefined_bootstrap_replicates')),
        'inner.csv':_project(inner34,('population','seed','policy','system','segment_f1_50_multi','segment_f1_50_pairs','pair_reference_positions','inner_pair_bank_absent')),
        'support.csv':[dict(table=v['table'],half=v['half'],seed=v['seed'],safety=v['safety'],readonly_A62_reference=v['readonly_A62_reference'],**{k+'_result':_support_kind(v['result'][k],k,v['safety']) for k in ep.KINDS}) for v in support56],
        'native.csv':[dict(population=v['population'],seed=v['seed'],policy=v['policy'],result=_native_public(v['result'])) for v in native34],
        'regression.csv':_project(regression32,('axis','seed','policy','top_exists','N_admitted','accepted','gate_score','tau','H','R','dFP16','short16_lost','actual_H','actual_R','actual_dFP16','actual_short16_lost','candidate_key_sha256','context_summary')),
        'training_curves.csv':_project(curves4,('seed','arm','update','loss')),
        'resources.csv':[dict(metric=k,value='unknown' if resources.get(k) is None else resources[k]) for k in ('known_CPU_seconds','CPU_accounting_seconds','incremental_GPU_seconds','incremental_tracked_storage_upper_bytes','actual_caps_fit','CPU_cap','GPU_cap','storage_cap')]}
    for name,data in exports.items():
        raw=_csv(data);_write(folder/name,raw);files.append(dict(path=str((folder/name).relative_to(pub)),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw)))
    text=('# SYNTHETIC FIXTURE - NO EXPERIMENT RESULT\n' if synthetic else '# A63 fixed-bank results presentation\n')
    text+='\nComplete238/140/34/56/34/32 input matrix; all4×6000 formal curve points.\n'
    text+='\nInput verdicts: '+json.dumps(verdict,ensure_ascii=False)+'\n'
    text+='\nResource fields remain unknown when null. Rendering does not establish actual costs, model provenance, clinical safety, independent visual QA or overall closure.\n'
    _write(folder/'README.md',text.encode());files.append(dict(path=str((folder/'README.md').relative_to(pub)),sha256=hashlib.sha256(text.encode()).hexdigest(),bytes=len(text.encode())))
    receipt=dict(synthetic=synthetic,actual_visual_QA_passed=False,counts=dict(outer=238,comparisons=140,inner=34,support=56,native=34,regression=32,curves=24000),files=files,
                 caller_actual_Git_input_private_audit_required=True,overall_complete=False)
    _write(folder/'render_receipt.json',(json.dumps(receipt,indent=2)+'\n').encode())
    return receipt


def render_stageS(pub, coverage, aux_curves, *, synthetic=False):
    """Same frozen23 coverage checks, new continuation namespace; no C0/C1 claim."""
    seeds,series,clean=stageS._validated(coverage,aux_curves)
    families=stageS.FAMILIES
    names=[n+'.'+ext for n in families for ext in ('png','pdf','csv')]+['render_receipt.json']
    pub,folder,plt=_setup(pub,'synthetic_stageS_figures' if synthetic else 'stageS_figures',synthetic,names)
    files=[];export={};status='COVERAGE PASS' if coverage['passed'] else 'COVERAGE FAIL - STOP BEFORE C0/C1'
    x=np.arange(len(stageS.FLOORS));fig,ax=plt.subplots(figsize=(13,6));floor_rows=[]
    for off,seed in [(-.18,ep.SEEDS[0]),(.18,ep.SEEDS[1])]:
        ratios=[]
        for key,(minimum,group,field) in stageS.FLOORS.items():
            actual=seeds[seed]['checks'][key]['actual'];ratios.append(actual/minimum)
            floor_rows.append(dict(seed=seed,metric=key,actual=actual,minimum=minimum,passed=seeds[seed]['checks'][key]['passed']))
        ax.bar(x+off,ratios,width=.36,label=str(seed)[-2:])
    ax.axhline(1,color='black',lw=.7);ax.set_xticks(x,list(stageS.FLOORS),rotation=75,ha='right',fontsize=8);ax.set_ylabel('Observed distinct support / fixed engineering floor');ax.legend();ax.set_title(status)
    _finish(fig,families[0],folder,pub,files,synthetic,'Coverage floors are engineering admission checks, not power/safety guarantees. Repeats/gamma variants do not create new independent events.')
    export[families[0]]=floor_rows
    fig,ax=plt.subplots(figsize=(12,5));types=stageS.UNION_TYPES;type_rows=[];x=np.arange(len(types))
    for off,seed in [(-.18,ep.SEEDS[0]),(.18,ep.SEEDS[1])]:
        vals=[seeds[seed]['union'][k] for k in types];ax.bar(x+off,vals,.36,label=str(seed)[-2:])
        type_rows.extend(dict(seed=seed,metric=k,value=v) for k,v in zip(types,vals))
    ax.set_xticks(x,types,rotation=35,ha='right',fontsize=8);ax.set_ylabel('Distinct event signatures');ax.legend()
    _finish(fig,families[1],folder,pub,files,synthetic,status+'. Harm, FP-only and geometry gaps remain visible; shared sources are not patients.')
    export[families[1]]=type_rows
    fig,axes=plt.subplots(1,3,figsize=(14,5));safe_rows=[]
    for seed in ep.SEEDS:
        r=seeds[seed];label=str(seed)[-2:]
        axes[0].plot(['D safe','X safe','D safe J','X safe J'],[r[g][f] for g,f in [('D','safe_useful_signatures'),('X_only','safe_useful_signatures'),('D','safe_J_signatures'),('X_only','safe_J_signatures')]],'o-',label=label)
        axes[1].plot(CLASSES,[r['union']['replacement_class_signatures'][str(c)] for c in range(5)],'o-',label=label)
        axes[2].plot(stageS.CONCENTRATION,[r[k] for k in stageS.CONCENTRATION],'o-',label=label)
        for g in ('D','X_only'):
            for f in stageS.MEMBER_FIELDS:safe_rows.append(dict(seed=seed,group=g,metric=f,value=r[g][f]))
        for c in range(5):safe_rows.append(dict(seed=seed,group='union',metric='replacement_'+CLASSES[c],value=r['union']['replacement_class_signatures'][str(c)]))
        for k in (*stageS.CONCENTRATION,'native_D_danger_actions','native_X_danger_actions'):safe_rows.append(dict(seed=seed,group='source',metric=k,value=r[k]))
    axes[0].set_title('Distinct safe-useful support');axes[1].set_title('All5 replacement classes, including0');axes[2].set_title('Responsible-source concentration');axes[2].axhline(.2,color='black',lw=.7)
    for ax in axes:ax.tick_params(axis='x',rotation=30,labelsize=8);ax.legend(fontsize=8)
    _finish(fig,families[2],folder,pub,files,synthetic,status+'. Native dangerous tops, including0, are retained in CSV; zero exposure is not learned native danger recognition.')
    export[families[2]]=safe_rows
    fig,axes=plt.subplots(2,4,figsize=(14,7))
    for ax,((seed,fold,role),points) in zip(axes.flat,series.items()):
        ax.plot([v['update'] for v in points],[v['loss'] for v in points],lw=.8)
        ax.set_title(f'{str(seed)[-2:]} held{fold} {role}',fontsize=10);ax.set_xlabel('Physical update');ax.set_ylabel('Loss')
    _finish(fig,families[3],folder,pub,files,synthetic,'8 auxiliary6000 models /48000 updates; complete saved cursors preserved. Curves do not select fold/seed/checkpoint or establish coverage sufficiency.')
    export[families[3]]=clean
    for name,data in export.items():
        raw=_csv([dict(row,synthetic_fixture=synthetic) for row in data]);path=folder/(name+'.csv');_write(path,raw)
        files.append(dict(path=str(path.relative_to(pub)),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw)))
    receipt=dict(synthetic=synthetic,actual_visual_QA_passed=False,coverage_passed=coverage['passed'],
                 aux_optimizer_updates=coverage['aux_optimizer_updates'],formal_optimizer_updates=coverage['formal_optimizer_updates'],
                 C0_C1_effectiveness_established=False,files=files,overall_complete=False)
    _write(folder/'render_receipt.json',(json.dumps(receipt,indent=2)+'\n').encode())
    return receipt
