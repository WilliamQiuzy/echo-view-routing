"""Explicit-call auxiliary-only incomplete figures; no model/GT/acceptance path."""
import csv
import hashlib
import json
import math
from pathlib import Path
import re

_SEEDS=(2026110461,2026110462)
_NOT_RUN={'coverage','CONFIRM','C0','C1','CAL_AUDIT','outer','paired_CI','support','native_regression'}
_RESOURCE_FIELDS=['metric','actual','forecast','cap','units','scope','source_pin_path','source_pin_sha256']


def _need(condition,message):
    if not condition:raise ValueError(message)


def _raw(pin):
    _need(isinstance(pin,dict) and set(pin)=={'path','sha256'} and isinstance(pin['path'],str)
          and pin['path'] and isinstance(pin['sha256'],str) and re.fullmatch('[0-9a-f]{64}',pin['sha256']),
          'exact external raw source pin required')
    path=Path(pin['path'])
    _need(not any(p.is_symlink() for p in [path,*path.parents]) and path.is_file(),'ordinary nonsymlink source required')
    raw=path.read_bytes();_need(hashlib.sha256(raw).hexdigest()==pin['sha256'],'raw source SHA differs')
    return raw


def _key(row):
    _need(type(row.get('seed')) is int and row['seed'] in _SEEDS and type(row.get('held_fold')) is int
          and row['held_fold'] in [0,1] and row.get('role') in ['utility','risk'],'exact auxiliary seed/fold/role required')
    return row['seed'],row['held_fold'],row['role']


def _finite(value):
    return not isinstance(value,bool) and isinstance(value,(int,float)) and math.isfinite(value)


def _prepare(aux_curves,aux_models,resource_rows,stop_pin):
    expected={(s,h,r) for s in _SEEDS for h in [0,1] for r in ['utility','risk']}
    _need(len(aux_models)==8 and {_key(r) for r in aux_models}==expected,'all eight auxiliary metadata rows required')
    for row in aux_models:
        _need(type(row.get('step')) is int and row['step']==6000,'every auxiliary model must be final6000')
        if 'sha256' in row:_need(isinstance(row['sha256'],str) and re.fullmatch('[0-9a-f]{64}',row['sha256']),'model metadata SHA malformed')
    _need(len(aux_curves)==48000,'all48000 original auxiliary loss rows required')
    seen=set();rows=[]
    for row in aux_curves:
        key=_key(row);u=row.get('update');loss=row.get('loss')
        _need(type(u) is int and 1<=u<=6000 and (*key,u) not in seen and _finite(loss),'unique full finite updates1..6000 required')
        seen.add((*key,u));rows.append(dict(seed=key[0],held_fold=key[1],role=key[2],update=u,loss=float(loss)))
    _need(len(seen)==8*6000,'complete original eight loss traces required')
    stop=json.loads(_raw(stop_pin))
    _need(stop.get('schema')=='A63_incomplete_resource_stop_decision_v1' and stop.get('actual_caps_refused') is True
          and stop.get('reason')=='complete_remaining_work_forecast_refused' and stop.get('coverage') is None
          and stop.get('coverage_evaluated') is False and stop.get('overall_complete') is False
          and (stop.get('aux_optimizer_updates'),stop.get('formal_optimizer_updates'),stop.get('C0_C1_instances'))==(48000,0,0)
          and set(stop.get('not_run',[]))==_NOT_RUN,'actual incomplete resource-stop metadata required, never coverage failure')
    _need(_finite(stop.get('T0_epoch')) and _finite(stop.get('deadline_epoch'))
          and stop['deadline_epoch']>stop['T0_epoch'],'original window metadata required')
    _need(isinstance(stop.get('runtime_authorization_sha256'),str) and re.fullmatch('[0-9a-f]{64}',stop['runtime_authorization_sha256']),'external owner SHA required')
    parent=json.loads(_raw(stop['refusal_parent_pin']));guardian=json.loads(_raw(stop['refusal_guardian_pin']))
    _need(parent.get('guardian_snapshot')==stop['refusal_guardian_pin']
          and isinstance(parent.get('stage'),str) and bool(parent['stage'].strip()) and parent['stage']==guardian.get('stage')
          and isinstance(parent.get('runtime_spec_sha256'),str) and re.fullmatch('[0-9a-f]{64}',parent['runtime_spec_sha256'])
          and parent['runtime_spec_sha256']==guardian.get('runtime_spec_sha256')
          and type(parent.get('guardian_pid')) is int and parent['guardian_pid']>0
          and type(guardian.get('supervisor_pid')) is int and parent['guardian_pid']==guardian['supervisor_pid']
          and type(parent.get('guardian_start_ticks')) is int and parent['guardian_start_ticks']>0
          and type(guardian.get('supervisor_start_ticks')) is int and parent['guardian_start_ticks']==guardian['supervisor_start_ticks'],
          'parent must bind exact guardian raw pin, process birth, runtime spec and stage')
    bounds=guardian.get('actual_fresh_completed_aux_admission',{})
    _need(parent.get('stage_exit_status')==1 and guardian.get('stage_exit_status')==1
          and parent.get('model_pid') is None and guardian.get('model_pid') is None
          and parent.get('whole_child_accounting_complete') is True and bounds.get('all_caps_fit') is False
          and isinstance(stop.get('failed_checks'),list) and bool(stop['failed_checks'])
          and stop['failed_checks']==bounds.get('failed_checks')
          and parent.get('runtime_authorization_sha256')==guardian.get('runtime_authorization_sha256')==stop.get('runtime_authorization_sha256'),
          'actual canonical prelaunch caps refusal/current raw parent and guardian required')
    resources=[]
    _need(bool(resource_rows),'explicit resource rows required')
    for row in resource_rows:
        _need(all(isinstance(row.get(k),str) and row[k].strip() for k in ['metric','units','scope']),'named units and evidence scope required')
        _need(all(k in row and (row[k] is None or _finite(row[k]) and row[k]>=0) for k in ['actual','forecast','cap'])
              and (row['cap'] is None or row['cap']>0),'unknown resource values must be explicit null, never inferred zero')
        _raw(row['source_pin'])
        resources.append({**{k:row[k] for k in ['metric','actual','forecast','cap','units','scope']},
                          'source_pin_path':row['source_pin']['path'],'source_pin_sha256':row['source_pin']['sha256']})
    return sorted(rows,key=lambda r:(r['seed'],r['held_fold'],r['role'],r['update'])),aux_models,resources,stop


def _csv(path,fields,rows):
    with Path(path).open('x',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
        writer.writerows({k:('unknown' if row[k] is None else row[k]) for k in fields} for row in rows)


def render_incomplete(output,aux_curves,aux_models,resource_rows,stop_pin):
    """Write exactly two PNG/PDF/CSV families into a fresh exclusive directory.

    Curves are flat seed/held_fold/role/update/loss rows; models have the same
    eight keys and step6000. Resources use metric,actual,forecast,cap,units,scope,
    source_pin. Source pins resolve from caller cwd or absolute paths. Caller
    must verify archive/model causality and numeric extraction; no weights/GT
    or scores are read here. A file hash is not human visual QA.
    """
    rows,models,resources,stop=_prepare(aux_curves,aux_models,resource_rows,stop_pin)
    output=Path(output)
    _need(not any(p.is_symlink() for p in [output,*output.parents]),'nonsymlink output ancestry required')
    output.mkdir(parents=True,exist_ok=False) # Preserve any partial prior result.
    names=['auxiliary_learning_curves','resource_admission']
    _csv(output/(names[0]+'.csv'),['seed','held_fold','role','update','loss'],rows)
    _csv(output/(names[1]+'.csv'),_RESOURCE_FIELDS,resources)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    title=('SYNTHETIC TEMPLATE\n' if stop.get('synthetic') is True else '')
    title+='Auxiliary / incomplete: coverage and C0/C1 not_run'
    with matplotlib.rc_context({'font.size':9,'path.simplify':False}):
        fig,axes=plt.subplots(2,4,figsize=(16,7),constrained_layout=True)
        try:
            for ri,seed in enumerate(_SEEDS):
                for ci,(held,role) in enumerate([(0,'utility'),(0,'risk'),(1,'utility'),(1,'risk')]):
                    data=[r for r in rows if (r['seed'],r['held_fold'],r['role'])==(seed,held,role)]
                    ax=axes[ri,ci];ax.plot([r['update'] for r in data],[r['loss'] for r in data],linewidth=.65)
                    ax.set(title=f'seed{seed} / fold{held} / {role}',xlabel='Optimizer update',ylabel='Raw training loss')
                    ax.grid(alpha=.2)
            fig.suptitle(title+'\nFull complement training loss; no held outcome or smoothing')
            for ext in ['png','pdf']:fig.savefig(output/(names[0]+'.'+ext),dpi=160)
        finally:plt.close(fig)
        fig,ax=plt.subplots(figsize=(12,max(4,1.0*len(resources)+2)),constrained_layout=True)
        try:
            labelled=set()
            for i,row in enumerate(resources):
                for key,offset,color in [('actual',-.18,'#336C99'),('forecast',.18,'#D68B27')]:
                    if row[key] is None or row['cap'] is None:
                        text=key+': unknown' if row[key] is None else key+': '+str(row[key])+' (cap unknown)'
                        ax.text(.02,i+offset,text,color='gray',transform=ax.get_yaxis_transform(),va='center')
                    else:
                        ax.barh(i+offset,row[key]/row['cap'],height=.30,color=color,label=key if key not in labelled else None)
                        labelled.add(key)
            ax.axvline(1.,color='black',linestyle='--',label='cap')
            ax.set_yticks(range(len(resources)),[r['metric']+' ['+r['units']+']' for r in resources]);ax.invert_yaxis()
            ax.set_xlabel('Fraction of each declared cap; raw values and scopes in CSV')
            ax.set_title(title+'\nActual accounting and future forecast shown separately; unknown stays unknown')
            ax.legend(loc='best');ax.grid(axis='x',alpha=.2)
            for ext in ['png','pdf']:fig.savefig(output/(names[1]+'.'+ext),dpi=160)
        finally:plt.close(fig)
    files=[output/(name+'.'+ext) for name in names for ext in ['png','pdf','csv']]
    return {'passed':True,'files':[{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in files],
            'counts':{'aux_models':8,'aux_loss_rows':48000,'resource_rows':len(resources),'families':2,'files':6},
            'stop_pin':stop_pin,'stop_decision':'actual_caps_refused','coverage':'not_run','C0_C1':'not_run',
            'scope':'auxiliary-only incomplete administrative rendering; no scientific acceptance',
            'synthetic':stop.get('synthetic') is True,'visual_QA_passed':False,'overall_complete':False}
