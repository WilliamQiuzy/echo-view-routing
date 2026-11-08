"""A63 monotone explicitly registered local costs with declared charge coverage.

Only new A63 direct public receipts and exact A63 plan receipts qualify.
Expected raw SHA, charge_group and covers_raw_sha256 bind provenance. Covers
relations are supplied by the registration compiler/reviewer, not guessed from
arbitrary JSON references. No server/admission/cumulative total is discovered.
"""
import copy,hashlib,json,math,time
from pathlib import Path,PurePosixPath
PUB='docs/results/2026-11-05_selected_action_risk/'
PLANS='docs/superpowers/plans/'


def qualified_path(path):
    pp=PurePosixPath(path)
    if pp.is_absolute() or '..' in pp.parts or str(pp)!=path:raise ValueError('canonical relative registered path required')
    allowed_pub=pp.parent==PurePosixPath(PUB)
    allowed_plan=pp.parent==PurePosixPath(PLANS) and pp.name.startswith('2026-11-05-a63-') and (pp.name.endswith('.receipt.json') or pp.name.endswith('verification.json'))
    if not (allowed_pub or allowed_plan):raise ValueError('closed predecessor or context alias cannot be charged to A63')
    name=pp.name
    allowed=name.startswith(('task','review_','root_','admin_','spec_a63_','a63_initial_')) or name=='owner_approval.json' or allowed_plan or name.startswith('initial_inventory_') and name.endswith('_CPU.json')
    if not allowed or name.startswith(('process_','entry_','resources','local_cpu_receipts')) or 'admission' in name and not name.startswith('root_') or 'summary' in name:raise ValueError('actual server/admission/resource aggregate is not a local fee')
    return pp

def get_raw(root,path):
    pp=qualified_path(path);p=root.joinpath(*pp.parts)
    for v in [p,*p.parents]:
        if v==root:break
        if v.is_symlink():raise ValueError('fee source ancestor/leaf symlink forbidden')
    raw=p.read_bytes();return hashlib.sha256(raw).hexdigest(),json.loads(raw)

def scalar(value,field):
    for key in field.split('/'):value=value[key]
    return value

def collect(root,registrations,*,event_epoch=None,prior=None):
    root=Path(root).resolve();event=time.time() if event_epoch is None else event_epoch
    rows=copy.deepcopy(prior['receipts']) if prior else [];pins=dict(prior['file_pins']) if prior else {};seen={};groups={};covered=set()
    for path,h in pins.items():
        if get_raw(root,path)[0]!=h:raise ValueError('previously charged raw receipt changed/missing; no silent replacement')
    def register_identity(h,field,cpu,allowance,group,covers):
        identity=(field,cpu,allowance,group,tuple(covers))
        if h in seen:
            if seen[h]!=identity:raise ValueError('same raw receipt has conflicting aggregate/child/charge alias')
            return False
        if h in covered or set(covers)&covered or any(x in seen for x in covers) or h in covers:raise ValueError('aggregate and included child/component cannot both be billed')
        if group in groups and groups[group]!=h:raise ValueError('one physical charge group cannot have different raw charges')
        seen[h]=identity;groups[group]=h;covered.update(covers);return True
    for row in rows:
        h=row['source_sha256'];rawh,raw=get_raw(root,row['source_receipt'])
        if rawh!=h or scalar(raw,row['source_CPU_field'])!=row['measured_CPU_core_seconds'] or row['receipt_id']!='local_raw:'+h:raise ValueError('old canonical charge no longer matches original raw')
        if not register_identity(h,row['source_CPU_field'],row['measured_CPU_core_seconds'],row['CPU_accounting_allowance_seconds'],row['charge_group'],row['covers_raw_sha256']):raise ValueError('duplicate old canonical charge')
    for spec in registrations:
        path=spec['path'];h,value=get_raw(root,path);kind=spec['kind'];group=spec['charge_group'];covers=sorted(set(spec.get('covers_raw_sha256',[])))
        if spec['sha256']!=h:raise ValueError('registered expected raw SHA differs')
        if kind not in ['software','review','admin','admin_unknown'] or not isinstance(group,str) or not group or any(not isinstance(x,str) or len(x)!=64 or any(c not in '0123456789abcdef' for c in x) for x in covers):raise ValueError('explicit process charge/coverage identity required')
        cpu=scalar(value,spec['CPU_field'])
        if cpu is None:
            if kind!='admin_unknown':raise ValueError('unknown CPU needs explicit administrative allowance')
            allowance=spec.get('accounting_allowance_seconds')
            if isinstance(allowance,bool) or not isinstance(allowance,(float,int)) or not math.isfinite(allowance) or allowance<0:raise ValueError('finite disclosed administrative allowance required')
        else:
            if isinstance(cpu,bool) or not isinstance(cpu,(float,int)) or not math.isfinite(cpu) or cpu<0:raise ValueError('finite nonnegative measured CPU required')
            allowance=0.
        pins[path]=h
        if not register_identity(h,spec['CPU_field'],cpu,allowance,group,covers):continue
        rows.append({'receipt_id':'local_raw:'+h,'measurement_kind':'administrative_partial_receipt','source_receipt':path,'source_sha256':h,'source_CPU_field':spec['CPU_field'],'source_scope_kind':kind,'charge_group':group,'covers_raw_sha256':covers,'started_epoch':event,'finished_epoch':event,'full_process_wall_seconds':0.,'original_process_interval_not_inferred_from_collection_event':True,'measured_CPU_core_seconds':cpu,'CPU_accounting_allowance_seconds':allowance,'GPU_job':False,'CUDA_context_occupied':False,'observed_worker_exit_code':None,'SELF_OS_null_not_inferred_from_passing_fixture':True,'raw_child_or_tool_OS_preserved_in_original_receipt':True,'measurement_host':None,'model_pid':None,'partial_snapshot_finalwrite_tool_modeltail_unknown':True})
    known=sum(r['measured_CPU_core_seconds'] or 0 for r in rows);allowances=sum(r['CPU_accounting_allowance_seconds'] for r in rows)
    return {'receipts':rows,'file_pins':pins,'known_measured_CPU_seconds':known,'administrative_CPU_allowance_seconds':allowances,'CPU_accounting_seconds':known+allowances,'registered_files':len(pins),'unique_raw_process_receipts':len(rows),'collection_event_not_original_interval':True,'monotone_prior_costs_and_pins_preserved':True,'closed_predecessor_baseline_not_recharged':True,'aggregate_component_coverage_from_explicit_registration_declarations_not_automatic_inference':True}
