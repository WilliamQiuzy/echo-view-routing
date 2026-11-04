"""A62-only namespace, explicit user authorization and actual-Git stage contracts."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from echo_routing.temporal.anchor_cache import atomic_json,sha,durable_directory,ProbabilityCache
from echo_routing.temporal.safety_git import verify as verify_Git
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/a62_anchor_context_risk_v1'
PUB=ROOT/'docs/results/2026-11-04_anchor_context_risk'
CFG=ROOT/'configs/experiments/anchor_context_risk_v1.json'
CARD_SHA='1f01eb3b9ad83df0b91f783c37d6ada70386384722090442d0c200b37335f5eb'
AUTH_SHA='171cbecf92f6b9cb92faa8f0069e8f19e7b14326b5e85a287fa629f8c167166f'


def read(path):return json.loads(Path(path).read_text())
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def pin(path):return {'path':str(Path(path).relative_to(ROOT)),'sha256':sha(path)}
def verify_pin(value):
    p=ROOT/value['path']
    if sha(p)!=value['sha256']:raise ValueError('frozen input/source changed: '+value['path'])
    return p

def checked_output(path):
    p=Path(path).resolve()
    if not any(p.is_relative_to(root.resolve()) for root in [RUN,PUB]):raise ValueError('A62 writes restricted to its new RUN/PUB')
    return p

def write(path,value):atomic_json(checked_output(path),value)
def save_array(path,value):
    p=checked_output(path);durable_directory(p.parent);tmp=p.with_suffix(p.suffix+'.tmp')
    import os
    with tmp.open('xb') as f:np.save(f,value,allow_pickle=True);f.flush();os.fsync(f.fileno())
    tmp.replace(p)
    from echo_routing.temporal.anchor_cache import fsync_directory
    fsync_directory(p.parent)

def execution_parser(description):
    p=argparse.ArgumentParser(description=description);p.add_argument('--execute',action='store_true');p.add_argument('--approved-plan-sha');return p

def validate_config(cfg):
    top={'run_id','card','authorization','basis','prior_scientific_git','prior_source_pins','readonly_code_pins','source','channel_mapping','training','data','scope','budget','prior_bookkeeping','legacy_SL_tau','prior_A61'}
    if set(cfg)!=top or cfg['run_id']!='a62_anchor_context_risk_v1' or cfg['prior_scientific_git']!='0aeb8527423d491920c0f9e9e578a8496592cf85':raise ValueError('exact authorized A62 config fields/parent required; unused fields reject')
    training={'seeds':[2026110461,2026110462],'arms':['R1','R2'],'model_dims':[205,32,1],'parameters_each':6625,'formal_steps_each':6000,'formal_updates':24000,'batch_candidate_kinds':[32,32],'old_draw_role':'risk','BCE_weights':'unweighted','AdamW':{'lr':.001,'weight_decay':.01,'betas':[.9,.999],'eps':1e-8},'warmup':100,'clip_norm':1.,'milestones':[2000,4000,6000],'latest_every':100,'engineering_updates':72}
    data={'partition_prefix':'A62-inner-v1\0','partition_sessions':[50,50],'recipe_seed':2026110463,'recipe_roles':['CAL_mixed','AUDIT_mixed','a62_confirmation_selection','a62_confirmation_calibration'],'recipe_counts':[150,150,120,120],'recipe_domains':[0,1,2,3],'source_role_natives':['inner_cal','selection','calibration'],'old_inner_mixed_for_new_thresholds':False}
    if cfg['training']!=training or cfg['data']!=data or cfg['legacy_SL_tau']!={'2026110461':.007963286712765694,'2026110462':.006616456434130669}:raise ValueError('training/data/gates differ from fixed card or contain unused fields')
    basis=read(verify_pin(cfg['basis']));old=read(verify_pin(cfg['prior_A61']['config']));auth=read(verify_pin(cfg['authorization']))
    budget={k:auth[k] for k in ['T0_epoch','deadline_epoch','incremental_GPU_job_seconds_cap','incremental_CPU_accounting_seconds_cap','incremental_tracked_storage_upper_bytes_cap','cumulative_GPU_job_hours_cap','cumulative_CPU_accounting_hours_cap','cumulative_tracked_storage_upper_bytes_cap','GPU_closing_reserve_seconds','CPU_closure_reserve_within_increment_seconds','CPU_forecast_uncertainty_buffer_seconds','storage_forecast_reserve_bytes']}
    if cfg['scope']!=basis['fixed_endpoint_counts'] or cfg['readonly_code_pins']!=basis['readonly_code_pins'] or cfg['prior_source_pins']!=basis['source_pins'] or cfg['prior_bookkeeping']!=basis['bookkeeping_basis'] or cfg['source']!=old['source'] or cfg['channel_mapping']!=old['channel_mapping'] or cfg['budget']!=budget:raise ValueError('frozen basis/source/channel/budget contract differs')
    return True


def authorize(args):
    if not args.execute:
        print(json.dumps({'dry_run':True,'new_stage_started':False,'required_execution_flag':'--execute --approved-plan-sha '+CARD_SHA}));return False
    cfg=read(CFG);validate_config(cfg);auth=read(verify_pin(cfg['authorization']))
    if cfg['authorization']['sha256']!=AUTH_SHA or auth['approved_card_sha256']!=CARD_SHA or args.approved_plan_sha!=CARD_SHA or sha(verify_pin(cfg['card']))!=CARD_SHA or not auth['execution_authorized'] or auth['T0_epoch']!=1793824361 or auth['deadline_epoch']!=1793867561 or not auth['T0_epoch']<=time.time()<auth['deadline_epoch']:raise ValueError('exact explicit A62 authorization/fixed12h deadline required; no implicit reset')
    return True

def verify_freeze(name):
    contract=read(RUN/name)
    for path,h in contract['objects'].items():
        if sha(ROOT/path)!=h:raise ValueError('current stage object changed: '+path)
    proof=read(verify_pin(contract['actual_Git_object_proof']))
    verify_Git(proof,contract['git_commit'],contract['objects'])
    return contract

def sequences(role):return np.load(verify_pin(read(PUB/'input_manifest.json')['roles'][role]),allow_pickle=True).tolist()
def actual_receipts():
    rows=[json.loads(x) for x in (RUN/'processes.jsonl').read_text().splitlines()] if (RUN/'processes.jsonl').exists() else []
    fees=read(PUB/'local_cpu_receipts.json')
    for path,h in fees['file_pins'].items():
        if sha(ROOT/path)!=h:raise ValueError('measured external receipt changed')
    return rows+fees['receipts']+[read(p) for p in PUB.glob('entry_CPU_*.json')]

def probability_reader(source,role,purpose='head'):
    if purpose not in ['head','geometry']:raise ValueError('explicit head or cost-only geometry reader purpose required')
    if purpose=='head' and role.startswith('a62_confirmation_'):
        if not (PUB/'source_adoption.json').exists():raise ValueError('fresh outer head reader requires final-Git readonly adoption')
        bound=read(PUB/'source_adoption.json');models=read(PUB/'models_manifest.json');seal=read(PUB/'outer_identity.json')
        from echo_routing.temporal.event_risk_audit import validate_barrier
        validate_barrier(models,seal,sha(PUB/'models_manifest.json'),sha(PUB/'calibration.json'),sha(PUB/'readout_plan.json'),read(verify_pin(seal['actual_Git_object_proof'])))
        if bound['formal_updates_at_binding']!=24000 or bound['new_source_forwards']!=0 or bound['new_event_updates_at_production']!=0 or bound['models_manifest_sha256']!=sha(PUB/'models_manifest.json') or bound['actual_head_Git_seal_sha256']!=sha(PUB/'outer_identity.json') or bound['producer_manifest_sha256']!=sha(PUB/'producer_manifest.json') or bound['records']!=read(PUB/'producer_manifest.json')['records']:raise ValueError('actual frozen head/producer adoption binding changed')
    if purpose=='head' and role in ['CAL_mixed','AUDIT_mixed']:
        if not (PUB/'model_audit.json').exists() or read(PUB/'model_audit.json')['formal_optimizer_updates']!=24000:raise ValueError('CAL/AUDIT new head input requires allfourfinal audit')
    m=read(PUB/'input_manifest.json');row=m['source_cache_roles'].get(role,{}).get(source)
    if row is None:
        produced=read(PUB/'producer_manifest.json');row=next((r for r in produced['records'] if r['source']==source and r['role']==role),None)
        if row is None or not produced['complete'] or produced['new_event_updates_at_production']!=0:raise ValueError('complete pre-admitted independent teacher cache required; no fallback forward')
    folder=ROOT/row['path']
    if sha(folder/'committed.jsonl')!=row['journal_sha256'] or row['input_sha256']!=m['roles'][role]['sha256']:raise ValueError('complete context/journal/input identity changed')
    return ProbabilityCache(folder,row['identity'],[len(q['refs']) for q in sequences(role)],readonly=True)
