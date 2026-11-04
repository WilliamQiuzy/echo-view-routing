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

def authorize(args):
    if not args.execute:
        print(json.dumps({'dry_run':True,'new_stage_started':False,'required_execution_flag':'--execute --approved-plan-sha '+CARD_SHA}));return False
    cfg=read(CFG);auth=read(verify_pin(cfg['authorization']))
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

def probability_reader(source,role):
    m=read(PUB/'input_manifest.json');row=m['source_cache_roles'].get(role,{}).get(source)
    if row is None:
        produced=read(PUB/'producer_manifest.json');row=next((r for r in produced['records'] if r['source']==source and r['role']==role),None)
        if row is None or not produced['complete'] or produced['new_event_updates_at_production']!=0:raise ValueError('complete pre-admitted independent teacher cache required; no fallback forward')
    folder=ROOT/row['path']
    if sha(folder/'committed.jsonl')!=row['journal_sha256'] or row['input_sha256']!=m['roles'][role]['sha256']:raise ValueError('complete context/journal/input identity changed')
    return ProbabilityCache(folder,row['identity'],[len(q['refs']) for q in sequences(role)],readonly=True)
