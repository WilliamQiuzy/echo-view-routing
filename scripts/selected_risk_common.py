"""A63-only IO, exact approval/basis binding, read-only predecessors and stage Git."""
import argparse,hashlib,json,time,os
from pathlib import Path,PurePosixPath
import numpy as np
from echo_routing.temporal.anchor_cache import atomic_json,sha,durable_directory,fsync_directory,ProbabilityCache
from echo_routing.temporal.safety_git import verify as verify_Git
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/a63_selected_action_risk_v1'
PUB=ROOT/'docs/results/2026-11-05_selected_action_risk'
CFG=ROOT/'configs/experiments/selected_action_risk_v1.json'
CARD_SHA='191acbf1204cd6bf3fbdb6066800a3c6e7d6208ff7ccc15fe752c7964daead7a'
BASIS_SHA='78b4f913064d7300238577c561384cc559ba09f2c54be84bffa8d876f9bbacbf'
AUTH_SHA='2981f97a33747a49cc30de8efb230870b7e799359e97d79e9e88c8665ac93c91'
POST_PREFIX='docs/results/2026-11-05_a62_frozen_score_diagnostic/'

def read(path):return json.loads(Path(path).read_text())
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def relative(path):return str(Path(path).relative_to(ROOT))
def pin(path):return {'path':relative(path),'sha256':sha(path)}
def _lexical(path):
    pp=PurePosixPath(path)
    if pp.is_absolute() or not pp.parts or any(x in ['.','..'] for x in pp.parts):raise ValueError('canonical relative path required')
    p=ROOT.joinpath(*pp.parts)
    for q in [p,*p.parents]:
        if q==ROOT:break
        if q.is_symlink():raise ValueError('ancestor and leaf symlinks forbidden')
    return p

def verify_pin(value):
    p=_lexical(value['path'])
    if sha(p)!=value['sha256']:raise ValueError('pinned input/source changed: '+value['path'])
    return p

EXTRA_HISTORICAL_PINS={'echo_routing/temporal/jepa21_source.py': '4e95c88d14c7535a37d301712ffd15db1685cd7bfbd98b076e0f85f0d056fa54', 'echo_routing/temporal/structured_inputs.py': 'c7a116cb02281f26efb0104da94480e6cbdf165ac29d73427474339d4f0b5568', 'echo_routing/temporal/dense_readout.py': '6efb392ace5433dd4f96ee5b268650b612b065b2adab62b6e8e62b6222ec5ddf'}
HISTORICAL_PATHS={'echo_routing/temporal/'+p+'.py' for p in ['dense_analysis','dense_runtime','dense_sequence','structured_metrics','structured_uncertainty']}|set(EXTRA_HISTORICAL_PINS)

def validate_source_refs(refs,basis):
    original=basis['predecessor_public_and_source_pins'];expected={**original,**EXTRA_HISTORICAL_PINS}
    if refs.get('additional_historical_import_pins')!=EXTRA_HISTORICAL_PINS:raise ValueError('only literal three further producer dependencies allowed')
    if refs['predecessor_public_and_source_pins']!=original:raise ValueError('operational prior pins must equal approved86 basis')
    if set(refs['historical_frozen_imports'])!=HISTORICAL_PATHS:raise ValueError('only literal five historical imports allowed')
    for p,v in refs['historical_frozen_imports'].items():
        raw=_lexical(p).read_bytes();blob=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
        if v!={'git_commit':'dda29a526bd1ac1d9611b7edabd81974a683811d','sha256':expected[p],'blob_oid':blob} or hashlib.sha256(raw).hexdigest()!=expected[p]:raise ValueError('historical import must match approved source commit/current bytes/basis SHA')
    return True

def validate_frozen_prior(objects):
    cfg=read(CFG);basis=read(verify_pin(cfg['basis']));expected={**basis['predecessor_public_and_source_pins'],**EXTRA_HISTORICAL_PINS};refs=read(verify_pin(cfg['source_refs']));validate_source_refs(refs,basis)
    for old,v in refs['read_aliases'].items():expected[v['path']]=expected[old]
    for p,h in objects.items():
        if p in expected and h!=expected[p]:raise ValueError('current valid Git cannot replace approved predecessor SHA: '+p)
    return True

def prior_path(original):
    cfg=read(CFG);basis=read(verify_pin(cfg['basis']));expected=basis['predecessor_public_and_source_pins']
    if original not in expected:raise ValueError('predecessor path not in approved basis')
    refs=read(verify_pin(cfg['source_refs']));validate_source_refs(refs,basis)
    if refs['predecessor_public_and_source_pins']!=expected:raise ValueError('predecessor operational pins drifted')
    aliases=refs['read_aliases']
    if set(aliases)!={p for p in expected if p.startswith(POST_PREFIX)}:raise ValueError('only exact six approved POST-A62 aliases allowed')
    for old,v in aliases.items():
        if v!={'path':relative(PUB/'readonly_context'/('POST_A62_'+Path(old).name)),'sha256':expected[old],'original_path':old}:raise ValueError('readonly context alias differs from approved bytes/path')
    return verify_pin(aliases[original] if original in aliases else {'path':original,'sha256':expected[original]})

def validate_config(cfg):
    if cfg['card']['sha256']!=CARD_SHA or cfg['basis']['sha256']!=BASIS_SHA or cfg['authorization']['sha256']!=AUTH_SHA:raise ValueError('literal approved card/basis/owner SHA required')
    b=read(verify_pin(cfg['basis']));verify_pin(cfg['card']);a=read(verify_pin(cfg['authorization']));old=read(prior_path('configs/experiments/anchor_context_risk_v1.json'))
    for field in ['supply','coverage_per_seed_all_required','risk_comparison','calibration','matrix']:
        if cfg[field]!=b[field]:raise ValueError('approved science contract changed: '+field)
    if cfg['source']!=old['source'] or cfg['channel_mapping']!=old['channel_mapping'] or cfg['legacy_SL_tau']!=old['legacy_SL_tau'] or cfg['budget']!=a['new_limits']:raise ValueError('fixed deployment/source/oldtau/budget drift')
    if cfg['T0_epoch']!=1793853840 or cfg['deadline_epoch']!=1793897040 or cfg['run_id']!='a63_selected_action_risk_v1' or cfg['parent_git']!='db0d04c6543813e2bc3d3456d3a6ecdfd09f327e' or cfg['membership_F_fit']!=.8917446116307443 or cfg['fixed_utility_RMS']!=1.229706395330679 or cfg['baseline_from_closed_post_A62']!=b['baseline_from_closed_post_A62_publication_receipt'] or cfg['storage_new_worktree_initial_bytes']!=127985311 or cfg['all_old_namespaces_readonly'] is not True:raise ValueError('A63 fixed account/time/source fields differ')
    expected={'run_id','parent_git','card','basis','authorization','source_refs','source','channel_mapping','supply','coverage_per_seed_all_required','risk_comparison','calibration','matrix','budget','T0_epoch','deadline_epoch','legacy_SL_tau','membership_F_fit','fixed_utility_RMS','baseline_from_closed_post_A62','storage_new_worktree_initial_bytes','all_old_namespaces_readonly'}
    if set(cfg)!=expected:raise ValueError('unknown/unused configuration fields forbidden')
    for original in b['predecessor_public_and_source_pins']:prior_path(original)
    return True

def execution_parser(description):
    p=argparse.ArgumentParser(description=description);p.add_argument('--execute',action='store_true');p.add_argument('--approved-plan-sha',default=CARD_SHA);return p

def authorize(a):
    if not a.execute:print(json.dumps({'dry_run':True,'new_stage_started':False,'required':'--execute with literal owner approval'}));return False
    cfg=read(CFG);validate_config(cfg);auth=read(verify_pin(cfg['authorization']))
    if a.approved_plan_sha!=CARD_SHA or auth['approved_plan_SHA256']!=CARD_SHA or auth['approved_basis_SHA256']!=BASIS_SHA or auth['execution_authorized'] is not True or auth['T0_epoch']!=1793853840 or auth['deadline_epoch']!=1793897040 or not auth['T0_epoch']<=time.time()<auth['deadline_epoch']:raise ValueError('explicit owner authorization and unreset fixed12h window required')
    return True

def checked_output(path):
    root=ROOT.resolve();p=Path(path)
    if not p.is_absolute():p=root/p
    lexical=Path(os.path.abspath(p))
    if not any(lexical.is_relative_to(q) for q in [RUN,PUB]):raise ValueError('new A63 RUN/PUB outputs only')
    if not lexical.is_relative_to(root):raise ValueError('A63 output escapes project')
    for v in [lexical,*lexical.parents]:
        if v==root:break
        if v.is_symlink():raise ValueError('output ancestor/leaf symlink forbidden')
    return lexical

def write(path,value):atomic_json(checked_output(path),value)
def save_array(path,value):
    p=checked_output(path);durable_directory(p.parent);tmp=p.with_suffix(p.suffix+'.tmp')
    with tmp.open('xb') as f:np.save(f,value,allow_pickle=True);f.flush();os.fsync(f.fileno())
    os.replace(tmp,p);fsync_directory(p.parent)

def load_array(value):return np.load(verify_pin(value),allow_pickle=True).tolist()
def sequences(role):return load_array(read(PUB/'input_manifest.json')['roles'][role])
def verify_freeze(stage):
    contract=read(RUN/'contracts'/(stage+'.json'));cfg=read(CFG)
    from echo_routing.temporal.selected_risk_contracts import validate_coverage
    base=[relative(CFG),cfg['card']['path'],cfg['basis']['path'],cfg['authorization']['path'],cfg['source_refs']['path'],'scripts/supervise_selected_risk.py']
    basis=read(verify_pin(cfg['basis']));base += [relative(prior_path(p)) for p in basis['predecessor_public_and_source_pins']]
    validate_coverage(ROOT,stage,contract['objects'],base);validate_frozen_prior(contract['objects'])
    for p,h in contract['objects'].items():
        if sha(_lexical(p))!=h:raise ValueError('current stage source/input bytes changed: '+p)
    expected=dict(contract['objects']);historical=contract['historical_frozen_imports'];refs=read(verify_pin(cfg['source_refs']))
    for p,v in historical.items():
        if v['sha256']!=expected.pop(p) or refs['historical_frozen_imports'].get(p)!={k:v[k] for k in ['git_commit','blob_oid','sha256']}:raise ValueError('historical frozen import not exact approved overlay')
        verify_Git(read(verify_pin(v['proof'])),v['git_commit'],{p:v['sha256']})
    verify_Git(read(verify_pin(contract['actual_Git_object_proof'])),contract['git_commit'],expected)
    return contract

def cpu_only():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU CLI must hide CUDA before Torch import')
    import torch
    torch.set_num_threads(1)
    if torch.cuda.is_initialized():raise ValueError('unexpected CUDA context: fail closed')

def actual_receipts():
    rows=[json.loads(x) for x in (RUN/'processes.jsonl').read_text().splitlines()] if (RUN/'processes.jsonl').exists() else []
    if (PUB/'local_cpu_receipts.json').exists():
        fees=read(PUB/'local_cpu_receipts.json')
        for p,h in fees['file_pins'].items():
            if sha(ROOT/p)!=h:raise ValueError('local fee raw pin changed')
        rows+=fees['receipts']
    rows+=[read(p) for p in PUB.glob('entry_CPU_*.json')]
    return rows



def verify_aux_seal():
    names=['aux_models_manifest.json','aux_model_audit.json','aux_training_inputs.json']
    objects={relative(PUB/n):sha(PUB/n) for n in names};seal=read(PUB/'aux_identity.json')
    if seal['objects']!=objects:raise ValueError('auxiliary seal must bind exact three current core bytes')
    verify_Git(read(verify_pin(seal['actual_Git_object_proof'])),seal['git_commit'],objects)
    models=read(PUB/names[0]);audit=read(PUB/names[1]);ih=objects[relative(PUB/names[2])]
    expected={(s,f,r) for s in [2026110461,2026110462] for f in [0,1] for r in ['utility','risk']}
    rows=models['models']
    if len(rows)!=8 or {(r['seed'],r['held_fold'],r['role']) for r in rows}!=expected or any(r['step']!=6000 for r in rows):raise ValueError('exact eight final auxiliary model identities required')
    if audit.get('passed') is not True or (audit.get('aux_optimizer_updates'),audit.get('physical_models'),audit.get('fullstate_count'))!=(48000,8,32) or audit['models']!=rows or audit['aux_training_inputs_sha256']!=ih or models['aux_training_inputs_sha256']!=ih:raise ValueError('independent complete auxiliary audit and inputs must match Git-sealed models')
    finals=[{k:r[k] for k in ['path','sha256']} for r in rows]
    if seal['all_final_checkpoints']!=finals:raise ValueError('all actual auxiliary final checkpoint pins required')
    for r in finals:verify_pin(r)
    return models
