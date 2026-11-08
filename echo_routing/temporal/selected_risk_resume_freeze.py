"""Local-only prospective source freeze builder; no scientific execution.

The final runtime spec is externally raw-pinned after its freeze pin is filled.
The contract binds every other spec field, avoiding a circular file hash. Shared
Git objects are only a lossless storage encoding; the unchanged actual commit,
tree and blob verifier checks expanded bytes. Original pools remain untouched.
"""
import hashlib
import os
from pathlib import Path,PurePosixPath
import stat

from echo_routing.temporal.selected_risk_resume_contract import RUN,digest,stage_graph

SOURCE='echo_routing/temporal/selected_risk_resume_freeze.py'
POOL=RUN+'/Git_proofs/shared_pool.json'
PASSIVE_SOURCE={'scripts/calibrate_event_risk.py':'b6a998ffb89ba6de4cb3fe90d640a44907ba7918593a3813c1bc8ba73ecd926d'}


def semantics(spec):return digest({k:v for k,v in spec.items() if k!='freeze'})


def _pin(value):
    if not isinstance(value,dict) or set(value)!={'path','sha256'}:raise ValueError('exact external path/raw SHA required')
    p=PurePosixPath(value['path']);h=value['sha256']
    if not p.parts or p.is_absolute() or str(p)!=value['path'] or '..' in p.parts or not isinstance(h,str) or len(h)!=64 or any(c not in '0123456789abcdef' for c in h):raise ValueError('canonical source/input pin required')
    return value['path']


def _base_paths(root):
    from scripts.selected_risk_common import ROOT,CFG,read,verify_pin,prior_path,relative,validate_config
    if Path(root).resolve()!=ROOT.resolve():raise ValueError('only the original isolated project root may build continuation contracts')
    cfg=read(CFG);validate_config(cfg);basis=read(verify_pin(cfg['basis']));refs=read(verify_pin(cfg['source_refs']))
    paths={SOURCE,'scripts/a63_resume_selected_risk.py','scripts/a63_wait_resume_selected_risk.py',
      'echo_routing/temporal/selected_risk_resume_guardian.py','echo_routing/temporal/selected_risk_resume_runtime.py',
      'echo_routing/temporal/selected_risk_resume_context.py','echo_routing/temporal/selected_risk_resume_contract.py',
      'echo_routing/temporal/selected_risk_resume_estimate.py',relative(CFG),
      'docs/results/2026-11-05_a63_resume_preparation/readonly_closed_publication_manifest.json'}
    from echo_routing.temporal.selected_risk_resume_contract import STORAGE_APPROVAL,STORAGE_AMENDMENT_PATH
    paths.update([STORAGE_APPROVAL['path'],STORAGE_AMENDMENT_PATH])
    from echo_routing.temporal.selected_risk_resume_context import HISTORICAL_ADMIN_VERSIONS
    paths.update(v['observation_path'] for v in HISTORICAL_ADMIN_VERSIONS.values())
    paths.update(cfg[k]['path'] for k in ['card','basis','authorization','source_refs'])
    paths.update(relative(prior_path(p)) for p in basis['predecessor_public_and_source_pins'])
    paths.update(refs['historical_frozen_imports'])
    return paths,refs['historical_frozen_imports']


def _public_arguments(value):
    """Bind explicit public JSON inputs; private weights/proofs stay raw-pinned.

    Private arrays/binary/full-object proofs are verified by their runtime
    consumers through these public manifests and the independent spec raw pin.
    They are deliberately not committed to Git by the builder.
    """
    out=set()
    if isinstance(value,dict):
        if set(value)=={'path','sha256'}:
            p=_pin(value)
            if p.startswith('docs/results/'):
                if not p.endswith('.json'):raise ValueError('public argument pins must name their explicit JSON manifests')
                out.add(p)
        else:
            for v in value.values():out.update(_public_arguments(v))
    elif isinstance(value,list):
        for v in value:out.update(_public_arguments(v))
    return out


def required(root,stage,spec_pin,spec):
    from echo_routing.temporal.selected_risk_contracts import source_closure
    if stage not in stage_graph() or spec.get('stage')!=stage:raise ValueError('exact registered continuation stage and matching spec required')
    _pin(spec_pin);paths,_=_base_paths(root)
    paths.update(_pin(spec[k]) for k in ['owner','proposal','baseline','metadata','prep_ledger','local_storage'])
    for k in ['FIT_producer','CONFIRM_producer']:
        if spec.get(k) is not None:paths.add(_pin(spec[k]))
    for descriptor in spec['predecessors'].values():
        if set(descriptor)!={'payload_pin','git_commit','actual_Git_object_proof','runtime_authorization_sha256'}:raise ValueError('structured actual predecessor evidence required')
        paths.add(_pin(descriptor['payload_pin']));_pin(descriptor['actual_Git_object_proof'])
    core=spec['arguments'].get('core',{})
    if not isinstance(core,dict):raise ValueError('explicit named public core pins required')
    for key,value in core.items():
        if key=='git_commit':
            from echo_routing.temporal.safety_git import oid
            oid(value);continue
        if key=='actual_Git_object_proof':
            _pin(value);continue
        p=_pin(value)
        if not p.startswith('docs/results/') or not p.endswith('.json'):raise ValueError('core seals must be explicit public JSON inputs, not private weights/proofs')
        paths.add(p)
    paths.update(_public_arguments(spec['arguments']))
    # The spec's freeze pointer is filled after this contract. Its own raw SHA
    # is independently supplied to the runtime CLI and live guardian ticket.
    closure=set(source_closure(root,sorted(paths)))
    for p,h in PASSIVE_SOURCE.items():
        if hashlib.sha256((Path(root)/p).read_bytes()).hexdigest()!=h:raise ValueError('qualified passive calibration source bytes changed')
        closure.add(p)
    return sorted(closure)


def _raw(ctx,path):
    fd,name=ctx._parent(path)
    try:
        leaf=os.open(name,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK,dir_fd=fd)
        with os.fdopen(leaf,'rb') as f:
            before=os.fstat(f.fileno())
            if not stat.S_ISREG(before.st_mode):raise ValueError('regular source/public file required')
            raw=f.read();after=os.fstat(f.fileno())
            if (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)!=(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns):raise ValueError('source file changed during freeze')
    finally:os.close(fd)
    h=hashlib.sha256(raw).hexdigest()
    if ctx.raw({'path':path,'sha256':h})!=raw:raise ValueError('source bytes changed before actual Git comparison')
    return raw,h


def make(context,stage,spec_pin,spec,commit):
    """Future new-owner operation. Return contract pin, not runtime success.

    Caller must fill this pin into the final raw spec and rebind it, then call
    Context.verify_stage. No new owner exists in preparation; real make is not
    invoked. All publication is exclusive under new RUN, never old/main paths.
    """
    from echo_routing.temporal.safety_git import build,verify,oid
    from echo_routing.temporal.selected_risk_storage_engineering import pack_proof,expand_proof
    from echo_routing.temporal.selected_risk_resume_context import strict_json
    from scripts.selected_risk_common import validate_frozen_prior
    context.revalidate();context.bind_runtime_spec(spec_pin,spec);commit=oid(commit)
    paths=required(context.root,stage,spec_pin,spec);_,historical=_base_paths(context.root)
    objects={p:_raw(context,p)[1] for p in paths};validate_frozen_prior(objects)
    current=dict(objects);hist_proofs={}
    for p,v in historical.items():
        if p not in current or current.pop(p)!=v['sha256']:raise ValueError('all literal historical sources must retain approved current SHA')
        proof=build(context.root,v['git_commit'],[p]);verify(proof,v['git_commit'],{p:v['sha256']})
        if proof['files'][p]['blob_oid']!=v['blob_oid']:raise ValueError('historical actual blob tuple changed')
        hist_proofs[p]=proof
    full=build(context.root,commit,sorted(current));verify(full,commit,current)
    # All actual Git/current-file comparisons precede any new output. Existing
    # partial stage files are never overwritten or silently repaired/replayed.
    version=stage+'.'+semantics(spec)[:12]
    names=['contracts/'+version+'.json','Git_proofs/'+version+'/current.shared.private.json']
    names+=['Git_proofs/'+version+'/historical/'+Path(p).name+'.json' for p in historical]
    for name in names:
        path=context.output(name)
        if path.exists() or path.is_symlink():raise FileExistsError('preserve existing/partial stage proof; explicit reconciliation required')
    poolpath=context.root/POOL
    if poolpath.exists() or poolpath.is_symlink():
        poolraw,poolsha=_raw(context,POOL);pool=strict_json(poolraw);pool_pin={'path':POOL,'sha256':poolsha}
    else:
        pool=full;pool_pin=None
    packed=pack_proof(full,pool)
    if expand_proof(packed,pool)!=full:raise ValueError('shared encoding must preserve every actual Git object byte')
    verify(expand_proof(packed,pool),commit,current)
    if pool_pin is None:pool_pin=context.write_json('Git_proofs/shared_pool.json',pool)
    hrows={}
    for p,proof in hist_proofs.items():
        pp=context.write_json('Git_proofs/'+version+'/historical/'+Path(p).name+'.json',proof)
        hrows[p]={**historical[p],'proof':pp}
    proof_pin=context.write_json('Git_proofs/'+version+'/current.shared.private.json',packed)
    contract={'schema':'A63_resume_stage_freeze_v1','stage':stage,'git_commit':commit,'objects':objects,
      'historical_frozen_imports':hrows,'actual_Git_object_proof':proof_pin,
      'runtime_authorization_sha256':context.owner_sha,'runtime_spec_semantics_sha256':semantics(spec),
      'proof_encoding':'A63_shared_Git_objects_v1','shared_object_pool':pool_pin}
    return context.write_json('contracts/'+version+'.json',contract)
