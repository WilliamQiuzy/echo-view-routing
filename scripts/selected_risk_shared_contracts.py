"""Future A63 contracts with lossless shared Git objects; old contracts untouched."""
from scripts.selected_risk_common import *
from echo_routing.temporal.selected_risk_contracts import stage_spec,source_closure
from echo_routing.temporal.selected_risk_storage_engineering import expand_proof
EXTENSIONS={'cost_shape_probe':(['scripts/probe_selected_risk_shape.py'],['input_manifest.json','cost_profile.json','sources_admission_v1.json','COST_STORAGE_REFINEMENT.md']),'sources_refined_admission':(['scripts/admit_selected_risk_refined.py'],['input_manifest.json','cost_profile.json','cost_shape_profile.json','sources_admission_v1.json','COST_STORAGE_REFINEMENT.md'])}

def spec(stage):
    if stage in EXTENSIONS:
        e,i=EXTENSIONS[stage];return e,[relative(PUB/n) for n in i]
    return stage_spec(stage)

def required(stage):
    cfg=read(CFG);basis=read(verify_pin(cfg['basis']));e,i=spec(stage)
    base=[relative(CFG),cfg['card']['path'],cfg['basis']['path'],cfg['authorization']['path'],cfg['source_refs']['path'],'scripts/supervise_selected_risk.py','scripts/selected_risk_shared_contracts.py']
    if stage in EXTENSIONS:base.append('scripts/supervise_selected_risk_extension.py')
    base.extend(relative(prior_path(p)) for p in basis['predecessor_public_and_source_pins'])
    return set(source_closure(ROOT,e+i+base))

def verify_shared_freeze(stage):
    # Metadata's original full proof remains byte-for-byte and independently
    # verified. No old proof/contract or old common verifier is rewritten.
    old=verify_freeze('metadata');contract=read(RUN/'contracts'/(stage+'.json'));objects=contract['objects'];cfg=read(CFG);refs=read(verify_pin(cfg['source_refs']))
    if contract.get('proof_encoding')!='A63_shared_Git_objects_v1' or contract['shared_object_pool']!=old['actual_Git_object_proof'] or not required(stage)<=set(objects):raise ValueError('exact original pool and complete new stage source/input/import coverage required')
    validate_frozen_prior(objects)
    for p,h in objects.items():
        if sha(ROOT/p)!=h:raise ValueError('new shared stage current source/input bytes changed: '+p)
    expected=dict(objects)
    for p,v in contract['historical_frozen_imports'].items():
        if expected.pop(p)!=v['sha256'] or refs['historical_frozen_imports'].get(p)!={k:v[k] for k in ['git_commit','blob_oid','sha256']}:raise ValueError('only exact existing historical overlays allowed')
        verify_Git(read(verify_pin(v['proof'])),v['git_commit'],{p:v['sha256']})
    packed=read(verify_pin(contract['actual_Git_object_proof']));pool=read(verify_pin(contract['shared_object_pool']));expanded=expand_proof(packed,pool)
    verify_Git(expanded,contract['git_commit'],expected);return contract
