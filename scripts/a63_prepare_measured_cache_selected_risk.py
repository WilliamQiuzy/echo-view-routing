#!/usr/bin/env python3
"""Preserve the base freeze and add actual-Git administrative source closure."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from echo_routing.temporal import selected_risk_resume_freeze as base
from scripts.a63_resume_measured_cache_selected_risk import EXTRA_SOURCES,verify_extra_sources


def make(context,stage,spec_pin,spec,commit):
    from echo_routing.temporal.selected_risk_contracts import source_closure
    from echo_routing.temporal.safety_git import build,verify,oid
    from echo_routing.temporal.selected_risk_storage_engineering import pack_proof,expand_proof
    from scripts.selected_risk_common import validate_frozen_prior
    # One prospective superset proof. Every old output remains untouched; the
    # original fixed pool and literal historical proofs retain their raw pins.
    context.revalidate();context.bind_runtime_spec(spec_pin,spec);commit=oid(commit)
    prior=context.read(spec['arguments']['measured_cache_prior_freeze'])
    if prior.get('schema')!='A63_resume_stage_freeze_v1' or prior.get('runtime_authorization_sha256')!=context.owner_sha:
        raise ValueError('same-owner immutable prior freeze required for historical proof reuse')
    paths=set(base.required(context.root,stage,spec_pin,spec))
    paths.update(source_closure(context.root,EXTRA_SOURCES))
    objects={p:base._raw(context,p)[1] for p in sorted(paths)}
    validate_frozen_prior(objects)
    _,historical=base._base_paths(context.root)
    if set(prior['historical_frozen_imports'])!=set(historical):
        raise ValueError('complete literal historical proof table required')
    current=dict(objects);hrows={}
    for path,expected in historical.items():
        row=prior['historical_frozen_imports'][path]
        if {k:row[k] for k in ['git_commit','blob_oid','sha256']}!=expected or current.pop(path,None)!=expected['sha256'] or prior['objects'].get(path)!=expected['sha256']:
            raise ValueError('all approved historical sources must retain their literal identities')
        proof=context.contract_proof(prior,row['proof'])
        verify(proof,expected['git_commit'],{path:expected['sha256']})
        if proof['files'][path]['blob_oid']!=expected['blob_oid']:
            raise ValueError('historical actual Git blob tuple changed')
        hrows[path]=row
    # contract_proof also enforces the unchanged original Context pool path;
    # a new pool cannot silently replace it or require a different reader.
    pool_pin=prior['shared_object_pool']
    if prior.get('proof_encoding')!='A63_shared_Git_objects_v1' or pool_pin.get('path')!=base.POOL:
        raise ValueError('exact original immutable Git pool required')
    pool=context.read(pool_pin)
    full=build(context.root,commit,sorted(current));verify(full,commit,current)
    packed=pack_proof(full,pool)
    if expand_proof(packed,pool)!=full:raise ValueError('administrative superset must preserve exact Git bytes')
    verify(expand_proof(packed,pool),commit,current)
    version=stage+'.'+base.semantics(spec)[:12]+'.measured_cache.single'
    for name in ['contracts/'+version+'.json','Git_proofs/'+version+'/current.shared.private.json']:
        path=context.output(name)
        if path.exists() or path.is_symlink():raise FileExistsError('preserve every prior or partial administrative freeze')
    proof_pin=context.write_json('Git_proofs/'+version+'/current.shared.private.json',packed)
    return context.write_json('contracts/'+version+'.json',{
      'schema':'A63_resume_stage_freeze_v1','stage':stage,'git_commit':commit,'objects':objects,
      'historical_frozen_imports':hrows,'actual_Git_object_proof':proof_pin,
      'runtime_authorization_sha256':context.owner_sha,'runtime_spec_semantics_sha256':base.semantics(spec),
      'proof_encoding':'A63_shared_Git_objects_v1','shared_object_pool':pool_pin})


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execute',action='store_true')
    p.add_argument('--make-freeze',action='store_true');p.add_argument('--runtime-spec');p.add_argument('--runtime-spec-sha')
    p.add_argument('--owner-sha');p.add_argument('--git-commit');a=p.parse_args(argv)
    if not a.execute:
        print(json.dumps({'preparation_only':True,'new_process_or_queue_or_model':False,'extra_source_paths':EXTRA_SOURCES}));return 0
    if not a.make_freeze or not all([a.runtime_spec,a.runtime_spec_sha,a.owner_sha,a.git_commit]):
        p.error('explicit freeze operation and all external owner/spec/Git pins required')
    from echo_routing.temporal.selected_risk_resume_runtime import bootstrap
    pin={'path':a.runtime_spec,'sha256':a.runtime_spec_sha};ctx,spec=bootstrap(ROOT,pin,a.owner_sha)
    contract=make(ctx,spec['stage'],pin,spec,a.git_commit)
    final={**spec,'freeze':contract}
    out=ctx.write_json('runtime_specs/'+spec['stage']+'.'+contract['sha256'][:12]+'.json',final)
    ctx.bind_runtime_spec(out,final);verify_extra_sources(ctx,final)
    print(json.dumps({'runtime_spec':out,'freeze':contract,'base_freeze_preserved':True,'actual_stage_executed':False}));return 0


if __name__=='__main__':raise SystemExit(main())
