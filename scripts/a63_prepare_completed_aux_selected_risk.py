#!/usr/bin/env python3
"""Preserve the base freeze and add actual-Git administrative source closure."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from echo_routing.temporal import selected_risk_resume_freeze as base
from scripts.a63_resume_completed_aux_selected_risk import EXTRA_SOURCES,verify_extra_sources


def make(context,stage,spec_pin,spec,commit):
    from echo_routing.temporal.selected_risk_contracts import source_closure
    from echo_routing.temporal.safety_git import build,verify
    from echo_routing.temporal.selected_risk_storage_engineering import pack_proof,expand_proof
    # The original maker independently validates authority, all original
    # sources, historical objects and inputs. Its immutable result is retained.
    basepin=base.make(context,stage,spec_pin,spec,commit);prior=context.read(basepin)
    objects=dict(prior['objects'])
    for path in source_closure(context.root,EXTRA_SOURCES):
        raw=(context.root/path).read_bytes();h=hashlib.sha256(raw).hexdigest()
        context.raw({'path':path,'sha256':h})
        if path in objects and objects[path]!=h:raise ValueError('base scientific source identity cannot change')
        objects[path]=h
    current={p:h for p,h in objects.items() if p not in prior['historical_frozen_imports']}
    full=build(context.root,commit,sorted(current));verify(full,commit,current)
    pool=context.read(prior['shared_object_pool']);packed=pack_proof(full,pool)
    if expand_proof(packed,pool)!=full:raise ValueError('administrative superset must preserve exact Git bytes')
    version=stage+'.'+prior['runtime_spec_semantics_sha256'][:12]+'.completed_aux'
    for name in ['contracts/'+version+'.json','Git_proofs/'+version+'/current.shared.private.json']:
        path=context.output(name)
        if path.exists() or path.is_symlink():raise FileExistsError('preserve every prior or partial administrative freeze')
    proof=context.write_json('Git_proofs/'+version+'/current.shared.private.json',packed)
    return context.write_json('contracts/'+version+'.json',{**prior,'objects':objects,'actual_Git_object_proof':proof})


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
