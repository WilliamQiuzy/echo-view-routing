"""Separate prospective cost-only contracts; executed science contracts stay exact."""
import re
from pathlib import Path
from echo_routing.temporal.event_risk_contracts import source_closure

PUB='docs/results/2026-11-04_anchor_context_risk/'


def stage_spec(name):
    if name=='refinement_probe_freeze.json':
        return ['scripts/probe_event_risk_refinement.py'],[PUB+p for p in ['input_manifest.json','cost_profile.json','COST_REFINEMENT.md','cost_preview_v1.json']]
    match=re.fullmatch(r'(sources|formal|outer)_refined_admission_v[1-9][0-9]*_freeze.json',name)
    if not match:raise ValueError('unrecognized independent cost-engineering contract')
    inputs=['input_manifest.json','cost_profile.json','cost_refinement_profile.json','COST_REFINEMENT.md','cost_preview_v1.json']
    if match[1]!='sources':inputs+=['training_inputs.json','FIT_checks.json','producer_manifest.json']
    if match[1]=='outer':inputs+=['models_manifest.json','calibration.json']
    return ['scripts/admit_event_risk_refined.py'],[PUB+p for p in inputs]


def validate_coverage(root,name,objects,base):
    entries,inputs=stage_spec(name)
    required=set(source_closure(root,entries+inputs+list(base)))
    if not required<=set(objects):raise ValueError('incomplete independent engineering source/input closure')
    return True


def verify(name):
    from scripts.event_risk_common import ROOT,RUN,CFG,read,sha,verify_pin
    from echo_routing.temporal.safety_git import verify as gitverify
    contract=read(RUN/'engineering_contracts'/name);cfg=read(CFG)
    base=[str(CFG.relative_to(ROOT)),cfg['card']['path'],cfg['authorization']['path'],cfg['basis']['path']]
    validate_coverage(ROOT,name,contract['objects'],base)
    if contract['stage']!=name or contract['fixed_deadline_epoch']!=1793867561:raise ValueError('engineering stage identity/deadline differs')
    for p,h in contract['objects'].items():
        if sha(ROOT/p)!=h:raise ValueError('executed engineering source/input changed: '+p)
    gitverify(read(verify_pin(contract['actual_Git_object_proof'])),contract['git_commit'],contract['objects'])
    return contract
