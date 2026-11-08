"""One independent versioned cost-unit contract; no old registry/proof rewrites."""
from scripts.selected_risk_common import *
from scripts.selected_risk_shared_contracts import required as old_required
from echo_routing.temporal.selected_risk_contracts import source_closure
from echo_routing.temporal.selected_risk_storage_engineering import expand_proof
STAGE='sources_costunit_admission_v3'
ENTRY='scripts/a63_admit_selected_risk_units.py'

def required(stage):
    if stage!=STAGE:raise ValueError('one literal versioned cost-unit stage only')
    return old_required('sources_refined_admission')|set(source_closure(ROOT,[ENTRY,'scripts/a63_supervise_selected_risk_units.py','scripts/a63_selected_risk_costunit_contract.py']))|{relative(PUB/'COST_UNITS_V3.md')}

def verify_shared_freeze(stage):
    if stage!=STAGE:raise ValueError('one literal versioned cost-unit stage only')
    old=verify_freeze('metadata');c=read(RUN/'contracts'/(STAGE+'.json'));objects=c['objects'];refs=read(verify_pin(read(CFG)['source_refs']))
    if c['stage']!=STAGE or c.get('proof_encoding')!='A63_shared_Git_objects_v1' or c['shared_object_pool']!=old['actual_Git_object_proof'] or not required(stage)<=set(objects):raise ValueError('exact old pool/mandatory cost-unit source and inputs required')
    validate_frozen_prior(objects)
    for p,h in objects.items():
        if sha(ROOT/p)!=h:raise ValueError('current cost-unit source/input drift: '+p)
    expected=dict(objects)
    for p,v in c['historical_frozen_imports'].items():
        if expected.pop(p)!=v['sha256'] or refs['historical_frozen_imports'].get(p)!={k:v[k] for k in ['git_commit','blob_oid','sha256']}:raise ValueError('historical source identity differs')
        verify_Git(read(verify_pin(v['proof'])),v['git_commit'],{p:v['sha256']})
    proof=expand_proof(read(verify_pin(c['actual_Git_object_proof'])),read(verify_pin(c['shared_object_pool'])));verify_Git(proof,c['git_commit'],expected);return c
