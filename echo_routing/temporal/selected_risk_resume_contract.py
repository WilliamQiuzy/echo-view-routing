"""A63 continuation contracts. Preparation never confers execution authority.

Old card/config/window/840-file closure remain immutable. A separate, explicit
new-window owner receipt plus fresh full-work caps and stage evidence are needed.
This pure module creates no files, acquires no queue and loads no model/data.
"""
import copy
import hashlib
import json
import math
from pathlib import PurePosixPath

RUN='runs/a63_selected_action_risk_resume_v1'
PUB='docs/results/2026-11-05_a63_resume_execution'
PREP='docs/results/2026-11-05_a63_resume_preparation'
SHARED_QUEUE='runs/a63_selected_action_risk_v1/exclusive_supervisor.lock'
CARD='191acbf1204cd6bf3fbdb6066800a3c6e7d6208ff7ccc15fe752c7964daead7a'
BASIS='78b4f913064d7300238577c561384cc559ba09f2c54be84bffa8d876f9bbacbf'
OLD_MANIFEST='13fb3877c80a794ee706316867f8ae6632750288f540ae6da7ba9cc7a84f18e6'
OLD_T0=1793853840
OLD_DEADLINE=1793897040
STORAGE_AMENDMENT_SHA='be8e892903892d91c7e1ef8d4c523671ec7d0b4e42e95482af3238c2aa64dc23'
STORAGE_AMENDMENT_PATH='docs/superpowers/plans/2026-11-05-a63-storage-window-amendment.md'
STORAGE_APPROVAL={'path':PREP+'/user_storage_window_approval_v1.json','sha256':'321a7c283bb66d27b232c407d99070d45cb5b020a8d20496135075b591fd4301'}

def storage_cap(owner):
    if owner.get('schema')=='A63_explicit_resume_execution_authorization_v2':
        if owner.get('approved_proposal_sha256')!=STORAGE_AMENDMENT_SHA or type(owner.get('storage_lifecycle_cap_bytes')) is not int or owner['storage_lifecycle_cap_bytes']!=1300000000 or owner.get('storage_amendment_sha256')!=STORAGE_AMENDMENT_SHA or owner.get('storage_approval_receipt')!=STORAGE_APPROVAL or owner.get('recorded_owner_user_authorization_sha256')!=STORAGE_APPROVAL['sha256']:raise ValueError('only the exact explicitly approved1.3GB lifecycle exception is allowed')
        return 1300000000
    if any(k in owner for k in ['storage_lifecycle_cap_bytes','storage_amendment_sha256','storage_approval_receipt']):raise ValueError('arbitrary cap fields without exact V2 approval are forbidden')
    return 1000000000

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def _sha(v,n=64):
    if not isinstance(v,str) or len(v)!=n or any(c not in '0123456789abcdef' for c in v):raise ValueError('canonical hash identity required')
    return v

def validate_execution_authority(raw_bytes,approved_sha,proposal_sha,baseline_sha,now):
    if not isinstance(raw_bytes,bytes) or approved_sha is None:raise ValueError('new explicit raw execution approval absent; preparation only')
    _sha(approved_sha);_sha(proposal_sha);_sha(baseline_sha)
    if hashlib.sha256(raw_bytes).hexdigest()!=approved_sha:raise ValueError('external owner file SHA differs')
    def unique_pairs(pairs):
        out={}
        for k,v in pairs:
            if k in out:raise ValueError('duplicate owner JSON keys forbidden')
            out[k]=v
        return out
    try:value=json.loads(raw_bytes,object_pairs_hook=unique_pairs,parse_constant=lambda s:(_ for _ in ()).throw(ValueError('nonfinite JSON token')))
    except (ValueError,UnicodeError,TypeError) as e:raise ValueError('strict owner raw JSON required') from e
    fields={'schema','execution_authorized','preparation_only','purpose','approved_proposal_sha256','scientific_card_sha256','scientific_basis_sha256','previous_public_manifest_sha256','runtime_RUN','runtime_PUB','T0_epoch','deadline_epoch','prior_T0_epoch','prior_deadline_epoch','new_budget_granted','recorded_owner_user_authorization_sha256','closed_baseline_account_sha256'}
    if isinstance(value,dict) and value.get('schema')=='A63_explicit_resume_execution_authorization_v2':fields|={'storage_lifecycle_cap_bytes','storage_amendment_sha256','storage_approval_receipt'}
    if not isinstance(value,dict) or set(value)!=fields:raise ValueError('exact separately pinned new owner receipt required')
    if value['schema'] not in ['A63_explicit_resume_execution_authorization_v1','A63_explicit_resume_execution_authorization_v2'] or value['execution_authorized'] is not True or value['preparation_only'] is not False or value['purpose']!='continue_A63_same_science' or value['new_budget_granted'] is not False:raise ValueError('this is not new-window continuation execution authority')
    if (value['approved_proposal_sha256'],value['scientific_card_sha256'],value['scientific_basis_sha256'],value['previous_public_manifest_sha256'],value['closed_baseline_account_sha256'])!=(proposal_sha,CARD,BASIS,OLD_MANIFEST,baseline_sha):raise ValueError('approved science/proposal/closed account mismatch')
    if type(value['prior_T0_epoch']) is not int or type(value['prior_deadline_epoch']) is not int or (value['runtime_RUN'],value['runtime_PUB'],value['prior_T0_epoch'],value['prior_deadline_epoch'])!=(RUN,PUB,OLD_T0,OLD_DEADLINE):raise ValueError('new namespaces and permanent old window required')
    a,b=value['T0_epoch'],value['deadline_epoch']
    if type(a) is not int or type(b) is not int or a<=OLD_T0 or b-a!=43200 or isinstance(now,bool) or not isinstance(now,(int,float)) or not math.isfinite(now) or not a<=now<b:raise ValueError('new fixed12h window required; old clock never resets')
    _sha(value['recorded_owner_user_authorization_sha256']);storage_cap(value);return True

_STAGES={
 'adopt_metadata':([],1), 'admit_FIT':(['adopt_metadata'],1),
 'source_FIT':(['admit_FIT'],6000), 'source_FIT_adoption':(['source_FIT'],1), 'observe_geometry':(['source_FIT','source_FIT_adoption'],1),
 'admit_aux':(['observe_geometry'],1), 'prepare_aux':(['admit_aux','observe_geometry'],2),
 'train_aux':(['prepare_aux','admit_aux'],8), 'audit_aux':(['train_aux','prepare_aux','admit_aux'],1),
 'seal_aux':(['audit_aux','prepare_aux'],1), 'held_score':(['seal_aux'],2),
 'seal_selectors':(['held_score','seal_aux','admit_aux'],1), 'coverage':(['seal_selectors','seal_aux'],2),
 'admit_formal':(['coverage','seal_aux'],1),
 'prepare_formal':(['admit_formal','coverage','seal_selectors','seal_aux'],2),
 'source_CONFIRM':(['admit_formal','coverage','seal_aux'],480),
 'admit_formal_after_CONFIRM':(['source_CONFIRM','prepare_formal'],1),
 'train_formal':(['prepare_formal','admit_formal_after_CONFIRM','coverage','seal_aux','source_CONFIRM','seal_selectors'],4),
 'audit_formal':(['train_formal','prepare_formal','coverage','seal_selectors'],1),
 'fit_risk_scores':(['audit_formal','train_formal','prepare_formal','coverage','seal_selectors'],1),
 'calibrate':(['audit_formal','coverage','fit_risk_scores','prepare_formal','seal_selectors'],34), 'seal_heads':(['calibrate','audit_formal','fit_risk_scores'],1),
 'adopt_confirmation':(['seal_heads','source_CONFIRM'],480),
 'admit_outer':(['seal_heads','source_CONFIRM'],1),
 'outer':(['adopt_confirmation','admit_outer','seal_heads','coverage'],238),
 'aggregate':(['outer','seal_heads','adopt_confirmation','coverage'],140), 'supports':(['aggregate','outer','seal_heads','adopt_confirmation','coverage'],56),
 'forensics':(['outer','aggregate','seal_heads','adopt_confirmation','coverage'],{'native':34,'regression':32}),
 'package':(['aggregate','supports','forensics','calibrate','audit_formal','audit_aux','coverage','train_formal','fit_risk_scores'],1),
 'scientific_review':(['package','aggregate','audit_formal'],1), 'resource_review':(['package','aggregate','audit_formal','scientific_review'],1),
 'close_negative':([],1), 'close_full':(['scientific_review','resource_review','package','aggregate'],1)}

def stage_graph():
    return {k:{'requires':list(deps),'expected_outputs':copy.deepcopy(n),'optimizer_updates':48000 if k=='train_aux' else 24000 if k=='train_formal' else 0,'GPU_source_only':k in ['source_FIT','source_CONFIRM'],'new_execution_authorization_required':True} for k,(deps,n) in _STAGES.items()}

def require_phase(stage,evidence,*,runtime_authorization_sha=None):
    """Check structured predecessor payloads AFTER IO verifies their Git proofs.

    This is not a substitute for actual byte/Git/private runtime verification.
    The driver must reconstruct coverage from sealed records before C0/C1.
    """
    if stage not in _STAGES or not isinstance(evidence,dict):raise ValueError('exact declared continuation stage required')
    _sha(runtime_authorization_sha)
    for dep in _STAGES[stage][0]:
        v=evidence.get(dep)
        if not isinstance(v,dict) or set(v)!={'payload','payload_sha256','git_commit','object_proof_sha256','runtime_authorization_sha256'} or not isinstance(v['payload'],dict):raise ValueError('structured verified predecessor evidence absent: '+dep)
        for k in ['payload_sha256','object_proof_sha256','runtime_authorization_sha256']:_sha(v[k])
        _sha(v['git_commit'],40)
        if digest(v['payload'])!=v['payload_sha256']:raise ValueError('predecessor payload digest differs')
        if v['runtime_authorization_sha256']!=runtime_authorization_sha:raise ValueError('predecessor from another window/authorization forbidden')
        _successful(dep,v['payload'])
    if stage=='train_formal':
        admission=evidence['admit_formal_after_CONFIRM']['payload']
        if admission.get('completed_CONFIRM_sha256')!=evidence['source_CONFIRM']['payload_sha256'] or admission.get('prepared_formal_sha256')!=evidence['prepare_formal']['payload_sha256']:raise ValueError('post-CONFIRM full admission must bind actually completed teacher and prepared formal inputs')
    if stage in ['admit_formal','prepare_formal','train_formal','source_CONFIRM']:
        cov=evidence['coverage']['payload'];audit=evidence['seal_aux']['payload']
        validate_coverage_pass(cov)
        if audit.get('passed') is not True or (audit.get('aux_optimizer_updates'),audit.get('physical_models'),audit.get('fullstate_count'))!=(48000,8,32):raise ValueError('complete actual auxiliary seal required')
    return True

def _successful(stage,p):
    """Git proves bytes, not success: check explicit stage verdict/counts too.

    Live receipt/ledger freshness and independent recomputation of the source
    payload remain runtime caller obligations. A historical admission is never
    permanent execution authority.
    """
    if stage.startswith('admit_'):
        if p.get('all_caps_fit') is not True:raise ValueError('complete remaining-work admission must pass')
        return
    if stage in ['source_FIT','source_CONFIRM']:
        phase='FIT' if stage=='source_FIT' else 'CONFIRM';n=6000 if phase=='FIT' else 480
        if p.get('complete') is not True or p.get('phase')!=phase or (p.get('committed_payloads'),p.get('aux_updates_at_production'),p.get('formal_updates_at_production'))!=(n,0 if phase=='FIT' else 48000,0):raise ValueError('actually complete exact-phase source payload/counters required')
        return
    if p.get('passed') is not True:raise ValueError('successful predecessor verdict required: '+stage)
    if stage in ['train_aux','audit_aux','seal_aux']:
        if (p.get('aux_optimizer_updates'),p.get('physical_models'),p.get('fullstate_count'))!=(48000,8,32):raise ValueError('actual complete auxiliary48000/eight/32 predecessor required')
    if stage in ['train_formal','audit_formal','seal_heads']:
        if (p.get('formal_optimizer_updates'),p.get('physical_models'),p.get('fullstate_count'))!=(24000,4,16):raise ValueError('actual complete formal24000/four/16 predecessor required')
    if stage=='prepare_aux' and (len(p.get('folds',[]))!=2 or {r.get('held_fold') for r in p['folds']}!={0,1}):raise ValueError('both independently prepared complement folds required')
    if stage=='prepare_formal' and p.get('seeds')!=[2026110461,2026110462]:raise ValueError('both independently prepared matched formal seeds required')
    if stage=='coverage':validate_coverage_pass(p)
    if stage=='source_FIT_adoption':
        if p.get('complete') is not True or (p.get('committed_payloads'),p.get('original_source_stage_exit_status'),p.get('original_teacher_inner_OS_exit_code'))!=(6000,1,0) or p.get('original_source_stage_cap_failure_preserved') is not True or any(type(p.get(k)) is not int or p[k]!=0 for k in ['new_teacher_forwards','new_optimizer_updates','cache_writes']):raise ValueError('actual complete readonly source adoption preserving original failure required')
    if stage=='fit_risk_scores':
        if p.get('complete') is not True or p.get('seeds')!=[2026110461,2026110462] or len(p.get('records',[]))!=2 or {r.get('seed') for r in p['records']}!={2026110461,2026110462} or p.get('FIT_q_completed_before_CAL') is not True or p.get('FIT_scores_used_to_select_model_or_tau') is not False or p.get('new_teacher_or_S_L_or_aux_forward_calls')!=0 or p.get('optimizer_updates')!=0:raise ValueError('complete pre-CAL two-seed final-risk-only FIT scoring required')
        for key in ['actual_new_MLP_calls_this_process','actual_new_candidate_outputs_this_process']:
            if type(p.get(key)) is not int or p[key]<0:raise ValueError('actual diagnostic MLP calls/outputs required, not inferred zero')
    counts={'outer':('row_count',238),'aggregate':('comparison_count',140),'supports':('record_count',56),'adopt_confirmation':('committed_payloads',480)}
    if stage in counts:
        key,n=counts[stage]
        if type(p.get(key)) is not int or p[key]!=n:raise ValueError('complete fixed endpoint count required: '+stage)
    if stage=='calibrate' and (len(p.get('inner_records',[]))!=34 or len(p.get('thresholds',[]))!=4):raise ValueError('all34 inner records and four frozen thresholds required')
    if stage=='forensics' and (p.get('native_count'),p.get('regression_count'))!=(34,32):raise ValueError('all native34 and regression32 endpoints required')

def validate_coverage_pass(cov):
    from echo_routing.temporal.selected_risk_figures import FLOORS,CONCENTRATION
    if cov.get('passed') is not True or set(cov.get('seeds',{}))!={'2026110461','2026110462'} or cov.get('source_isolation',{}).get('passed') is not True or type(cov.get('aux_optimizer_updates')) is not int or type(cov.get('formal_optimizer_updates')) is not int or (cov.get('aux_optimizer_updates'),cov.get('formal_optimizer_updates'))!=(48000,0):raise ValueError('complete isolated 48000 auxiliary coverage before formal required')
    for row in cov['seeds'].values():
        checks=row.get('checks',{})
        if row.get('passed') is not True or set(checks)!=set(FLOORS)|set(CONCENTRATION) or row.get('failed_checks')!=[]:raise ValueError('both seeds must pass all original23 checks')
        for key,(minimum,group,field) in FLOORS.items():
            c=checks[key];a=c.get('actual')
            if type(a) is not int or a<minimum or type(c.get('minimum')) is not int or c.get('minimum')!=minimum or c.get('passed') is not True or type(row.get(group,{}).get(field)) is not int or row[group][field]!=a:raise ValueError('original signature/session floors and source aggregates required')
        for key in CONCENTRATION:
            c=checks[key];a=c.get('actual')
            if isinstance(a,bool) or not isinstance(a,(float,int)) or not math.isfinite(a) or not 0<=a<=.2+1e-15 or c.get('maximum')!=.2 or c.get('passed') is not True or row.get(key)!=a:raise ValueError('original .5 actual signature-source concentration bound required')
    return True

def output_relative(path,*,public=False):
    p=PurePosixPath(path)
    if not isinstance(path,str) or not path or p.is_absolute() or str(p)!=path or any(x in ['.','..'] for x in p.parts):raise ValueError('plain new continuation relative output required')
    return (PUB if public else RUN)+'/'+path
