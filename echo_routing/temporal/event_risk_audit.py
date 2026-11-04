"""A62 independent physical-trace/fullstate/pre-outer Git audits; no forward.

A test fixture can satisfy a schema; only runtime file/OS/source receipts bind
these checks to actual training. A passed flag alone is never accepted.
"""
import json,math,re
import numpy as np
from echo_routing.temporal.event_risk_metrics import plan as endpoint_plan
from echo_routing.temporal.safety_git import verify as verify_Git,unpack

SEEDS=(2026110461,2026110462)
ARMS=('R1','R2')
NAMES={'network.0.weight','network.0.bias','network.2.weight','network.2.bias'}
PREFIX='docs/results/2026-11-04_anchor_context_risk/'
RUN='runs/a62_anchor_context_risk_v1/'
FIXED_A61=(
 {'seed':2026110461,'arm':'S-L','path':'runs/a61_action_safety_ranking_v1/models/seed2026110461/S-L/step6000.pt','sha256':'039bbec11f6be5b1379e50b9334934c09250ab7ef035bcf630014bb38f0c1a80'},
 {'seed':2026110462,'arm':'S-L','path':'runs/a61_action_safety_ranking_v1/models/seed2026110462/S-L/step6000.pt','sha256':'b3eaf1bdf3a33a0d98928ba834d05d5b6f945d6065db9d25e241468af51b8af2'},
 {'seed':2026110461,'arm':'risk','path':'runs/a61_action_safety_ranking_v1/models/seed2026110461/risk/step6000.pt','sha256':'313c7995c9189a4b527f8324dd8b37e75719524a0d021d9864014d8665af2311'},
 {'seed':2026110462,'arm':'risk','path':'runs/a61_action_safety_ranking_v1/models/seed2026110462/risk/step6000.pt','sha256':'f82c28f3f091644233d8f2c6cd83e68e9daf06bff2c9ab8d934cc3f746797432'},
)


def _hash(v):
    if not isinstance(v,str) or not re.fullmatch('[0-9a-f]{64}',v):raise ValueError('canonical SHA256 required')
    return v


def _identity(v):
    if not isinstance(v,dict) or set(v)!={'code','config','sources','input','draw_plan','normalizer','target_scales'}:raise ValueError('complete seven-field training identity required')
    for h in v.values():_hash(h)
    return v


def audit_trace(rows,plan,kind_indices,unsafe_targets,seed,identity=None):
    from echo_routing.temporal.safety_training import draw_plan
    from echo_routing.temporal.proposal_training import lr_factor
    if seed not in SEEDS or isinstance(seed,(bool,np.bool_)) or len(rows)!=6000 or set(kind_indices)!={'native','constructed'}:raise ValueError('complete6000 physical fixedseed event trace required')
    y=np.asarray(unsafe_targets)
    if y.shape!=(4183,) or not np.isfinite(y).all() or not np.isin(y,[0,1]).all():raise ValueError('exact4183 complete binary FIT unsafe targets required')
    pools={k:np.asarray(v) for k,v in kind_indices.items()}
    if any(pools[k].shape!=(n,) or not np.issubdtype(pools[k].dtype,np.integer) for k,n in [('native',265),('constructed',3918)]) or not np.array_equal(np.sort(np.r_[pools['native'],pools['constructed']]),np.arange(4183)):raise ValueError('exact disjoint265/3918 original FIT candidate row coverage required')
    expected=draw_plan(265,3918,seed,'risk')
    if set(plan)!=set(expected) or any(not np.array_equal(plan[k],v) if isinstance(v,np.ndarray) else plan[k]!=v for k,v in expected.items()):raise ValueError('original A61 risk draw bytes/domain changed')
    identity=_identity(rows[0]['identity'] if identity is None else identity);live={k:False for k in NAMES};unsafe=0
    for u,row in enumerate(rows):
        if isinstance(row['u'],(bool,np.bool_)) or not isinstance(row['u'],(int,np.integer)) or row['u']!=u or row['identity']!=identity or row['finite_gradients'] is not True or set(row['nonzero_gradient_tensors'])!=NAMES or any(not isinstance(v,bool) for v in row['nonzero_gradient_tensors'].values()):raise ValueError('physical cursor/identity/gradient tensor coverage changed')
        for key in ['loss','candidate_BCE','gradient_norm','optimizer_update_wall_seconds']:
            if not math.isfinite(row[key]) or row[key]<0:raise ValueError('finite actual BCE/gradient/full update timing required')
        if row['loss']!=row['candidate_BCE'] or not math.isclose(row['lr'],.001*lr_factor(u),rel_tol=1e-12,abs_tol=1e-15):raise ValueError('unweighted BCE or original LR schedule changed')
        idx=[]
        for block,kind in enumerate(expected['source_kinds']):
            idx.extend(pools[['native','constructed'][kind]][expected['indices'][u,32*block:32*(block+1)]].tolist())
        if row['candidate_indices']!=idx or row['source_kind_blocks']!=expected['source_kinds'] or row['physical_candidates']!=64 or row['unsafe_targets']!=int(y[idx].sum()):raise ValueError('exact64 full candidate/unsafe exposure differs from original draws')
        unsafe+=int(y[idx].sum())
        for name,flag in row['nonzero_gradient_tensors'].items():live[name]|=flag
    if not all(live.values()):raise ValueError('an active event tensor never received nonzero gradient')
    return {'physical_updates':6000,'candidate_exposures':384000,'native_candidate_exposures':192000,'constructed_candidate_exposures':192000,'unsafe_candidate_exposures':unsafe,'all_tensor_gradient_liveness_over_history':live,'no_model_forward_during_audit':True,'original_A61_risk_draw_verified':True}


def tensor_equal(a,b):
    import torch
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and a.dtype==b.dtype and a.shape==b.shape and torch.equal(a,b)
    if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and a.dtype==b.dtype and np.array_equal(a,b)
    if isinstance(a,dict):return isinstance(b,dict) and set(a)==set(b) and all(tensor_equal(v,b[k]) for k,v in a.items())
    if isinstance(a,(list,tuple)):return type(a)==type(b) and len(a)==len(b) and all(tensor_equal(x,y) for x,y in zip(a,b))
    return type(a)==type(b) and a==b


def validate_state(model,opt,state,identity,norms,cursor):
    import torch
    from echo_routing.temporal.event_risk_model import restore
    if isinstance(cursor,bool) or not isinstance(cursor,int) or not 0<=cursor<=6000 or torch.cuda.is_initialized():raise ValueError('canonical CPU cursor required before restoring any state')
    if sum(p.numel() for p in model.parameters())!=6625 or {n for n,_ in model.named_parameters()}!=NAMES or restore(model,opt,state,identity,norms)!=cursor:raise ValueError('actual event state/Adam cursor changed')
    if cursor and any(not bool(opt.state[p]['exp_avg_sq'].count_nonzero()) for p in model.parameters()):raise ValueError('active tensor has no nonzero Adam second moment')
    return {'cursor':cursor,'parameters':6625,'full_Adam_RNG_norm_identity_validated':True,'no_model_forward_during_audit':True}


def validate_fullstates(model,opt,states,identity,norms,initial_state):
    import torch
    if set(states)!={2000,4000,6000,'latest'} or set(initial_state)!=set(model.state_dict()):raise ValueError('exactthree milestones pluslatest and original initializer required')
    for cursor in [2000,4000,6000]:validate_state(model,opt,states[cursor],identity,norms,cursor)
    if not tensor_equal(states[6000],states['latest']):raise ValueError('latest/final complete state differs')
    validate_state(model,opt,states['latest'],identity,norms,6000)
    if any(initial_state[k].shape!=v.shape or initial_state[k].dtype!=v.dtype or torch.equal(initial_state[k],v) for k,v in model.state_dict().items()):raise ValueError('active tensor never drifted from matched initializer')
    return {'fullstates':4,'milestones':[2000,4000,6000],'latest_cursor':6000,'all_active_tensors_drifted':True,'latest_final_complete_state_equal':True,'no_model_forward_during_audit':True}


def _coverage(rows,fields,expected):
    if len(rows)!=len(expected) or {tuple(r[k] for k in fields) for r in rows}!=set(expected):raise ValueError('exact unique audited endpoint coverage required')


def validate_barrier(models,seal,modelsha,calsha,plansha,proof):
    import base64
    for h in [modelsha,calsha,plansha]:_hash(h)
    if models['formal_optimizer_updates']!=24000 or models['physical_models']!=4:raise ValueError('complete4models24000 actual audited barrier required')
    expected={(s,a) for s in SEEDS for a in ARMS};new=models['models'];fixed=models['fixed_A61_models'];states=models['fullstates']
    _coverage(new,('seed','arm'),expected)
    for r in new:
        if r['step']!=6000 or r['parameters']!=6625 or r['path']!=f"{RUN}models/seed{r['seed']}/{r['arm']}/step6000.pt":raise ValueError('fixed6000 matched6625 event final identity required')
        _hash(r['sha256'])
    fixed_keys={(r['seed'],r['arm'],r['path'],r['sha256']) for r in FIXED_A61}
    _coverage(fixed,('seed','arm','path','sha256'),fixed_keys)
    statekeys={(s,a,k,'milestone') for s,a in expected for k in [2000,4000,6000]}|{(s,a,6000,'latest') for s,a in expected}
    _coverage(states,('seed','arm','step','kind'),statekeys)
    for r in states:
        name='latest.pt' if r['kind']=='latest' else f"step{r['step']}.pt"
        if r['path']!=f"{RUN}models/seed{r['seed']}/{r['arm']}/{name}":raise ValueError('exact milestone/latest private path required')
        _hash(r['sha256'])
    final_states={(r['seed'],r['arm']):(r['path'],r['sha256']) for r in states if r['kind']=='milestone' and r['step']==6000}
    if any(final_states[(r['seed'],r['arm'])]!=(r['path'],r['sha256']) for r in new):raise ValueError('finalcheckpoint and actual6000state pins differ')
    _coverage(models['thresholds'],('seed','policy'),{(s,p) for s in SEEDS for p in ['M0','R1','R2']})
    _coverage(models['legacy_thresholds'],('seed','policy'),{(s,'L0') for s in SEEDS})
    for r in models['thresholds']+models['legacy_thresholds']:
        if isinstance(r['tau'],bool) or not math.isfinite(r['tau']) or not 0<=r['tau']<=1:raise ValueError('finite strict unit-range frozen thresholds required')
    old={2026110461:.007963286712765694,2026110462:.006616456434130669}
    if any(r['tau']!=old[r['seed']] for r in models['legacy_thresholds']):raise ValueError('original A61 SL threshold changed')
    expected_plan=endpoint_plan();_coverage(models['inner_records'],('population','system'),{(r['population'],r['system']) for r in expected_plan['inner_rows']})
    if models['calibration_sha256']!=calsha or models['readout_plan_sha256']!=plansha or models['audited_readout_plan_sha256']!=plansha or seal['models_manifest_sha256']!=modelsha or seal['calibration_sha256']!=calsha or seal['readout_plan_sha256']!=plansha:raise ValueError('current/seal/audited model/calibration/plan SHA binding differs')
    finals={(r['path'],r['sha256']) for r in [*new,*fixed]}
    _coverage(seal['all_final_checkpoints'],('path','sha256'),finals)
    if proof is None or 'git_commit' not in seal:raise ValueError('actual Git public commit/tree/blob proof required')
    expected={PREFIX+'models_manifest.json':modelsha,PREFIX+'calibration.json':calsha,PREFIX+'readout_plan.json':plansha}
    verify_Git(proof,seal['git_commit'],expected)
    def blob(name):
        oid=proof['files'][PREFIX+name]['blob_oid'];kind,raw=unpack(base64.b64decode(proof['objects'][oid],validate=True),oid)
        return json.loads(raw)
    if blob('models_manifest.json')!=models or blob('readout_plan.json')!=expected_plan:raise ValueError('actual Git audited manifest/endpoint data differs from current complete graph')
    calibration=blob('calibration.json')
    if any(calibration[k]!=models[k] for k in ['thresholds','legacy_thresholds','inner_records']):raise ValueError('actual calibration data/model manifest threshold or inner rows differ')
    return True
