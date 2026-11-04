"""A61 independent plan/exposure/optimizer/weight audits; no model forward."""
import json,math,hashlib
from pathlib import Path
import numpy as np,torch
from echo_routing.temporal.safety_training import Utility,Risk,optimizer,restore,SEEDS,ARMS,lr_factor
from echo_routing.temporal.anchor_cache import sha


def validate_trace(rows,data,plan,role,arm,expected_updates=6000,identity=None):
    if len(rows)!=expected_updates or not 0<expected_updates<=6000:raise ValueError('complete physical update prefix required')
    K=np.diff(data['position_offsets']);totals={'updates':len(rows),'query_group_exposures':0,'candidate_exposures':0,'changed_position_exposures':0,'unequal_pair_exposures':0};live={}
    pair_counts=[]
    if role=='utility':
        col=0 if arm.startswith('U') else 1
        for group in data['groups']:
            t=np.r_[data['targets'][group,col],0.];pair_counts.append(int(np.count_nonzero(np.triu(t[:,None]!=t,1))))
    for u,row in enumerate(rows):
        if row['u']!=u or identity is not None and row['identity']!=identity or not row['finite_gradients'] or not math.isfinite(row['loss']) or not math.isfinite(row['gradient_norm']) or not math.isclose(row['lr'],.001*lr_factor(u),rel_tol=1e-12,abs_tol=1e-15):raise ValueError('physical cursor/identity/finite/schedule mismatch')
        candidates=[];queries=[]
        for block,k in enumerate(plan['source_kinds']):
            kind=['native','constructed'][k];draw=plan['indices'][u,block*32:(block+1)*32]
            pool=data['kind_groups' if role=='utility' else 'kind_indices'][kind]
            if role=='utility':
                qs=pool[draw].tolist();queries.extend(qs)
                for q in qs:candidates.extend(data['groups'][q].tolist())
            else:candidates.extend(pool[draw].tolist())
        positions=int(K[candidates].sum());pairs=sum(pair_counts[q] for q in queries) if role=='utility' and arm.endswith('L') else 0
        if row['candidate_indices']!=candidates or row['query_sequence_indices']!=queries or row['source_kind_blocks']!=plan['source_kinds'] or row['physical_candidates']!=len(candidates) or row['changed_position_exposures']!=positions or row.get('physical_groups',0)!=len(queries) or row.get('physical_unequal_pairs',0)!=pairs or role=='risk' and row['physical_changed_positions']!=positions:raise ValueError('full query/candidate/changed-position/pair exposure differs from exact frozen draws')
        totals['query_group_exposures']+=len(queries);totals['candidate_exposures']+=len(candidates);totals['changed_position_exposures']+=positions;totals['unequal_pair_exposures']+=pairs
        for name,flag in row['nonzero_gradient_tensors'].items():live[name]=live.get(name,False) or flag
    return {**totals,'all_tensor_gradient_liveness_over_history':live}


def tensor_equal(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
    if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and np.array_equal(a,b)
    if isinstance(a,dict):return isinstance(b,dict) and set(a)==set(b) and all(tensor_equal(v,b[k]) for k,v in a.items())
    if isinstance(a,(list,tuple)):return type(a)==type(b) and len(a)==len(b) and all(tensor_equal(x,y) for x,y in zip(a,b))
    return a==b


def initial_digest(model):
    h=hashlib.sha256()
    for name,t in model.state_dict().items():h.update(name.encode());h.update(t.detach().numpy().tobytes())
    return h.hexdigest()


def verify_models(records,data,plans,root):
    expected={(seed,arm) for seed in SEEDS for arm in ['risk',*ARMS]}
    if len(records)!=10 or {(r['seed'],r['arm']) for r in records}!=expected:raise ValueError('exact2risk+8utility declaredmodels required')
    audited=[];initials={}
    for row in records:
        seed,arm,role=row['seed'],row['arm'],row['role'];model=Risk(seed) if arm=='risk' else Utility(arm,seed);initial={k:v.clone() for k,v in model.state_dict().items()};fresh=initial_digest(model)
        if role!=model.role or fresh!=row['initial_state_sha256']:raise ValueError('original common initializer or declared role changed')
        if role=='utility':
            initials.setdefault(seed,fresh)
            if initials[seed]!=fresh:raise ValueError('withinseed utility initializer mismatch')
        path=Path(root)/row['path'];folder=path.parent
        if sha(path)!=row['sha256']:raise ValueError('final checkpoint bytes changed')
        trace=[json.loads(s) for s in (folder/'physical_updates.jsonl').read_text().splitlines()]
        names={n for n,_ in model.named_parameters()}
        if any(set(r['nonzero_gradient_tensors'])!=names for r in trace):raise ValueError('gradient tensor coverage changed')
        exposure=validate_trace(trace,data,plans[str(seed)][role],role,arm,6000,row['identity'])
        if not all(exposure['all_tensor_gradient_liveness_over_history'].values()):raise ValueError('active tensor never received nonzero gradient')
        for r in trace:
            if r['physical_candidates']<=0:raise ValueError('fake no-domain update')
        states=[]
        for cursor in [2000,4000,6000]:
            p=folder/f'step{cursor}.pt';value=torch.load(p,map_location='cpu',weights_only=True);opt=optimizer(model)
            if restore(model,opt,value,row['identity'],data['normalizers'],data['scale'])!=cursor:raise ValueError('fullmilestone actual optimizer cursor changed')
            if any(not bool(opt.state[p]['exp_avg_sq'].count_nonzero()) for p in model.parameters()):raise ValueError('active tensor lacks nonzero Adam second moment')
            states.append({'step':cursor,'path':str(p.relative_to(root)),'sha256':sha(p),'fullstate_includes_Adam_RNG_norms_scale':True})
        latest=folder/'latest.pt';value=torch.load(latest,map_location='cpu',weights_only=True);opt=optimizer(model)
        if restore(model,opt,value,row['identity'],data['normalizers'],data['scale'])!=6000 or any(not torch.equal(v,value['model'][k]) for k,v in model.state_dict().items()):raise ValueError('latest differs from final cursor')
        final=torch.load(path,weights_only=True,map_location='cpu')
        if any(not tensor_equal(value[k],final[k]) for k in ['optimizer','rng','normalizers','utility_scale','identity','update']):raise ValueError('latest/final complete optimizer/RNG/input states differ')
        if any(not torch.equal(v,final['model'][k]) for k,v in model.state_dict().items()) or any(torch.equal(initial[k],v) for k,v in model.state_dict().items()):raise ValueError('latest/final tensors differ or active tensor never drifted')
        if row['result']['committed_physical_updates']!=6000 or row['result']['actual_Adam_update']!=6000:raise ValueError('reported optimizer counters differ from physical states')
        audited.append({**row,'milestones':states,'latest_sha256':sha(latest),'physical_trace_sha256':sha(folder/'physical_updates.jsonl'),'audited_exposure':exposure,'parameters':sum(p.numel() for p in model.parameters()),'no_model_forward_during_audit':True})
    return audited
