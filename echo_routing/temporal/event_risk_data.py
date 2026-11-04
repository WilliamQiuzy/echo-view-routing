"""A62 fixed source-separated CAL/AUDIT and unchanged FIT candidate exposure."""
import hashlib
import numpy as np
from echo_routing.temporal.anchor_inputs import native_group
from echo_routing.evaluate.failure_units import session_of
from echo_routing.compose.recipe_builder import Pool,make_multi_recipes,recipe_to_dict
from echo_routing.temporal.safety_training import draw_plan,seed_value

ROLES=('CAL_mixed','AUDIT_mixed','a62_confirmation_selection','a62_confirmation_calibration')


def partition_inner(natives):
    native_group(natives)
    sessions={session_of(q['id']) for q in natives}
    if len(natives)!=628 or len(sessions)!=100:raise ValueError('exact original628 native/100session inner source pool required')
    ordered=sorted(sessions,key=lambda s:(hashlib.sha256(b'A62-inner-v1\0'+s.encode()).digest(),s))
    cal=set(ordered[:50]);audit=set(ordered[50:]);groups=[[q for q in natives if session_of(q['id']) in ss] for ss in [cal,audit]]
    if cal&audit or any(np.bincount([int(q['labels'][0]) for q in group],minlength=5).min()==0 for group in groups):raise ValueError('fixed partition lacks classes; fail without redraw')
    return *groups,{'sessions':[50,50],'native_counts':[len(g) for g in groups],'class_counts':[np.bincount([int(q['labels'][0]) for q in g],minlength=5).tolist() for g in groups],'source_overlap':0,'session_overlap':0,'partition_rule':'SHA256(A62-inner-v1\\0 + session), lexical tie','original_inner_mixed_used':0,'retries':0}


def construct(natives,role):
    native_group(natives)
    if role not in ROLES:raise ValueError('exact four prospective A62 role names required')
    domain=ROLES.index(role);count=150 if domain<2 else 120;lookup={q['id']:q for q in natives}
    pool=Pool.from_index([q['id'] for q in natives],[int(q['labels'][0]) for q in natives],[len(q['refs']) for q in natives])
    recipes=make_multi_recipes(pool,count,np.random.default_rng([2026110463,62,domain]),n_fragments=(4,8),min_len=4,max_len=80,edit_variant='gamma090',edit_prob=.5,reuse=True,p_same_next=.35)
    if len(recipes)!=count:raise ValueError('fixed recipes failed; no retry/selection')
    sequences=[];lengths=[]
    for i,recipe in enumerate(recipes):
        refs=[];truth=[];joins=[];seen=set()
        if not 4<=len(recipe.fragments)<=8:raise ValueError('declared fragment count changed')
        for f in recipe.fragments:
            if f.video_id in seen or f.video_id not in lookup:raise ValueError('source duplicate or outside declared native group')
            seen.add(f.video_id);q=lookup[f.video_id]
            if refs:joins.append(len(refs))
            gamma=.9 if f.variant=='gamma090' else 1.;n=f.end-f.start
            refs.extend((v,int(frame),gamma) for v,frame,_ in q['refs'][f.start:f.end]);truth.extend([f.label]*n);lengths.append(n)
        sequences.append({'id':f'a62_{role}_{i:04d}','kind':'multi','refs':refs,'labels':np.asarray(truth,np.int64),'joins':np.asarray(joins,np.int64)})
    return sequences,[recipe_to_dict(r) for r in recipes],{'RNG_domain':[2026110463,62,domain],'streams':count,'requested_fragment_support':[4,80],'realized_fragment_minmax':[min(lengths),max(lengths)],'same_source_repeated_within_stream':0,'source_outside_group':0,'generation_retries_or_metric_selection':0}


def unsafe_targets(costs):
    if not costs:raise ValueError('complete nonempty FIT action domain required')
    for c in costs:
        if any(isinstance(c[k],bool) or not isinstance(c[k],(int,np.integer)) for k in ['H','R','dFP']) or min(c['H'],c['R'])<0:raise ValueError('exact nonnegative gross counts and signed integer shortFP delta required')
    return np.asarray([c['H']>0 or c['dFP']>0 for c in costs],np.float32)


def validate_draw_plan(plan,native,constructed,seed):
    expected=draw_plan(native,constructed,seed_value(seed),'risk')
    if set(plan)!=set(expected) or any(not np.array_equal(plan[k],v) if isinstance(v,np.ndarray) else plan[k]!=v for k,v in expected.items()):raise ValueError('unchanged A61 candidate-risk draw required, not a new RNG domain')
    return plan


def candidate_indices(plan,update,native,constructed):
    if isinstance(update,bool) or int(update)!=update or not 0<=update<6000:raise ValueError('canonical optimizer cursor required')
    pools=[np.asarray(native,np.int64),np.asarray(constructed,np.int64)];indices=[]
    if plan['role']!='risk' or np.asarray(plan['indices']).shape!=(6000,64):raise ValueError('complete old candidate plan required')
    for block,kind in enumerate(plan['source_kinds']):indices.extend(pools[kind][plan['indices'][update,32*block:32*(block+1)]].tolist())
    return np.asarray(indices,np.int64)
