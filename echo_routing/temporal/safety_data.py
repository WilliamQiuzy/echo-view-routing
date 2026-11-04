"""A61 role-pure complete-query/candidate/position training arrays and plans."""
import numpy as np
from echo_routing.temporal.anchor_edits import counts
from echo_routing.temporal.proposal_actions import propose,partition,apply
from echo_routing.temporal.proposal_features import fit_normalizer,normalize
from echo_routing.temporal.safety_targets import effects,position_labels
from echo_routing.temporal.safety_features import utility_features,position_features,fit_position_normalizer,normalize_positions
from echo_routing.temporal.safety_training import draw_plan,SEEDS


def prepare_training(sequences,a08,jepa,baselines):
    if not len(sequences)==len(a08)==len(jepa)==len(baselines):raise ValueError('complete matched FIT axes required')
    pooled=np.asarray([counts(b,q['labels']) for q,b in zip(sequences,baselines) if q['kind']=='multi'],np.int64)
    if not len(pooled):raise ValueError('complete FIT multi bank required for unchanged operating point')
    m,p,t=map(int,pooled.sum(0));f=2*m/(p+t)
    features=[];positions=[];labels=[];targets=[];rows=[];groups=[];offsets=[0];costs=[]
    kinds={k:[] for k in ['native','constructed']};kg={k:[] for k in kinds};all_count=0;rejected_count=0
    for qi,(q,a,j,b) in enumerate(zip(sequences,a08,jepa,baselines)):
        actions=propose(b,np.asarray(a).argmax(1),np.asarray(j).argmax(1));eligible,rejected=partition(b,actions)
        all_count+=len(actions);rejected_count+=len(rejected);group=[];kind='native' if q['kind']=='native' else 'constructed'
        for action in eligible:
            index=len(rows);group.append(index);kinds[kind].append(index)
            x=utility_features(a,j,b,action);px=position_features(a,j,b,action);py=position_labels(b,action,q['labels']);cost=effects(b,action,q['labels'],f)
            if len(px)!=len(py):raise ValueError('changed-position feature/label axis differs')
            features.append(x);positions.append(px);labels.append(py);targets.append([cost['u'],cost['v']]);costs.append(cost);offsets.append(offsets[-1]+len(px))
            rows.append({'sequence_index':qi,'start':action.start,'end':action.end,'replacement':action.replacement,'old':action.old,'jepa':action.jepa,'kind':kind,'changed_positions':len(px),'T':len(b)})
        groups.append(np.asarray(group,np.int64))
        if group:kg[kind].append(qi)
    if not features:raise ValueError('complete G domain empty: no pretend optimizer steps')
    features=np.asarray(features,np.float64);targets=np.asarray(targets,np.float64);pos=np.concatenate(positions)
    scale=max(float(np.sqrt(np.mean(targets**2))),1.)
    # Pair target ties/signs use exact stored values. Assert common scaling preserves them.
    for group in groups:
        for col in [0,1]:
            raw=np.r_[targets[group,col],0.];scaled=raw/scale
            if not np.array_equal(np.sign(raw[:,None]-raw),np.sign(scaled[:,None]-scaled)):raise ValueError('utility scaling alters exact target pair ties/order')
    return {'F_fit':f,'FIT_operating_counts':{'M':m,'P':p,'T':t},'features':features,'position_features':pos,'position_labels':np.concatenate(labels),'position_groups':positions,'position_offsets':np.asarray(offsets,np.int64),'targets':targets,'effects':costs,'candidate_rows':rows,'groups':groups,'kind_indices':{k:np.asarray(v,np.int64) for k,v in kinds.items()},'kind_groups':{k:np.asarray(v,np.int64) for k,v in kg.items()},'normalizers':{'candidate':fit_normalizer(features),'position':fit_position_normalizer(pos)},'scale':scale,'all_candidates':all_count,'G_rejected_candidates':rejected_count}


def plans(data):
    return {str(seed):{role:draw_plan(*(len(data['kind_groups' if role=='utility' else 'kind_indices'][k]) for k in ['native','constructed']),seed,role) for role in ['utility','risk']} for seed in SEEDS}


def exact(left,right):
    if isinstance(left,np.ndarray):return isinstance(right,np.ndarray) and left.dtype==right.dtype and np.array_equal(left,right)
    if isinstance(left,dict):return isinstance(right,dict) and set(left)==set(right) and all(exact(v,right[k]) for k,v in left.items())
    if isinstance(left,(tuple,list)):return type(left)==type(right) and len(left)==len(right) and all(exact(a,b) for a,b in zip(left,right))
    return left==right


def validate(data,sequences,a,j,b,frozen_plans):
    if set(frozen_plans)!={str(s) for s in SEEDS}:raise ValueError('exact canonical two seed keys required')
    expected=prepare_training(sequences,a,j,b)
    if not exact(data,expected) or not exact(frozen_plans,plans(expected)):raise ValueError('complete FIT targets/groups/positions/normalizers/scales/plans changed')
    return {'passed':True,'candidates':len(data['targets']),'changed_position_multiset':len(data['position_labels']),'all_targets_features_groups_normalizers_scales_and_both_role_plans_recounted':True}


def tensor_data(data):
    import torch
    return {'x':torch.from_numpy(normalize(data['features'],data['normalizers']['candidate'])),
            'positions':torch.from_numpy(normalize_positions(data['position_features'],data['normalizers']['position'])),
            'labels':torch.from_numpy(data['position_labels']),'targets':torch.from_numpy(data['targets']/data['scale']),
            'fp':torch.tensor([c['dFP']>0 for c in data['effects']],dtype=torch.float32)}


def batch(data,tensors,plan,update,role,arm=None):
    indices=[];queries=[]
    for block,k in enumerate(plan['source_kinds']):
        kind=['native','constructed'][k];picked=plan['indices'][update,block*32:(block+1)*32]
        pool=data['kind_groups' if role=='utility' else 'kind_indices'][kind]
        if role=='utility':
            for local in picked:
                q=int(pool[local]);queries.append(q);indices.extend(data['groups'][q].tolist())
        else:indices.extend(pool[picked].tolist())
    idx=np.asarray(indices,np.int64)
    if role=='utility':
        out={'x':tensors['x'][idx],'target':tensors['targets'][idx,0 if arm.startswith('U') else 1],'lengths':[len(data['groups'][q]) for q in queries]}
    else:
        positions=np.concatenate([np.arange(data['position_offsets'][i],data['position_offsets'][i+1]) for i in idx]);out={'x':tensors['x'][idx],'positions':tensors['positions'][positions],'labels':tensors['labels'][positions],'fp':tensors['fp'][idx],'lengths':np.diff(data['position_offsets'])[idx].tolist()}
    return out,{'candidate_indices':indices,'query_sequence_indices':queries,'changed_position_exposures':int(np.diff(data['position_offsets'])[idx].sum()),'source_kind_blocks':plan['source_kinds']}


from echo_routing.temporal.anchor_inputs import native_group
from echo_routing.compose.recipe_builder import Pool,make_multi_recipes,recipe_to_dict

def construct_confirmation(natives,count,half):
    native_group(natives)
    if half not in ['selection','calibration'] or int(count)!=count or count<1:raise ValueError('fixed new confirmation half required')
    seed=2026110461;domain=2+['selection','calibration'].index(half)
    by_id={q['id']:q for q in natives}
    pool=Pool.from_index([q['id'] for q in natives],[int(q['labels'][0]) for q in natives],[len(q['refs']) for q in natives])
    recipes=make_multi_recipes(pool,count,np.random.default_rng([seed,61,domain]),n_fragments=(4,8),edit_variant='gamma090',edit_prob=.5,min_len=4,max_len=80,reuse=True,p_same_next=.35)
    if len(recipes)!=count:raise ValueError('new bank cannot retry or select recipes')
    sequences=[];lengths=[]
    for index,recipe in enumerate(recipes):
        refs=[];truth=[];joins=[];seen=set()
        for fragment in recipe.fragments:
            if fragment.video_id in seen:raise ValueError('source repeated in one construction')
            seen.add(fragment.video_id)
            if refs:joins.append(len(refs))
            q=by_id[fragment.video_id];gamma=.9 if fragment.variant=='gamma090' else 1.
            refs.extend((v,int(f),gamma) for v,f,_ in q['refs'][fragment.start:fragment.end])
            n=fragment.end-fragment.start;truth.extend([fragment.label]*n);lengths.append(n)
        sequences.append({'id':f'a61_confirmation_{half}_{index:04d}','kind':'multi','refs':refs,'labels':np.asarray(truth,np.int64),'joins':np.asarray(joins,np.int64)})
    return sequences,[recipe_to_dict(r) for r in recipes],{'RNG_domain':[seed,61,domain],'streams':len(sequences),'requested_fragment_support':[4,80],'realized_fragment_minmax':[min(lengths),max(lengths)],'realized_fragment_quantiles':np.quantile(lengths,[.1,.5,.9]).tolist(),'same_source_repeated_within_stream':0,'generation_retries_or_metric_selection':0}

