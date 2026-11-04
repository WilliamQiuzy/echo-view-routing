"""A60 role-pure complete-action supervision and new confirmation construction."""
import numpy as np
from echo_routing.temporal.anchor_inputs import native_group
from echo_routing.compose.recipe_builder import Pool,make_multi_recipes,recipe_to_dict
from echo_routing.temporal.anchor_edits import counts
from echo_routing.temporal.proposal_actions import propose,partition,apply
from echo_routing.temporal.proposal_features import candidate_features,targets,fit_target_scales
from echo_routing.temporal.proposal_training import draw_plan


def construct_confirmation(natives,count,half):
    native_group(natives)
    if half not in ['selection','calibration'] or int(count)!=count or count<1:raise ValueError('fixed new confirmation half required')
    seed=2026110360;domain=2+['selection','calibration'].index(half)
    by_id={q['id']:q for q in natives}
    pool=Pool.from_index([q['id'] for q in natives],[int(q['labels'][0]) for q in natives],[len(q['refs']) for q in natives])
    recipes=make_multi_recipes(pool,count,np.random.default_rng([seed,60,domain]),n_fragments=(4,8),edit_variant='gamma090',edit_prob=.5,min_len=4,max_len=80,reuse=True,p_same_next=.35)
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
        sequences.append({'id':f'a60_confirmation_{half}_{index:04d}','kind':'multi','refs':refs,'labels':np.asarray(truth,np.int64),'joins':np.asarray(joins,np.int64)})
    return sequences,[recipe_to_dict(r) for r in recipes],{'RNG_domain':[seed,60,domain],'streams':len(sequences),'requested_fragment_support':[4,80],'realized_fragment_minmax':[min(lengths),max(lengths)],'realized_fragment_quantiles':np.quantile(lengths,[.1,.5,.9]).tolist(),'same_source_repeated_within_stream':0,'generation_retries_or_metric_selection':0}


def prepare_training(sequences,a08,jepa,baselines):
    if not len(sequences)==len(a08)==len(jepa)==len(baselines):raise ValueError('whole matched FIT axes required')
    pooled=np.asarray([counts(b,q['labels']) for q,b in zip(sequences,baselines) if q['kind']=='multi'],np.int64)
    if not len(pooled):raise ValueError('unchanged FIT multi bank required')
    m,p,t=pooled.sum(0).tolist();f=2*m/(p+t)
    features={'A':[],'J':[]};ys=[];rows=[];kinds={'native':[],'constructed':[]};all_count=0;rejected_count=0
    for qi,(q,a,j,b) in enumerate(zip(sequences,a08,jepa,baselines)):
        candidates=propose(b,np.asarray(a).argmax(1),np.asarray(j).argmax(1));eligible,rejected=partition(b,candidates)
        all_count+=len(candidates);rejected_count+=len(rejected)
        for action in eligible:
            kind='native' if q['kind']=='native' else 'constructed';kinds[kind].append(len(rows))
            for arm in ['A','J']:features[arm].append(candidate_features(a,j,b,action,arm))
            ys.append(targets(b,action,q['labels'],f));before=counts(b,q['labels']);after=counts(apply(b,action),q['labels'])
            rows.append({'sequence_index':qi,'span':(action.start,action.end),'replacement':action.replacement,'old':action.old,'jepa':action.jepa,'kind':kind,'delta_M':after[0]-before[0],'delta_P':after[1]-before[1],'feature_positions':min(len(b),action.end+16)-max(0,action.start-16)})
    return {'F_fit':f,'FIT_operating_counts':{'M':m,'P':p,'T':t},'features':{k:np.asarray(v,np.float64).reshape(-1,109) for k,v in features.items()},'targets':np.asarray(ys,np.float64).reshape(-1,3),'candidate_rows':rows,'kind_indices':{k:np.asarray(v,np.int64) for k,v in kinds.items()},'all_candidates':all_count,'G_rejected_candidates':rejected_count,'native_target_is_count_stability_proxy_not_macroF1_surrogate':True}


def validate(data,sequences,a08,jepa,baselines,normalizers,scales,plans):
    normalized={}
    for key,value in plans.items():
        if isinstance(key,bool) or not isinstance(key,(int,np.integer,str)):raise ValueError('canonical seed keys required')
        seed=int(key)
        if isinstance(key,str) and key!=str(seed) or seed in normalized:raise ValueError('duplicate/noncanonical seed alias')
        normalized[seed]=value
    if set(normalized)!={2026110360,2026110361}:raise ValueError('exact both declared seeds required')
    expected=prepare_training(sequences,a08,jepa,baselines)
    for name in ['F_fit','FIT_operating_counts','candidate_rows','all_candidates','G_rejected_candidates']:
        if data[name]!=expected[name]:raise ValueError('complete FIT candidate/target identity changed')
    if not np.array_equal(data['targets'],expected['targets']) or not np.array_equal(scales,fit_target_scales(expected['targets'])):raise ValueError('full signed targets/shared uncentered RMS changed')
    for arm in ['A','J']:
        a=data['features'][arm];e=expected['features'][arm]
        if not np.array_equal(a,e) or not np.array_equal(a.mean(0),normalizers[arm]['mean']) or not np.array_equal(np.maximum(a.std(0),1e-6),normalizers[arm]['std']):raise ValueError('all FIT features/normalizer changed')
    if not np.array_equal(data['features']['A'][:,:69],data['features']['J'][:,:69]) or not np.array_equal(data['features']['A'][:,29:69],data['features']['A'][:,69:]):raise ValueError('common69/A duplicate mismatch')
    for k in ['native','constructed']:
        if not np.array_equal(data['kind_indices'][k],expected['kind_indices'][k]):raise ValueError('FIT kind indices changed')
    for seed,p in normalized.items():
        e=draw_plan(*(len(expected['kind_indices'][k]) for k in ['native','constructed']),seed)
        if p['no_action_domain']!=e['no_action_domain'] or any(not np.array_equal(p[k],e[k]) for k in ['native','constructed']):raise ValueError('fixed shared full plans changed')
    return {'passed':True,'candidate_count':len(data['targets']),'all_targets_features_normalizers_RMS_and_two_seed_plans_recounted':True,'F_fit':data['F_fit']}
