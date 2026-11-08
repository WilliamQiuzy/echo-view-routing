"""A63 fixed new within-fold recipes; original constructor semantics unchanged."""
import numpy as np
from echo_routing.temporal.anchor_inputs import native_group
from echo_routing.compose.recipe_builder import Pool,make_multi_recipes,recipe_to_dict

def construct(natives,domain,*,count=None):
    native_group(natives)
    if domain not in [0,1,2,3]:raise ValueError('exact A63 fold/confirmation domain required')
    count=(1500 if domain<2 else 120) if count is None else count;lookup={q['id']:q for q in natives}
    pool=Pool.from_index([q['id'] for q in natives],[int(q['labels'][0]) for q in natives],[len(q['refs']) for q in natives])
    recipes=make_multi_recipes(pool,count,np.random.default_rng([2026110563,63,domain]),n_fragments=(4,8),min_len=4,max_len=80,edit_variant='gamma090',edit_prob=.5,reuse=True,p_same_next=.35)
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
        sequences.append({'id':f'a63_domain{domain}_{i:04d}','kind':'multi','refs':refs,'labels':np.asarray(truth,np.int64),'joins':np.asarray(joins,np.int64)})
    return sequences,[recipe_to_dict(r) for r in recipes],{'RNG_domain':[2026110563,63,domain],'streams':count,'requested_fragment_support':[4,80],'realized_fragment_minmax':[min(lengths),max(lengths)],'same_source_repeated_within_stream':0,'source_outside_group':0,'generation_retries_or_metric_selection':0}

