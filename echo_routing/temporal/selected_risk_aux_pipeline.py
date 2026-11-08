"""Independent auxiliary identities, full draw accounting and phase guards.

No training, weights, sources or filesystem writes. Predraw byte counts retain
every actual selected candidate/query index; they never substitute a mean C/K.
"""
import hashlib
import json
from pathlib import Path
import numpy as np

KEYS=('code','config','sources','input','draw_plan','normalizer','target_scales')
SEEDS=(2026110461,2026110462)

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def expected_identity(parts,seed,held_fold,role):
    if set(parts)!=set(KEYS) or any(not isinstance(v,str) or len(v)!=64 or any(c not in '0123456789abcdef' for c in v) for v in parts.values()):raise ValueError('independently verified seven SHA inputs required')
    if isinstance(seed,bool) or seed not in SEEDS or type(held_fold) is not int or held_fold not in [0,1] or role not in ['utility','risk']:raise ValueError('exact declared auxiliary model identity required')
    return {k:digest({'input_SHA':v,'seed':seed,'held_fold':held_fold,'train_fold':1-held_fold,'role':role,'steps':6000}) for k,v in parts.items()}

def guard_head_start(folder):
    folder=Path(folder)
    if folder.is_symlink() or any((folder/n).exists() or (folder/n).is_symlink() for n in ['physical_updates.jsonl.gz','physical_updates.jsonl.archive.json','completion.json','pending.json']):raise ValueError('archived/completed/pending head cannot reenter original update journal')
    return True

def validate_measured_shapes(value):
    if set(value)!={'native','mixed'}:raise ValueError('native/mixed measured shape strata required')
    caps={'native':{'max_C':9,'max_candidate_K':86,'max_T':1081},'mixed':{'max_C':16,'max_candidate_K':383,'max_T':383}}
    for kind,bounds in caps.items():
        for key,upper in bounds.items():
            v=value[kind][key]
            if isinstance(v,(bool,np.bool_)) or not isinstance(v,(int,np.integer)) or not 0<=v<=upper:raise ValueError('actual teacher shape exceeds measured full-batch domain: '+kind+'/'+key)
    return True

def _indices(value,upper):
    v=np.asarray(value)
    if v.ndim!=1 or v.dtype.kind not in 'iu' or np.any(v>np.iinfo(np.int64).max) or np.any(v<0) or np.any(v>=upper):raise ValueError('actual canonical in-bank integer indices required')
    return v.astype(np.int64)

def _fixed_row(role):
    # All finite Python float reprs fit 24 characters. Count fields use a
    # conservative signed-int64 maximum; indices themselves are counted exactly.
    names=['network.0.weight','network.0.bias','network.2.weight','network.2.bias'] if role=='utility' else [p+'.'+n for p in ['position','candidate'] for n in ['0.weight','0.bias','2.weight','2.bias']]
    row={'u':5999,'identity':{k:'a'*64 for k in KEYS},'loss':-1.7976931348623157e308,'lr':-1.7976931348623157e308,'gradient_norm':-1.7976931348623157e308,'finite_gradients':True,'nonzero_gradient_tensors':{n:False for n in names},'optimizer_update_wall_seconds':-1.7976931348623157e308,'candidate_indices':[],'query_sequence_indices':[],'changed_position_exposures':2**63-1,'source_kind_blocks':[0,1],'physical_candidates':2**63-1}
    if role=='utility':row.update(pointwise_MSE=-1.7976931348623157e308,pairwise_logistic=-1.7976931348623157e308,physical_groups=64,physical_unequal_pairs=2**63-1)
    else:row.update(position_CE=-1.7976931348623157e308,candidate_BCE=-1.7976931348623157e308,physical_changed_positions=2**63-1)
    return len((json.dumps(row,sort_keys=True)+'\n').encode())

def predraw_summary(data,plan,role):
    """Offsets may be observable metadata; full arrays bind later in data audit.

    When actual concatenated arrays are provided, their complete lengths are
    checked here as well. Python integer sums avoid sampled-exposure overflow.
    """
    if role not in ['utility','risk'] or plan.get('role')!=role or plan.get('seed') not in SEEDS or plan.get('fallback') is not False or plan.get('source_kinds')!=[0,1]:raise ValueError('original two-kind no-fallback full draw plan required')
    raw=np.asarray(plan['indices'])
    if raw.shape!=(6000,64) or raw.dtype.kind not in 'iu' or np.any(raw<0) or np.any(raw>np.iinfo(np.int64).max):raise ValueError('exact 6000 by64 physical integer draw required')
    offsets=np.asarray(data['position_offsets'])
    if offsets.ndim!=1 or offsets.dtype.kind not in 'iu' or len(offsets)<2 or np.any(offsets<0) or np.any(offsets>np.iinfo(np.int64).max) or offsets[0]!=0 or np.any(offsets[1:]<=offsets[:-1]):raise ValueError('complete ordered nonnegative int64 all-K offsets required')
    offsets=offsets.astype(np.int64);C=len(offsets)-1
    if any(k in data and len(data[k])!=int(offsets[-1]) for k in ['position_features','position_labels']):raise ValueError('full concatenated position arrays differ from final offset')
    lengths=np.diff(offsets);poolname='kind_groups' if role=='utility' else 'kind_indices';pools=data[poolname]
    if set(pools)!={'native','constructed'}:raise ValueError('exact original native/constructed pools required')
    ixp=[]
    for b,k in enumerate(['native','constructed']):
        p=_indices(pools[k],len(data['groups']) if role=='utility' else C)
        if not len(p) or len(set(p.tolist()))!=len(p) or np.any(raw[:,b*32:(b+1)*32]>=len(p)):raise ValueError('both full kind pools must be nonempty and in bank; no fallback')
        ixp.append(p)
    totalC=totalQ=totalK=variable=maximum=0
    for update in range(6000):
        picked=np.r_[ixp[0][raw[update,:32]],ixp[1][raw[update,32:]]]
        if role=='utility':
            queries=picked.tolist();groups=[_indices(data['groups'][q],C) for q in queries]
            if any(not len(g) for g in groups):raise ValueError('selected utility query may not be empty')
            indices=np.concatenate(groups)
        else:queries=[];indices=picked
        n=len(indices);totalC+=n;totalQ+=len(queries);totalK+=sum(map(int,lengths[indices]));maximum=max(maximum,n)
        # Empty [] is already present in the fixed JSON template.
        variable+=len(json.dumps(indices.tolist()).encode())-2+len(json.dumps(queries).encode())-2
    rawbound=6000*_fixed_row(role)+variable
    return {'updates':6000,'candidate_exposures':totalC,'query_exposures':totalQ,'changed_position_exposures':totalK,'max_candidates':maximum,'variable_JSON_bytes':variable,'trace_fixed_JSON_per_row_upper':_fixed_row(role),'trace_raw_byte_upper':rawbound,'all_6000_actual_full_query_or_candidate_indices_counted':True,'float_repr_bound':24}
