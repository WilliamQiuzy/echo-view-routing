"""A63 private FIT-only input alignment and readonly complete teacher loading.

No image/source forward, cache fallback or CAL/AUDIT/outer access is exposed.
The caller may use labels only for authorized complement targets/effects.
"""
import numpy as np

def original_native_indices(original,requested):
    index={q['id']:i for i,q in enumerate(original) if q['kind']=='native'}
    if len(index)!=sum(q['kind']=='native' for q in original) or len({q['id'] for q in requested})!=len(requested):raise ValueError('unique immutable native axes required')
    out=[]
    for q in requested:
        if q['kind']!='native' or q['id'] not in index:raise ValueError('requested native not in original FIT bank')
        i=index[q['id']];r=original[i]
        if q['refs']!=r['refs'] or not np.array_equal(q['labels'],r['labels']):raise ValueError('native source refs/truth cannot drift with role mapping')
        out.append(i)
    return out

def minimal_aux_data(data):
    # The normalized tensor path and independent audit consume the single
    # concatenated matrix plus exact offsets. Redundant per-action slices carry
    # no additional evidence and are deliberately not serialized a second time.
    if 'position_groups' not in data or sum(len(v) for v in data['position_groups'])!=len(data['position_features']):raise ValueError('complete original position concatenation required')
    return {k:v for k,v in data.items() if k!='position_groups'}

def load_bank(role):
    from scripts.selected_risk_common import ROOT,PUB,CFG,read,sequences,ProbabilityCache,sha
    from echo_routing.temporal.baselines.registry import decode_with_prob
    if role not in ['FIT_native','F0_native','F1_native','F0_mixed','F1_mixed']:raise ValueError('FIT-only auxiliary input reader')
    m=read(PUB/'input_manifest.json');qs=sequences(role);native=role.endswith('native')
    if native:
        original=sequences('fit');indices=original_native_indices(original,qs);source=m['source_cache_roles']['fit'];lengths=[len(q['refs']) for q in original]
    else:
        producer=read(PUB/'producer_FIT_manifest.json')
        if producer['complete'] is not True or producer['phase']!='FIT' or producer['aux_updates_at_production']!=0 or producer['formal_updates_at_production']!=0 or producer['input_manifest_sha256']!=sha(PUB/'input_manifest.json'):raise ValueError('exact complete pre-aux FIT producer required')
        rows=producer['records']
        if len(rows)!=4 or {(r['source'],r['role']) for r in rows}!={(s,k) for s in ['a08','jepa'] for k in ['F0_mixed','F1_mixed']}:raise ValueError('all four genuine FIT producer records required')
        source={r['source']:r for r in rows if r['role']==role};lengths=[len(q['refs']) for q in qs];indices=list(range(len(qs)))
    probs={}
    for name in ['a08','jepa']:
        r=source[name];target=ProbabilityCache(ROOT/r['path'],r['identity'],lengths,readonly=True)
        input_role='fit' if native else role
        if r['input_sha256']!=m['roles'][input_role]['sha256'] or sha(target.journal)!=r['journal_sha256'] or target.cursor!=len(lengths):raise ValueError('complete eager readonly payload/journal/input binding required')
        probs[name]=[target.load(i) for i in indices]
    params=read(CFG)['source']['a08']['original_B3'];bases=[decode_with_prob('b3',p,{'decode':{'b3':params}})[0] for p in probs['a08']]
    return qs,probs['a08'],probs['jepa'],bases
