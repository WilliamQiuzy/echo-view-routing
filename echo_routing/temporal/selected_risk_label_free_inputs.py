"""New prospective FIT geometry/held-selector reader, no truth comparisons.

The original executed selected_risk_inputs bytes remain intact. Labels in the
already-pinned metadata serialization are parsed, but never queried here.
Targets/effects remain a separate post-selector-seal path.
"""
def native_indices(original,requested):
    index={q['id']:i for i,q in enumerate(original) if q['kind']=='native'}
    if len(index)!=sum(q['kind']=='native' for q in original) or len({q['id'] for q in requested})!=len(requested):raise ValueError('unique complete original native identities required')
    out=[]
    for q in requested:
        if q['kind']!='native' or q['id'] not in index:raise ValueError('native outside exact original FIT domain')
        i=index[q['id']]
        if q['refs']!=original[i]['refs']:raise ValueError('held observable references differ from pinned original native')
        out.append(i)
    return out

def load_bank(role,*,purpose):
    if purpose not in ['geometry','selector'] or role not in ['FIT_native','F0_native','F1_native','F0_mixed','F1_mixed']:raise ValueError('only authorized FIT observable geometry/selector reader')
    if role.endswith('mixed'):
        from echo_routing.temporal.selected_risk_inputs import load_bank as original_mixed_reader
        return original_mixed_reader(role)
    from scripts.selected_risk_common import ROOT,PUB,CFG,read,sequences,ProbabilityCache,sha
    from echo_routing.temporal.baselines.registry import decode_with_prob
    m=read(PUB/'input_manifest.json');qs=sequences(role);original=sequences('fit');indices=native_indices(original,qs)
    lengths=[len(q['refs']) for q in original];source=m['source_cache_roles']['fit'];probs={}
    for name in ['a08','jepa']:
        r=source[name];c=ProbabilityCache(ROOT/r['path'],r['identity'],lengths,readonly=True)
        if r['input_sha256']!=m['roles']['fit']['sha256'] or sha(c.journal)!=r['journal_sha256'] or c.cursor!=len(lengths):raise ValueError('complete original payload/journal/input binding required')
        probs[name]=[c.load(i) for i in indices]
    params=read(CFG)['source']['a08']['original_B3'];bases=[decode_with_prob('b3',p,{'decode':{'b3':params}})[0] for p in probs['a08']]
    return qs,probs['a08'],probs['jepa'],bases
