"""A56 source-only five-class replay contracts; no target evaluator."""
import numpy as np
import torch.nn.functional as F

SEED=2026110256


def native_allowlist(sequences, allowed):
    result=[]
    for q in sequences:
        if q['kind']!='native':
            continue
        ident=q['id']
        if ident not in allowed or any(r[0]!=ident or r[0] not in allowed for r in q['refs']):
            raise ValueError('native recording/reference outside FIT allowlist')
        if len(q['labels'])!=len(q['refs']) or not np.all(np.asarray(q['labels'])==allowed[ident]):
            raise ValueError('native label differs from protocol FIT label')
        result.append(q)
    if len({q['id'] for q in result})!=len(result):
        raise ValueError('duplicate native allowlist recording')
    return result


def exposure_plans(source_labels, natives, updates=1200):
    y=np.asarray(source_labels);rng=np.random.default_rng(SEED);streams=[]
    for c in (0,1):
        pool=np.flatnonzero(y==c)
        if not len(pool) or len(pool)>4*updates:raise ValueError('source exposure cannot cover FIT cohort')
        values=[]
        while len(values)<4*updates:values.extend(rng.permutation(pool).tolist())
        streams.append(np.asarray(values[:4*updates],np.int64).reshape(updates,4))
    source=np.concatenate(streams,axis=1)
    rng=np.random.default_rng(np.random.SeedSequence([SEED,56]));ry=np.array([q['labels'][0] for q in natives]);replay=np.empty(updates*8,np.int64)
    for c in range(5):
        slots=np.flatnonzero(np.arange(updates*8)%5==c);pool=np.flatnonzero(ry==c)
        if not len(pool) or len(pool)>len(slots):raise ValueError('replay exposure cannot cover native allowlist')
        values=[]
        while len(values)<len(slots):values.extend(rng.permutation(pool).tolist())
        replay[slots]=values[:len(slots)]
    return source,replay.reshape(updates,8)


def accumulate(forward,source_labels,replay_labels,coefficient):
    if coefficient not in (0.,1.) or len(source_labels)!=8 or len(replay_labels)!=8:
        raise ValueError('A56 fixed8+8 exposure/coefficient required')
    totals=[]
    for offset,labels,weight in ((0,source_labels,1.),(8,replay_labels,float(coefficient))):
        total=0.
        for k,label in enumerate(labels):
            logits=forward(offset+k)
            loss=F.cross_entropy(logits,logits.new_tensor([int(label)],dtype=__import__('torch').long))/8
            (weight*loss).backward();total+=float(loss.detach())
        totals.append(total)
    return {'source_loss':totals[0],'replay_loss':totals[1],'loss':totals[0]+coefficient*totals[1],
            'physical_forwards':16,'physical_backwards':16}
