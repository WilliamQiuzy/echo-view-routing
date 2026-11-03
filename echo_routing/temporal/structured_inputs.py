"""A58 metadata/posterior checks; no label, join or membership enters window generation."""
import numpy as np
from echo_routing.temporal.windowed import window_centres
from echo_routing.evaluate.window_target import window_frames
from echo_routing.vit.target_mvit import marker_slot

def windows(n,stride):
    if int(n)!=n or n<1 or stride not in [2,5]:raise ValueError('positive original length and declared stride2/5 required')
    centres=window_centres(n,stride);samples=np.stack([window_frames(int(c),n,16,16) for c in centres]);markers=np.array([marker_slot(s,int(c)) for s,c in zip(samples,centres)],np.int64)
    return {'centre':centres,'samples':samples,'marker':markers}

def verify_roles(roles):
    sources={}
    for role,seqs in roles.items():
        if not seqs or len({q['id'] for q in seqs})!=len(seqs):raise ValueError('nonempty unique sequence role required')
        allow={q['id'] for q in seqs if q['kind']=='native'};seen=set()
        for q in seqs:
            if not len(q['labels']) or len(q['refs'])!=len(q['labels']):raise ValueError('complete original-axis references required')
            used={r[0] for r in q['refs']}
            if not used<=allow:raise ValueError('whole media footprint crosses role allowlist')
            seen|=used
        sources[role]=seen
    for i,(role,a) in enumerate(sources.items()):
        if any(a&b for other,b in list(sources.items())[i+1:]):raise ValueError('source recording crosses roles')
    return {'cross_role_sources':0,'role_source_counts':{k:len(v) for k,v in sources.items()}}

def check_probabilities(p,length):
    p=np.asarray(p)
    if p.shape!=(length,5) or not np.isfinite(p).all() or (p<0).any() or not np.allclose(p.sum(1),1.,atol=1e-5,rtol=0):raise ValueError('complete finite normalized original-axis5class cache required')

def error_overlap(left,right,truth,mask):
    a,b,y,mask=map(np.asarray,[left,right,truth,mask])
    if a.ndim!=1 or a.shape!=b.shape or a.shape!=y.shape or a.shape!=mask.shape or mask.dtype!=bool:raise ValueError('aligned label predictions/truth/boolean group mask required')
    if any(((x<0)|(x>4)).any() for x in [a,b,y]):raise ValueError('original five-class labels only')
    x,z=(a==y)&mask,(b==y)&mask
    return {'positions':int(mask.sum()),'both_correct':int((x&z).sum()),'left_only_correct':int((x&~z).sum()),'right_only_correct':int((~x&z).sum()),'both_wrong':int((~x&~z&mask).sum()),'position_oracle_correct':int((x|z).sum())}
