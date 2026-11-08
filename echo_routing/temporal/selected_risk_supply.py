"""A63 source partition, copy-invariant coverage and matched draw contracts.

Identifiers/signatures and source memberships returned here are PRIVATE. Public
coverage() omits all identifiers/signatures and publishes counts/probabilities.
No model, source forward, writing CLI, or hidden data dependency.
"""
import hashlib
import json
from collections import defaultdict
import numpy as np

PREFIX=bytes.fromhex('4136332d737570706c792d763100')
SEEDS=(2026110461,2026110462)

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode('utf8')).hexdigest()

def session_of(clip):
    if not isinstance(clip,str) or '_' not in clip:raise ValueError('source clip with session suffix required')
    return clip.rsplit('_',1)[0]

def partition_sources(natives):
    if len({q['id'] for q in natives})!=len(natives):raise ValueError('unique complete original native bank required')
    sessions=sorted({session_of(q['id']) for q in natives},key=lambda s:(hashlib.sha256(PREFIX+s.encode()).digest(),s))
    if len(sessions)!=402:raise ValueError('exact 402 original FIT sessions required; no redraw')
    folds=[sessions[:201],sessions[201:]];indices=[];classes=[]
    for fold in folds:
        ss=set(fold);ix=[i for i,q in enumerate(natives) if session_of(q['id']) in ss];cc=[0]*5
        for i in ix:
            q=natives[i];y=np.asarray(q['labels'])
            if y.ndim!=1 or not len(y) or not np.isin(y,range(5)).all() or np.any(y!=y[0]):raise ValueError('original native constant five-class reference required')
            if len(q['refs'])!=len(y) or any(session_of(v)!=session_of(q['id']) for v,*_ in q['refs']):raise ValueError('native source/session alignment required')
            cc[int(y[0])]+=1
        if min(cc)==0:raise ValueError('all five classes in each fixed fold required; no redraw')
        indices.append(ix);classes.append(cc)
    return {'sessions':folds,'native_indices':indices,'native_class_counts':classes,'prefix_bytes_hex':PREFIX.hex()}

def _arrays(refs,base,edited,truth=None):
    b=np.asarray(base);e=np.asarray(edited)
    if b.ndim!=1 or e.shape!=b.shape or len(refs)!=len(b) or not np.isin(b,range(5)).all() or not np.isin(e,range(5)).all():raise ValueError('aligned complete original-reference axis required')
    for r in refs:
        if len(r)!=3 or not isinstance(r[0],str) or isinstance(r[1],bool) or int(r[1])!=r[1] or r[1]<0:raise ValueError('source/frame/gamma reference required')
        session_of(r[0])
    if truth is None:return b,e
    t=np.asarray(truth)
    if t.shape!=b.shape or not np.isin(t,range(5)).all():raise ValueError('aligned five-class reference required')
    return b,e,t

def draw_signature(refs,base,edited):
    b,e=_arrays(refs,base,edited)
    return digest([[refs[i][0],int(refs[i][1]),int(b[i]),int(e[i])] for i in np.flatnonzero(b!=e)])

def event_signature(refs,base,edited,truth,*,new_fp_segments):
    """new_fp_segments must be actual newly-appearing unmatched <=16 segments.

    The caller must obtain them from the qualified full-axis matcher; this pure
    function validates geometry/class but does not pretend to match segments.
    H tuples sorted; FP ordered refs preserved and canonical segments sorted.
    """
    b,e,t=_arrays(refs,base,edited,truth);hi=np.flatnonzero((b==t)&(e!=t))
    harms=sorted([[refs[i][0],int(refs[i][1]),int(b[i]),int(e[i])] for i in hi])
    fps=[];sources={refs[i][0] for i in hi}
    for start,end,label in new_fp_segments:
        if not 0<=start<end<=len(b) or end-start>16 or not np.all(e[start:end]==label):raise ValueError('new unmatched short segment geometry required')
        rr=[[refs[i][0],int(refs[i][1])] for i in range(start,end)]
        sources.update(v for v,_ in rr);fps.append({'refs':rr,'predicted_class':int(label)})
    fps.sort(key=lambda v:json.dumps(v,sort_keys=True,separators=(',',':')))
    if not harms and not fps:raise ValueError('danger event requires actual H or newly unmatched short FP')
    return {'signature':digest({'H':harms,'new_FP16':fps}),
      'responsible_sessions':sorted({session_of(v) for v in sources}),
      'H':len(harms),'harm_transitions':sorted({(x[2],x[3]) for x in harms}) and [list(v) for v in sorted({(x[2],x[3]) for x in harms})],
      'new_FP16_segments':len(fps)}

def top(keys,utility,pred_h,pred_r,pfp):
    u=np.asarray(utility,dtype=np.float64);h=np.asarray(pred_h,dtype=np.float64);r=np.asarray(pred_r,dtype=np.float64);p=np.asarray(pfp,dtype=np.float64)
    if any(x.shape!=(len(keys),) or not np.isfinite(x).all() for x in (u,h,r,p)) or np.any(h<0) or np.any(r<0) or np.any((p<0)|(p>1)) or len(set(keys))!=len(keys):raise ValueError('finite unique aligned original physical score rows required')
    positive=np.flatnonzero(u>0).tolist();i=min(positive,key=lambda i:(-u[i],keys[i])) if positive else None
    return {'top_index':i,'N_admitted':bool(i is not None and h[i]<=r[i] and p[i]<.5)}

def _unsafe(r):return r['H']>0 or r['dFP']>0

def _useful(r):return not _unsafe(r) and r['short16_lost']==0 and (r['R']>0 or r['u']>0 or r['Jexclusive_short16_recovered']>0)

def _groups(rows,key):
    out=defaultdict(list)
    for r in rows:out[r[key]].append(r)
    return dict(sorted(out.items()))

def _danger_stats(groups):
    ss=set();probs=defaultdict(float)
    for rs in groups.values():
        canonical=[tuple(sorted(set(r['responsible_sessions']))) for r in rs]
        if not canonical[0] or any(v!=canonical[0] for v in canonical):raise ValueError('same event signature must have identical nonempty responsible sessions')
        if len({r['H'] for r in rs})!=1 or len({tuple(tuple(x) for x in r['harm_transitions']) for r in rs})!=1:raise ValueError('same event signature must preserve actual harm event')
        for s in canonical[0]:ss.add(s);probs[s]+=1/len(groups)
    def selected(predicate):return {k:v for k,v in groups.items() if predicate(v)}
    def sessions(g):return len({s for rs in g.values() for s in rs[0]['responsible_sessions']})
    pure=selected(lambda rs:rs[0]['H']==0 and all(r['dFP']>0 for r in rs))
    av=selected(lambda rs:any(tuple(x) in [(2,3),(3,2)] for x in rs[0]['harm_transitions']))
    pp=selected(lambda rs:[0,1] in rs[0]['harm_transitions'])
    full=selected(lambda rs:all(r['min_overlap']==1. for r in rs));partial=selected(lambda rs:all(r['min_overlap']<.5 for r in rs))
    conflicts=sum(any(r['min_overlap']==1. for r in rs) and not all(r['min_overlap']==1. for r in rs) or any(r['min_overlap']<.5 for r in rs) and not all(r['min_overlap']<.5 for r in rs) for rs in groups.values())
    return {'dangerous_signatures':len(groups),'responsible_sessions':len(ss),'H_positive_signatures':sum(rs[0]['H']>0 for rs in groups.values()),'H_ge8_signatures':sum(rs[0]['H']>=8 for rs in groups.values()),'pure_FP_signatures':len(pure),'pure_FP_sessions':sessions(pure),'A4C_A5C_signatures':len(av),'A4C_A5C_sessions':sessions(av),'PLAX_PSAX_signatures':len(pp),'PLAX_PSAX_sessions':sessions(pp),'full_run_signatures':len(full),'partial_run_signatures':len(partial),'geometry_conflicts':int(conflicts),'replacement_class_signatures':{str(c):sum(any(r['replacement_class']==c for r in rs) for rs in groups.values()) for c in range(5)}},dict(probs)

def _member(rows):
    danger=_groups([r for r in rows if _unsafe(r)],'event_signature');out,prob=_danger_stats(danger)
    safe_groups=_groups(rows,'draw_signature');safe={k:rs for k,rs in safe_groups.items() if any(_useful(r) for r in rs)}
    # Safety/effects conflicts stay in the audit; only eligible actions enter the
    # training membership. J quota is stricter: ALL original selected members.
    useful={k:[r for r in rs if _useful(r)] for k,rs in safe.items()}
    out.update({'selected_actions':len(rows),'dangerous_actions':sum(_unsafe(r) for r in rows),'safe_useful_actions':sum(_useful(r) for r in rows),'safe_useful_signatures':len(useful),'safe_J_signatures':sum(all(_useful(r) and r['Jexclusive_short16_recovered']>0 for r in rs) for rs in safe.values()),'safe_J_conflicts':sum(any(_useful(r) and r['Jexclusive_short16_recovered']>0 for r in rs) and not all(_useful(r) and r['Jexclusive_short16_recovered']>0 for r in rs) for rs in safe.values()),'same_draw_effect_conflicts':sum(len({(r['H'],r['R'],r['dFP'],r['short16_lost'],r['Jexclusive_short16_recovered']) for r in rs})>1 for rs in safe_groups.values()),'safe_replacement_class_signatures':{str(c):sum(any(r['replacement_class']==c for r in rs) for rs in useful.values()) for c in range(5)}})
    return out,prob,danger,useful

def coverage(rows,*,private_details=False):
    """Coverage only; no learned models may be constructed based on a failed gate."""
    unique={}
    for r in rows:
        if r['canonical_id'] in unique and r!=unique[r['canonical_id']]:raise ValueError('canonical action has conflicting selection/effects records')
        unique[r['canonical_id']]=r
    rows=list(unique.values());native=[r for r in rows if r['kind']=='native'];mixed=[r for r in rows if r['kind']=='multi']
    if len(native)+len(mixed)!=len(rows):raise ValueError('only FIT native/multi supply allowed')
    d=[r for r in mixed if r['D']];x=[r for r in mixed if r['X'] and not r['D']]
    D,pd,dg,ds=_member(d);X,px,xg,xs=_member(x);ug=_groups([r for r in d+x if _unsafe(r)],'event_signature');U,_=_danger_stats(ug)
    maxd=max(pd.values(),default=0.);maxx=max(px.values(),default=0.);maxmix=max((.5*pd.get(s,0)+.5*px.get(s,0) for s in set(pd)|set(px)),default=0.)
    requirements={'D_danger30':(D['dangerous_signatures'],30),'D_sessions10':(D['responsible_sessions'],10),'X_danger30':(X['dangerous_signatures'],30),'X_sessions10':(X['responsible_sessions'],10),'union60':(U['dangerous_signatures'],60),'union_sessions20':(U['responsible_sessions'],20),'Hpositive40':(U['H_positive_signatures'],40),'Hge8_15':(U['H_ge8_signatures'],15),'pureFP8':(U['pure_FP_signatures'],8),'pureFP_sessions5':(U['pure_FP_sessions'],5),'A4C_A5C15':(U['A4C_A5C_signatures'],15),'A4C_A5C_sessions6':(U['A4C_A5C_sessions'],6),'PLAX_PSAX8':(U['PLAX_PSAX_signatures'],8),'PLAX_PSAX_sessions4':(U['PLAX_PSAX_sessions'],4),'fullrun15':(U['full_run_signatures'],15),'partial15':(U['partial_run_signatures'],15),'D_safe100':(D['safe_useful_signatures'],100),'X_safe100':(X['safe_useful_signatures'],100),'D_safeJ30':(D['safe_J_signatures'],30),'X_safeJ30':(X['safe_J_signatures'],30)}
    checks={k:{'actual':a,'minimum':b,'passed':a>=b} for k,(a,b) in requirements.items()}
    for k,v in [('max_p_D',maxd),('max_p_X',maxx),('max_p_mix',maxmix)]:checks[k]={'actual':v,'maximum':.2,'passed':v<=.2+1e-15}
    out={'passed':all(v['passed'] for v in checks.values()),'checks':checks,'failed_checks':[k for k,v in checks.items() if not v['passed']],'D':D,'X_only':X,'union':U,'max_p_D':maxd,'max_p_X':maxx,'max_p_mix':maxmix,'native_D_danger_actions':sum(r['D'] and _unsafe(r) for r in native),'native_X_danger_actions':sum(r['X'] and _unsafe(r) for r in native),'source_unit':'source session, not patient; event/signature not IID'}
    if private_details:out['private_pools']=[{k:sorted(r['canonical_id'] for r in rs) for k,rs in g.items()} for g in [dg,xg,ds,xs]]
    return out

def draws(native,mixed,pools,seed,*,steps=6000):
    def indices(values):
        raw=np.asarray(values)
        if raw.ndim!=1 or raw.dtype.kind not in 'iu' or raw.dtype.kind=='b' or not len(raw) or np.any(raw<0) or np.any(raw>np.iinfo(np.int64).max) or len(set(raw.tolist()))!=len(raw):raise ValueError('unique nonnegative one-dimensional exact integer candidate indices required')
        return raw.astype(np.int64)
    native=indices(native);mixed=indices(mixed)
    if set(native.tolist())&set(mixed.tolist()):raise ValueError('native and mixed candidate banks must be disjoint')
    if len(pools)!=4:raise ValueError('four selected mixed memberships required')
    mixedset=set(mixed.tolist())
    for pool in pools:
        for members in pool.values():
            checked_members=indices(members)
            if not set(checked_members.tolist())<=mixedset:raise ValueError('every selected signature member must belong to complete mixed bank')
    if seed not in SEEDS or steps<1 or not len(native) or not len(mixed) or len(pools)!=4 or any(not p or any(not len(v) or len(set(v))!=len(v) for v in p.values()) for p in pools):raise ValueError('four nonempty signature/canonical-member pools and complete source pools required')
    rn=np.random.default_rng([seed,63,0]);rm=np.random.default_rng([seed,63,1]);n=rn.choice(native,(steps,32));m=rm.choice(mixed,(steps,32));c0=np.concatenate([n,m],axis=1);c1=c0.copy()
    for slot,pool in enumerate(pools):
        keys=sorted(pool);rs=np.random.default_rng([seed,63,2,slot,0]);rr=np.random.default_rng([seed,63,2,slot,1]);si=rs.integers(len(keys),size=(steps,4))
        for update in range(steps):
            for j in range(4):
                members=sorted(pool[keys[int(si[update,j])]]);c1[update,48+slot*4+j]=members[int(rr.integers(len(members)))]
    return {'C0':c0,'C1':c1,'RNG_domains':{'native':[seed,63,0],'mixed':[seed,63,1],'selected_signature_member':[[[seed,63,2,s,0],[seed,63,2,s,1]] for s in range(4)]},'replay_fraction':.75}
