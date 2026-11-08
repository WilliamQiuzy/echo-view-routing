"""Real fixed-head scoring, CAL-only thresholds,34inner/238outer/140paired.

Only new C0/C1 scores are generated for old banks. New confirmation uses the
frozen old heads once in independent durable caches. Main native/pair scores
are reused with bitwise teacher/Base/C6/q checks, never re-forwarded for stress.
Private paired bundles use the reviewed no-pickle typed codec and stay on RUN.
"""
import hashlib
import io
import json
import time
import math
from pathlib import PurePosixPath
import numpy as np
from echo_routing.temporal.selected_risk_resume_contract import RUN,PUB,digest,validate_coverage_pass,_successful
from echo_routing.temporal.selected_risk_resume_S import _cpu,SEEDS,array_sha
from echo_routing.temporal.selected_risk_metrics import readout_plan,TABLES,POLICIES,BASE

SOURCE='echo_routing/temporal/selected_risk_resume_readout.py'
F_FIT=.8917446116307443

# Pure definitions below retain the frozen POST-A62 AST after renaming only.
POST_RISK_MATH_SOURCE_PIN={'path':'docs/results/2026-11-05_a62_frozen_score_diagnostic/code/diagnostic_math.py','sha256':'f70f7e6220b496974e00d800f56644adbd8667ff4f2d7709bbc4275d79d78dc2'}
_POST_TAILS=(.01,.05,.10,.25,.50,1.)
def _post_flags(H,dFP,shortlost,R,u,J16):
 unsafe=H>0 or dFP>0;safe=not unsafe;acceptsafe=safe and shortlost==0
 return {'unsafe':bool(unsafe),'event_safe':bool(safe),'acceptance_safe':bool(acceptsafe),'safe_R':bool(acceptsafe and R>0),'safe_u':bool(acceptsafe and u>0),'safe_J16':bool(acceptsafe and J16>0),'H_only':bool(H>0 and dFP<=0),'FP_only':bool(H==0 and dFP>0),'H_and_FP':bool(H>0 and dFP>0),'shortloss_only':bool(safe and shortlost>0)}

def _post_tail(q,fraction):
 if not 0<fraction<=1:raise ValueError('fixedpositivefraction required')
 if not len(q):return []
 a=np.asarray(q);value=np.sort(a)[math.ceil(len(a)*fraction)-1]
 return (a<=value).tolist()

def _post_classification(q,unsafe):
 a=np.asarray(q,np.float64);y=np.asarray(unsafe,bool)
 if a.shape!=y.shape or a.ndim!=1 or not np.isfinite(a).all() or ((a<0)|(a>1)).any():raise ValueError('aligned finite unitrange uncalibrated scores required')
 n=len(a);p=int(y.sum());negative=n-p
 if not n:return {'N':0,'unsafe':0,'prevalence':None,'ROC_AUC':None,'average_precision':None,'Brier':None,'BCE':None,'infinite_BCE_count':0}
 _,inv=np.unique(a,return_inverse=True);pos=np.bincount(inv,weights=y.astype(int));neg=np.bincount(inv,weights=(~y).astype(int));before=np.cumsum(neg)-neg
 auc=float(np.sum(pos*(before+.5*neg))/(p*negative)) if p and negative else None
 cumpos=np.cumsum(pos[::-1]);cumn=np.cumsum((pos+neg)[::-1]);ap=float(np.sum(pos[::-1]/p*cumpos/cumn)) if p else None
 loss=np.empty(n,np.float64)
 with np.errstate(divide='ignore'):
  loss[y]=-np.log(a[y]);loss[~y]=-np.log1p(-a[~y])
 inf=int(np.isinf(loss).sum());bce=None if inf else float(loss.mean())
 return {'N':n,'unsafe':p,'prevalence':p/n,'ROC_AUC':auc,'average_precision':ap,'Brier':float(np.mean((a-y)**2)),'BCE':bce,'infinite_BCE_count':inf}

def _post_distribution(x):
 a=np.asarray(x,np.float64)
 if not len(a):return {'N':0,'median':None,'IQR':None,'quantile_levels':[0,.01,.05,.25,.5,.75,.95,.99,1.],'quantiles':None}
 return {'N':len(a),'median':float(np.median(a)),'IQR':np.quantile(a,[.25,.75]).tolist(),'quantile_levels':[0,.01,.05,.25,.5,.75,.95,.99,1.],'quantiles':np.quantile(a,[0,.01,.05,.25,.5,.75,.95,.99,1.]).tolist()}

def _post_summary(rows,arm,tau):
 q=np.asarray([r[arm] for r in rows],np.float64);y=np.asarray([r['unsafe'] for r in rows],bool);out={**_post_classification(q,y),**{'q_'+k:v for k,v in _post_distribution(q).items()},'tau':tau,'strict_below_tau':int((q<tau).sum())};names=['H_only','FP_only','H_and_FP','shortloss_only','event_safe','acceptance_safe','safe_R','safe_u','safe_J16']
 for name in names:out[name]=sum(r[name] for r in rows)
 out['subgroup_q']={name:_post_distribution([r[arm] for r in rows if r[name]]) for name in ['unsafe','event_safe','acceptance_safe','safe_R','safe_u','safe_J16']}
 out['tail_rows']=[]
 for fraction in _POST_TAILS:
  keep=np.asarray(_post_tail(q,fraction),bool);n=int(keep.sum());tailrow={'nominal_fraction':fraction,'N':n,'actual_fraction':n/len(rows) if rows else None,'boundary_q':float(q[keep].max()) if n else None,'unsafe':int(y[keep].sum()),'unsafe_fraction':float(y[keep].mean()) if n else None,'unsafe_recall':int(y[keep].sum())/int(y.sum()) if y.sum() else None}
  for name in ['safe_R','safe_u','safe_J16']:
   allflag=np.asarray([r[name] for r in rows],bool);num=int((allflag&keep).sum());tailrow[name]=num;tailrow[name+'_recall']=num/int(allflag.sum()) if allflag.sum() else None
  out['tail_rows'].append(tailrow)
 out['strict_tau_counts']={name:sum(r[name] and r[arm]<tau for r in rows) for name in ['unsafe','safe_R','safe_u','safe_J16']};out['safe_useful_rejected']={name:sum(r[name] and r[arm]>=tau for r in rows) for name in ['safe_R','safe_u','safe_J16']}
 return out


def _actual_global_exposure(candidate_count,actual_rows,identity):
    """Count an independently verified observed prefix, never a planned schedule.

    Caller must bind the real journal/identity and candidate-index map. This
    pure function alone does not authenticate a physical training execution.
    A63 explicit indices are global; do not apply the old local-kind remap.
    """
    if type(candidate_count) is not int or candidate_count<0 or not isinstance(identity,dict) or not identity or len(actual_rows)>6000:raise ValueError('bound candidate domain and actual prefix required')
    counts=np.zeros(candidate_count,np.int64)
    for u,row in enumerate(actual_rows):
        ix=np.asarray(row['candidate_indices'])
        if type(row.get('u')) is not int or row['u']!=u or row.get('identity')!=identity or ix.shape!=(64,) or ix.dtype.kind not in 'iu' or np.any(ix<0) or np.any(ix>=candidate_count):raise ValueError('ordered actual global64-candidate trace prefix required')
        np.add.at(counts,ix.astype(np.int64),1)
    return {'counts':counts,'actual_updates_observed':len(actual_rows),'actual_candidate_exposures':int(counts.sum()),'full6000':len(actual_rows)==6000,'scheduled_future_exposures_included':False,'repeated_draws_are_independent_n':False}



FIT_SCORE_HEADER_UPPER=640
FIT_SCORE_RECEIPT_UPPER=512
FIT_PUBLIC_AGGREGATE_UPPER=2_000_000
FIT_CANDIDATE_UPPER=15823
FIT_NONEMPTY_AXIS_UPPER=3265


def _fit_candidate_map(rows,data):
    from echo_routing.temporal.selected_risk_supply import digest as candidate_digest
    ids=np.asarray(data['candidate_digests']);unsafe=np.asarray(data['unsafe'])
    if ids.dtype!=np.dtype('V32') or ids.ndim!=1 or unsafe.dtype!=np.float32 or unsafe.shape!=ids.shape or len(ids)>FIT_CANDIDATE_UPPER or not np.isin(unsafe,[0,1]).all():raise ValueError('bound prepared global candidate/unsafe domain required')
    offsets=[0];D=np.zeros(len(ids),np.uint8);X=np.zeros(len(ids),np.uint8);actual=[];ys=[];kind_indices={'native':[],'constructed':[]}
    for q,b,v in rows:
        C=len(v['actions']);start=offsets[-1]
        from echo_routing.temporal.safety_targets import effects
        for action in v['actions']:
            c=effects(b,action,q['labels'],F_FIT);kind_indices['native' if q['kind']=='native' else 'constructed'].append(len(actual));actual.append(bytes.fromhex(candidate_digest([q['id'],action.key])));ys.append(int(c['H']>0 or c['dFP']>0))
        for name,mask in [('D',D),('X',X)]:
            selected=v[name];i=selected['top_index']
            if type(selected['N_admitted']) is not bool:raise ValueError('original label-free selected flag required')
            if selected['N_admitted']:
                if isinstance(i,(bool,np.bool_)) or not isinstance(i,(int,np.integer)) or not 0<=i<C or start+i>=len(mask):raise ValueError('canonical selected candidate index required')
                mask[start+i]=1
        offsets.append(start+C)
    if offsets[-1]!=len(ids) or not np.array_equal(np.asarray(actual,dtype='V32'),ids) or len(set(actual))!=len(actual) or np.count_nonzero(np.diff(offsets))>FIT_NONEMPTY_AXIS_UPPER:raise ValueError('FIT global hashes/order or observed nonempty geometry exceeds admitted bound')
    if not np.array_equal(np.asarray(ys,np.float32),unsafe) or 'kind_indices' in data and (set(data['kind_indices'])!=set(kind_indices) or any(not np.array_equal(data['kind_indices'][k],v) for k,v in kind_indices.items())):raise ValueError('prepared unsafe targets/global native-mixed pools changed')
    return {'axis_offsets':np.asarray(offsets,np.int64),'D':D,'X':X}


def _fit_native_q(original,rows,oldfit,post):
    from echo_routing.temporal.selected_risk_label_free_inputs import native_indices
    c=np.asarray(post['C6']);q=np.asarray(post['q'])
    if c.dtype!=np.float64 or c.ndim!=2 or c.shape[1]!=6 or q.dtype!=np.float64 or q.shape!=(len(c),2) or not np.isfinite(c).all() or not np.isfinite(q).all() or np.any((q<0)|(q>1)):raise ValueError('exact frozen POST native C6/C2 required')
    native=[v[0] for v in rows if v[0]['kind']=='native'];ix=iter(native_indices(original,native));out=[]
    for axis,b,value in rows:
        if axis['kind']!='native':out.append(None);continue
        group=np.asarray(oldfit['groups'][next(ix)])
        if group.dtype.kind not in 'iu' or group.ndim!=1 or len(group)!=len(value['actions']) or np.any(group<0) or np.any(group>=len(c)) or not np.array_equal(c[group],value['D_C6']):raise ValueError('native source/action C6 differs from original group, never infer replacement old scores')
        out.append(q[group].copy())
    return out


def _fit_score_axis(ctx,folder,index,C,identity,expected_calls,predict):
    """One durable pending, compact typed q and bounded completion; no replay."""
    from echo_routing.temporal.selected_risk_resume_context import strict_json
    from echo_routing.temporal.selected_risk_resume_freeze import _raw
    from echo_routing.temporal.selected_risk_resume_storage import storage_bound
    import os
    if type(index) is not int or index<0 or type(C) is not int or not 0<=C<=FIT_CANDIDATE_UPPER or type(identity) is not str or len(identity)!=64 or any(c not in '0123456789abcdef' for c in identity) or expected_calls not in ([0] if C==0 else [2,4]):raise ValueError('exact bounded axis identity/count required')
    stem=folder+f'/axes/{index:05d}';score=ctx.output(stem+'.npz');done=ctx.output(stem+'.done.json');pending=ctx.output(folder+'/pending.json')
    if pending.exists() or pending.is_symlink():raise ValueError('FIT pending is ambiguous: no NN replay or automatic recovery')
    if not C:
        if any(p.exists() or p.is_symlink() for p in [score,done]):raise ValueError('empty axis must have no score artifact or completion')
        return np.empty((0,4),np.float64),{'score_pin':None,'receipt_sha256':None,'new_calls':0,'new_outputs':0,'score_bytes':0,'receipt_bytes':0,'reused':False}
    for directory in [score.parent,pending.parent]:
        if directory.exists() and any('.tmp-' in x.name or '.owned-' in x.name for x in directory.glob('.*')):raise ValueError('partial FIT atomic artifact requires explicit reconciliation')
    def checked(value):
        if not isinstance(value,dict) or set(value)!={'identity','q'} or value['identity']!=identity:raise ValueError('exact two-field FIT score identity required')
        a=value['q']
        if type(a) is not np.ndarray or a.dtype!=np.float64 or a.shape!=(C,4) or not np.isfinite(a).all() or np.any((a<0)|(a>1)):raise ValueError('complete finite float64 all-G four-head q required')
        return a
    if done.exists() or done.is_symlink():
        if done.stat().st_size>FIT_SCORE_RECEIPT_UPPER:raise ValueError('FIT completion exceeds hard bound before read')
        raw,h=_raw(ctx,str(done.relative_to(ctx.root)));r=strict_json(raw)
        if len(raw)>FIT_SCORE_RECEIPT_UPPER or set(r)!={'identity','sha256','C','calls','outputs','CPU_seconds'} or (r['identity'],r['C'],r['calls'],r['outputs'])!=(identity,C,expected_calls,expected_calls*C) or not isinstance(r['CPU_seconds'],(int,float)) or not math.isfinite(r['CPU_seconds']) or r['CPU_seconds']<0:raise ValueError('bounded committed FIT completion differs')
        if score.stat().st_size>32*C+FIT_SCORE_HEADER_UPPER:raise ValueError('FIT typed score exceeds hard header bound')
        pin={'path':str(score.relative_to(ctx.root)),'sha256':r['sha256']};a=checked(ctx.load_value(pin))
        return a,{'score_pin':pin,'receipt_sha256':h,'new_calls':0,'new_outputs':0,'score_bytes':score.stat().st_size,'receipt_bytes':len(raw),'reused':True,'successful_axis_calls':r['calls'],'successful_axis_outputs':r['outputs'],'successful_axis_CPU_seconds':r['CPU_seconds']}
    if score.exists() or score.is_symlink():raise ValueError('orphaned FIT q artifact requires explicit reconciliation, never re-forward')
    if predict is None:raise ValueError('complete FIT q cache missing: report cannot perform a forward')
    if storage_bound({'identity':identity,'q':np.zeros((C,4),np.float64)})['artifact_bytes']>32*C+FIT_SCORE_HEADER_UPPER:raise ValueError('typed FIT schema exceeds pre-admitted header')
    intent={'identity':identity,'axis':index,'C':C,'uncompleted_forward_counts_unknown':True}
    if len(json.dumps(intent,sort_keys=True,separators=(',',':')).encode())>FIT_SCORE_RECEIPT_UPPER:raise ValueError('FIT pending exceeds hard bound')
    pp=ctx.write_json(folder+'/pending.json',intent) # file+parent durable before model callback
    q,cost=predict();a=checked({'identity':identity,'q':q})
    if (cost['calls'],cost['candidate_outputs'])!=(expected_calls,expected_calls*C) or not isinstance(cost['CPU_seconds'],(int,float)) or not math.isfinite(cost['CPU_seconds']) or cost['CPU_seconds']<0:raise ValueError('actual successful per-axis forward count/CPU required')
    value={'identity':identity,'q':a}
    if storage_bound(value)['artifact_bytes']>32*C+FIT_SCORE_HEADER_UPPER:raise ValueError('actual typed FIT score exceeds hard header')
    pin=ctx.save_value(stem+'.npz',value)
    if pin['bytes']>32*C+FIT_SCORE_HEADER_UPPER:raise ValueError('actual FIT artifact bytes exceed hard bound')
    r={'identity':identity,'sha256':pin['sha256'],'C':C,'calls':cost['calls'],'outputs':cost['candidate_outputs'],'CPU_seconds':cost['CPU_seconds']}
    raw=json.dumps(r,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    if len(raw)>FIT_SCORE_RECEIPT_UPPER:raise ValueError('actual FIT completion exceeds hard bound')
    rp=ctx.write_json(stem+'.done.json',r)
    ctx.raw(pp);fd,name=ctx._parent(pp['path'])
    try:os.unlink(name,dir_fd=fd);os.fsync(fd)
    finally:os.close(fd)
    return a,{'score_pin':{k:pin[k] for k in ['path','sha256']},'receipt_sha256':rp['sha256'],'new_calls':cost['calls'],'new_outputs':cost['candidate_outputs'],'score_bytes':pin['bytes'],'receipt_bytes':len(raw),'reused':False,'successful_axis_calls':r['calls'],'successful_axis_outputs':r['outputs'],'successful_axis_CPU_seconds':r['CPU_seconds']}


def _write_fit_aggregate(ctx,value,*,already_public_bytes=0,path='FIT_risk_diagnostics.json'):
    raw=json.dumps(plain(value),sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    if type(already_public_bytes) is not int or already_public_bytes<0 or len(raw)+already_public_bytes>FIT_PUBLIC_AGGREGATE_UPPER:raise ValueError('complete FIT public aggregate exceeds decimal2M; do not truncate diagnostics')
    target=ctx.output(path,public=True)
    if target.parent.exists() and any(target.parent.glob('.'+target.name+'.owned-*')):raise ValueError('partial FIT public atomic artifact requires explicit reconciliation')
    return ctx.write_json(path,plain(value),public=True)


def _fit_axis_identity(global_identity,index,q,b,value,ids):
    return digest({'global':global_identity,'axis':index,'axis_source_identity':digest([q['id'],q.get('refs')]),'axis_kind':q['kind'],'truth_SHA':array_sha(q['labels']),'candidate_digests':array_sha(ids),'Base':array_sha(b),'D_C6':array_sha(value['D_C6']),'X_C6':array_sha(value['X_C6']),'raw109':array_sha(value['x109']),'context96':array_sha(value['context96'])})


class _FitRiskHeads:
    """Lazy final risk heads only; no S-L, auxiliary, teacher or optimizer."""
    def __init__(self,ctx,seed,norms,rows):self.ctx=ctx;self.seed=seed;self.norms=norms;self.rows=rows;self.loaded={}
    def _head(self,arm):
        import torch
        from echo_routing.temporal.event_risk_model import EventRisk,norm_tensors
        if arm not in self.loaded:
            r=next(r for r in self.rows if r['arm']==arm);pin={k:r[k] for k in ['path','sha256']};self.ctx.raw(pin);v=torch.load(self.ctx.root/pin['path'],map_location='cpu',weights_only=True);expected=norm_tensors(self.norms)
            if (v['seed'],v['arm'],v['update'])!=(self.seed,arm,6000) or set(v['normalizers'])!=set(expected) or any(set(v['normalizers'][k])!=set(d) or any(not torch.equal(v['normalizers'][k][n],x) for n,x in d.items()) for k,d in expected.items()) or 'identity' in r and v['identity']!=r['identity']:raise ValueError('final risk head/normalizer/audit identity differs')
            m=EventRisk(arm if arm in ['R1','R2'] else 'R2',self.seed);m.arm=arm;m.load_state_dict(v['model'],strict=True);m.eval();m.requires_grad_(False);self.loaded[arm]=m
        return self.loaded[arm]
    def predict(self,value,native):
        import torch,resource
        from echo_routing.temporal.event_risk_features import event_matrix
        _cpu();t=resource.getrusage(resource.RUSAGE_SELF);start=t.ru_utime+t.ru_stime;raw=value['x109'];cx=value['context96'];C=len(raw)
        if not C:raise ValueError('empty axis must skip all feature/model calls')
        x2=event_matrix(raw,cx,'R2',self.norms['candidate'],self.norms['context']);x1=None if native is not None else event_matrix(raw,None,'R1',self.norms['candidate']);columns=[];calls=0
        with torch.no_grad():
            for i,arm in enumerate(['R1','R2','C0','C1']):
                if i<2 and native is not None:columns.append(native[:,i]);continue
                model=self._head(arm);x=x1 if arm=='R1' else x2;columns.append(model(torch.from_numpy(x)).sigmoid().cpu().numpy().astype(np.float64));calls+=1
        q=np.column_stack(columns).astype(np.float64)
        if native is not None and not np.array_equal(q[:,:2],native):raise ValueError('native original q changed during reuse')
        t=resource.getrusage(resource.RUSAGE_SELF);return q,{'calls':calls,'candidate_outputs':calls*C,'CPU_seconds':t.ru_utime+t.ru_stime-start}


def role_of(table,half):
    if table not in TABLES or half not in ['selection','calibration']:raise ValueError('one original fixed table/half required')
    return half if table=='main' else 'old_stress_'+half if table=='a58_stress' else 'confirmation_'+half if table=='a59_confirmation' else table+'_'+half


def full_context(main,stress,main_values,stress_values):
    """Historical stress_context order: all main nonmulti, then120 new multi.

    Reuse complete scored axes only. Never join teacher logits or re-forward
    native/pair references for a different table. Original objects stay intact.
    """
    if len(main)!=4 or len(stress)!=4 or any(len(x)!=len(main[0]) for x in main) or any(len(x)!=len(stress[0]) for x in stress):raise ValueError('complete aligned physical banks required')
    if len(main_values)!=len(main[0]) or len(stress_values)!=len(stress[0]):raise ValueError('complete per-axis scores required')
    if len(stress[0])!=120 or any(q['kind']!='multi' for q in stress[0]) or {q['id'] for q in main[0]}&{q['id'] for q in stress[0]}:raise ValueError('original disjoint120 whole multi streams required')
    ix=[i for i,q in enumerate(main[0]) if q['kind']!='multi']
    if not ix or not any(main[0][i]['kind']=='native' for i in ix) or not any(main[0][i]['kind'] in ['pair','pairs'] for i in ix):raise ValueError('main native and pair reference banks required')
    bank=tuple([v[i] for i in ix]+list(w) for v,w in zip(main,stress))
    if len({q['id'] for q in bank[0]})!=len(bank[0]):raise ValueError('unique complete fulltask axes required')
    return bank,[main_values[i] for i in ix]+list(stress_values)


def input_identity(role_pins,bank):
    """Composite data digest of original raw role pins and complete arrays.

    This is explicitly a composite digest, not the raw SHA of one metadata file.
    Producer-manifest identity separately uses actual written manifest bytes.
    """
    return digest({'physical_role_metadata_raw_sha256':role_pins,'a08':[array_sha(v) for v in bank[1]],'jepa':[array_sha(v) for v in bank[2]],'Base':[array_sha(v) for v in bank[3]],'ordered_axis_kinds':[q['kind'] for q in bank[0]]})


def decoded(base,old,new_q,old_taus,new_taus):
    from echo_routing.temporal.event_risk_policy import decode as old_decode
    from echo_routing.temporal.selected_risk_policy import decode as new_decode
    c=np.asarray(old['old_scores'],np.float64);q=np.asarray(old.get('old_q',old.get('q')),np.float64);actions=old['actions']
    return {**old_decode(base,actions,c,q[:,0],q[:,1],old_taus),**new_decode(base,actions,c,new_q,new_taus),'actions':actions,'old_scores':c,'old_q':q,'q':np.asarray(new_q,np.float64)}


def validate_CAL_parity(seed,counts):
    expected={2026110461:(80,10,469),2026110462:(92,14,469)}
    if seed not in expected or tuple(counts.get(k) for k in ['N_admitted','unsafe_N_top','total_axes'])!=expected[seed]:raise ValueError('original CAL C6/N-top denominator or unsafe count differs: repair IO before any threshold fitting')
    return True


def original_CAL_counts(qs,bases,values):
    from echo_routing.temporal.selected_risk_supply import top
    from echo_routing.temporal.safety_targets import effects
    admitted=unsafe=0
    for q,b,v in zip(qs,bases,values):
        c=v['old_scores'];selected=top([a.key for a in v['actions']],c[:,0],c[:,1],c[:,2],c[:,4])
        if selected['N_admitted']:
            admitted+=1;cost=effects(b,v['actions'][selected['top_index']],q['labels'],F_FIT);unsafe+=cost['H']>0 or cost['dFP']>0
    return {'N_admitted':admitted,'unsafe_N_top':int(unsafe),'total_axes':len(qs)}


def calibration_inputs(qs,bases,values,population):
    if population!='CAL':raise ValueError('only exposed fixed CAL may fit thresholds; never AUDIT/outer')
    if not len(qs)==len(bases)==len(values):raise ValueError('complete original CAL axes required')
    from echo_routing.temporal.selected_risk_supply import top
    from echo_routing.temporal.safety_targets import effects
    rows={a:[] for a in ['C0','C1']}
    for q,b,v in zip(qs,bases,values):
        c=v['old_scores'];s=top([a.key for a in v['actions']],c[:,0],c[:,1],c[:,2],c[:,4]);i=s['top_index'];cost=None if not s['N_admitted'] else effects(b,v['actions'][i],q['labels'],F_FIT)
        for col,arm in enumerate(['C0','C1']):rows[arm].append({'N_admitted':s['N_admitted'],'q':None if i is None else float(v['q'][i,col]),'H':None if cost is None else int(cost['H']),'dFP':None if cost is None else int(cost['dFP'])})
    return rows


def plain(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {str(k):plain(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [plain(v) for v in value]
    if isinstance(value,set):raise ValueError('private per-unit sets must never be serialized into public aggregates')
    return value


class Inference:
    def __init__(self,context,inputs,references,audit_pin,coverage,code_objects):
        _cpu();validate_coverage_pass(coverage);self.ctx=context;self.inputs=inputs;self.refs=references;self.audit_pin=audit_pin;self.audit=context.read(audit_pin);_successful('audit_formal',self.audit);self.coverage=coverage;self.code={p:h for p,h in code_objects.items() if p.endswith('.py')};self.heads={};self.records=[];self.loaded_pins=[]
        from scripts.selected_risk_common import prior_path
        old=json.loads(prior_path('docs/results/2026-11-04_anchor_context_risk/training_inputs.json').read_text());self.norm_pin=old['normalizers'];self.norms=context.read(self.norm_pin)

    def _models(self,seed,old=False):
        import torch
        from echo_routing.temporal.event_risk_model import EventRisk,norm_tensors
        names=['R1','R2'] if old else ['C0','C1'];rows=self.refs.model_pins() if old else self.audit['models']
        for arm in names:
            key=(seed,arm)
            if key in self.heads:continue
            row=next(r for r in rows if (r['seed'],r['arm'])==(seed,arm));pin={k:row[k] for k in ['path','sha256']};self.ctx.raw(pin);state=torch.load(self.ctx.root/pin['path'],map_location='cpu',weights_only=True)
            if (state['seed'],state['arm'],state['update'])!=(seed,arm,6000):raise ValueError('only exact final6000 frozen heads required')
            norms=norm_tensors(self.norms)
            if set(state['normalizers'])!=set(norms) or any(not torch.equal(state['normalizers'][k][n],v) for k,d in norms.items() for n,v in d.items()):raise ValueError('fixed A62 normalizer changed; never refit on new FIT/CAL')
            model=EventRisk(arm if old else 'R2',seed);model.arm=arm;model.load_state_dict(state['model'],strict=True);model.eval();model.requires_grad_(False);self.heads[key]=model;self.loaded_pins.append(pin)
        return [self.heads[(seed,a)] for a in names]

    def _features(self,a,j,b,actions):
        from echo_routing.temporal.safety_features import utility_features
        from echo_routing.temporal.event_risk_features import context_features,event_matrix
        raw=np.stack([utility_features(a,j,b,x) for x in actions]) if actions else np.empty((0,109),np.float64)
        ctx=np.stack([context_features(a,j,b,x) for x in actions]) if actions else np.empty((0,96),np.float64)
        return raw,ctx,event_matrix(raw,ctx,'R2',self.norms['candidate'],self.norms['context'])

    def _fresh_old(self,seed,role,bank,readonly):
        from echo_routing.temporal.selected_risk_resume_supply import Supply
        from echo_routing.temporal.safety_policy_cache import SafetyCache
        from echo_routing.temporal.safety_runtime import infer
        from echo_routing.temporal.event_risk_cache import EventCache
        from echo_routing.temporal.event_risk_features import event_matrix
        from echo_routing.temporal.anchor_cache import sha
        import torch
        qs,aa,jj,bb=bank;ra=[a.argmax(1) for a in aa];rj=[j.argmax(1) for j in jj];tau=self.refs.thresholds()[seed]['L0'];ident=digest({'seed':seed,'role':role,'source_input':self.inputs.manifest['roles'][role]['sha256'],'fixed_models':self.refs.model_pins(),'code':self.code});folder=self.ctx.output(f'fixed_scores/seed{seed}/{role}');fixed=SafetyCache(folder,ident,bb,ra,rj,tau,readonly=readonly);prefix=fixed.cursor;calls=0
        if not readonly and fixed.cursor<len(qs):
            heads,norms,scale,_=Supply(self.ctx,self.inputs,None,{})._fixed(seed)
            for i in range(fixed.cursor,len(qs)):
                def predict(i=i):
                    nonlocal calls
                    self.ctx.revalidate();_cpu()
                    with torch.no_grad():v=infer(heads['utility'],heads['risk'],norms,scale,aa[i],jj[i],bb[i],tau)
                    calls+=3*bool(v['actions']);return v
                fixed.run_one(i,predict)
        values=[fixed.load(i) for i in range(len(qs))];refs=[{'path':str(fixed.file(i).relative_to(self.ctx.root)),'sha256':sha(fixed.file(i)),'scores_sha256':hashlib.sha256(np.asarray(v['score_matrix'],np.float64).tobytes()).hexdigest()} for i,v in enumerate(values)]
        identity=digest({'fixed_identity':ident,'fixed_journal':sha(fixed.journal),'seed':seed,'old_R1_R2_models':self.refs.model_pins(),'norms':self.norm_pin,'code':self.code});qcache=EventCache(self.ctx.output(f'fixed_q/seed{seed}/{role}'),identity,bb,ra,rj,None,refs,readonly=readonly)
        for i in range(qcache.cursor,len(qs)):
            def predict(i=i):
                nonlocal calls
                self.ctx.revalidate();_cpu();t=time.perf_counter();v=values[i];raw,ctx,x2=self._features(aa[i],jj[i],bb[i],v['actions'])
                if len(raw):
                    x1=event_matrix(raw,None,'R1',self.norms['candidate']);models=self._models(seed,old=True)
                    with torch.no_grad():q=np.column_stack([models[0](torch.from_numpy(x1)).sigmoid().numpy(),models[1](torch.from_numpy(x2)).sigmoid().numpy()]).astype(np.float64)
                    calls+=2
                else:q=np.empty((0,2),np.float64)
                return {'old_scores':v['score_matrix'],'q':q,'old_reference':refs[i],'seconds':time.perf_counter()-t}
            qcache.run_one(i,predict)
        self.records.append({'seed':seed,'role':role,'scope':'new confirmation frozen S-L/risk/R1/R2','new_MLP_calls_this_process':calls,'resumed_prefix_axes':prefix,'fixed_journal_sha256':sha(fixed.journal),'old_q_journal_sha256':sha(qcache.journal)})
        return [{**qcache.load(i),'old_q':qcache.load(i)['q']} for i in range(len(qs))]

    def score(self,seed,role,bank,*,readonly=False):
        from echo_routing.temporal.event_risk_cache import EventCache
        from echo_routing.temporal.anchor_cache import sha
        qs,aa,jj,bb=bank
        old=self._fresh_old(seed,role,bank,readonly) if role.startswith('a63_confirmation') else self.refs.scores(seed,role,bank)
        refs=[v['old_reference'] for v in old]
        identity=digest({'seed':seed,'role':role,'input':self.inputs.manifest['roles'][role]['sha256'],'old_C6_references':refs,'audit':self.audit_pin,'norms':self.norm_pin,'code':self.code,'columns':['C0','C1']});folder=self.ctx.output(f'q_scores/seed{seed}/{role}');cache=EventCache(folder,identity,bb,[a.argmax(1) for a in aa],[j.argmax(1) for j in jj],None,refs,readonly=readonly);prefix=cache.cursor;calls=0;candidate_outputs=0
        import torch
        for i in range(cache.cursor,len(qs)):
            def predict(i=i):
                nonlocal calls,candidate_outputs
                self.ctx.revalidate();_cpu();t=time.perf_counter()
                _,_,x=self._features(aa[i],jj[i],bb[i],old[i]['actions'])
                if len(x):
                    models=self._models(seed)
                    with torch.no_grad():q=np.column_stack([m(torch.from_numpy(x)).sigmoid().numpy() for m in models]).astype(np.float64)
                    calls+=2;candidate_outputs+=2*len(x)
                else:q=np.empty((0,2),np.float64)
                return {'old_scores':old[i]['old_scores'],'q':q,'old_reference':refs[i],'seconds':time.perf_counter()-t}
            cache.run_one(i,predict)
        for p in self.loaded_pins:self.ctx.raw(p)
        values=[{**cache.load(i),'old_q':old[i].get('old_q',old[i]['q'])} for i in range(len(qs))]
        self.records.append({'seed':seed,'role':role,'path':str(folder.relative_to(self.ctx.root)),'identity':identity,'journal_sha256':sha(cache.journal),'sequences':len(qs),'candidate_vectors':sum(len(v['q']) for v in values),'columns':['C0','C1'],'new_MLP_calls_this_process':calls,'new_candidate_outputs_this_process':candidate_outputs,'reused_main_native_pair_axes':0,'resumed_prefix_axes':prefix,'readonly':readonly})
        return values


class Readout:
    def __init__(self,context,inputs,freeze_pin,descriptors):self.ctx=context;self.inputs=inputs;self.freeze_pin=freeze_pin;self.descriptors=descriptors;self._main={};self._scores={}
    def _enter(self,stage):
        required={SOURCE,'echo_routing/temporal/selected_risk_resume_reference.py','echo_routing/temporal/selected_risk_analysis.py','scripts/calibrate_event_risk.py'}
        if stage in ['outer','aggregate']:required.add(PUB+'/readout_producer_manifest.json')
        f,e=self.ctx.enter(stage,self.freeze_pin,self.descriptors,required=required);_cpu();return f,e
    def _reference(self):
        from echo_routing.temporal.selected_risk_resume_reference import ReferenceBank
        return ReferenceBank(self.ctx,self.inputs)
    def _model_identity(self,audit,refs,head_sha,policy_sha,producer_sha):
        return {'producer_manifest_sha256':producer_sha,'head_seal_sha256':head_sha,'policy_manifest_sha256':policy_sha,'model_roles':{f'seed{r["seed"]}:{r["arm"]}':r['sha256'] for r in audit['models']+refs.model_pins()}}
    def _data_identity(self,role,bank,*,half=None):
        pins={role:self.inputs.manifest['roles'][role]['sha256']}
        if half is not None and role!=half:pins[half]=self.inputs.manifest['roles'][half]['sha256']
        return input_identity(pins,bank)
    def _producer(self,*,create=False):
        spec=self.ctx.read(self.ctx._bound_runtime_spec_pin)
        if any(spec.get(k) is None for k in ['FIT_producer','CONFIRM_producer']):raise ValueError('actual complete FIT and CONFIRM producer manifest rawpins required before readout')
        for k in ['FIT_producer','CONFIRM_producer']:self.ctx.raw(spec[k])
        value={'schema':'A63_readout_composite_physical_producer_manifest_v1','metadata':self.inputs.metadata_pin,'physical_old_cache_records':self.inputs.manifest['source_cache_roles'],'physical_new_FIT_producer_manifest':spec['FIT_producer'],'physical_new_CONFIRM_producer_manifest':spec['CONFIRM_producer'],'scope':'Composite manifest of physical producer records; adoption receipt is a separate gate and is never called a producer manifest'}
        path=PUB+'/readout_producer_manifest.json'
        if create:return self.ctx.write_json('readout_producer_manifest.json',value,public=True)
        cal=self.ctx.read({'path':PUB+'/calibration.json','sha256':self.ctx.evidence(self.descriptors['seal_heads'])['payload']['calibration']['sha256']})
        pin=cal['readout_producer_manifest']
        if pin['path']!=path or self.ctx.read(pin)!=value:raise ValueError('physical composite producer manifest changed')
        for k in ['FIT_producer','CONFIRM_producer']:self.ctx.raw(spec[k])
        return pin
    def _outer_bank(self,table,half,refs,infer,*,readonly,risk_sources=None):
        role=role_of(table,half)
        if half not in self._main:self._main[half]=refs.bank(half)
        main=self._main[half]
        for seed in SEEDS:
            if (half,seed) not in self._scores:self._scores[(half,seed)]=infer.score(seed,half,main,readonly=readonly)
            if risk_sources is not None:risk_sources[(half,seed)]=(main,self._scores[(half,seed)])
        if table=='main':return main,{s:self._scores[(half,s)] for s in SEEDS},self._data_identity(half,main)
        bank=self.inputs.bank(role,purpose='outer') if table=='a63_confirmation' else refs.bank(role)
        values={};complete=None
        for seed in SEEDS:
            original=infer.score(seed,role,bank,readonly=readonly)
            if risk_sources is not None:risk_sources[(role,seed)]=(bank,original)
            assembled,values[seed]=full_context(main,bank,self._scores[(half,seed)],original)
            complete=assembled
        return complete,values,self._data_identity(role,complete,half=half)

    def _risk_diagnostics(self,bank,values,*,role,seed,taus,return_candidates=False):
        """Post-score source diagnostics; no forward, fitting or policy changes."""
        from echo_routing.temporal.event_risk_metrics import risk_summary,_effect
        from echo_routing.temporal.safety_targets import effects
        from echo_routing.temporal.selected_risk_supply import top
        from echo_routing.temporal.proposal_actions import propose,partition
        heads=('R1','R2','C0','C1');kinds=('native','multi','pair');groups=('all',*kinds)
        if seed not in SEEDS or set(taus)!=set(heads) or any(isinstance(t,bool) or not isinstance(t,(int,float)) or not math.isfinite(t) or not 0<=t<=1 for t in taus.values()) or len(bank) not in [2,4] or any(len(v)!=len(bank[0]) for v in bank) or len(values)!=len(bank[0]):raise ValueError('aligned complete cache, exact head seed and frozen unit-range taus required')
        names=('axes','nonempty_axes','pre_G_candidates','G_eligible_candidates','G_vetoed_unscored_candidates','positive_Etop_axes','original_Ntop_axes')
        counts={n:0 for n in names};counts.update({n+'_'+k:0 for n in names for k in kinds});rows=[]
        regret={a:{k:[[],[]] for k in groups} for a in heads}
        for axis,(q,b,v) in enumerate(zip(bank[0],bank[-1],values)):
            kind='pair' if q['kind']=='pairs' else q['kind']
            if kind not in kinds:raise ValueError('fixed native/multi/pair source kind required')
            actions=v['actions'];c=np.asarray(v['old_scores']);oq=np.asarray(v['old_q']);nq=np.asarray(v['q']);C=len(actions)
            if c.shape!=(C,6) or oq.shape!=(C,2) or nq.shape!=(C,2):raise ValueError('complete unchanged C6/oldC2/newC2 rows required')
            selected=top([a.key for a in actions],c[:,0],c[:,1],c[:,2],c[:,4]);ei=selected['top_index'];ni=ei if selected['N_admitted'] else None
            if 'all_actions' in v and 'G_rejected' in v:all_actions=v['all_actions'];rejected=v['G_rejected']
            else:
                if len(bank)!=4:raise ValueError('cached FIT rows must provide original all-actions/G-rejected')
                all_actions=propose(b,np.asarray(bank[1][axis]).argmax(1),np.asarray(bank[2][axis]).argmax(1));canonical,rejected=partition(b,all_actions)
                if canonical!=actions:raise ValueError('post-score canonical action set changed')
            values_count=(1,int(bool(C)),len(all_actions),C,len(rejected),int(ei is not None),int(ni is not None))
            for name,n in zip(names,values_count):counts[name]+=n;counts[name+'_'+kind]+=n
            targets=[]
            for i,action in enumerate(actions):
                cost=effects(b,action,q['labels'],F_FIT);detail,*_=_effect(b,action,q['labels'],F_FIT);targets.append(cost['v'])
                rows.append({'kind':kind,'Ntop':i==ni,**_post_flags(cost['H'],cost['dFP'],detail['short16_lost'],cost['R'],cost['u'],detail['JEPA_exclusive_short16_recovered']),'R1':float(oq[i,0]),'R2':float(oq[i,1]),'C0':float(nq[i,0]),'C1':float(nq[i,1])})
            # Frozen S-L action_summary: best actual v including fixed noop0,
            # minus original positive top / its post-veto actual v. Not F1.
            best=max([0.,*targets]);top_actual=0. if ei is None else targets[ei]
            for h,head in enumerate(heads):
                qvalues=oq[:,h] if h<2 else nq[:,h-2];chosen=ni if ni is not None and qvalues[ni]<taus[head] else None
                after=best-(0. if chosen is None else targets[chosen])
                for k in ['all',kind]:regret[head][k][0].append(best-top_actual);regret[head][k][1].append(after)
        def describe(part,head):
            q=np.asarray([r[head] for r in part],np.float64);y=np.asarray([r['unsafe'] for r in part],bool)
            return {**risk_summary(q,y),**_post_summary(part,head,taus[head]),'q_zero_count':int((q==0).sum()),'q_one_count':int((q==1).sum()),'unsafe_at_zero':int(((q==0)&y).sum()),'safe_at_one':int(((q==1)&~y).sum())}
        result={'role':role,'seed':seed,'scope':'physical source axes only; full-task native/pair reuse is not new source exposure','axis_denominators':counts,'training_exposure':{'counts':None,'actual_candidate_exposures':None,'reason':'No exact verified training-index map supplied for this source bank; not inferred as0 from candidate or reporting counts'},'new_NN_forwards':0,'new_optimizer_updates':0}
        for head in heads:
            all_stats=describe(rows,head);admitted=[r for r in rows if r['Ntop']];ntop_stats=describe(admitted,head);coverage={};reg={}
            all_stats['by_kind']={k:describe([r for r in rows if r['kind']==k],head) for k in kinds};ntop_stats['by_kind']={k:describe([r for r in admitted if r['kind']==k],head) for k in kinds}
            for kind in groups:
                part=admitted if kind=='all' else [r for r in admitted if r['kind']==kind];accepted=sum(r[head]<taus[head] for r in part);axes=counts['axes' if kind=='all' else 'axes_'+kind]
                coverage[kind]={'N_top':len(part),'accepted_N_top':accepted,'q_extra_vetoes':len(part)-accepted,'unsafe_accepted':sum(r['unsafe'] and r[head]<taus[head] for r in part),'coverage_given_N_top':accepted/len(part) if part else None,'coverage_of_source_axes':accepted/axes if axes else None}
                reg[kind]={label:{**_post_distribution(v),'mean':float(np.mean(v)) if v else None} for label,v in zip(['top_action_regret','after_veto_regret'],regret[head][kind])}
            result[head]={'all_G_eligible_candidates':all_stats,'original_N_top':ntop_stats,'gate_coverage':coverage,'regret':reg,'regret_scope':'within-axis original S-L v including fixed noop0; not pooled F1, joint capacity or a safe-only oracle gap'}
        return (result,rows) if return_candidates else result

    def _source_risk_diagnostics(self,sources,old_taus,thresholds,fit_pin=None):
        roles=[role_of(t,h) for t in TABLES for h in ['selection','calibration']]
        if len(set(roles))!=14 or set(sources)!={(r,s) for r in roles for s in SEEDS}:raise ValueError('all14 complete scored physical source banks for each seed required; no missing-cache forward fallback')
        records=[]
        for role in roles:
            for seed in SEEDS:
                bank,values=sources[(role,seed)]
                if role not in ['selection','calibration'] and (len(bank[0])!=120 or any(q['kind']!='multi' for q in bank[0])):raise ValueError('stress/confirmation source is120multi only; reused main native/pair is a different view')
                tau={a:old_taus[seed][a] for a in ['R1','R2']}
                for arm in ['C0','C1']:
                    found=[r['tau'] for r in thresholds if (r['seed'],r['policy'])==(seed,arm)]
                    if len(found)!=1:raise ValueError('one already frozen CAL threshold per new head required')
                    tau[arm]=found[0]
                records.append(self._risk_diagnostics(bank,values,role=role,seed=seed,taus=tau))
        fit_scope={'score_cache_available':None,'derived_here':False,'required_scope_status':'FIT cache/report not supplied; required scope remains pending, never forward as fallback'}
        if fit_pin is not None:
            fit=self.ctx.read(fit_pin)
            if fit.get('passed') is not True or fit.get('complete') is not True or fit.get('FIT_q_completed_before_CAL') is not True or fit.get('FIT_report_generated_after_CAL') is not True or fit.get('FIT_scores_used_to_select_model_or_tau') is not False or fit.get('seeds')!=list(SEEDS) or len(fit.get('records',[]))!=2 or {r['seed'] for r in fit['records']}!=set(SEEDS) or any(set(r['selected_supply_views'])!={'all','D','X'} for r in fit['records']):raise ValueError('one shared complete FIT diagnostic report required')
            fit_scope={'score_cache_available':True,'diagnostics_pin':fit_pin,'derived_here':False}
        return {'physical_source_roles':14,'records':records,'source_role_scope_complete':True,'CAL_AUDIT_scope':'separate calibration.json.inner_records risk_diagnostics, never pooled as outer','FIT_scope':fit_scope,'required_risk_scope_complete':fit_pin is not None,'new_NN_forwards':0,'new_optimizer_updates':0,'source_vs_fulltask_scope':'only original14 physical cache banks; main native/pair reused in stress/confirmation full-task tables is not repeated source or NN exposure'}

    def _fit_inputs(self,e,freeze,seed,refs):
        from echo_routing.temporal.selected_risk_resume_formal import Formal
        prepared=e['prepare_formal']['payload'];audit=e['audit_formal']['payload'];_successful('audit_formal',audit)
        if len(audit.get('models',[]))!=4 or {(r['seed'],r['arm'],r['step'],r['parameters']) for r in audit['models']}!={(s,a,6000,6625) for s in SEEDS for a in ['C0','C1']}:raise ValueError('exact four audited final risk model endpoints required')
        if audit['formal_training_inputs_sha256']!=self.descriptors['prepare_formal']['payload_pin']['sha256']:raise ValueError('independent formal audit must bind exact prepared inputs')
        found=[r for r in prepared['records'] if r['seed']==seed]
        if len(found)!=1:raise ValueError('one prepared global candidate/draw domain per seed required')
        pr=found[0];data=self.ctx.load_value(pr['data'])
        if set(data)!={'candidate_digests','unsafe','kind_indices','draws'} or set(data['draws'])!={'C0','C1'} or any(array_sha(data['draws'][a])!=pr['draw_SHA'][a] for a in ['C0','C1']):raise ValueError('exact original global draws and candidate domain required')
        formal=Formal(self.ctx,self.inputs,self.freeze_pin,self.descriptors);selector=e['seal_selectors']['payload'];manifest=self.ctx.read(selector['cache_manifest']);rows=formal._cache_rows(manifest,seed);maps=_fit_candidate_map(rows,data)
        raw=np.concatenate([v['x109'] for q,b,v in rows]);cx=np.concatenate([v['context96'] for q,b,v in rows])
        if array_sha(raw)!=pr['raw109_sha256'] or array_sha(cx)!=pr['context96_sha256']:raise ValueError('all original raw109/context96 feature bits must match prepared inputs')
        del raw,cx
        norms,normpin=formal._norms()
        if normpin!=pr['original_fixed_normalizers']:raise ValueError('only original frozen A62 normalizers; no FIT refit')
        heads=[r for r in refs.model_pins() if r['seed']==seed and r['arm'] in ['R1','R2']]+[r for r in audit['models'] if r['seed']==seed]
        if len(heads)!=4 or {r['arm'] for r in heads}!={'R1','R2','C0','C1'}:raise ValueError('exact old/new final event heads required')
        modelpins=[{k:r[k] for k in ['arm','path','sha256']} for r in sorted(heads,key=lambda r:r['arm'])]
        from echo_routing.temporal.selected_risk_resume_supply import POST_NATIVE_PINS
        from scripts.selected_risk_common import prior_path
        oldcfg=json.loads(prior_path('configs/experiments/anchor_context_risk_v1.json').read_text());oldtrain=self.ctx.read(oldcfg['prior_A61']['training_inputs.json'])
        native_contract={'post_q':POST_NATIVE_PINS[seed],'old_A61_groups_dataset':oldtrain['dataset'],'original_FIT_metadata_SHA':self.inputs.manifest['roles']['fit']['sha256']}
        identity_inputs={'native_reuse_contract':native_contract,'seed':seed,'runtime_owner_raw_SHA':self.ctx.owner_sha,'prepared_manifest':self.descriptors['prepare_formal']['payload_pin'],'prepared_data':pr['data'],'candidate_digests_SHA':array_sha(data['candidate_digests']),'unsafe_SHA':array_sha(data['unsafe']),'draw_SHA':pr['draw_SHA'],'normalizers':normpin,'models':modelpins,'selector':self.descriptors['seal_selectors']['payload_pin'],'metadata':self.inputs.metadata_pin,'readout_code_SHA':freeze['objects'][SOURCE]}
        return data,rows,maps,norms,heads,identity_inputs

    def fit_risk_scores(self):
        """Final-risk-only FIT q cache before CAL; reporting never selects a head."""
        import resource
        from echo_routing.temporal.selected_risk_resume_formal import Formal
        from echo_routing.temporal.selected_risk_resume_supply import POST_NATIVE_PINS
        from echo_routing.temporal.selected_risk_resume_freeze import _raw
        from echo_routing.temporal.selected_risk_resume_context import strict_json
        from echo_routing.temporal.selected_risk_storage_engineering import load_trace
        from echo_routing.temporal.selected_risk_audit import audit_event_trace
        from echo_routing.temporal.selected_risk_resume_storage import storage_bound
        from echo_routing.temporal.safety_data import exact
        from scripts.selected_risk_common import prior_path
        from echo_routing.temporal.selected_risk_resume_contract import RUN
        start=resource.getrusage(resource.RUSAGE_SELF);cpu0=start.ru_utime+start.ru_stime
        cov=self.ctx.evidence(self.descriptors['coverage'])['payload']
        if cov.get('passed') is False:return {'status':'not_run','reason':'coverage-negative: four final risk heads not_run','new_MLP_calls_this_process':0,'new_candidate_outputs_this_process':0}
        validate_coverage_pass(cov);freeze,e=self._enter('fit_risk_scores');_successful('audit_formal',e['audit_formal']['payload']);_successful('train_formal',e['train_formal']['payload'])
        if any(self.ctx.output(name,public=True).exists() or self.ctx.output(name,public=True).is_symlink() for name in ['calibration_thresholds.json','calibration.json']):raise ValueError('FIT q cache must finish before any CAL threshold fitting')
        refs=self._reference();oldcfg=json.loads(prior_path('configs/experiments/anchor_context_risk_v1.json').read_text());oldtraining=self.ctx.read(oldcfg['prior_A61']['training_inputs.json']);oldfit=np.load(io.BytesIO(self.ctx.raw(oldtraining['dataset'])),allow_pickle=True).item();original=self.inputs.sequences('fit');records=[];newcalls=newoutputs=0
        physical=e['train_formal']['payload']['records']
        if len(physical)!=4 or {(r['seed'],r['arm']) for r in physical}!={(s,a) for s in SEEDS for a in ['C0','C1']}:raise ValueError('all original actual-Git physical completion records required')
        for seed in SEEDS:
            data,rows,maps,norms,heads,identity_inputs=self._fit_inputs(e,freeze,seed,refs);identity=digest(identity_inputs);n=len(data['unsafe']);exposures={};trace_pins=[]
            for arm in ['C0','C1']:
                model=next(r for r in heads if r['arm']==arm);r=next(r for r in physical if (r['seed'],r['arm'])==(seed,arm));completion=self.ctx.read(r['completion']);folder=str(PurePosixPath(model['path']).parent)
                if folder!=RUN+f'/models/seed{seed}/{arm}' or r['completion']['path']!=folder+'/completion.json' or completion.get('passed') is not True or completion['identity']!=model['identity'] or (completion['physical_updates'],completion['fullstate_count'])!=(6000,4):raise ValueError('actual audited full6000 head completion required before FIT inference')
                raw,arsha=_raw(self.ctx,folder+'/physical_updates.jsonl.archive.json');ar=strict_json(raw);gzpin={'path':folder+'/physical_updates.jsonl.gz','sha256':r['archive_sha256']}
                if ar['identity']!=model['identity'] or ar['archive_sha256']!=gzpin['sha256'] or ar['uncompressed_sha256']!=completion['raw_trace_sha256']:raise ValueError('FIT exposure trace must reproduce original audited completion')
                self.ctx.raw(gzpin);trace=load_trace(self.ctx.root/gzpin['path'],ar);audit_event_trace(trace,data['draws'][arm],data['kind_indices'],data['unsafe'],model['identity']);exposure=_actual_global_exposure(n,trace,model['identity'])
                if not exposure['full6000']:raise ValueError('full-pass FIT diagnostics requires all6000 actual updates, not a plan/prefix guess')
                exposures[arm]=exposure['counts'];trace_pins.append({'arm':arm,'completion':r['completion'],'archive':gzpin,'archive_receipt_SHA':arsha,'draw_SHA':array_sha(data['draws'][arm])})
            table={'identity':identity,**maps,'C0_exposures':exposures['C0'],'C1_exposures':exposures['C1']};table_bound=18*n+8*(len(rows)+1)+4096
            if storage_bound(table)['artifact_bytes']>table_bound:raise ValueError('global DX/offset/actual-exposure table exceeds hard18N+offsets+4096 bound')
            folder=f'FIT_risk/seed{seed}';tablepath=self.ctx.output(folder+'/global.private.npz')
            if tablepath.parent.exists() and any('.tmp-' in x.name or '.owned-' in x.name for x in tablepath.parent.glob('.*')):raise ValueError('partial FIT table atomic artifact requires reconciliation')
            if tablepath.exists() or tablepath.is_symlink():
                tb,th=_raw(self.ctx,str(tablepath.relative_to(self.ctx.root)));tablepin={'path':str(tablepath.relative_to(self.ctx.root)),'sha256':th}
                if len(tb)>table_bound or not exact(self.ctx.load_value(tablepin),table):raise ValueError('existing FIT global table is not exact same source/draw identity')
            else:tablepin={k:v for k,v in self.ctx.save_value(folder+'/global.private.npz',table).items() if k in ['path','sha256']}
            with np.load(io.BytesIO(self.ctx.raw(POST_NATIVE_PINS[seed])),allow_pickle=False) as z:
                if set(z.files)!={'C6','q'} or z['C6'].shape!=(4183,6) or z['q'].shape!=(4183,2):raise ValueError('entire original pinned POST FIT C6/C2 required')
                native=_fit_native_q(original,rows,oldfit,{k:z[k] for k in z.files})
            for r in heads:self.ctx.raw({k:r[k] for k in ['path','sha256']})
            runner=_FitRiskHeads(self.ctx,seed,norms,heads);ordered=hashlib.sha256();scorebytes=receiptbytes=reused=successful_calls=successful_outputs=0;neural_cpu=0.
            for i,(q,b,v) in enumerate(rows):
                lo,hi=maps['axis_offsets'][i:i+2];C=int(hi-lo);axisid=_fit_axis_identity(identity,i,q,b,v,data['candidate_digests'][lo:hi]);expected=0 if not C else 2 if native[i] is not None else 4
                result,info=_fit_score_axis(self.ctx,folder,i,C,axisid,expected,lambda v=v,native_q=native[i]:runner.predict(v,native_q))
                if info['receipt_sha256'] is not None:ordered.update(bytes.fromhex(info['receipt_sha256']))
                newcalls+=info['new_calls'];newoutputs+=info['new_outputs'];scorebytes+=info['score_bytes'];receiptbytes+=info['receipt_bytes'];reused+=int(info['reused']);successful_calls+=info.get('successful_axis_calls',0);successful_outputs+=info.get('successful_axis_outputs',0);neural_cpu+=info.get('successful_axis_CPU_seconds',0.)
            for r in heads:self.ctx.raw({k:r[k] for k in ['path','sha256']})
            records.append({'seed':seed,'identity':identity,'identity_inputs':identity_inputs,'table':tablepin,'folder':str(self.ctx.output(folder).relative_to(self.ctx.root)),'axes':len(rows),'candidates':n,'nonempty_axes':int(np.count_nonzero(np.diff(maps['axis_offsets']))),'ordered_completion_SHA256':ordered.hexdigest(),'score_artifact_bytes':scorebytes,'completion_receipt_bytes':receiptbytes,'reused_completed_axes':reused,'risk_event_head_instances_loaded_this_process':len(runner.loaded),'successful_axis_receipt_MLP_calls':successful_calls,'successful_axis_receipt_candidate_outputs':successful_outputs,'successful_axis_receipt_model_load_forward_CPU_seconds':neural_cpu,'native_original_q_pin':POST_NATIVE_PINS[seed],'actual_training_trace_bindings':trace_pins})
        if any(self.ctx.output(name,public=True).exists() or self.ctx.output(name,public=True).is_symlink() for name in ['calibration_thresholds.json','calibration.json']):raise ValueError('CAL appeared before final FIT q completion seal')
        t=resource.getrusage(resource.RUSAGE_SELF);value={'passed':True,'complete':True,'runtime_authorization_sha256':self.ctx.owner_sha,'seeds':list(SEEDS),'records':records,'FIT_q_completed_before_CAL':True,'FIT_scores_used_to_select_model_or_tau':False,'actual_new_MLP_calls_this_process':newcalls,'actual_new_candidate_outputs_this_process':newoutputs,'actual_all_attempt_MLP_calls':None,'all_attempt_counts_require_independent_closed_caller_recount':True,'informational_consumer_CPU_seconds_to_snapshot':t.ru_utime+t.ru_stime-cpu0,'axis_CPU_scope':'SELF CPU for feature normalization, lazy final head load and FP32 forwards; nested descriptor only','CPU_scope':'scoring consumer SELF through snapshot; full caller/entry/write/exit cost is canonical elsewhere; receipt-axis CPU is a nested descriptor, never an extra fee','new_teacher_or_S_L_or_aux_forward_calls':0,'optimizer_updates':0,'score_header_upper':FIT_SCORE_HEADER_UPPER,'axis_completion_upper':FIT_SCORE_RECEIPT_UPPER,'ordered_completion_scope':'SHA256 of nonempty completion receipt SHA256 bytes in physical global axis order; offsets describe empty axes'}
        return _write_fit_aggregate(self.ctx,value,path='FIT_risk_scores_manifest.json')

    def _fit_risk_report(self,e,freeze,thresholds,old_taus):
        """Read fixed q only after original CAL tau fitting; one shared report."""
        from echo_routing.temporal.selected_risk_resume_freeze import _raw
        from echo_routing.temporal.safety_data import exact
        pin=self.descriptors['fit_risk_scores']['payload_pin'];scores=e['fit_risk_scores']['payload'];_successful('audit_formal',e['audit_formal']['payload'])
        if scores.get('passed') is not True or scores.get('complete') is not True or scores.get('FIT_q_completed_before_CAL') is not True or scores.get('FIT_scores_used_to_select_model_or_tau') is not False or scores.get('seeds')!=list(SEEDS) or len(scores.get('records',[]))!=2:raise ValueError('independently sealed complete pre-CAL FIT q manifest required')
        refs=self._reference();public=[]
        for seed in SEEDS:
            data,rows,maps,norms,heads,identity_inputs=self._fit_inputs(e,freeze,seed,refs);r=next(v for v in scores['records'] if v['seed']==seed);identity=digest(identity_inputs)
            if r['identity']!=identity or r['identity_inputs']!=identity_inputs or r['candidates']!=len(data['unsafe']) or r['axes']!=len(rows):raise ValueError('FIT report inputs/models differ from sealed q production')
            n=len(data['unsafe']);tablepath=self.ctx.output(f'FIT_risk/seed{seed}/global.private.npz')
            if tablepath.stat().st_size>18*n+8*(len(rows)+1)+4096:raise ValueError('FIT global table exceeds hard bound before read')
            table=self.ctx.load_value(r['table'])
            if set(table)!={'identity','axis_offsets','D','X','C0_exposures','C1_exposures'} or table['identity']!=identity or any(np.asarray(table[k]).dtype!=maps[k].dtype or not np.array_equal(table[k],maps[k]) for k in maps) or any(np.asarray(table[a+'_exposures']).dtype!=np.int64 or np.asarray(table[a+'_exposures']).shape!=(n,) or np.any(table[a+'_exposures']<0) or int(table[a+'_exposures'].sum())!=384000 for a in ['C0','C1']):raise ValueError('exact sealed global candidate/DX/offset/full actual exposure table required')
            ordered=hashlib.sha256();values=[];qs=[];bb=[];folder=f'FIT_risk/seed{seed}'
            if r['folder']!=RUN+'/'+folder or r['table']['path']!=RUN+'/'+folder+'/global.private.npz':raise ValueError('exact seeded FIT cache namespace required')
            for i,(q,b,v) in enumerate(rows):
                lo,hi=maps['axis_offsets'][i:i+2];C=int(hi-lo);axisid=_fit_axis_identity(identity,i,q,b,v,data['candidate_digests'][lo:hi]);expected=0 if not C else 2 if q['kind']=='native' else 4
                q4,info=_fit_score_axis(self.ctx,folder,i,C,axisid,expected,None)
                if info['receipt_sha256'] is not None:ordered.update(bytes.fromhex(info['receipt_sha256']))
                values.append({**v,'old_scores':v['D_C6'],'old_q':q4[:,:2],'q':q4[:,2:]});qs.append(q);bb.append(b)
            if ordered.hexdigest()!=r['ordered_completion_SHA256']:raise ValueError('FIT q completion sequence changed after pre-CAL Git seal')
            taus={a:old_taus[seed][a] for a in ['R1','R2']}
            for arm in ['C0','C1']:
                found=[x['tau'] for x in thresholds if (x['seed'],x['policy'])==(seed,arm)]
                if len(found)!=1:raise ValueError('one original CAL-fitted threshold required per final head')
                taus[arm]=found[0]
            all_stats,flat=self._risk_diagnostics((qs,bb),values,role='FIT_all',seed=seed,taus=taus,return_candidates=True)
            if not np.array_equal(np.asarray([v['unsafe'] for v in flat],np.float32),data['unsafe']) or not np.array_equal(np.asarray([v['Ntop'] for v in flat],np.uint8),maps['D']):raise ValueError('FIT reporting unsafe/originalD membership changed from prepared global domain')
            views={}
            for membership,mask in [('all',np.ones(n,bool)),('D',maps['D'].astype(bool)),('X',maps['X'].astype(bool))]:
                parts=[v for v,keep in zip(flat,mask) if keep];head_stats={}
                for head in ['R1','R2','C0','C1']:
                    def summary(part):
                        a=np.asarray([v[head] for v in part]);y=np.asarray([v['unsafe'] for v in part],bool)
                        return {**_post_summary(part,head,taus[head]),'q_zero_count':int((a==0).sum()),'q_one_count':int((a==1).sum()),'unsafe_at_zero':int(((a==0)&y).sum()),'safe_at_one':int(((a==1)&~y).sum())}
                    head_stats[head]={'all':summary(parts),'by_kind':{k:summary([v for v in parts if v['kind']==k]) for k in ['native','multi','pair']}}
                views[membership]={'candidate_count':len(parts),'heads':head_stats,'actual_training_candidate_exposures':{arm:{**_post_distribution(table[arm+'_exposures'][mask]),'sum':int(table[arm+'_exposures'][mask].sum())} for arm in ['C0','C1']}}
            all_stats['training_exposure']={'counts_table':r['table'],'actual_global_C0_exposures':384000,'actual_global_C1_exposures':384000,'actual_training_trace_bindings':r['actual_training_trace_bindings'],'R1_R2_exposures_on_new_FIT_domain':None,'scope':'actual independently audited global6000x64 journals, not output vectors or repeated reporting views'}
            public.append({'seed':seed,'all_diagnostics':all_stats,'selected_supply_views':views,'q_score_manifest_seed_identity':identity,'score_table':r['table']})
        value={'passed':True,'complete':True,'records':public,'seeds':list(SEEDS),'FIT_q_scores_manifest':pin,'FIT_q_completed_before_CAL':True,'FIT_report_generated_after_CAL':True,'FIT_scores_used_to_select_model_or_tau':False,'new_MLP_calls_during_reporting':0,'new_optimizer_updates':0,'D_X_membership':'the original sealed deployment and opposite-fold selected-action flags; not recomputed from final q','training_vs_inference_exposure':'actual formal global draws are separate from newly recorded diagnostic MLP calls/candidate outputs in the score manifest'}
        return _write_fit_aggregate(self.ctx,value,already_public_bytes=len(self.ctx.raw(pin)))

    def calibrate(self):
        freeze,e=self._enter('calibrate');refs=self._reference();cov=self.ctx.evidence(self.descriptors['coverage'])['payload'];audit=e['audit_formal']['payload'];infer=Inference(self.ctx,self.inputs,refs,self.descriptors['audit_formal']['payload_pin'],cov,freeze['objects']);oldtau=refs.thresholds();thresholds=[];cal_info={};cached={};banks={};diagnostics={};plan=readout_plan()
        producer=self._producer(create=True)
        from echo_routing.temporal.selected_risk_policy import calibrate
        from echo_routing.temporal.selected_risk_resume_inner import evaluate_inner,inner_provenance
        for seed in SEEDS:
            bank=banks['CAL'] if 'CAL' in banks else banks.setdefault('CAL',refs.bank('CAL'));validate_CAL_parity(seed,original_CAL_counts(bank[0],bank[3],refs.scores(seed,'CAL',bank)));v=infer.score(seed,'CAL',bank);cached[(seed,'CAL')]=v;rows=calibration_inputs(bank[0],bank[3],v,'CAL')
            for arm,r in rows.items():
                result=calibrate(r);validate_CAL_parity(seed,result);thresholds.append({'seed':seed,'policy':arm,**result})
                if arm=='C0':cal_info[str(seed)]={k:result[k] for k in ['N_admitted','unsafe_N_top','total_axes']}
        fixed=[{'seed':s,'policy':p,'tau':oldtau[s][p]} for s in SEEDS for p in ['M0','R1','R2']];legacy=[{'seed':s,'policy':'L0','tau':oldtau[s]['L0']} for s in SEEDS]
        threshold_pin=self.ctx.write_json('calibration_thresholds.json',{'thresholds':thresholds,'fixed_thresholds':fixed,'legacy_thresholds':legacy,'CAL_only':True,'independent_or_clinical_safety_guarantee':False},public=True);inner=[]
        for population in ['CAL','AUDIT']:
            bank=banks[population] if population in banks else banks.setdefault(population,refs.bank(population));input_id=self._data_identity(population,bank)
            for spec in [r for r in plan['inner_rows'] if r['population']==population]:
                seed=spec['seed'];policy=spec['policy'];values=None
                if seed is not None:
                    if (seed,population) not in cached:cached[(seed,population)]=infer.score(seed,population,bank)
                    nt={a:next(r['tau'] for r in thresholds if (r['seed'],r['policy'])==(seed,a)) for a in ['C0','C1']};values=[decoded(b,v,v['q'],oldtau[seed],nt) for b,v in zip(bank[3],cached[(seed,population)])]
                models=self._model_identity(audit,refs,self.descriptors['audit_formal']['payload_pin']['sha256'],threshold_pin['sha256'],producer['sha256'])
                expected=inner_provenance(bank[0],bank[3],values,population,seed,policy,input_id,models,F_FIT)
                result=evaluate_inner(bank[0],bank[3],values,population,seed,policy,input_id,models,F_FIT,expected_provenance=expected);record=plain(result['record'])
                if seed is not None:
                    if (seed,population) not in diagnostics:diagnostics[(seed,population)]=self._risk_diagnostics(bank,cached[(seed,population)],role=population,seed=seed,taus={**{a:oldtau[seed][a] for a in ['R1','R2']},**nt})
                    record['risk_diagnostics_key']=population+'/seed'+str(seed)
                inner.append(record)
        fit_score_pin=self.descriptors['fit_risk_scores']['payload_pin']
        fitpin=self._fit_risk_report(e,freeze,thresholds,oldtau)
        inner_risk_pin=self.ctx.write_json('inner_risk_diagnostics.json',{'records':{p+'/seed'+str(seed):v for (seed,p),v in diagnostics.items()},'FIT_risk_diagnostics':fitpin,'CAL_AUDIT_are_exposed_development_sets':True},public=True)
        for r in inner:
            if 'risk_diagnostics_key' in r:r['risk_diagnostics']={'pin':inner_risk_pin,'key':r.pop('risk_diagnostics_key')}
        value={'passed':True,'FIT_risk_scores_manifest':fit_score_pin,'FIT_risk_diagnostics':fitpin,'risk_diagnostics':inner_risk_pin,'thresholds':thresholds,'fixed_thresholds':fixed,'legacy_thresholds':legacy,'inner_records':inner,'calibration_information':cal_info,'q_cache_records':infer.records,'model_audit_sha256':self.descriptors['audit_formal']['payload_pin']['sha256'],'readout_producer_manifest':producer,'aux_optimizer_updates':48000,'formal_optimizer_updates':24000,'new_teacher_forwards':0,'no_AUDIT_threshold_or_model_selection':True}
        calpin=self.ctx.write_json('calibration.json',value,public=True);planpin=self.ctx.write_json('readout_plan.json',plan,public=True)
        models={**audit,'FIT_risk_scores_manifest':fit_score_pin,'FIT_risk_diagnostics':fitpin,'risk_diagnostics':inner_risk_pin,'thresholds':thresholds,'fixed_thresholds':fixed,'legacy_thresholds':legacy,'inner_records':inner,'model_audit_sha256':self.descriptors['audit_formal']['payload_pin']['sha256'],'fixed_reference_models':refs.model_pins(),'calibration_sha256':calpin['sha256'],'readout_plan_sha256':planpin['sha256']}
        self.ctx.write_json('models_manifest.json',models,public=True);return calpin

    def outer(self):
        freeze,e=self._enter('outer');core=self.inputs.authorize_outer(self.descriptors['seal_heads'],self.descriptors['adopt_confirmation']);refs=self._reference();cov=self.ctx.evidence(self.descriptors['coverage'])['payload'];audit=self.ctx.read({'path':PUB+'/formal_model_audit.json','sha256':core['models_manifest']['model_audit_sha256']});infer=Inference(self.ctx,self.inputs,refs,{'path':PUB+'/formal_model_audit.json','sha256':core['models_manifest']['model_audit_sha256']},cov,freeze['objects']);cal=core['calibration'];oldtau=refs.thresholds();rows=[];bundles=[];risk_sources={}
        from echo_routing.temporal.selected_risk_analysis import endpoint_provenance
        from echo_routing.temporal.selected_risk_resume_pairing import evaluate_and_prepare
        for table in TABLES:
            for half in ['selection','calibration']:
                bank,scored,input_id=self._outer_bank(table,half,refs,infer,readonly=False,risk_sources=risk_sources)
                for spec in [r for r in readout_plan()['rows'] if (r['table'],r['half'])==(table,half)]:
                    seed=spec['seed'];policy=spec['policy'];values=None
                    if seed is not None:
                        nt={a:next(r['tau'] for r in cal['thresholds'] if (r['seed'],r['policy'])==(seed,a)) for a in ['C0','C1']};values=[decoded(b,v,v['q'],oldtau[seed],nt) for b,v in zip(bank[3],scored[seed])]
                    modelid=self._model_identity(audit,refs,self.descriptors['seal_heads']['payload_pin']['sha256'],e['seal_heads']['payload']['calibration']['sha256'],self._producer()['sha256'])
                    expected=endpoint_provenance(bank[0],bank[3],values,table,half,seed,policy,input_id,modelid,F_FIT)
                    result=evaluate_and_prepare(bank[0],bank[3],values,table,half,seed,policy,input_id,modelid,F_FIT,expected_provenance=expected);pin=self.ctx.save_value(f'paired/{table}/{half}/{spec["system"]}.private.npz',result['pair_bundle']);rows.append(plain(result['row']));bundles.append({'table':table,'half':half,'system':spec['system'],'pin':{k:pin[k] for k in ['path','sha256']}})
        if len(rows)!=238 or len(bundles)!=238:raise ValueError('all238 paired-provenance-bound endpoints must complete')
        return self.ctx.write_json('outer_metrics.json',{'passed':True,'row_count':238,'rows':rows,'private_bundles':bundles,'q_cache_records':infer.records,'source_risk_diagnostics':self._source_risk_diagnostics(risk_sources,oldtau,cal['thresholds'],fit_pin=cal['FIT_risk_diagnostics']),'head_seal_sha256':self.descriptors['seal_heads']['payload_pin']['sha256'],'source_adoption_sha256':self.descriptors['adopt_confirmation']['payload_pin']['sha256'],'new_teacher_forwards':0,'new_optimizer_updates':0},public=True)

    def aggregate(self):
        freeze,e=self._enter('aggregate');outer=e['outer']['payload'];core=self.inputs.authorize_outer(self.descriptors['seal_heads'],self.descriptors['adopt_confirmation']);refs=self._reference();cov=self.ctx.evidence(self.descriptors['coverage'])['payload'];auditpin={'path':PUB+'/formal_model_audit.json','sha256':core['models_manifest']['model_audit_sha256']};audit=self.ctx.read(auditpin);infer=Inference(self.ctx,self.inputs,refs,auditpin,cov,freeze['objects']);bundles={};expected={};rows=outer['rows'];oldtau=refs.thresholds()
        from echo_routing.temporal.selected_risk_analysis import endpoint_provenance
        from echo_routing.temporal.selected_risk_resume_aggregate import aggregate
        for table in TABLES:
            for half in ['selection','calibration']:
                bank,scored,inputid=self._outer_bank(table,half,refs,infer,readonly=True)
                for spec in [r for r in readout_plan()['rows'] if (r['table'],r['half'])==(table,half)]:
                    seed=spec['seed'];policy=spec['policy'];values=None
                    if seed is not None:
                        nt={a:next(r['tau'] for r in core['calibration']['thresholds'] if (r['seed'],r['policy'])==(seed,a)) for a in ['C0','C1']};values=[decoded(b,v,v['q'],oldtau[seed],nt) for b,v in zip(bank[3],scored[seed])]
                    modelid=self._model_identity(audit,refs,self.descriptors['seal_heads']['payload_pin']['sha256'],e['seal_heads']['payload']['calibration']['sha256'],self._producer()['sha256']);key=(table,half,spec['system']);expected[key]=endpoint_provenance(bank[0],bank[3],values,table,half,seed,policy,inputid,modelid,F_FIT);record=next(r for r in outer['private_bundles'] if (r['table'],r['half'],r['system'])==key);bundles[key]=self.ctx.load_value(record['pin'])
        result=aggregate(rows,bundles,expected)
        acceptance=self.ctx.write_json('acceptance.json',plain(result['acceptance']),public=True)
        return self.ctx.write_json('summary.json',plain({'passed':True,'comparison_count':140,'paired_comparisons':result['comparisons'],'rows':rows,'counts':result['counts'],'acceptance_sha256':acceptance['sha256'],'outer_metrics_sha256':e['outer']['payload_sha256'],'bootstrap_replicates':5000,'primary_family_size':8,'independence':result['independence']}),public=True)
