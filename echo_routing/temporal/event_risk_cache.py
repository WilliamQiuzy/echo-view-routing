"""A62 old C6 reference plus two event scores; sealed teacher read-only adoption."""
import json,math,os
from pathlib import Path
import numpy as np
from echo_routing.temporal.anchor_cache import ProbabilityCache,atomic_json,fsync_directory,sha
from echo_routing.temporal.proposal_actions import propose,partition,apply
from echo_routing.temporal.proposal_policy_cache import payload
from echo_routing.temporal.event_risk_policy import decode,POLICIES


class EventCache(ProbabilityCache):
    def __init__(self,folder,identity,bases,raw_a,raw_j,taus,old_references,readonly=False):
        if not bases or not len(bases)==len(raw_a)==len(raw_j)==len(old_references):raise ValueError('complete original source axes and oldscore references required')
        self.bases,self.raw_a,self.raw_j,self.taus,self.old_references=bases,raw_a,raw_j,dict(taus),old_references
        super().__init__(folder,identity,[len(b) for b in bases],readonly=readonly)
    def _check(self,v,i):
        b=self.bases[i];all_actions=propose(b,self.raw_a[i],self.raw_j[i]);actions,rejected=partition(b,all_actions);old=np.asarray(v['old_scores'],np.float64);q=np.asarray(v['q'],np.float64)
        if old.shape!=(len(actions),6) or q.shape!=(len(actions),2) or not np.isfinite(old).all() or not np.isfinite(q).all() or (q<0).any() or (q>1).any() or v['old_reference']!=self.old_references[i]:raise ValueError('aligned finite C6/C2 and exact frozen oldscore reference required')
        for a,s in zip(actions,old):
            K=np.count_nonzero(apply(b,a)!=b);u,h,r,n,fp,e=s
            if min(h,r,n)<0 or abs(h+r+n-K)>max(1e-5,K*1e-6) or not 0<=fp<=1 or not max(fp,h/K)-1e-7<=e<=max(fp,min(h,1))+1e-7:raise ValueError('old gross probability sums/evidence changed')
        if 'action_payload' in v and v['action_payload']!=payload(actions):raise ValueError('canonical action/axis changed')
        expected=decode(b,actions,old,q[:,0],q[:,1],self.taus)
        if any(p in v and not np.array_equal(v[p]['prediction'],r['prediction']) for p,r in expected.items()):raise ValueError('same-top frozen-gate labels disagree')
        seconds=float(v['seconds'])
        if not math.isfinite(seconds) or seconds<0:raise ValueError('finite cached inference time required')
        return {**expected,'old_scores':old,'q':q,'old_reference':v['old_reference'],'actions':actions,'all_actions':all_actions,'G_rejected':rejected,'seconds':seconds}
    def load(self,i):
        if not 0<=i<self.cursor or sha(self.file(i))!=self.rows[i]['sha256']:raise ValueError('uncommitted/changed q payload')
        with np.load(self.file(i)) as z:
            expected={'labels_'+p for p in POLICIES}|{'old_scores','q','actions_json','old_reference_json','taus_json','seconds'}
            if set(z.files)!=expected or json.loads(str(z['taus_json']))!=self.taus:raise ValueError('complete joint policy payload/frozen thresholds required')
            v={p:{'prediction':z['labels_'+p].copy()} for p in POLICIES};v.update(old_scores=z['old_scores'].copy(),q=z['q'].copy(),action_payload=json.loads(str(z['actions_json'])),old_reference=json.loads(str(z['old_reference_json'])),seconds=float(z['seconds']))
        return self._check(v,i)
    def run_one(self,i,predict):
        if self.readonly or i!=self.cursor or i>=self.total or self.pending.exists():raise ValueError('exactly-next writable q claim required; ambiguous work never replayed')
        atomic_json(self.pending,{'index':i,'identity':self.identity});v=self._check(predict(),i);path=self.file(i);temp=path.with_suffix(path.suffix+'.tmp')
        with temp.open('xb') as f:
            np.savez(f,**{'labels_'+p:v[p]['prediction'].astype(np.int8) for p in POLICIES},old_scores=v['old_scores'],q=v['q'],actions_json=json.dumps(payload(v['actions'])),old_reference_json=json.dumps(v['old_reference'],sort_keys=True),taus_json=json.dumps(self.taus,sort_keys=True),seconds=v['seconds']);f.flush();os.fsync(f.fileno())
        os.replace(temp,path);fsync_directory(self.path);row={'index':i,'identity':self.identity,'positions':len(self.bases[i]),'sha256':sha(path)}
        with self.journal.open('a') as f:f.write(json.dumps(row,sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())
        fsync_directory(self.path);self.pending.unlink();fsync_directory(self.path);self.rows.append(row);self.cursor+=1


def adopt_sources(manifest,expected,root,formal_updates):
    roles={'CAL_mixed':150,'AUDIT_mixed':150,'a62_confirmation_selection':120,'a62_confirmation_calibration':120};rows=manifest['records'];root=Path(root)
    if formal_updates!=24000 or not manifest['complete'] or manifest['new_event_updates_at_production']!=0 or set(expected)!=set(roles) or len(rows)!=8 or {(r['source'],r['role']) for r in rows}!={(s,r) for s in ['a08','jepa'] for r in roles}:raise ValueError('complete independent teacher production0 / actualfinal24000 binding required')
    count=0
    for r in rows:
        input_sha,lengths=expected[r['role']];folder=root/r['path']
        if not r['complete'] or len(lengths)!=roles[r['role']] or r['committed_sequences']!=len(lengths) or r['input_sha256']!=input_sha or sha(folder/'committed.jsonl')!=r['journal_sha256']:raise ValueError('teacher source/recipe/axis/journal changed')
        cache=ProbabilityCache(folder,r['identity'],lengths,readonly=True);count+=cache.cursor
    if count!=1080:raise ValueError('exact1080 complete whole-context teacher payloads required')
    return {'passed':True,'verified_payloads':count,'new_source_forwards':0,'new_event_updates_at_production':0,'formal_updates_at_binding':formal_updates,'records':rows,'actual_head_Git_seal_verified_by_caller_before_adoption':True}
