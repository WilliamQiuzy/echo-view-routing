"""A61 single joint E/N/C forward, full-axis source binding and durable prefix."""
import json,math,os
import numpy as np
from echo_routing.temporal.anchor_cache import ProbabilityCache,atomic_json,fsync_directory,sha
from echo_routing.temporal.proposal_actions import propose,partition,apply
from echo_routing.temporal.proposal_policy_cache import payload
from echo_routing.temporal.safety_runtime import policies


class SafetyCache(ProbabilityCache):
    def __init__(self,folder,identity,bases,raw_a,raw_j,tau,readonly=False):
        if not bases or not len(bases)==len(raw_a)==len(raw_j) or not math.isfinite(tau) or not 0<=tau<=1:raise ValueError('complete source axes and frozen unit-range tau required')
        self.bases,self.raw_a,self.raw_j,self.tau=bases,raw_a,raw_j,float(tau)
        super().__init__(folder,identity,[len(b) for b in bases],readonly=readonly)
    def _check(self,value,index):
        b=self.bases[index];all_actions=propose(b,self.raw_a[index],self.raw_j[index]);actions,rejected=partition(b,all_actions);scores=np.asarray(value['score_matrix'],np.float64)
        if value['action_payload']!=payload(actions) or value['tau']!=self.tau or scores.shape!=(len(actions),6) or not np.isfinite(scores).all():raise ValueError('source/action/class/scalar-score/tau binding changed')
        for a,row in zip(actions,scores):
            K=np.count_nonzero(apply(b,a)!=b);u,h,r,n,fp,evidence=row
            if min(h,r,n)<0 or abs(h+r+n-K)>max(1e-5,K*1e-6) or not 0<=fp<=1 or not max(fp,h/K)-1e-7<=evidence<=max(fp,min(h,1))+1e-7:raise ValueError('changed-position gross probability aggregation/evidence range changed')
        expected=policies(b,actions,scores,self.tau)
        for p in expected:
            if not np.array_equal(value[p]['prediction'],expected[p]['prediction']):raise ValueError('cached E/N/C labels differ from same-top exact frozen scores/gates')
        seconds=float(value['cached_feature_heads_policy_seconds'])
        if not math.isfinite(seconds) or seconds<0:raise ValueError('finite cached-only timing required')
        return {**expected,'score_matrix':scores,'actions':actions,'all_actions':all_actions,'G_rejected':rejected,'cached_feature_heads_policy_seconds':seconds,'tau':self.tau}
    def load(self,index):
        if not 0<=index<self.cursor or sha(self.file(index))!=self.rows[index]['sha256']:raise ValueError('uncommitted/changed score cache')
        with np.load(self.file(index)) as z:
            if set(z.files)!={'labels_E','labels_N','labels_C','score_matrix','actions_json','seconds','tau'}:raise ValueError('complete E/N/C physical score schema required')
            value={p:{'prediction':z['labels_'+p].copy()} for p in ['E','N','C']}
            value.update(score_matrix=z['score_matrix'].copy(),action_payload=json.loads(str(z['actions_json'])),cached_feature_heads_policy_seconds=float(z['seconds']),tau=float(z['tau']))
        if self.rows[index]['positions']!=len(self.bases[index]):raise ValueError('original cache label axis changed')
        return self._check(value,index)
    def run_one(self,index,predict):
        if self.readonly or index!=self.cursor or index>=self.total or self.pending.exists():raise ValueError('exactly-next unambiguous writable score claim required')
        atomic_json(self.pending,{'index':index,'identity':self.identity});value=predict();value.update(action_payload=payload(value['actions']),tau=self.tau);value=self._check(value,index)
        path=self.file(index);temp=path.with_suffix(path.suffix+'.tmp')
        with temp.open('xb') as h:
            np.savez(h,**{'labels_'+p:value[p]['prediction'].astype(np.int8) for p in ['E','N','C']},score_matrix=value['score_matrix'],actions_json=json.dumps(payload(value['actions'])),seconds=value['cached_feature_heads_policy_seconds'],tau=self.tau);h.flush();os.fsync(h.fileno())
        os.replace(temp,path);fsync_directory(self.path);row={'index':index,'identity':self.identity,'positions':len(self.bases[index]),'sha256':sha(path)}
        with self.journal.open('a') as h:h.write(json.dumps(row,sort_keys=True)+'\n');h.flush();os.fsync(h.fileno())
        fsync_directory(self.path);self.pending.unlink();fsync_directory(self.path);self.rows.append(row);self.cursor+=1
