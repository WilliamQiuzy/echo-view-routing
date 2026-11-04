"""Joint O/E/R durable full-axis cache: one forward and canonical source binding."""
import json,math,os
import numpy as np
from echo_routing.temporal.anchor_cache import ProbabilityCache,atomic_json,fsync_directory,sha
from echo_routing.temporal.proposal_actions import propose,partition,decode


def payload(actions):return [[a.start,a.end,list(a.replacement),a.old,a.jepa] for a in actions]

class PolicyCache(ProbabilityCache):
    def __init__(self,folder,identity,bases,raw_a,raw_j,readonly=False):
        if not bases or not len(bases)==len(raw_a)==len(raw_j):raise ValueError('matching full source axes required')
        self.bases=bases;self.raw_a=raw_a;self.raw_j=raw_j
        super().__init__(folder,identity,[len(b) for b in bases],readonly=readonly)

    def _check(self,value,index):
        all_actions=propose(self.bases[index],self.raw_a[index],self.raw_j[index]);actions,rejected=partition(self.bases[index],all_actions)
        if value['action_payload']!=payload(actions):raise ValueError('cached canonical action/replacement/provenance differs from frozen source axes')
        outputs=np.asarray(value['outputs'],np.float64)
        if outputs.shape!=(len(actions),3) or not np.isfinite(outputs).all():raise ValueError('finite physical signed N3 score axis required')
        expected={p:decode(self.bases[index],actions,outputs,p) for p in ['O','E','R']}
        for p in expected:
            if not np.array_equal(value[p]['prediction'],expected[p]['prediction']):raise ValueError('cached labels differ from same-weight O/E/R decoding')
        seconds=float(value['cached_feature_head_policy_seconds'])
        if not math.isfinite(seconds) or seconds<0:raise ValueError('finite cached-only head timing required')
        return {**expected,'outputs':outputs,'actions':actions,'all_actions':all_actions,'G_rejected':rejected,'cached_feature_head_policy_seconds':seconds}

    def load(self,index):
        if not 0<=index<self.cursor or sha(self.file(index))!=self.rows[index]['sha256']:raise ValueError('uncommitted or changed joint output')
        with np.load(self.file(index)) as z:
            if set(z.files)!={'labels_O','labels_E','labels_R','outputs','actions_json','seconds'}:raise ValueError('complete joint cache keys changed')
            value={p:{'prediction':z['labels_'+p].copy()} for p in ['O','E','R']}
            value.update({'outputs':z['outputs'].copy(),'action_payload':json.loads(str(z['actions_json'])),'cached_feature_head_policy_seconds':float(z['seconds'])})
        if self.rows[index]['positions']!=len(self.bases[index]):raise ValueError('joint original axis changed')
        return self._check(value,index)

    def run_one(self,index,predict):
        if self.readonly or index!=self.cursor or index>=self.total or self.pending.exists():raise ValueError('exactly-next durable unambiguous joint policy claim required')
        atomic_json(self.pending,{'index':index,'identity':self.identity})
        value=predict();value['action_payload']=payload(value['actions']);value=self._check(value,index)
        path=self.file(index);tmp=path.with_suffix(path.suffix+'.tmp')
        with tmp.open('xb') as h:
            np.savez(h,**{'labels_'+p:value[p]['prediction'].astype(np.int8) for p in ['O','E','R']},outputs=value['outputs'],actions_json=json.dumps(payload(value['actions'])),seconds=value['cached_feature_head_policy_seconds']);h.flush();os.fsync(h.fileno())
        os.replace(tmp,path);fsync_directory(self.path)
        row={'index':index,'identity':self.identity,'positions':len(self.bases[index]),'sha256':sha(path)}
        with self.journal.open('a') as h:h.write(json.dumps(row,sort_keys=True)+'\n');h.flush();os.fsync(h.fileno())
        fsync_directory(self.path);self.pending.unlink();fsync_directory(self.path);self.rows.append(row);self.cursor+=1
