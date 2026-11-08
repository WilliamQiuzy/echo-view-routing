"""A63 immutable label-free joint D/X scores/features/selection before effects.

The cache contains no reference labels/effects. Pending or uncommitted output
never replays NN work. Complete original axes and canonical G actions bind every
payload, including empty axes that skip all MLP calls in the caller.
"""
import json,os
import numpy as np
from echo_routing.temporal.anchor_cache import ProbabilityCache,atomic_json,fsync_directory,sha
from echo_routing.temporal.proposal_actions import propose,partition,apply
from echo_routing.temporal.proposal_policy_cache import payload
from echo_routing.temporal.selected_risk_supply import top

class SupplyCache(ProbabilityCache):
    def __init__(self,folder,identity,bases,raw_a,raw_j,readonly=False):
        if not bases or not len(bases)==len(raw_a)==len(raw_j):raise ValueError('complete matched source axes required')
        self.bases,self.raw_a,self.raw_j=bases,raw_a,raw_j
        super().__init__(folder,identity,[len(b) for b in bases],readonly=readonly)
    def _check(self,value,index):
        b=self.bases[index];all_actions=propose(b,self.raw_a[index],self.raw_j[index]);actions,rejected=partition(b,all_actions);C=len(actions);out={}
        for name,width in [('D_C6',6),('X_C6',6),('x109',109),('context96',96)]:
            a=np.asarray(value[name])
            if a.dtype!=np.float64 or a.shape!=(C,width) or not np.isfinite(a).all():raise ValueError('complete finite exact float64 source feature/score matrices required')
            out[name]=a
        for name in ['D_C6','X_C6']:
            for action,row in zip(actions,out[name]):
                K=np.count_nonzero(apply(b,action)!=b);u,h,r,n,fp,evidence=row
                if min(h,r,n)<0 or abs(h+r+n-K)>max(1e-5,K*1e-6) or not 0<=fp<=1 or not max(fp,h/K)-1e-7<=evidence<=max(fp,min(h,1))+1e-7:raise ValueError('original changed-position probability mass/evidence schema changed')
            s=out[name];out[name[0]]=top([a.key for a in actions],s[:,0],s[:,1],s[:,2],s[:,4])
        if value['action_payload']!=payload(actions) or value['selection']!={k:out[k] for k in ['D','X']}:raise ValueError('canonical G/action/positive same-top oldN selection binding differs')
        return {**out,'actions':actions,'all_actions':all_actions,'G_rejected':rejected}
    def load(self,index):
        if not 0<=index<self.cursor or sha(self.file(index))!=self.rows[index]['sha256']:raise ValueError('uncommitted or changed joint supply payload')
        with np.load(self.file(index),allow_pickle=False) as z:
            if set(z.files)!={'D_C6','X_C6','x109','context96','actions_json','selection_json'}:raise ValueError('raw label-free joint supply schema changed')
            value={k:z[k].copy() for k in ['D_C6','X_C6','x109','context96']};value.update(action_payload=json.loads(str(z['actions_json'])),selection=json.loads(str(z['selection_json'])))
        return self._check(value,index)
    def run_one(self,index,predict):
        if self.readonly or index!=self.cursor or index>=self.total or self.pending.exists():raise ValueError('exact next unambiguous supply score claim required')
        atomic_json(self.pending,{'index':index,'identity':self.identity});value=predict();b=self.bases[index];actions,_=partition(b,propose(b,self.raw_a[index],self.raw_j[index]));value['action_payload']=payload(actions);value['selection']={}
        for name in ['D','X']:
            s=np.asarray(value[name+'_C6']);value['selection'][name]=top([a.key for a in actions],s[:,0],s[:,1],s[:,2],s[:,4])
        checked=self._check(value,index);path=self.file(index);tmp=path.with_suffix(path.suffix+'.tmp')
        with tmp.open('xb') as f:
            np.savez(f,**{k:checked[k] for k in ['D_C6','X_C6','x109','context96']},actions_json=json.dumps(value['action_payload']),selection_json=json.dumps(value['selection']));f.flush();os.fsync(f.fileno())
        os.replace(tmp,path);fsync_directory(self.path);row={'index':index,'identity':self.identity,'positions':len(b),'sha256':sha(path)}
        with self.journal.open('a') as f:f.write(json.dumps(row,sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())
        fsync_directory(self.path);self.pending.unlink();fsync_directory(self.path);self.rows.append(row);self.cursor+=1
