"""New lossless label-free supply layout; the old executed cache is untouched.

Ordered actions are reconstructed from the same frozen teacher/Base/G then
checked against their saved canonical SHA. The second seed stores its own C6
and a per-payload SHA to the first seed's complete readonly raw109/context96.
"""
import json,os
import numpy as np
from echo_routing.temporal.selected_risk_cache import SupplyCache
from echo_routing.temporal.selected_risk_supply import top,digest
from echo_routing.temporal.proposal_actions import propose,partition
from echo_routing.temporal.proposal_policy_cache import payload
from echo_routing.temporal.anchor_cache import atomic_json,sha,fsync_directory


def payload_upper(C,*,reference):
    if isinstance(C,bool) or not isinstance(C,int) or C<0:raise ValueError('nonnegative canonical candidate count required')
    return 8*C*(12 if reference else 217)+4096


class CompactSupplyCache(SupplyCache):
    def __init__(self,folder,identity,bases,raw_a,raw_j,readonly=False,feature_reference=None):
        self.feature_reference=feature_reference
        if feature_reference is not None:
            ref=feature_reference
            if not isinstance(ref,CompactSupplyCache) or not ref.readonly or ref.cursor!=ref.total or ref.feature_reference is not None or len(bases)!=ref.total:raise ValueError('one complete readonly first-seed feature provider, no reference chain')
            if str(ref.path)==str(folder) or any(not np.array_equal(a,b) for xx,yy in [(bases,ref.bases),(raw_a,ref.raw_a),(raw_j,ref.raw_j)] for a,b in zip(xx,yy)):raise ValueError('identical canonical teacher/Base original axes required for raw-feature sharing')
        super().__init__(folder,identity,bases,raw_a,raw_j,readonly=readonly)
    def _actions(self,index):
        b=self.bases[index];return partition(b,propose(b,self.raw_a[index],self.raw_j[index]))[0]
    def _reference(self,index):
        r=self.feature_reference
        if r is None:return None,None
        return r.load(index),sha(r.file(index))
    def load(self,index):
        if not 0<=index<self.cursor or sha(self.file(index))!=self.rows[index]['sha256']:raise ValueError('uncommitted/changed compact score payload')
        actions=self._actions(index);base={'D_C6','X_C6','actions_SHA','selection_JSON'};ref,refsha=self._reference(index)
        with np.load(self.file(index),allow_pickle=False) as z:
            required=base|({'reference_SHA'} if ref is not None else {'x109','context96'})
            if set(z.files)!=required or str(z['actions_SHA'])!=digest(payload(actions)):raise ValueError('exact ordered canonical action digest/schema differs')
            value={k:z[k].copy() for k in ['D_C6','X_C6']};value.update(action_payload=payload(actions),selection=json.loads(str(z['selection_JSON'])))
            if ref is None:value.update({k:z[k].copy() for k in ['x109','context96']})
            else:
                if str(z['reference_SHA'])!=refsha:raise ValueError('immutable first-seed feature payload SHA changed')
                value.update({k:ref[k] for k in ['x109','context96']})
        return self._check(value,index)
    def run_one(self,index,predict):
        if self.readonly or index!=self.cursor or index>=self.total or self.pending.exists():raise ValueError('exact next nonambiguous compact score claim required')
        atomic_json(self.pending,{'index':index,'identity':self.identity});value=predict();actions=self._actions(index);value['action_payload']=payload(actions);value['selection']={}
        for name in ['D','X']:
            s=np.asarray(value[name+'_C6']);value['selection'][name]=top([a.key for a in actions],s[:,0],s[:,1],s[:,2],s[:,4])
        checked=self._check(value,index);ref,refsha=self._reference(index)
        if ref is not None and any(checked[k].dtype!=ref[k].dtype or checked[k].shape!=ref[k].shape or checked[k].tobytes()!=ref[k].tobytes() for k in ['x109','context96']):raise ValueError('shared raw features must match every original bit, including signed zero')
        fields={k:checked[k] for k in ['D_C6','X_C6']};fields.update(actions_SHA=digest(value['action_payload']),selection_JSON=json.dumps(value['selection'],sort_keys=True))
        if ref is None:fields.update({k:checked[k] for k in ['x109','context96']})
        else:fields['reference_SHA']=refsha
        path=self.file(index);tmp=path.with_suffix(path.suffix+'.tmp')
        with tmp.open('xb') as f:np.savez(f,**fields);f.flush();os.fsync(f.fileno())
        if tmp.stat().st_size>payload_upper(len(actions),reference=ref is not None):raise ValueError('compact payload header/coexistence byte bound exceeded; retain pending prefix')
        os.replace(tmp,path);fsync_directory(self.path);row={'index':index,'identity':self.identity,'positions':len(self.bases[index]),'sha256':sha(path)}
        with self.journal.open('a') as f:f.write(json.dumps(row,sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())
        fsync_directory(self.path);self.rows.append(row);self.cursor+=1;self.pending.unlink();fsync_directory(self.path)
