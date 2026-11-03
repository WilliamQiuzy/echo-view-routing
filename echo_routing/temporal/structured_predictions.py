"""A58 integer-label inference journal with exact clean-prefix continuation."""
from pathlib import Path
import json,os
import numpy as np
from echo_routing.temporal.dense_readout import atomic_json,sha

class LabelReadout:
    def __init__(self,path,identity,lengths):
        self.path=Path(path);self.path.mkdir(parents=True,exist_ok=True);self.identity=identity;self.lengths=tuple(map(int,lengths));self.total=len(self.lengths);self.pending=self.path/'pending.json';self.journal=self.path/'committed.jsonl'
        if not self.total or any(n<1 for n in self.lengths):raise ValueError('one positive expected length per sequence required')
        binding=self.path/'identity.json';value={'identity':identity,'total':self.total}
        if binding.exists():
            if json.loads(binding.read_text())!=value:raise ValueError('head/cache/input inference identity changed')
        else:atomic_json(binding,value)
        if self.pending.exists():raise RuntimeError('ambiguous pending label inference: retain evidence and stop')
        self.rows=[json.loads(s) for s in self.journal.read_text().splitlines()] if self.journal.exists() else [];self.cursor=len(self.rows)
        if self.cursor>self.total:raise ValueError('label prefix exceeds declared inputs')
        for i,row in enumerate(self.rows):
            if row['index']!=i or row['identity']!=identity:raise ValueError('label prefix is not contiguous or has changed identity')
            self.load(i)
    def file(self,index):return self.path/f'sequence{index:06d}.private.npz'
    def _check(self,value,index):
        value=np.asarray(value)
        if value.shape!=(self.lengths[index],) or value.dtype.kind not in 'iu' or (value<0).any() or (value>=5).any():raise ValueError('complete original-axis integer five-class labels required')
        return value.astype(np.int8)
    def run_one(self,index,predict):
        if index!=self.cursor or index>=self.total:raise ValueError('only exactly next committed label cursor may be claimed')
        atomic_json(self.pending,{'index':index,'identity':self.identity});value=self._check(predict(),index);path=self.file(index);temp=path.with_suffix(path.suffix+'.tmp')
        with temp.open('wb') as f:np.savez(f,labels=value);f.flush();os.fsync(f.fileno())
        temp.replace(path);directory=os.open(self.path,os.O_RDONLY)
        try:os.fsync(directory)
        finally:os.close(directory)
        row={'index':index,'identity':self.identity,'positions':len(value),'sha256':sha(path)}
        with self.journal.open('a') as f:f.write(json.dumps(row)+'\n');f.flush();os.fsync(f.fileno())
        self.pending.unlink();self.rows.append(row);self.cursor+=1
    def load(self,index):
        if not 0<=index<self.cursor:raise ValueError('uncommitted label inference')
        row=self.rows[index];path=self.file(index)
        if row['positions']!=self.lengths[index] or sha(path)!=row['sha256']:raise ValueError('label cache axis/hash differs from committed prefix')
        with np.load(path) as z:value=z['labels'].copy()
        return self._check(value,index)
