"""A57 independently resumable whole-sequence probability cache."""
from pathlib import Path
import hashlib,json,os
import numpy as np

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def atomic_json(path,value):
 path=Path(path);temp=path.with_suffix(path.suffix+'.tmp')
 with temp.open('w') as f:json.dump(value,f);f.write('\n');f.flush();os.fsync(f.fileno())
 temp.replace(path)

class SequenceReadout:
 def __init__(self,path,identity,total):
  self.path=Path(path);self.path.mkdir(parents=True,exist_ok=True);self.identity=identity;self.total=total;self.journal=self.path/'committed.jsonl';self.pending=self.path/'pending.json';self.cursor=0
  binding=self.path/'identity.json';value={'identity':identity,'total':total}
  if binding.exists():
   if json.loads(binding.read_text())!=value:raise ValueError('readout model/input identity changed')
  else:atomic_json(binding,value)
  if self.pending.exists():raise RuntimeError('pending sequence is ambiguous; stop rather than silently rerun')
  rows=[json.loads(x) for x in self.journal.read_text().splitlines()] if self.journal.exists() else []
  for i,row in enumerate(rows):
   if row['index']!=i or row['identity']!=identity or row['sha256']!=sha(self.file(i)):raise ValueError('readout committed prefix inconsistent')
  self.cursor=len(rows)
  if self.cursor>total:raise ValueError('readout exceeds declared sequence count')
 def file(self,index):return self.path/f'sequence{index:06d}.private.npz'
 def run_one(self,index,predict):
  if index!=self.cursor or index>=self.total:raise ValueError('claim must be exactly next committed cursor')
  atomic_json(self.pending,{'index':index,'identity':self.identity});prob=np.asarray(predict())
  if prob.ndim!=2 or prob.shape[1]!=5 or not len(prob) or not np.isfinite(prob).all() or (prob<0).any() or not np.allclose(prob.sum(1),1,atol=1e-5):raise ValueError('complete finite five-class probabilities required')
  path=self.file(index);temp=path.with_suffix(path.suffix+'.tmp')
  with temp.open('wb') as f:np.savez(f,prob=prob.astype(np.float32));f.flush();os.fsync(f.fileno())
  temp.replace(path);row={'index':index,'identity':self.identity,'sha256':sha(path),'positions':len(prob)}
  with self.journal.open('a') as f:f.write(json.dumps(row)+'\n');f.flush();os.fsync(f.fileno())
  self.pending.unlink();self.cursor+=1
 def load(self,index):
  if not 0<=index<self.cursor:raise ValueError('uncommitted sequence')
  with np.load(self.file(index)) as z:return z['prob'].copy()
