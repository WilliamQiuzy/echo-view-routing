import importlib
import numpy as np
import pytest
import torch

class Reader:
    def __init__(self,values,cursor=None):self.values=values;self.cursor=len(values) if cursor is None else cursor;self.identity='source'
    def load(self,i):return self.values[i]

def test_full_cached_axis_and_native_mixed_plan_roles_are_enforced():
    m=importlib.import_module('echo_routing.temporal.structured_bank');qs=[{'id':str(i),'kind':k,'labels':np.arange(n)%5,'refs':[None]*n} for i,(k,n) in enumerate([('native',9),('mixed',11)])];p=[np.full((len(q['labels']),5),.2,np.float32) for q in qs]
    bank=m.PosteriorBank(qs,Reader(p),Reader(p));batch=bank.batch([0,1]);assert len(batch)==2 and batch[0][0].shape==(9,5) and batch[1][2].dtype==torch.int64
    with pytest.raises(ValueError):bank.batch([1,0])
    with pytest.raises(ValueError):m.PosteriorBank(qs,Reader(p,cursor=1),Reader(p))
    bad=[p[0],np.full((10,5),.2,np.float32)]
    with pytest.raises(ValueError):m.PosteriorBank(qs,Reader(bad),Reader(p))

def test_cached_probabilities_and_label_axis_cannot_silently_change():
    m=importlib.import_module('echo_routing.temporal.structured_bank');qs=[{'id':'q','kind':'native','labels':np.array([0,1]),'refs':[None,None]}];p=[np.full((2,5),.2,np.float32)];bad=[p[0].copy()];bad[0][0,0]=float('nan')
    with pytest.raises(ValueError):m.PosteriorBank(qs,Reader(bad),Reader(p))
    qs[0]['labels']=np.array([0,5])
    with pytest.raises(ValueError):m.PosteriorBank(qs,Reader(p),Reader(p))
