import importlib
import numpy as np
import pytest

def mod():
 try:m=importlib.import_module('echo_routing.temporal.selected_risk_fit')
 except ModuleNotFoundError:m=None
 assert m is not None,'A63 complement-only all-axis Ffit data preparer missing'
 return m

def test_fold_Ffit_includes_native_and_mixed_not_predecessor_multi_only():
 m=mod();b=np.zeros(12,np.int64);a=np.full((12,5),.025);a[:,0]=.9;j1=a.copy();j1[5]=[.025,.9,.025,.025,.025];j2=a.copy();j2[5:7]=[.025,.025,.025,.9,.025]
 qs=[{'id':'own_0','kind':'native','labels':b.copy(),'refs':[('own_0',i,1.) for i in range(12)]},{'id':'axis','kind':'multi','labels':np.array([0]*5+[3]*2+[0]*5),'refs':[('own_1',i,1.) for i in range(12)]}]
 out=m.prepare_aux_training(qs,[a,a],[j1,j2],[b,b],{'own'},{'held'})
 assert out['FIT_operating_counts']=={'M':1,'P':2,'T':4} and out['F_fit']==1/3
 assert out['targets'][1,0]==pytest.approx(6-2/3)
 assert len(out['kind_indices']['native']) and len(out['kind_indices']['constructed'])
 # Training source guard must precede any label-dependent target/normalizer work.
 qs[1]['refs'][0]=('held_0',0,1.)
 with pytest.raises(ValueError):m.prepare_aux_training(qs,[a,a],[j1,j2],[b,b],{'own'},{'held'})
