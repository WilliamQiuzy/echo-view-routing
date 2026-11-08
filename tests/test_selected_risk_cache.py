import importlib,json
import numpy as np
import pytest
from echo_routing.temporal.proposal_actions import propose,partition

def mod():
 try:m=importlib.import_module('echo_routing.temporal.selected_risk_cache')
 except ModuleNotFoundError:m=None
 assert m is not None,'A63 label-free joint D/X supply cache absent'
 return m

def test_complete_empty_and_nonempty_joint_prefix_no_replay(tmp_path):
 m=mod();b=np.zeros(8,np.int64);j=b.copy();j[3]=1;actions,_=partition(b,propose(b,b,j));c=m.SupplyCache(tmp_path/'cache','frozen',[b,b],[b,b],[j,b]);calls=[]
 def predict():
  calls.append(1);return {'D_C6':np.array([[1.,.2,.7,.1,.1,.2]]),'X_C6':np.array([[2.,.1,.8,.1,.1,.1]]),'x109':np.zeros((1,109)),'context96':np.zeros((1,96))}
 c.run_one(0,predict);assert c.load(0)['D']['N_admitted'] and c.load(0)['X']['N_admitted']
 with pytest.raises(ValueError):m.SupplyCache(tmp_path/'cache','frozen',[b,b],[b,b],[j,b],readonly=True)
 c=m.SupplyCache(tmp_path/'cache','frozen',[b,b],[b,b],[j,b]);c.run_one(1,lambda:{'D_C6':np.empty((0,6)),'X_C6':np.empty((0,6)),'x109':np.empty((0,109)),'context96':np.empty((0,96))})
 assert m.SupplyCache(tmp_path/'cache','frozen',[b,b],[b,b],[j,b],readonly=True).load(1)['D']['top_index'] is None and len(calls)==1
 (tmp_path/'cache/pending.json').write_text('{}')
 with pytest.raises(ValueError):m.SupplyCache(tmp_path/'cache','frozen',[b,b],[b,b],[j,b])

def test_invalid_gross_probability_mass_fails_before_commit(tmp_path):
 m=mod();b=np.zeros(8,np.int64);j=b.copy();j[3]=1;c=m.SupplyCache(tmp_path/'cache','frozen',[b],[b],[j])
 with pytest.raises(ValueError):c.run_one(0,lambda:{'D_C6':np.array([[1.,2.,0.,0.,.1,.2]]),'X_C6':np.array([[1.,.2,.7,.1,.1,.2]]),'x109':np.zeros((1,109)),'context96':np.zeros((1,96))})
 assert c.cursor==0 and c.pending.exists()
