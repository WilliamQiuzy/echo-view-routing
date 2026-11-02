import importlib
import numpy as np
import pytest

def api():return importlib.import_module('echo_routing.temporal.dense_readout')
def test_three_sequences_partial_prefix_resume_has_no_duplicate_forward(tmp_path):
 a=api();r=a.SequenceReadout(tmp_path,'fixedmodel-and-input',3);calls=[]
 def predict(i):calls.append(i);return np.full((i+1,5),.2,np.float32)
 r.run_one(0,lambda:predict(0));r.run_one(1,lambda:predict(1));resumed=a.SequenceReadout(tmp_path,'fixedmodel-and-input',3)
 assert resumed.cursor==2
 for i in range(resumed.cursor,3):resumed.run_one(i,lambda i=i:predict(i))
 assert calls==[0,1,2] and resumed.cursor==3 and resumed.load(1).shape==(2,5)
 with pytest.raises(ValueError):resumed.run_one(0,lambda:predict(0))
def test_cursor_binds_identity_tamper_and_pending_is_ambiguous_stop(tmp_path):
 a=api();r=a.SequenceReadout(tmp_path,'identity',2);r.run_one(0,lambda:np.full((2,5),.2))
 with pytest.raises(ValueError):a.SequenceReadout(tmp_path,'other',2)
 (tmp_path/'pending.json').write_text('{"index":1}')
 with pytest.raises(RuntimeError):a.SequenceReadout(tmp_path,'identity',2)
 (tmp_path/'pending.json').unlink();(tmp_path/'sequence000000.private.npz').write_bytes(b'tampered')
 with pytest.raises(ValueError):a.SequenceReadout(tmp_path,'identity',2)
