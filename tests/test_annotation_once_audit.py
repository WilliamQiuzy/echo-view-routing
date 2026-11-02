import json
import numpy as np
import pytest
from scripts.audit_annotation_coverage_terminal import verify_cell,sha

def test_once_audit_rejects_pending_identity_and_rebound_scores(tmp_path):
 key="DC";identity={"model":"frozen","rows":133};scores=np.arange(133,dtype=float)
 np.save(tmp_path/(key+".npy"),scores)
 progress={"identity":identity,"n":133,"done":133,"pending":None}
 def write():
  (tmp_path/(key+".progress.json")).write_text(json.dumps(progress))
  (tmp_path/(key+".complete.json")).write_text(json.dumps({"identity":identity,"rows":133,"sha256":sha(tmp_path/(key+".npy"))}))
 write();assert verify_cell(tmp_path,key,identity,scores)==133
 progress["pending"]=132;write()
 with pytest.raises(AssertionError):verify_cell(tmp_path,key,identity,scores)
 progress["pending"]=None;write()
 with pytest.raises(AssertionError):verify_cell(tmp_path,key,{"model":"other","rows":133},scores)
 changed=scores.copy();changed[0]=9;np.save(tmp_path/(key+".npy"),changed);write()
 with pytest.raises(AssertionError):verify_cell(tmp_path,key,identity,scores)
