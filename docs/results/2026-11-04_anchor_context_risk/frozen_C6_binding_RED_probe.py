import subprocess,types,hashlib
import numpy as np
import pytest

def test_previous_snapshot_rejects_changed_valid_C6(tmp_path):
 source=subprocess.check_output(['git','show','5887afceb1ec02d5fe8f31b4102f1bf959982905:echo_routing/temporal/event_risk_cache.py'],text=True)
 module=types.ModuleType('old_event_cache');exec(compile(source,'frozen5887_cache.py','exec'),module.__dict__)
 b=np.zeros(5,int);raw=b.copy();raw[1]=1;old=np.array([[.2,.1,.9,0.,.1,.1]],np.float64);ref={'source':'frozen_sha_verified','sha256':'a'*64,'scores_sha256':hashlib.sha256(old.tobytes()).hexdigest()};taus={k:.5 for k in ['L0','M0','R1','R2']};cache=module.EventCache(tmp_path/'cache','identity',[b],[raw],[b],taus,[ref]);changed=old.copy();changed[0,0]=3
 with pytest.raises(ValueError):cache._check({'old_scores':changed,'q':np.array([[.1,.2]]),'old_reference':ref,'seconds':0.},0)
