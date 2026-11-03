import copy,importlib,math
import numpy as np
import pytest
from echo_routing.temporal.dev_train import lr_factor

def test_physical_trace_requires_exact_plan_lr_positions_and_complete_live_gradients():
    m=importlib.import_module('echo_routing.temporal.structured_trace');plan=np.tile([0,1],(4,1));names=['first','last'];rows=[{'u':u,'identity':'fixed','plan_row':[0,1],'finite_gradients':True,'loss':1.,'native_NLL':1.,'constructed_NLL':1.,'gradient_norm':1.,'lr':.001*lr_factor(u,100,6000),'physical_sequences':2,'valid_original_positions':20,'nonzero_gradient_tensors':{n:u>0 for n in names},'model_digest_after_update':'digest'} for u in range(4)]
    assert m.audit_trace(rows,plan,'fixed',[9,11],names,expected_updates=4)['physical_optimizer_updates']==4
    for mutation in [lambda r:r[2].update(lr=.001),lambda r:r[2].update(plan_row=[1,0]),lambda r:r[2].update(valid_original_positions=19),lambda r:r[2].update(loss=float('nan')),lambda r:r[-1]['nonzero_gradient_tensors'].update(first=False)]:
        bad=copy.deepcopy(rows);mutation(bad)
        with pytest.raises(ValueError):m.audit_trace(bad,plan,'fixed',[9,11],names,expected_updates=4)
