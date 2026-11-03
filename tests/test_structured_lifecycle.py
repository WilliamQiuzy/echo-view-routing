import importlib,copy
import pytest,torch
from echo_routing.temporal.structured_evidence import StructuredEvidence
from echo_routing.temporal import structured_training as training

def test_inner_selection_uses_declared_tolerance_native_shortFP_and_earlier_step():
    m=importlib.import_module('echo_routing.temporal.structured_lifecycle');rows=[{'step':s,'segment_f1_50_multi':.8,'native_macro_f1':.9,'short_spurious_segments_16samples':5} for s in [2000,4000,6000]];assert m.choose_inner(rows)['step']==2000
    rows[1]['segment_f1_50_multi']+=5e-13;rows[2]['short_spurious_segments_16samples']=4;assert m.choose_inner(rows)['step']==6000
    rows[0]['native_macro_f1']=.91;assert m.choose_inner(rows)['step']==2000
    rows[1]['segment_f1_50_multi']+=1e-9;assert m.choose_inner(rows)['step']==4000

def test_independent_checkpoint_moment_shape_finiteness_and_recipe_validation():
    m=importlib.import_module('echo_routing.temporal.structured_lifecycle');model=StructuredEvidence('D2',58);opt=training.optimizer(model);batch=[(torch.full((n,5),.2),torch.full((n,5),.2),torch.arange(n)%5) for n in [9,11]]
    for u in range(2):training.step(model,opt,batch,u)
    z=training.checkpoint(model,opt,2,'fixed');assert m.audit_payload(model,z,'fixed')['optimizer_tensors']==5
    for kind in ['negative_second_moment','nonfinite_first_moment','wrong_shape','missing_state','wrong_cursor']:
        bad=copy.deepcopy(z);state=next(iter(bad['optimizer']['state'].values()))
        if kind=='negative_second_moment':state['exp_avg_sq'].flatten()[0]=-1
        elif kind=='nonfinite_first_moment':state['exp_avg'].flatten()[0]=float('nan')
        elif kind=='wrong_shape':state['exp_avg_sq']=torch.ones(1)
        elif kind=='missing_state':bad['optimizer']['state'].pop(next(iter(bad['optimizer']['state'])))
        else:bad['update']=3
        with pytest.raises(ValueError):m.audit_payload(model,bad,'fixed')
