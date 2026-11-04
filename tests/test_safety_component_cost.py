import importlib
import numpy as np
from echo_routing.evaluate.failure_units import session_of
from echo_routing.temporal.safety_audit import validate_trace


def test_component_bench_synthetic_units_and_full_draw_metadata_have_no_optimizer():
    m=importlib.import_module('scripts.probe_action_safety_components');q,b,p,ctx=m.task(16)
    assert len(q)==640 and len({session_of(x['id']) for x in q if x['kind']=='native'})==320
    assert len(ctx['eval_set'].multi)==160 and len(ctx['eval_set'].pairs)==160
    _,base,a,j,value,g=m.case(16,False);assert g['C']>0 and value['score_matrix'].shape==(g['C'],6)
    data={'position_offsets':np.array([0,1,3]),'groups':[np.array([0]),np.array([1])],'targets':np.array([[0.,-1.],[1.,1.]]),'kind_groups':{'native':np.array([0]),'constructed':np.array([1])},'kind_indices':{'native':np.array([0]),'constructed':np.array([1])}}
    plan={'source_kinds':[0,1],'indices':np.zeros((6000,64),int)}
    for role in ['risk','utility']:
        rows=m.trace_rows(data,plan,role,2);r=validate_trace(rows,data,plan,role,'S-L' if role=='utility' else 'risk',expected_updates=2)
        assert r['updates']==2 and r['candidate_exposures']==128 and all(x['synthetic_cost_trace_no_optimizer_executed'] for x in rows)
