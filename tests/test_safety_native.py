import importlib
import numpy as np
from echo_routing.temporal.proposal_actions import Action
from echo_routing.temporal.safety_runtime import policies


def test_native_cube_gross_cost_and_predicted_gate_are_independently_bound():
    m=importlib.import_module('echo_routing.temporal.safety_native');b=np.r_[np.full(20,2),np.full(18,3)];q={'id':'synthetic','kind':'native','labels':np.full(38,3)};a=Action(20,38,tuple([2]*18),jepa=True);score=np.array([[1.,.1,17.8,.1,.01,.05]])
    v={**policies(b,[a],score,.2),'actions':[a],'score_matrix':score,'tau':.2}
    result=m.diagnose([q],[b],[v],'C','S-L',.8917446116307443)
    cube=np.asarray(result['true_Base_prediction_joint5cube'])
    assert cube.sum()==38 and result['Base_correct_to_wrong']==18 and result['Base_wrong_to_correct']==0
    assert result['frame_error_delta']==18 and result['changed_native_sequences']==1 and result['majority_changed_native_sequences']==0
    group=result['accepted_groups'][0]
    assert group['actual_H']==18 and group['actual_u']>0 and group['actual_v']<0
    assert group['newly_wrong_positions_extending_old_wrong_run']==18 and group['wrong_run_crossed_short16_to_long']==0
    assert group['predicted_H']<group['predicted_R'] and group['positive_safety_score_false_accepts']==1
