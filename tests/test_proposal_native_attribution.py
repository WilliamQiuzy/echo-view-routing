import numpy as np
from echo_routing.temporal.proposal_native_attribution import joint_counts,effects

def test_native_joint_correct_wrong_accounting_and_count_proxy_blindspot():
    r=joint_counts(np.array([0,0,1]),np.array([0,1,0]),np.array([1,0,2]))
    assert r['Base_correct_to_wrong']==1 and r['Base_wrong_to_correct']==1 and r['both_wrong_changed_class']==1 and r['frame_error_delta']==0
    truth=np.zeros(40,int);base=truth.copy();base[:10]=1;pred=base.copy();pred[10:18]=1
    r=effects(base,pred,truth,.9)
    assert r['delta_M']==r['delta_P']==0 and r['u']==0 and r['dE']==8 and r['dFP16']==-1
    assert r['newly_wrong_positions_extending_old_wrong_run']==8 and r['newly_wrong_positions_new_wrong_segment']==0
    assert r['wrong_run_crossed_short16_to_long'] and r['added_cuts']==0
    base=truth.copy();pred=truth.copy();pred[15:20]=1;r=effects(base,pred,truth,.9)
    assert r['newly_wrong_positions_new_wrong_segment']==5 and r['added_cuts']==2 and r['dFP16']==2
