import importlib
import numpy as np
import pytest

def test_position_bins_distinguish_stable_flicker_and_labeled_change():
    a=importlib.import_module('echo_routing.temporal.dense_bottleneck');gt=np.array([2]*20+[3]*20);pred=gt.copy();pred[3]=3;pred[20]=2;p=np.eye(5)[pred]*.8+.04
    r=a.position_diagnostics(gt,pred,p)
    assert r['near_true_change_5samples']['positions']==11 and r['stable_interior']['positions']==29
    assert r['near_true_change_5samples']['errors']==1 and r['stable_interior']['errors']==1
    assert r['predicted_cuts_far_from_true_change']==2
    with pytest.raises(ValueError):a.position_diagnostics(gt,pred,p[:-1])

def test_four_correctness_cells_and_class_stats_keep_denominators():
    a=importlib.import_module('echo_routing.temporal.dense_bottleneck')
    assert a.correctness_cells(np.array([1,1,0,0],bool),np.array([1,0,1,0],bool))=={'both_correct':1,'candidate_only_correct':1,'A08_only_correct':1,'both_wrong':1,'denominator':4}
    cm=np.diag([2,3,4,5,6]);stats=a.class_stats(cm);assert [x['f1'] for x in stats]==[1.]*5 and sum(x['true'] for x in stats)==20
