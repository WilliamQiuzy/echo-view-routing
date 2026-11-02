import importlib
import numpy as np
import pytest


def api():return importlib.import_module('scripts.diagnose_post_a56_roc')


def test_strict_threshold_ties_cannot_be_split():
    # The high tied group has one true positive and two false positives.
    y=np.array([1,0,0,1,0],bool);scores=np.array([3.,3.,3.,2.,1.])
    points=api().roc_points(y,scores)
    assert [(p['tp'],p['fp']) for p in points]==[(0,0),(0,0),(1,2),(2,2),(2,3)]
    result=api().budget_envelope(points,[0,1,2,3])
    assert [r['max_tp'] for r in result]==[0,0,2,2]
    assert result[2]['attained_fp']==2
    assert result[0]['admissible_thresholds']==2


def test_known_auc_endpoints_and_score_shift_invariance():
    a=api();y=np.array([0,0,1,1],bool);s=np.array([.1,.4,.35,.8])
    assert a.auc(y,s)==.75
    assert a.auc(y,np.ones(4))==.5
    assert a.auc(y,y.astype(float))==1.
    assert a.auc(y,-y.astype(float))==0.
    pts=a.roc_points(y,s);shift=a.roc_points(y,100+2*s)
    assert [(p['tp'],p['fp']) for p in pts]==[(p['tp'],p['fp']) for p in shift]
    assert pts[0]['threshold']==float('inf') and pts[-1]['threshold']==-float('inf')
    with pytest.raises(ValueError):a.roc_points(y,[1,float('nan'),2,3])


def test_original_zero_point_and_empty_budget_are_explicit():
    a=api();y=np.array([1,0,1,0],bool);s=np.array([0.,0.,-1.,1.])
    assert a.operating(y,s,0.)['tp']==0 and a.operating(y,s,0.)['fp']==1
    with pytest.raises(ValueError):a.budget_envelope(a.roc_points(y,s),[-1])
