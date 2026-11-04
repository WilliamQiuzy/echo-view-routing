import importlib
import inspect
import numpy as np
import pytest
from echo_routing.temporal.proposal_actions import Action,apply
from echo_routing.temporal.proposal_features import candidate_features


def api(name):return importlib.import_module('echo_routing.temporal.safety_'+name)


def test_count_proxy_rewards_cut_removal_with_gross_harm_and_safety_target_rejects():
    m=api('targets');b=np.r_[np.full(20,2),np.full(18,3)];y=np.full(38,3)
    a=Action(20,38,tuple([2]*18),jepa=True);r=m.effects(b,a,y,.8917446116307443)
    assert r['delta_P']==-1 and r['delta_M']==0 and r['u']==.8917446116307443
    assert r['H']==18 and r['R']==0 and r['dE']==18
    assert not r['safe'] and r['v']==-1-r['u']


def test_gross_repair_cannot_cancel_harm_and_positions_exclusive():
    m=api('targets');b=np.array([0,1,2,3]);y=np.array([0,0,2,2]);a=Action(0,4,(1,0,2,4),jepa=True)
    # changed 0:H,1:R,3:neutral; unchanged interior position2 is excluded.
    labels=m.position_labels(b,a,y)
    assert labels.tolist()==[0,1,2]
    r=m.effects(b,a,y,.5)
    assert r['H']==1 and r['R']==1 and r['dE']==0 and not r['safe'] and r['v']<0
    safe=m.effects(np.zeros(6,int),Action(2,4,(1,1)),np.array([0,0,1,1,0,0]),.5)
    assert safe['safe'] and safe['v']==safe['u']==5 and safe['dFP']==-1
    harmful=m.effects(np.zeros(20,int),Action(10,11,(1,)),np.zeros(20,int),.5)
    assert harmful['dFP']==2 # full-axis matching, not changed singleton only


def test47_only_changed_positions_and_observable_flanks_K_not_interval_length():
    m=api('features');a=np.tile([.6,.1,.1,.1,.1],(40,1));j=np.tile([.1,.2,.3,.2,.2],(40,1))
    b=np.zeros(40,int);replacement=tuple([1]+[0]*37+[1]);action=Action(0,39,replacement)
    x=m.position_features(a,j,b,action)
    assert x.shape==(2,47) and x.dtype==np.float64
    assert np.allclose(x[:,:5],a[[0,38]]) and np.allclose(x[:,5:10],j[[0,38]])
    assert np.array_equal(x[0,20:25],np.zeros(5)) and np.allclose(x[0,25:30],a[1:17].mean(0))
    assert np.array_equal(x[:,40:42],[[0,1],[1,1]])
    assert np.allclose(x[:,42:45],np.log1p([40,39,2]))
    assert np.allclose(x[:,45],np.array([.5,38.5])/39)
    assert np.allclose(x[:,46],np.array([.5,38.5])/40)
    assert np.array_equal(m.utility_features(a,j,b,action),candidate_features(a,j,b,action,'J'))
    assert list(inspect.signature(m.position_features).parameters)==['prob_a','prob_j','base','action']
    norm=m.fit_position_normalizer(x);assert m.normalize_positions(x,norm).shape==(2,47)
    with pytest.raises(ValueError):m.normalize_positions(x,{'mean':np.zeros(47),'std':np.zeros(47)})
