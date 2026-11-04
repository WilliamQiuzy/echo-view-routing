import importlib
import numpy as np
import pytest
from echo_routing.temporal.proposal_actions import Action


def api(name):return importlib.import_module('echo_routing.temporal.safety_'+name)


def test_same_top_first_no_fallback_empty_and_strict_probability_gates():
    m=api('policy');b=np.zeros(20,int);actions=[Action(2,3,(1,)),Action(5,6,(1,))]
    risk=[{'predicted_H':1.,'predicted_R':0.,'pFP':.1,'r':.4},{'predicted_H':0.,'predicted_R':1.,'pFP':.1,'r':.1}]
    assert m.decode(b,actions,[2.,1.],risk,'E')['accepted_index']==0
    assert m.decode(b,actions,[2.,1.],risk,'N')['accepted_index'] is None
    assert m.decode(b,actions,[2.,1.],risk,'C',1.)['accepted_index'] is None
    risk[0].update(predicted_H=0.,predicted_R=0.,pFP=.5)
    assert m.decode(b,actions,[2.,1.],risk,'N')['accepted_index'] is None
    risk[0].update(pFP=.1,r=.4)
    assert m.decode(b,actions,[2.,2.],risk,'N')['accepted_index']==0
    assert m.decode(b,actions,[2.,1.],risk,'C',.4)['accepted_index'] is None
    assert m.decode(b,actions,[2.,1.],risk,'C',.4001)['accepted_index']==0
    assert m.decode(b,actions,[0.,-1.],risk,'E')['top_index'] is None
    assert np.array_equal(m.decode(b,[],[],[],'C',0.)['prediction'],b)


def test_first_unsafe_calibration_ties_no_accept_and_saturation():
    m=api('policy');rows=[{'N_admitted':True,'r':r,'H':h,'dFP':f} for r,h,f in [(.2,0,0),(.4,1,0),(.4,0,1),(.6,0,0)]]
    result=m.calibrate(rows)
    assert result['tau']==.4 and result['unsafe_tied_at_threshold']==2 and result['C_accepted']==1 and result['C_unsafe']==0
    assert m.calibrate([])['tau']==0 and m.calibrate([{'N_admitted':False,'r':None}])['tau']==0
    assert m.calibrate([{'N_admitted':True,'r':1.,'H':0,'dFP':0}])['tau']==1
    with pytest.raises(ValueError):m.calibrate([{'N_admitted':True,'r':float('nan'),'H':0,'dFP':0}])


def test_exact314_240_60_56_graph_no_checkpoint_selection_and_historical_seed_mapping():
    m=api('readout_plan');p=m.plan()
    assert len(p['rows'])==314 and len(p['comparisons'])==240 and len(p['oracle_cells'])==60
    assert len(p['native_cells'])==56 and len(p['inner_policy_sets'])==56
    assert len({(x['table'],x['half'],x['system']) for x in p['rows']})==314
    for x in p['comparisons']:
        existing={r['system'] for r in p['rows'] if (r['table'],r['half'])==(x['table'],x['half'])}
        assert x['left'] in existing and x['right'] in existing
    hist=[r for r in p['rows'] if r['role']=='A60_historical']
    assert len(hist)==40 and {r['old_seed'] for r in hist}=={2026110360,2026110361}
    assert all(r['old_seed']-2026110360==r['replicate_seed']-2026110461 for r in hist)
    assert not p['checkpoint_selection'] and all(r['guard']=='G' for r in p['oracle_cells'])
