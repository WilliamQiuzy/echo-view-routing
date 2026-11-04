import numpy as np
from scripts.probe_action_safety_role_costs import synthetic,timed_role


def test_synthetic_truth_and_actual_geometry_cost_path(tmp_path):
    assert len(synthetic(9,1))==9 and (np.count_nonzero(np.diff(synthetic(701,2)))+1)<=8
    base=[np.array([0,0,0,0]),np.array([0,0,0,0])];pa=[np.eye(5)[[0,1,0,0]],np.eye(5)[base[1]]];pj=[np.eye(5)[[0,2,0,0]],np.eye(5)[base[1]]]
    class NoReference(dict):
        def __getitem__(self,key):
            if key=='labels':raise AssertionError('actual GT used')
            return super().__getitem__(key)
    out=timed_role('tiny',[NoReference(kind='native',refs=[('x',i,1.) for i in range(4)],labels='forbidden'),NoReference(kind='native',refs=[('y',i,1.) for i in range(4)],labels='forbidden')],pa,pj,base,tmp_path,.8)
    assert out['N']==2 and out['T']==8 and len(out['all6_descriptor_seconds'])==6 and len(out['all6_oracle_seconds'])==6


def test_fixed_fit_numeric_and_structural_truth_run_ceiling():
    import pytest
    from scripts.probe_action_safety_role_costs import operating_point,truth_run_bound
    assert operating_point({'F_fit':.8917446116307443})==.8917446116307443
    with pytest.raises(ValueError):operating_point({'F_fit':'descriptive'})
    q={'kind':'multi','refs':[(f'clip{i//2}',i,1.) for i in range(16)],'labels':'must_not_read'}
    assert truth_run_bound(q)==8
    with pytest.raises(ValueError):truth_run_bound({**q,'refs':[(f'clip{i}',i,1.) for i in range(9)]})
    with pytest.raises(ValueError):truth_run_bound({**q,'kind':'native'})
