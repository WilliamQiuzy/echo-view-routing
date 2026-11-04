import importlib.util
from pathlib import Path
import pytest


def test_engineering_coverage_is_registered_not_caller_selected(tmp_path):
    from echo_routing.temporal.event_risk_engineering import stage_spec,validate_coverage
    entries,inputs=stage_spec('refinement_probe_freeze.json')
    assert entries==['scripts/probe_event_risk_refinement.py']
    assert 'docs/results/2026-11-04_anchor_context_risk/cost_profile.json' in inputs
    with pytest.raises(ValueError):stage_spec('anything_freeze.json')
    (tmp_path/'scripts').mkdir();(tmp_path/'scripts/probe_event_risk_refinement.py').write_text('')
    with pytest.raises(ValueError):validate_coverage(tmp_path,'refinement_probe_freeze.json',{},[])


def test_profiles_include_long_envelope_and_synthetic_truth_only(tmp_path):
    from scripts.probe_event_risk_refinement import synthetic_axis,profile_value
    import numpy as np
    for name in ['empty','envelope','legal_RLE','long_K']:
        b,a,j=synthetic_axis(37,name)
        assert len(b)==len(a)==len(j)==37
        if name=='legal_RLE':assert np.diff(np.flatnonzero(b[1:]!=b[:-1])+1).min()>=6
        v=profile_value(b,a,j)
        assert v['sequences'][0]['labels'].max()<5
        assert len(v['sequences'][0]['labels'])==37
    b,a,j=synthetic_axis(37,'envelope')
    assert np.unique(b).size==1
    assert len(profile_value(b,a,j)['actions'])>10


def test_complete_integer_solver_pool_is_synthetic_and_fixed_truth():
    from scripts.probe_event_risk_refinement import solver_pools
    from echo_routing.temporal.anchor_oracle import fractional_oracle
    pools=solver_pools(3,7)
    assert len(pools)==3 and all(len(p)==8 for p in pools)
    assert all(p[0]['span'] is None and {v['T'] for v in p}=={8} for p in pools)
    assert fractional_oracle(pools)['T']==24
