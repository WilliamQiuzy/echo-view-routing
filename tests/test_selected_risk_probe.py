import importlib
import numpy as np
from echo_routing.temporal.proposal_actions import propose,partition

def test_synthetic_profiles_retains_empty_longK_and_legal_full_envelope():
    try:m=importlib.import_module('scripts.probe_selected_risk')
    except ModuleNotFoundError:m=None
    assert m is not None,'prospective synthetic cost probe absent'
    for name in ['empty','envelope','legal_RLE','long_K']:
        b,a,j=m.synthetic_axis(48,name);assert b.shape==a.shape==j.shape==(48,)
        acts,_=partition(b,propose(b,a,j))
        if name=='empty':assert not acts
        else:assert acts
        if name=='long_K':assert max(np.count_nonzero(np.array(x.replacement)!=b[x.start:x.end]) for x in acts)>=24

def test_cost_stage_storage_keeps_closing_reserve_separate():
    import scripts.probe_selected_risk as m
    assert getattr(m,'stage_storage_upper',lambda x:x)(450000000)==450000000+300000000+64000000+128000000
