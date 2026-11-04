import importlib
import numpy as np
import pytest


def test_stress_reuses_native_pairs_exactly_and_checks_original_labels():
    m=importlib.import_module('echo_routing.temporal.safety_tasks')
    native={'id':'native','kind':'native','labels':np.array([0,0])};pair={'id':'pair','kind':'pair','labels':np.array([0,1])};old={'id':'oldmulti','kind':'multi','labels':np.array([1,1])};fresh={'id':'fresh','kind':'multi','labels':np.array([2,2])}
    n,p,o,f=object(),object(),object(),object()
    out=m.compose([native,pair,old],([n,p,o],),[fresh],([f],),[pair,native,fresh])
    assert [q['id'] for q in out[0]]==['pair','native','fresh'] and out[1]==[p,n,f]
    with pytest.raises(ValueError):m.compose([native,pair,old],([n,p,o],),[fresh],([f],),[{**native,'labels':np.array([1,1])},pair,fresh])
    with pytest.raises(ValueError):m.compose([native,pair,old],([n,p,o],),[native],([f],),[native,pair])
