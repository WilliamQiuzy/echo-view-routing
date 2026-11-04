import importlib,numpy as np


def test_rle_work_is_observable_and_keeps_all_candidate_volumes():
    m=importlib.import_module('echo_routing.temporal.safety_rle_cost');T=31;b=np.zeros(T,np.int64);a=b.copy();a[1::2]=1;j=2+np.arange(T)%2
    g=m.geometry(b,a,j)
    assert g['T']==T and g['C']>0 and g['CR']>=2*g['C'] and g['TC']==T*g['C'] and g['C2']==g['C']**2
    assert g['all_C']>=g['C'] and g['max_C']==g['C']
    assert m.pair_kernel(16)['informative_pair_operations']>100
