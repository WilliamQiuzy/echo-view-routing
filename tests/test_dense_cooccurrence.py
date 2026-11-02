import numpy as np
from echo_routing.temporal.dense_cooccurrence import descriptor, projection

def test_channel_shifts_preserve_means_and_diagonal():
 t=np.random.default_rng(3).normal(size=(256,384));p=projection();a=descriptor(t,p,False);b=descriptor(t,p,True)
 np.testing.assert_allclose(a[:384],b[:384],atol=1e-13)
 ids=np.triu_indices(32);np.testing.assert_allclose(a[384:][ids[0]==ids[1]],b[384:][ids[0]==ids[1]],atol=1e-13)
 assert a.shape==b.shape==(912,) and np.max(abs(a[384:]-b[384:]))>1e-3

def test_projection_and_descriptor_deterministic_and_reconstruction():
 p=projection();np.testing.assert_array_equal(p,projection());np.testing.assert_allclose(p.T@p,np.eye(32),atol=1e-12)
 t=np.random.default_rng(4).normal(size=(256,384));z=t@p;z-=z.mean(0);cov=sum(np.outer(row,row) for row in z)/256
 np.testing.assert_allclose(descriptor(t,p,False)[384:],cov[np.triu_indices(32)],atol=1e-13)

def test_invalid_tokens_rejected():
 import pytest
 for t in [np.zeros((0,384)),np.zeros((256,383)),np.full((256,384),np.nan)]:
  with pytest.raises(ValueError):descriptor(t,projection(),False)
