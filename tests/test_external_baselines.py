import numpy as np

from echo_routing.evaluate.external_baselines import grouped_probabilities, stfm_probabilities


def test_coarse_families_sum_probabilities_and_keep_unsupported_mass():
    raw=np.array([[.25,.25,.4,.1],[.05,.05,.1,.8]])
    got=grouped_probabilities(raw,{0:1,1:1,2:2})
    assert got.shape==(2,6)
    np.testing.assert_allclose(got.sum(1),1)
    assert got.argmax(1).tolist()==[1,5]


def test_stfm_prior_once_is_distinct_from_legacy_per_key_prior():
    logits=np.array([[0.,2.],[0.,2.]])
    old=stfm_probabilities(logits,prior_once=False)
    corrected=stfm_probabilities(logits,prior_once=True)
    assert corrected[1]>old[1]
    assert old.sum()==1 and corrected.sum()==1
