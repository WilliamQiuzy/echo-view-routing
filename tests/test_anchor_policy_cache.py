import importlib

import numpy as np
import pytest

from echo_routing.temporal.anchor_edits import components, decode


def test_both_policies_commit_together_clean_resume_scores_and_labels_recount(tmp_path):
    m = importlib.import_module('echo_routing.temporal.anchor_policy_cache')
    bases = [np.array([0, 0, 0, 1, 1])] * 3
    raws = [np.array([1, 1, 0, 1, 0])] * 3
    calls = []
    def predict(i):
        calls.append(i)
        scores = np.array([2., 1.], np.float32)
        return {'scores': scores, 'spans': components(bases[i], raws[i]),
                **{p: decode(bases[i], raws[i], scores, p == 'G') for p in ['U', 'G']},
                'cached_feature_head_policy_seconds': .01}
    cache = m.PolicyPairCache(tmp_path, 'model+normalizer+sources+input+policy', bases, raws)
    for i in range(2):
        cache.run_one(i, lambda i=i: predict(i))
    reopened = m.PolicyPairCache(tmp_path, cache.identity, bases, raws)
    reopened.run_one(2, lambda: predict(2))
    assert calls == [0, 1, 2]
    out = reopened.load(0)
    assert out['U']['accepted'] == (0, 2) and out['G']['accepted'] == (4, 5)
    assert out['G']['vetoed'] == [(0, 2)]
    assert out['cached_feature_head_policy_seconds'] == .01
    read = m.PolicyPairCache(tmp_path, cache.identity, bases, raws, readonly=True)
    assert read.cursor == 3 and calls == [0, 1, 2]
    with pytest.raises(ValueError):
        read.run_one(3, lambda: predict(3))


def test_wrong_policy_output_or_spans_stops_with_durable_ambiguous_claim(tmp_path):
    m = importlib.import_module('echo_routing.temporal.anchor_policy_cache')
    base, raw = np.zeros(7, int), np.array([0, 0, 1, 1, 0, 0, 0])
    cache = m.PolicyPairCache(tmp_path, 'fixed', [base], [raw])
    scores = np.array([1.])
    result = {'scores': scores, 'spans': [(2, 4)],
              **{p: decode(base, raw, scores, p == 'G') for p in ['U', 'G']},
              'cached_feature_head_policy_seconds': .01}
    result['G']['prediction'] = np.ones(7, int)
    with pytest.raises(ValueError):
        cache.run_one(0, lambda: result)
    assert cache.pending.exists()
    with pytest.raises(ValueError):
        m.PolicyPairCache(tmp_path, 'fixed', [base], [raw])
