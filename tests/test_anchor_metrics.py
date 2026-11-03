import importlib
from types import SimpleNamespace

import numpy as np

from echo_routing.temporal.dense_runtime import metric_stream
from echo_routing.temporal.structured_uncertainty import prepare_system


def test_pair_short_fp_fn_and_undefined_boundary_precision_remain_visible():
    m = importlib.import_module('echo_routing.temporal.anchor_metrics')
    def q(name, kind, y, joins):
        return {'id': name, 'kind': kind, 'labels': np.array(y), 'joins': np.array(joins, int),
                'refs': [(name, i, 1.) for i in range(len(y))]}
    native = q('n_0', 'native', [0] * 8, [])
    multi = q('m', 'multi', [0, 0, 1, 1, 0, 0, 0, 0], [2, 4])
    pair = q('p', 'pair', [0] * 4 + [1] * 4, [4])
    qs = [native, multi, pair]
    context = {'sequences': qs, 'eval_set': SimpleNamespace(natives=[SimpleNamespace(seq_id='n_0', labels=native['labels'])],
                multi=[metric_stream(multi)], pairs=[metric_stream(pair)])}
    left = {q['id']: q['labels'].copy() for q in qs}
    left['p'] = np.array([0, 2, 0, 0, 1, 1, 1, 1])
    right = {q['id']: np.zeros(8, int) for q in qs}
    lm, lt = m.evaluate(left, context)
    rm, rt = m.evaluate(right, context)
    assert lm['pairs_short_true_unmatched_16samples'] == 0
    assert rm['pairs_short_true_unmatched_16samples'] == 1
    assert lm['pairs_short_spurious_segments_16samples'] == 2
    assert rm['pairs_short_spurious_segments_16samples'] == 0
    assert lm['boundary_precision_pairs'] == 1 / 3 and rm['boundary_precision_pairs'] is None
    lp, rp = prepare_system(left, qs, lt), prepare_system(right, qs, rt)
    result = m.paired(lp, rp, n_boot=20, seed=2026110359)
    assert result['delta95']['pairs_short_true_unmatched_16samples'] == [-1., -1.]
    assert result['delta95']['pairs_short_spurious_segments_16samples'] == [2., 2.]
    assert result['delta95']['boundary_precision_pairs'] is None
    assert result['undefined_bootstrap_replicates']['boundary_precision_pairs'] == 20
