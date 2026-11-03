"""Independent entry-point recount of the actual FIT artifact before optimization."""
import numpy as np

from echo_routing.evaluate.metrics_segmentation import segment_f1
from echo_routing.temporal.segments import labels_to_segments
from echo_routing.temporal.anchor_edits import components
from echo_routing.temporal.anchor_training import draw_plan


def metric_counts(prediction, truth):
    p, t = labels_to_segments(prediction), labels_to_segments(truth)
    return segment_f1(p, t, .5)['matched'], len(p), len(t)


def validate(data, sequences, a08, baselines, normalizers, plans):
    if not len(sequences) == len(a08) == len(baselines):
        raise ValueError('whole fixed FIT axes differ')
    pooled = np.asarray([metric_counts(b, q['labels']) for q, b in zip(sequences, baselines) if q['kind'] != 'native']).sum(0)
    fit_f1 = 2 * pooled[0] / (pooled[1] + pooled[2])
    if data['F_fit'] != fit_f1:
        raise ValueError('actual FIT pooled operating point did not recount')
    expected, targets, deltas = [], [], []
    for qi, (q, p, b) in enumerate(zip(sequences, a08, baselines)):
        raw = p.argmax(1)
        before = metric_counts(b, q['labels'])
        for span in components(b, raw):
            edited = np.array(b, copy=True)
            edited[span[0]:span[1]] = raw[span[0]:span[1]]
            after = metric_counts(edited, q['labels'])
            delta_m, delta_p = after[0] - before[0], after[1] - before[1]
            targets.append(2 * delta_m - fit_f1 * delta_p)
            expected.append((qi, span, 'native' if q['kind'] == 'native' else 'constructed'))
            deltas.append((delta_m, delta_p))
    rows = data['candidate_rows']
    if [(r['sequence_index'], tuple(r['span']), r['kind']) for r in rows] != expected:
        raise ValueError('complete candidate row order/domain changed')
    if [(r['delta_M'], r['delta_P']) for r in rows] != deltas:
        raise ValueError('candidate full-sequence matching deltas changed')
    if data['targets'].dtype != np.float32 or not np.array_equal(data['targets'], np.asarray(targets, np.float32)):
        raise ValueError('actual MSE target values did not independently recount')
    for kind in ['native', 'constructed']:
        indices = np.asarray([i for i, row in enumerate(expected) if row[2] == kind], np.int64)
        if not np.array_equal(data['kind_indices'][kind], indices):
            raise ValueError('eligible-kind candidate indices changed')
    for arm in ['A', 'J']:
        features = data['features'][arm]
        if features.dtype != np.float64 or features.shape != (len(rows), 98) or not np.isfinite(features).all():
            raise ValueError('raw feature matrix dtype/axis/finite contract changed')
        mean, std = features.mean(0), np.maximum(features.std(0), 1e-6)
        if not np.array_equal(mean, normalizers[arm]['mean']) or not np.array_equal(std, normalizers[arm]['std']):
            raise ValueError('FIT-only population normalizer differs from complete feature artifact')
    a, j = data['features']['A'], data['features']['J']
    if not np.array_equal(a[:, :62], j[:, :62]) or not np.array_equal(a[:, 26:62], a[:, 62:]):
        raise ValueError('matched first62/A-duplicate feature contract changed')
    for seed, plan in plans.items():
        wanted = draw_plan(len(data['kind_indices']['native']), len(data['kind_indices']['constructed']), int(seed))
        if any(not np.array_equal(plan[k], wanted[k]) for k in ['native', 'constructed']) or plan['no_action_domain'] != wanted['no_action_domain']:
            raise ValueError('actual pre-frozen shared kind-balanced draw plan changed')
    return {'all_candidate_targets_independently_recounted': True, 'candidate_count': len(rows),
            'F_fit': float(fit_f1), 'FIT_operating_counts': {'M': int(pooled[0]), 'P': int(pooled[1]), 'T': int(pooled[2])},
            'complete_normalizers_and_seed_plans_recounted': True,
            'targets_positive_zero_negative': [int(np.count_nonzero(data['targets'] > 0)),
                                             int(np.count_nonzero(data['targets'] == 0)),
                                             int(np.count_nonzero(data['targets'] < 0))]}
