"""Stream counts and whole-cluster bootstrap. Matching uses the project's IoU50 class-aware one-to-one functions."""
import numpy as np

from echo_routing.evaluate.failure_units import match_changes, match_segments
from echo_routing.temporal.segments import labels_to_segments

A4C, A5C = 2, 3
FIELDS = ('M', 'P', 'T', 'n', 'correct', 'change_true', 'change_pred', 'change_matched',
          'short10_true', 'short10_fn', 'short10_pred', 'short10_fp',
          'short16_true', 'short16_fn', 'short16_pred', 'short16_fp', 'a4c_to_a5c', 'a5c_to_a4c')


def merge_apical(labels):
    """A5C becomes A4C. Used only for the merged-apical readout."""
    return np.where(np.asarray(labels) == A5C, A4C, labels)


def _short(true, pred, matched, used, maximum):
    return {f'short{maximum}_true': sum(t.end - t.start <= maximum for t in true),
            f'short{maximum}_fn': sum(t.end - t.start <= maximum and i not in matched for i, t in enumerate(true)),
            f'short{maximum}_pred': sum(p.end - p.start <= maximum for p in pred),
            f'short{maximum}_fp': sum(p.end - p.start <= maximum and i not in used for i, p in enumerate(pred))}


def stream_counts(truth, pred, near=10):
    truth, pred = np.asarray(truth), np.asarray(pred)
    if truth.shape != pred.shape or truth.ndim != 1 or not truth.size:
        raise ValueError('aligned nonempty one-dimensional label axes required')
    true_seg, pred_seg = labels_to_segments(truth), labels_to_segments(pred)
    matched = match_segments(pred_seg, true_seg)
    used = set(matched.values())
    true_change = np.flatnonzero(np.diff(truth)) + 1
    pred_change = np.flatnonzero(np.diff(pred)) + 1
    row = {'M': len(matched), 'P': len(pred_seg), 'T': len(true_seg), 'n': int(truth.size),
           'correct': int((truth == pred).sum()), 'change_true': int(true_change.size),
           'change_pred': int(pred_change.size), 'change_matched': len(match_changes(pred_change, true_change, near)),
           'a4c_to_a5c': int(((truth == A4C) & (pred == A5C)).sum()),
           'a5c_to_a4c': int(((truth == A5C) & (pred == A4C)).sum())}
    return {**row, **_short(true_seg, pred_seg, matched, used, 10), **_short(true_seg, pred_seg, matched, used, 16)}


def segment_f1(counts):
    return 2 * counts['M'] / (counts['P'] + counts['T'])


def as_rows(counts_list):
    return np.array([[c[f] for f in FIELDS] for c in counts_list], float)


def pooled(rows):
    """Readable pooled summary from a rows array."""
    total = dict(zip(FIELDS, rows.sum(0).tolist()))
    ratio = lambda a, b: a / b if b else None
    return {**{k: int(v) for k, v in total.items()},
            'segment_f1': ratio(2 * total['M'], total['P'] + total['T']),
            'frame_accuracy': ratio(total['correct'], total['n']),
            'change_recall': ratio(total['change_matched'], total['change_true']),
            'change_precision': ratio(total['change_matched'], total['change_pred'])}


def f1_statistic(sums):
    """Segment F1 from summed rows; field order is FIELDS."""
    return 2 * sums[..., 0] / (sums[..., 1] + sums[..., 2])


def field_statistic(name):
    index = FIELDS.index(name)
    return lambda sums: sums[..., index]


def ratio_statistic(top, bottom):
    a, b = FIELDS.index(top), FIELDS.index(bottom)
    return lambda sums: sums[..., a] / sums[..., b]


def macro_f1(confusion):
    """Macro-F1 over classes with truth or predictions; accepts (..., C, C). True classes are rows."""
    c = np.asarray(confusion, float)
    tp = np.diagonal(c, axis1=-2, axis2=-1)
    denominator = c.sum(-1) + c.sum(-2)
    with np.errstate(divide='ignore', invalid='ignore'):
        f1 = np.where(denominator > 0, 2 * tp / denominator, np.nan)
    return np.nanmean(f1, axis=-1)


def _cluster_sums(rows, clusters):
    keys = {k: i for i, k in enumerate(sorted(set(clusters)))}
    out = np.zeros((len(keys), *np.asarray(rows).shape[1:]))
    for row, key in zip(np.asarray(rows, float), clusters):
        out[keys[key]] += row
    return out


def _weights(k, n_boot, seed):
    return np.random.default_rng(seed).multinomial(k, np.full(k, 1 / k), size=n_boot).astype(float)


def paired_replicates(a, b, clusters, statistic, n_boot=5000, seed=2026110399):
    """statistic(a) - statistic(b) over whole-cluster bootstrap draws; both arms share every draw."""
    if len(a) != len(b) or len(a) != len(clusters):
        raise ValueError('one row per cluster member required in each arm')
    sum_a, sum_b = _cluster_sums(a, clusters), _cluster_sums(b, clusters)
    w = _weights(len(sum_a), n_boot, seed)
    with np.errstate(divide='ignore', invalid='ignore'):
        return statistic(np.tensordot(w, sum_a, axes=1)) - statistic(np.tensordot(w, sum_b, axes=1))


def paired_delta(a, b, clusters, statistic, n_boot=5000, seed=2026110399):
    values = paired_replicates(a, b, clusters, statistic, n_boot, seed)
    finite = values[np.isfinite(values)]
    with np.errstate(divide='ignore', invalid='ignore'):
        delta = statistic(np.asarray(a, float).sum(0)) - statistic(np.asarray(b, float).sum(0))
    return {'delta': float(delta), 'ci95': np.quantile(finite, [.025, .975]).tolist(),
            'undefined_replicates': int(values.size - finite.size), 'n_clusters': len(set(clusters))}


def unpaired_ci(rows, clusters, statistic, n_boot=5000, seed=2026110399):
    """Percentile interval of one arm under whole-cluster resampling."""
    sums = _cluster_sums(rows, clusters)
    with np.errstate(divide='ignore', invalid='ignore'):
        values = statistic(np.tensordot(_weights(len(sums), n_boot, seed), sums, axes=1))
    return np.quantile(values[np.isfinite(values)], [.025, .975]).tolist()
