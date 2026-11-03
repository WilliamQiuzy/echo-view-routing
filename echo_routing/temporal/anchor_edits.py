"""A59 single raw-supported action and observable occupancy-majority constraint.

Inference functions accept only Base/raw labels and predicted utilities. The
explicitly supervised helpers below are for private training preparation only.
"""
import numpy as np

from echo_routing.evaluate.failure_units import match_segments
from echo_routing.temporal.segments import labels_to_segments


def labels(value):
    value = np.asarray(value)
    if value.ndim != 1 or not len(value) or value.dtype.kind not in 'iu' or (value < 0).any() or (value >= 5).any():
        raise ValueError('nonempty original-axis integer five-class labels required')
    return value


def aligned(base, raw):
    base, raw = labels(base), labels(raw)
    if base.shape != raw.shape:
        raise ValueError('Base/raw original axes differ')
    return base, raw


def components(base, raw):
    base, raw = aligned(base, raw)
    difference = np.r_[False, base != raw, False].astype(np.int8)
    return list(zip(np.flatnonzero(np.diff(difference) == 1).tolist(),
                    np.flatnonzero(np.diff(difference) == -1).tolist()))


def apply_action(base, raw, span):
    base, raw = aligned(base, raw)
    if tuple(span) not in components(base, raw):
        raise ValueError('action must be one complete maximal disagreement component')
    start, end = span
    prediction = base.copy()
    prediction[start:end] = raw[start:end]
    return prediction


def majority(value):
    return int(np.bincount(labels(value), minlength=5).argmax())


def decode(base, raw, scores, guarded=False):
    base, raw = aligned(base, raw)
    spans = components(base, raw)
    scores = np.asarray(scores, dtype=np.float64)
    if scores.shape != (len(spans),) or not np.isfinite(scores).all():
        raise ValueError('one finite predicted utility per ordered candidate required')
    order = sorted((i for i, score in enumerate(scores) if score > 0),
                   key=lambda i: (-scores[i], *spans[i]))
    vetoed, accepted, prediction = [], None, base.copy()
    for index in order:
        candidate = apply_action(base, raw, spans[index])
        if guarded and majority(candidate) != majority(base):
            vetoed.append(spans[index])
            continue
        accepted, prediction = spans[index], candidate
        break
    return {'prediction': prediction, 'accepted': accepted, 'vetoed': vetoed,
            'candidate_count': len(spans), 'positive_count': len(order)}


def counts(prediction, truth):
    prediction, truth = aligned(prediction, truth)
    pred_segments, true_segments = labels_to_segments(prediction), labels_to_segments(truth)
    return len(match_segments(pred_segments, true_segments, .5)), len(pred_segments), len(true_segments)


def predicted_count(value):
    value = labels(value)
    return 1 + int(np.count_nonzero(value[1:] != value[:-1]))


def utility(base, raw, span, truth, fit_f1):
    if not np.isfinite(fit_f1) or not 0 <= fit_f1 <= 1:
        raise ValueError('frozen pooled FIT operating point required')
    before = counts(base, truth)
    after = counts(apply_action(base, raw, span), truth)
    delta_m, delta_p = after[0] - before[0], after[1] - before[1]
    return {'utility': float(2 * delta_m - fit_f1 * delta_p),
            'delta_M': delta_m, 'delta_P': delta_p,
            'base_counts': before, 'edited_counts': after}
