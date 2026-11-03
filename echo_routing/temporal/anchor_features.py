"""A59's fixed 26 common plus two 36-posterior feature blocks; no truth inputs."""
import numpy as np

from echo_routing.temporal.anchor_edits import aligned, apply_action, predicted_count


def probabilities(value):
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 5 or not len(value) or not np.isfinite(value).all() or (value < 0).any() or not np.allclose(value.sum(1), 1., rtol=0, atol=1e-5):
        raise ValueError('nonempty finite normalized original-axis five-class probabilities required')
    return value


def regions(length, span):
    start, end = span
    if int(start) != start or int(end) != end or not 0 <= start < end <= length:
        raise ValueError('nonempty candidate inside complete unpadded axis required')
    return [(max(0, start - 16), start), (start, end), (end, min(length, end + 16))]


def source_features(prob, span):
    prob = probabilities(prob)
    rows = []
    for start, end in regions(len(prob), span):
        region = prob[start:end]
        if not len(region):
            rows.extend(np.zeros(12))
            continue
        absolute_difference = np.abs(np.diff(region, axis=0)).mean(0) if len(region) > 1 else np.zeros(5)
        entropy = -np.sum(region * np.log(np.maximum(region, np.finfo(np.float64).tiny)), axis=1).mean()
        sorted_prob = np.sort(region, axis=1)
        margin = (sorted_prob[:, -1] - sorted_prob[:, -2]).mean()
        rows.extend([*region.mean(0), *absolute_difference, entropy, margin])
    return np.asarray(rows, dtype=np.float64)


def candidate_features(prob_a, prob_j, base, span, arm):
    prob_a, prob_j = probabilities(prob_a), probabilities(prob_j)
    if prob_a.shape != prob_j.shape or arm not in ['A', 'J']:
        raise ValueError('matched A/J original axes and declared feature arm required')
    raw = prob_a.argmax(1)
    base, raw = aligned(base, raw)
    edited = apply_action(base, raw, span)
    start, end = span
    bounds = regions(len(base), span)
    onehot = np.eye(5)
    common = [*np.log1p([b - a for a, b in bounds]),
              *onehot[base[start:end]].mean(0), *onehot[raw[start:end]].mean(0),
              *(onehot[base[start - 1]] if start else np.zeros(5)),
              *(onehot[base[end]] if end < len(base) else np.zeros(5)),
              int(start > 0), int(end < len(base)), predicted_count(edited) - predicted_count(base)]
    left = source_features(prob_a, span)
    return np.concatenate([common, left, left if arm == 'A' else source_features(prob_j, span)])


def feature_matrix(value):
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 98 or not len(value) or not np.isfinite(value).all():
        raise ValueError('nonempty finite N by 98 fixed feature matrix required')
    return value


def fit_normalizer(fit_features):
    value = feature_matrix(fit_features)
    return {'mean': value.mean(0), 'std': np.maximum(value.std(0), 1e-6)}


def normalize(features, normalizer):
    value = feature_matrix(features)
    mean, std = np.asarray(normalizer['mean']), np.asarray(normalizer['std'])
    if mean.shape != (98,) or std.shape != (98,) or not np.isfinite(mean).all() or not np.isfinite(std).all() or (std < 1e-6).any():
        raise ValueError('frozen finite 98-feature FIT normalizer required')
    return ((value - mean) / std).astype(np.float32)
