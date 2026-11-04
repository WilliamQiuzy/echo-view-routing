"""A62 label-free full-anchor context96 and matched event-model input205.

The original109 block and its normalizer are never refitted here. R1's extra
96 inputs duplicate existing information; only R2 receives the context block.
"""
import numpy as np

from echo_routing.temporal.anchor_edits import labels
from echo_routing.temporal.anchor_features import probabilities
from echo_routing.temporal.proposal_actions import apply
from echo_routing.temporal.proposal_features import normalize
from echo_routing.temporal.baselines.b3_js_divergence import js_divergence
from echo_routing.temporal.segments import labels_to_segments


CONTEXT_DIM = 96
EVENT_DIM = 205
GEOMETRY_NAMES = (
    'log_T', 'log_K', 'log_action_length', 'log_envelope_length',
    'action_start_relative_to_envelope', 'action_end_relative_to_envelope',
    'log_Base_boundaries_strictly_inside_action', 'log_removed_boundaries',
    'log_added_boundaries', 'log_shortest_intersecting_full_Base_run',
    'log_longest_intersecting_full_Base_run', 'minimum_overlap_over_full_run',
    'changed_over_action_length', 'minimum_full_Base_run_A08_margin',
    'full_Base_run_length_weighted_A08_margin', 'maximum_endpoint_B3_JSD_h7',
)
CONTEXT_NAMES = tuple(
    f'{source}_envelope_bin{bin_index}_class{class_index}_mean'
    for source in ('A08', 'J')
    for bin_index in range(8)
    for class_index in range(5)
) + GEOMETRY_NAMES


def _matrix(value, width, nonempty=False):
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != width or (nonempty and not len(value)) or not np.isfinite(value).all():
        raise ValueError(f'finite original-order N{width} matrix required')
    return value


def _normalizer(value, width):
    if not isinstance(value, dict) or set(value) != {'mean', 'std'}:
        raise ValueError('exact FIT mean/std normalizer required')
    mean = np.asarray(value['mean'], dtype=np.float64)
    std = np.asarray(value['std'], dtype=np.float64)
    if mean.shape != (width,) or std.shape != (width,) or not np.isfinite(mean).all() or not np.isfinite(std).all() or (std < 1e-6).any():
        raise ValueError(f'complete finite FIT{width} normalizer required')
    return mean, std


def context_features(prob_a, prob_j, base, action):
    """Eight ordered posterior bins over full Base-run envelope +16 observables.

    No reference labels, fragment boundaries or IDs enter this interface. Full
    run margins use only runs intersecting the action, excluding added flanks.
    """
    a, j = probabilities(prob_a), probabilities(prob_j)
    base = labels(base)
    if a.shape != j.shape or len(a) != len(base):
        raise ValueError('same complete A08/J/Base original axes required')
    edited = apply(base, action)
    runs = labels_to_segments(base)
    indices = [i for i, run in enumerate(runs) if run.start < action.end and run.end > action.start]
    intersecting = [runs[i] for i in indices]
    left = runs[max(0, indices[0] - 1)].start
    right = runs[min(len(runs) - 1, indices[-1] + 1)].end
    n = right - left
    bins = [(left + k * n // 8, left + (k + 1) * n // 8) for k in range(8)]
    result = []
    for probability in (a, j):
        for lo, hi in bins:
            result.extend(probability[lo:hi].mean(0) if hi > lo else np.zeros(5))

    before = set((np.flatnonzero(base[1:] != base[:-1]) + 1).tolist())
    after = set((np.flatnonzero(edited[1:] != edited[:-1]) + 1).tolist())
    lengths = np.asarray([run.length for run in intersecting], dtype=np.float64)
    overlaps = np.asarray([min(run.end, action.end) - max(run.start, action.start) for run in intersecting], dtype=np.float64)
    margins = []
    for run in intersecting:
        mean = a[run.start:run.end].mean(0)
        margins.append(float(mean[run.label] - np.delete(mean, run.label).max()))
    margins = np.asarray(margins)
    # Identical original B3 prefix arithmetic, evaluated only at the two used
    # endpoints. Do not recompute all T Python JSD scores for every candidate.
    cs = np.cumsum(np.pad(a, ((1, 0), (0, 0))), axis=0)
    endpoint_values = []
    for p in (action.start, action.end):
        if not 0 < p < len(base):
            endpoint_values.append(0.)
            continue
        lo, hi = max(0, p - 7), min(len(base), p + 7)
        left_mean = (cs[p] - cs[lo]) / (p - lo)
        right_mean = (cs[hi] - cs[p]) / (hi - p)
        endpoint_values.append(float(js_divergence(left_mean, right_mean)))
    endpoint_score = max(endpoint_values)
    k = int(np.count_nonzero(edited != base))
    length = action.end - action.start
    result.extend([
        *np.log1p([len(base), k, length, n]),
        (action.start - left) / n, (action.end - left) / n,
        *np.log1p([sum(action.start < p < action.end for p in before), len(before - after), len(after - before)]),
        *np.log1p([lengths.min(), lengths.max()]),
        float(np.min(overlaps / lengths)), k / length,
        float(margins.min()), float(np.average(margins, weights=lengths)), endpoint_score,
    ])
    value = np.asarray(result, dtype=np.float64)
    if value.shape != (CONTEXT_DIM,) or not np.isfinite(value).all():
        raise ValueError('complete finite context96 required')
    return value


def fit_context_normalizer(values):
    values = _matrix(values, CONTEXT_DIM, nonempty=True)
    return {'mean': values.mean(0), 'std': np.maximum(values.std(0), 1e-6)}


def normalize_contexts(values, normalizer):
    values = _matrix(values, CONTEXT_DIM)
    mean, std = _normalizer(normalizer, CONTEXT_DIM)
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        normalized = ((values - mean) / std).astype(np.float32)
    if not np.isfinite(normalized).all():
        raise ValueError('finite normalized context96 required')
    return normalized


def event_matrix(raw_candidates, contexts, arm, original_candidate_normalizer, context_normalizer=None):
    """Build complete N205; empty source axes return zero rows without NN work."""
    raw_candidates = _matrix(raw_candidates, 109)
    _normalizer(original_candidate_normalizer, 109)
    if arm not in ('R1', 'R2'):
        raise ValueError('declared R1/R2 event arm required')
    if arm == 'R1':
        if contexts is not None or context_normalizer is not None:
            raise ValueError('R1 receives no new context or unused context normalizer')
    else:
        if contexts is None or context_normalizer is None:
            raise ValueError('R2 requires complete context96 and FIT normalizer')
        contexts = _matrix(contexts, CONTEXT_DIM)
        if len(contexts) != len(raw_candidates):
            raise ValueError('candidate/context original row order and length must align')
        _normalizer(context_normalizer, CONTEXT_DIM)
    if not len(raw_candidates):
        return np.empty((0, EVENT_DIM), dtype=np.float32)
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        original = normalize(raw_candidates, original_candidate_normalizer)
        extra = original[:, :96] if arm == 'R1' else normalize_contexts(contexts, context_normalizer)
        values = np.concatenate([original, extra], axis=1)
    if values.shape != (len(raw_candidates), EVENT_DIM) or not np.isfinite(values).all():
        raise ValueError('complete finite matched event205 required')
    return values
