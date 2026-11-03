"""Historical task metrics plus explicitly scoped pair short-unit supplements."""
import numpy as np

from echo_routing.temporal.structured_metrics import evaluate as historical_evaluate
from echo_routing.temporal.structured_uncertainty import paired_bootstrap, _weights
from echo_routing.evaluate.failure_units import match_segments
from echo_routing.temporal.segments import labels_to_segments


def evaluate(predictions, context):
    result, tables = historical_evaluate(predictions, context)
    confusion = np.asarray(result['native_confusion'])
    result['native_per_class_recall'] = np.divide(np.diag(confusion), confusion.sum(1),
        out=np.zeros(5, float), where=confusion.sum(1) > 0).tolist()
    fragment = tables['pairs_frag'].sum(numeric_only=True)
    ratio = lambda a, b: float(a / b) if b else None
    result['boundary_precision_pairs'] = ratio(fragment.t_matched, fragment.p_pred)
    result['boundary_f1_pairs'] = ratio(2 * fragment.t_matched, fragment.t_true + fragment.p_pred)
    frame = tables['pairs_seg'].sum(numeric_only=True)
    result['pairs_frame_accuracy'] = ratio(frame.correct, frame.n)
    result['same_class_cut_rate_pairs'] = ratio(fragment.false_split, fragment.same_join)
    result['counts'].update({'pairs_true_changes': int(fragment.t_true), 'pairs_predicted_changes': int(fragment.p_pred),
                             'pairs_matched_changes': int(fragment.t_matched), 'pairs_same_joins': int(fragment.same_join)})
    for maximum in [10, 16]:
        total, missed, predicted, false = 0, 0, 0, 0
        for q in context['sequences']:
            if q['kind'] in ['native', 'multi']:
                continue
            true, pred = labels_to_segments(q['labels']), labels_to_segments(predictions[q['id']])
            matched = match_segments(pred, true)
            used = set(matched.values())
            total += sum(t.length <= maximum for t in true)
            missed += sum(t.length <= maximum and i not in matched for i, t in enumerate(true))
            predicted += sum(p.length <= maximum for p in pred)
            false += sum(p.length <= maximum and i not in used for i, p in enumerate(pred))
        result.update({f'pairs_short_true_segments_{maximum}samples': total,
                       f'pairs_short_true_unmatched_{maximum}samples': missed,
                       f'pairs_short_predicted_segments_{maximum}samples': predicted,
                       f'pairs_short_spurious_segments_{maximum}samples': false})
    result['short_metric_scope'] = 'unprefixed historical primary=multi; pairs_short supplement=pair; same IoU50 classawareone-to-one'
    return result, tables


def paired(left, right, n_boot=5000, seed=2026110359):
    result = paired_bootstrap(left, right, n_boot=n_boot, seed=seed)
    rng = np.random.default_rng(seed)
    _weights(rng, len(left['session_order']), n_boot)
    _weights(rng, len(left['streams']['multi']), n_boot)
    weights = _weights(rng, len(left['streams']['pair']), n_boot)
    a = weights @ np.stack([row[1] for row in left['streams']['pair']])
    b = weights @ np.stack([row[1] for row in right['streams']['pair']])
    for maximum, start in [(10, 10), (16, 14)]:
        for name, offset in [('true_unmatched', 1), ('spurious_segments', 3)]:
            values = a[:, start + offset] - b[:, start + offset]
            key = f'pairs_short_{name}_{maximum}samples'
            result['delta95'][key] = np.quantile(values, [.025, .975]).tolist()
            result['undefined_bootstrap_replicates'][key] = 0
    return result
