#!/usr/bin/env python3
"""EX1 post-hoc checks added after the independent fact-check, so its new numbers are reproducible. CPU, aggregates only."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
from scipy.stats import rankdata

import ex1_clip_context as cc
from exam_level import clip_models as cm, io, metrics as mx, selective as sel, streams as st
from exam_level.decode import decode_b3

A4C, A5C = cm.A4C, cm.A5C
N_CLASSES = 5
OUT_OF_SCOPE = 'PMPALA'
CONFIDENCE_LEVELS = (.5, .7, .8, .9, .95, .99)
BOOT_DRAWS = 5000
BOOT_SEED = 2026110399
O_FAMILY = ('O0', 'O', 'OT')


def constructed_joins(half):
    """Where the source recording changes inside a constructed stream, and whether both sides share a session."""
    total = same = same_apical = 0
    for q in io.role_sequences(half):
        if q['kind'] != 'multi':
            continue
        ids = [r[0] for r in q['refs']]
        for j in range(1, len(ids)):
            if ids[j] != ids[j - 1]:
                total += 1
                if st.session_of(ids[j]) == st.session_of(ids[j - 1]):
                    same += 1
                    same_apical += {int(q['labels'][j - 1]), int(q['labels'][j])} == {A4C, A5C}
    return {'source_changes': total, 'same_session': same, 'cross_session': total - same, 'same_session_a4c_a5c': same_apical}


def exam_joins_spanning_excluded(half):
    """Exam joins between two five-class clips that have an excluded clip between them in k order."""
    raw = {}
    for row in io.manifest_rows('validation'):
        raw.setdefault(st.session_of(row['video_id']), []).append((st.clip_index(row['video_id']), row['raw_label']))
    total = spanning = 0
    for session, clips in st.group_sessions(io.natives(half)).items():
        kept = [st.clip_index(q['id']) for q in clips]
        excluded = [k for k, label in raw[session] if label == OUT_OF_SCOPE]
        for a, b in zip(kept, kept[1:]):
            total += 1
            spanning += any(a < k < b for k in excluded)
    return {'joins': total, 'spanning_excluded_clip': spanning}


def d1_extra():
    """OT, O and O0 without coordinate-ascent sessions, and T with the class marginal removed from its chain."""
    folds = cc.load_d1()
    decoded = {arm: [] for arm in (*cc.ARMS, 'T_no_marginal')}
    ordered, skipped = [], {arm: [] for arm in O_FAMILY}
    for k in sorted(folds):
        fit = [s for j, v in folds.items() if j != k for s in v]
        sequences = cc.train_label_sequences(fit)
        fitted = cc.fit_arms(fit, sequences)
        result, fallback = cc.decode_arms(folds[k], fitted)
        pi = np.bincount(np.concatenate(sequences), minlength=N_CLASSES) + 1.
        log_pi = np.log(pi / pi.sum())
        log_start, log_trans = fitted['chains']['T']
        start, trans = log_start - log_pi, log_trans - log_pi[None, :]
        tau = cm.fit_temperature([s['L'] for s in fit], [s['y'] for s in fit], start, trans)
        for arm, indices in fallback.items():
            skipped[arm] += [len(ordered) + i for i in indices]
        ordered += folds[k]
        for arm in cc.ARMS:
            decoded[arm] += result[arm]
        decoded['T_no_marginal'] += [cm.viterbi(s['L'] / tau, start, trans) for s in folds[k]]
    flat = {arm: np.concatenate(v) for arm, v in decoded.items()}
    y = np.concatenate([s['y'] for s in ordered])
    clusters = [s['timestamp'] for s in ordered for _ in s['y']]
    out = {'T_no_marginal_minus_B': cc.compare(y, clusters, flat['T_no_marginal'], flat['B']),
           'T_no_marginal_minus_T': cc.compare(y, clusters, flat['T_no_marginal'], flat['T'])}
    for arm in O_FAMILY:
        keep = [i for i in range(len(ordered)) if i not in set(skipped[arm])]
        kept_y = np.concatenate([ordered[i]['y'] for i in keep])
        kept_clusters = [ordered[i]['timestamp'] for i in keep for _ in ordered[i]['y']]
        out[f'{arm}_minus_B_without_fallback'] = cc.compare(kept_y, kept_clusters, np.concatenate([decoded[arm][i] for i in keep]),
                                                            np.concatenate([decoded['B'][i] for i in keep]))
        out[f'{arm}_minus_B_without_fallback']['clips'] = len(kept_y)
    return out


def auroc(score, positive):
    """Chance that a correct clip has higher confidence than a wrong one."""
    pos = int(positive.sum())
    neg = len(positive) - pos
    if not pos or not neg:
        return None
    return float((rankdata(score)[positive].sum() - pos * (pos + 1) / 2) / (pos * neg))


def external_hindsight():
    """Fixed confidence levels chosen with hindsight. These cannot serve as thresholds selected in advance."""
    out = []
    for seed in io.SEEDS:
        for dataset in io.EXTERNAL_DATASETS:
            records = io.external_records(dataset)
            prob = np.stack([io.softmax_mean(r[f'logits_s{seed}']) for r in records])
            truth = np.array([r['true'] for r in records])
            for space, merged in (('five_class', False), ('merged_apical', True)):
                decision = sel.clip_decision(prob, merged)
                target = sel.merged_labels(truth) if merged else truth
                ok = decision['label'] == target
                rows = []
                for level in CONFIDENCE_LEVELS:
                    keep = decision['confidence'] >= level
                    errors = int((keep & ~ok).sum())
                    rows.append({'level': level, 'coverage': float(keep.mean()), 'errors': errors, 'risk': errors / int(keep.sum()) if keep.any() else None})
                out.append({'seed': seed, 'dataset': dataset, 'space': space, 'clips': len(ok), 'errors_without_rejection': int((~ok).sum()),
                            'auroc_correct_over_wrong': auroc(decision['confidence'], ok), 'levels': rows})
    return out


def exam_predictions(half, seed):
    sequences = np.load(io.RUN / 'streams' / f'exam_{half}.npy', allow_pickle=True).tolist()
    params = io.b3_params(half, seed)
    with np.load(io.RUN / 'forward' / f'exam_{half}_s{seed}.npz') as z:
        edges = np.cumsum([0, *z['position_counts']])
        probs = [z['probs'][a:b] for a, b in zip(edges, edges[1:])]
    return sequences, [decode_b3(p, params) for p in probs]


def frame_accuracy_mix(half, seed):
    """Exam frame accuracy, and the same per-class recalls weighted by the constructed bank's class mix."""
    sequences, predictions = exam_predictions(half, seed)
    truth, pred = np.concatenate([q['labels'] for q in sequences]), np.concatenate(predictions)
    constructed = np.concatenate([q['labels'] for q in io.role_sequences(half) if q['kind'] == 'multi'])
    weights = np.bincount(constructed, minlength=N_CLASSES) / len(constructed)
    recalls = np.array([(pred[truth == c] == c).mean() for c in range(N_CLASSES)])
    return {'half': half, 'seed': seed, 'exam_frame_accuracy': float((truth == pred).mean()), 'exam_at_constructed_mix': float(recalls @ weights),
            'class_recalls': recalls.tolist()}


def error_timestamps(half, seed):
    """How many timestamps hold the segment errors, and how often a whole-timestamp resample holds none of them."""
    sequences, predictions = exam_predictions(half, seed)
    errors = {}
    for q, p in zip(sequences, predictions):
        stamp = st.timestamp_of(q['clips'][0][0])
        counts = mx.stream_counts(q['labels'], p)
        errors[stamp] = errors.get(stamp, 0) + counts['T'] + counts['P'] - 2 * counts['M']
    has_error = np.array([v > 0 for v in errors.values()])
    draws = np.random.default_rng(BOOT_SEED).multinomial(len(has_error), np.full(len(has_error), 1 / len(has_error)), BOOT_DRAWS)
    return {'half': half, 'seed': seed, 'timestamps': len(has_error), 'timestamps_with_errors': int(has_error.sum()),
            'total_errors': int(sum(errors.values())), 'draws_without_any_error': float((draws[:, has_error].sum(1) == 0).mean())}


def main():
    results = {'constructed_joins': {h: constructed_joins(h) for h in io.HALVES},
               'exam_joins_spanning_excluded': {h: exam_joins_spanning_excluded(h) for h in io.HALVES},
               'd1': d1_extra(), 'external_hindsight': external_hindsight(),
               'frame_accuracy_mix': [frame_accuracy_mix(h, s) for h in io.HALVES for s in io.SEEDS],
               'error_timestamps': [error_timestamps(h, s) for h in io.HALVES for s in io.SEEDS]}
    io.write_json(io.RUN / 'public/posthoc_checks.json', results)
    print('post-hoc checks written')


if __name__ == '__main__':
    main()
