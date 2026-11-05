#!/usr/bin/env python3
"""EX1 E3 readout: selective prediction on A08 clip-level probabilities, across halves and on external data. CPU only."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np

from exam_level import io, metrics as mx, selective as sel

ALPHAS = (.01, .02, .05)
SPACES = {'five_class': False, 'merged_apical': True}
PUBLISHED_EXTERNAL_RECALL = {'dynamic': .947794, 'lvh': .957240, 'camus': .733333}
RECALL_TOLERANCE = 1e-4
N_CLASSES = {'five_class': 5, 'merged_apical': 4}


def dev_clips(half, seed):
    cache = io.A08Cache(half, seed)
    natives = io.natives(half)
    return np.stack([cache.window_mean(q['id']) for q in natives]), np.array([int(q['labels'][0]) for q in natives])


def judged(prob, truth, merged):
    decision = sel.clip_decision(prob, merged)
    target = sel.merged_labels(truth) if merged else np.asarray(truth)
    return decision['label'], decision['confidence'], decision['label'] == target, target


def cross_half(seed):
    """Threshold on one half, measured on the other."""
    rows = []
    for space, merged in SPACES.items():
        for fit, test in (('selection', 'calibration'), ('calibration', 'selection')):
            _, conf_f, ok_f, _ = judged(*dev_clips(fit, seed), merged)
            pred, conf_t, ok_t, target = judged(*dev_clips(test, seed), merged)
            for alpha in ALPHAS:
                threshold = sel.threshold_for_risk(conf_f, ok_f, alpha)
                measured = sel.apply_threshold(conf_t, ok_t, threshold)
                rows.append({'seed': seed, 'space': space, 'fit': fit, 'test': test, 'alpha': alpha,
                             'threshold': None if np.isinf(threshold) else threshold, **measured,
                             'risk_upper_95': sel.risk_upper_bound(measured['errors'], measured['accepted']),
                             'base_accuracy': float(ok_t.mean()), 'base_errors': int((~ok_t).sum()),
                             'destinations': sel.destination_errors(pred, target, conf_t >= threshold, N_CLASSES[space])})
    return rows


def external_parity(records, dataset):
    """Majority over frozen B3 positions, three-seed mean, must equal the A50 published recall."""
    recalls = []
    for seed in io.SEEDS:
        votes = [np.bincount(r[f'frozen_B3_s{seed}'], minlength=5).argmax() for r in records]
        recalls.append(np.mean([v == r['true'] for v, r in zip(votes, records)]))
    mean = float(np.mean(recalls))
    if abs(mean - PUBLISHED_EXTERNAL_RECALL[dataset]) > RECALL_TOLERANCE:
        raise SystemExit(f'A50 external recall does not reproduce on {dataset}: {mean}')
    return mean


def external_rows(seed):
    """Thresholds from all development clips of this seed, measured on data A08 never trained on."""
    prob = np.concatenate([dev_clips(h, seed)[0] for h in io.HALVES])
    truth = np.concatenate([dev_clips(h, seed)[1] for h in io.HALVES])
    rows = []
    for dataset in io.EXTERNAL_DATASETS:
        records = io.external_records(dataset)
        published = external_parity(records, dataset)
        mean_prob = np.stack([io.softmax_mean(r[f'logits_s{seed}']) for r in records])
        labels = np.array([r['true'] for r in records])
        groups = [r['group'] for r in records]
        for space, merged in SPACES.items():
            _, conf_dev, ok_dev, _ = judged(prob, truth, merged)
            pred, conf, ok, target = judged(mean_prob, labels, merged)
            confusion = np.bincount(pred, minlength=N_CLASSES[space]).tolist()
            for alpha in ALPHAS:
                threshold = sel.threshold_for_risk(conf_dev, ok_dev, alpha)
                measured = sel.apply_threshold(conf, ok, threshold)
                keep = conf >= threshold
                error_rows = np.stack([(keep & ~ok).astype(float), keep.astype(float)], 1)
                ci = mx.unpaired_ci(error_rows, groups, lambda v: v[..., 0] / v[..., 1]) if keep.any() and (keep & ~ok).any() else None
                rows.append({'seed': seed, 'dataset': dataset, 'space': space, 'alpha': alpha, 'threshold': None if np.isinf(threshold) else threshold,
                             **measured, 'risk_ci95': ci, 'risk_upper_95': sel.risk_upper_bound(measured['errors'], measured['accepted']),
                             'destinations': sel.destination_errors(pred, target, keep, N_CLASSES[space]),
                             'base_accuracy': float(ok.mean()), 'base_prediction_counts': confusion,
                             'a50_three_seed_recall_parity': published, 'groups': len(set(groups))})
    return rows


def out_of_scope_rows(seed):
    """PMPALA clips are outside the five classes. Each one the system accepts is a confident wrong route."""
    prob = np.concatenate([dev_clips(h, seed)[0] for h in io.HALVES])
    truth = np.concatenate([dev_clips(h, seed)[1] for h in io.HALVES])
    with np.load(io.RUN / 'forward' / f'out_of_scope_s{seed}.npz') as z:
        edges = np.cumsum([0, *z['window_counts']])
        logits = [z['logits'][a:b] for a, b in zip(edges, edges[1:])]
    mean_prob = np.stack([io.softmax_mean(x) for x in logits])
    rows = []
    for space, merged in SPACES.items():
        _, conf_dev, ok_dev, _ = judged(prob, truth, merged)
        decision = sel.clip_decision(mean_prob, merged)
        for alpha in ALPHAS:
            threshold = sel.threshold_for_risk(conf_dev, ok_dev, alpha)
            keep = decision['confidence'] >= threshold
            rows.append({'seed': seed, 'space': space, 'alpha': alpha, 'clips': len(keep), 'accepted': int(keep.sum()),
                         'accepted_share': float(keep.mean()), 'routed_to': np.bincount(decision['label'][keep], minlength=N_CLASSES[space]).tolist()})
    return rows


def curves():
    """Risk-coverage curves on pooled development clips, per seed and label space."""
    out = {}
    for seed in io.SEEDS:
        prob = np.concatenate([dev_clips(h, seed)[0] for h in io.HALVES])
        truth = np.concatenate([dev_clips(h, seed)[1] for h in io.HALVES])
        for space, merged in SPACES.items():
            _, conf, ok, _ = judged(prob, truth, merged)
            out[f's{seed}_{space}'] = sel.risk_coverage_curve(conf, ok)
    return out


def main():
    io.write_json(io.RUN / 'public/selective.json', {
        'cross_half': [r for s in io.SEEDS for r in cross_half(s)],
        'external': [r for s in io.SEEDS for r in external_rows(s)],
        'out_of_scope': [r for s in io.SEEDS for r in out_of_scope_rows(s)],
        'curves': curves()})
    print('selective readout written')


if __name__ == '__main__':
    main()
