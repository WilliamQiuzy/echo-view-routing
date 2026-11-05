#!/usr/bin/env python3
"""EX2 X3: can A08 refuse views outside the five classes? Four frozen scores, thresholds from development clips. CPU."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np

from exam_level import io
from open_routing import data, openset as os_

RETAIN = .95
PARITY_TOLERANCE = 1e-3
METHODS =('msp', 'energy', 'knn', 'mahalanobis')
N_CLASSES = 5


def load(name, seed):
    with np.load(data.RUN / 'forward' / f'embed_{name}_s{seed}.npz') as z:
        return {k: z[k] for k in z.files}


def train_labels(ids):
    labels = {r['video_id']: int(float(r['family5_index'])) for r in io.manifest_rows('train') if r['family5_label']}
    return np.array([labels[str(i)] for i in ids])


def scorers(reference):
    """One function per declared score, each mapping a loaded group to one value per recording."""
    model = os_.fit_mahalanobis(reference['embed_mean'], train_labels(reference['ids']))
    return {'msp': lambda g: os_.msp(g['prob_mean']), 'energy': lambda g: g['energy'],
            'knn': lambda g: os_.knn_score(reference['embed_mean'], g['embed_mean']),
            'mahalanobis': lambda g: os_.mahalanobis_score(model, g['embed_mean'])}


def routes(group, keep=None):
    """Where the five-class argmax sends these recordings."""
    labels = group['prob_mean'].argmax(1)
    return np.bincount(labels if keep is None else labels[keep], minlength=N_CLASSES).tolist()


def check_external(groups, seed):
    """Stored external groups must equal the published A50 outputs, recording by recording."""
    worst = 0.
    dynamic = io.external_records('dynamic')
    picked = [i for i, _ in data.dynamic_sample()]
    if groups['dynamic']['ids'].tolist() != [r['id'] for _, r in data.dynamic_sample()]:
        raise SystemExit('Dynamic sample order changed')
    for prob, i in zip(groups['dynamic']['prob_mean'], picked):
        worst = max(worst, float(np.abs(prob - io.softmax_mean(dynamic[i][f'logits_s{seed}'])).max()))
    for record, prob in zip(io.external_records('lvh'), groups['lvh']['prob_mean']):
        worst = max(worst, float(np.abs(prob - io.softmax_mean(record[f'logits_s{seed}'])).max()))
    camus = {f'{r["id"]}_4CH': i for i, r in enumerate(data.external_rows('camus'))}
    records = io.external_records('camus')
    compared = 0
    for name, prob in zip(groups['camus_4ch']['ids'].tolist(), groups['camus_4ch']['prob_mean']):
        if name in camus:
            worst = max(worst, float(np.abs(prob - io.softmax_mean(records[camus[name]][f'logits_s{seed}'])).max()))
            compared += 1
    if compared != len(camus) or worst > PARITY_TOLERANCE:
        raise SystemExit(f'external groups do not reproduce A50: worst {worst}, CAMUS {compared}')
    return {'worst_probability_difference': worst, 'camus_recordings_compared': compared, 'dynamic_compared': len(picked), 'lvh_compared': len(groups['lvh']['ids'])}


def seed_rows(seed):
    groups = {name: load(name, seed) for name in ('dev_selection', 'dev_calibration', 'train_reference', 'out_of_scope', 'dynamic', 'lvh', 'camus_2ch', 'camus_4ch')}
    if groups['camus_2ch']['groups'].tolist() != groups['camus_4ch']['groups'].tolist():
        raise SystemExit('CAMUS views are not paired by patient')
    parity = check_external(groups, seed)
    rows = []
    for method, score in scorers(groups['train_reference']).items():
        values = {name: score(g) for name, g in groups.items() if name != 'train_reference'}
        dev = np.concatenate([values['dev_selection'], values['dev_calibration']])
        threshold = os_.threshold_for_retention(dev, RETAIN)
        accept = {name: os_.accept_share(v, threshold) for name, v in values.items()}
        cross = {'selection_threshold_kept_on_calibration': os_.accept_share(values['dev_calibration'], os_.threshold_for_retention(values['dev_selection'], RETAIN)),
                 'calibration_threshold_kept_on_selection': os_.accept_share(values['dev_selection'], os_.threshold_for_retention(values['dev_calibration'], RETAIN))}
        kept_2ch = values['camus_2ch'] >= threshold
        rows.append({'seed': seed, 'method': method, 'threshold': threshold, 'accept': accept, 'cross_half': cross, 'a50_parity': parity,
                     'auroc': os_.auroc(values['camus_4ch'], values['camus_2ch']),
                     'auroc_ci95': os_.paired_auroc_ci(values['camus_4ch'], values['camus_2ch']),
                     'auroc_dev_vs_camus_2ch': os_.auroc(dev, values['camus_2ch']),
                     'auroc_dev_vs_out_of_scope': os_.auroc(dev, values['out_of_scope']),
                     'auroc_dev_vs_camus_4ch': os_.auroc(dev, values['camus_4ch']),
                     'camus_2ch_accept': accept['camus_2ch'], 'camus_4ch_retain': accept['camus_4ch'],
                     'dynamic_retain': accept['dynamic'], 'lvh_retain': accept['lvh'],
                     'camus_2ch_routes_all': routes(groups['camus_2ch']), 'camus_2ch_routes_accepted': routes(groups['camus_2ch'], kept_2ch),
                     'camus_4ch_routes_all': routes(groups['camus_4ch']), 'out_of_scope_routes_all': routes(groups['out_of_scope']),
                     'recordings': {name: len(v) for name, v in values.items()}})
    return rows


def main():
    rows = [r for seed in io.SEEDS for r in seed_rows(seed)]
    verdict = {m: os_.usable([r for r in rows if r['method'] == m]) for m in METHODS}
    io.write_json(data.RUN / 'public/x3_open_set.json', {'rows': rows, 'usable': verdict, 'any_usable': any(verdict.values()), 'retain': RETAIN})
    print('x3 written', len(rows))


if __name__ == '__main__':
    main()
