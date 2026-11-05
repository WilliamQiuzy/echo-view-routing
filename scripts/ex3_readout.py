#!/usr/bin/env python3
"""EX3 external and open-set readout, CPU. A08's values come from EX2's stored groups and must reproduce EX2's published table.

  --tags H_s0 Z_s0 A08onH_s0 A08onZ_s0
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import argparse
import json

import numpy as np

from exam_level import io
from helena_input import data, verdict as vd
from open_routing import data as ex2, openset as os_

PLAX, A4C, A5C = 0, 2, 3
RETAIN = .95
METHODS = ('msp', 'energy', 'knn', 'mahalanobis')
GROUPS = ('dev_selection', 'dev_calibration', 'train_reference', 'out_of_scope', 'dynamic', 'lvh', 'camus_2ch', 'camus_4ch')
ENDPOINTS = {'dynamic_a4c': ('dynamic', (A4C,)), 'lvh_plax': ('lvh', (PLAX,)), 'camus_4ch_a4c': ('camus_4ch', (A4C,)), 'camus_4ch_apical': ('camus_4ch', (A4C, A5C))}
PARITY_RELATIVE_TOLERANCE = 1e-7
FORWARD = data.RUN / 'forward'


def load_group(folder, name, tag):
    with np.load(folder / f'embed_{name}_{tag}.npz') as z:
        return {k: z[k] for k in z.files}


def load_all(folder, tag):
    return {name: load_group(folder, name, tag) for name in GROUPS}


def train_labels(ids):
    labels = {r['video_id']: int(float(r['family5_index'])) for r in io.manifest_rows('train') if r['family5_label']}
    return np.array([labels[str(i)] for i in ids])


def open_rows(groups):
    """EX2's scorers, thresholds and readouts, for one model."""
    reference = groups['train_reference']
    model = os_.fit_mahalanobis(reference['embed_mean'], train_labels(reference['ids']))
    scorers = {'msp': lambda g: os_.msp(g['prob_mean']), 'energy': lambda g: g['energy'],
               'knn': lambda g: os_.knn_score(reference['embed_mean'], g['embed_mean']), 'mahalanobis': lambda g: os_.mahalanobis_score(model, g['embed_mean'])}
    rows = []
    for method, score in scorers.items():
        values = {name: score(g) for name, g in groups.items() if name != 'train_reference'}
        threshold = os_.threshold_for_retention(np.concatenate([values['dev_selection'], values['dev_calibration']]), RETAIN)
        accept = {name: os_.accept_share(v, threshold) for name, v in values.items()}
        rows.append({'method': method, 'threshold': threshold, 'accept': accept, 'auroc': os_.auroc(values['camus_4ch'], values['camus_2ch']),
                     'camus_2ch_accept': accept['camus_2ch'], 'camus_4ch_retain': accept['camus_4ch'], 'dynamic_retain': accept['dynamic'], 'lvh_retain': accept['lvh']})
    return rows


def check_a08_parity():
    """My readout on EX2's own stored groups must equal EX2's published open-set table."""
    published = json.loads((ex2.RUN / 'public/x3_open_set.json').read_text())['rows']
    worst = 0.
    for seed in io.SEEDS:
        for row in open_rows(load_all(ex2.RUN / 'forward', f's{seed}')):
            match = next(r for r in published if r['seed'] == seed and r['method'] == row['method'])
            relative = abs(row['threshold'] - match['threshold']) / max(1., abs(match['threshold']))
            worst = max(worst, relative, abs(row['auroc'] - match['auroc']), abs(row['camus_4ch_retain'] - match['camus_4ch_retain']))
    if worst > PARITY_RELATIVE_TOLERANCE:
        raise SystemExit(f'the readout does not reproduce EX2: {worst}')
    return worst


def correct(group, classes):
    return np.isin(group['prob_mean'].argmax(1), classes).astype(float)


def clusters_of(group, name):
    return group['groups'] if 'groups' in group else group['ids']


def endpoints(system, a08):
    out = {}
    for endpoint, (name, classes) in ENDPOINTS.items():
        if system[name]['ids'].tolist() != a08[name]['ids'].tolist():
            raise SystemExit(f'{name}: the recordings differ from EX2')
        out[endpoint] = vd.paired_recall_delta(correct(system[name], classes), correct(a08[name], classes), clusters_of(a08[name], name))
    return out


def routes(group):
    return np.bincount(group['prob_mean'].argmax(1), minlength=5).tolist()


def tag_seed(tag):
    return int(tag.rsplit('_s', 1)[1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tags', nargs='+', required=True)
    args = parser.parse_args()
    parity = check_a08_parity()
    results = {'a08_open_set_parity_worst': parity, 'tags': {}}
    for tag in args.tags:
        seed = tag_seed(tag)
        system, a08 = load_all(FORWARD, tag), load_all(ex2.RUN / 'forward', f's{seed}')
        rows = open_rows(system)
        results['tags'][tag] = {'seed': seed, 'endpoints': endpoints(system, a08), 'open_set': rows, 'usable_this_seed': {r['method']: os_.usable([r]) for r in rows},
                                'routes': {name: routes(system[name]) for name in ('camus_2ch', 'camus_4ch', 'dynamic', 'lvh', 'out_of_scope')}}
        print(tag, {k: (round(v['a'], 3), round(v['b'], 3)) for k, v in results['tags'][tag]['endpoints'].items()}, flush=True)
    data.write_public(f"external_{'_'.join(args.tags)}.json", results)


if __name__ == '__main__':
    main()
