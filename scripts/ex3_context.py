#!/usr/bin/env python3
"""EX3 context added after reading seed 0. CPU, aggregates only, no reading rule depends on it.

It records what limits the seed-0 result: how far A08's own external recall moves between seeds, how often Helena's
analysis fell back to the fan geometry alone, which readings further seeds could still reach, and the resource ledger.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import json
import pickle
from collections import Counter

import numpy as np

from exam_level import io
from helena_input import data, verdict as vd
from open_routing import data as ex2

PLAX, A4C, A5C = 0, 2, 3
ENDPOINTS = {'dynamic_a4c': ('dynamic', (A4C,)), 'lvh_plax': ('lvh', (PLAX,)), 'camus_4ch_a4c': ('camus_4ch', (A4C,)), 'camus_4ch_apical': ('camus_4ch', (A4C, A5C))}
MEDIA = ('dynamic', 'lvh', 'camus_4ch', 'camus_2ch')
METRIC = 'multi segment F1@50'
FORWARD = data.RUN / 'forward'
A08_FORWARD = ex2.RUN / 'forward'


def load(folder, name, tag):
    with np.load(folder / f'embed_{name}_{tag}.npz') as z:
        return {k: z[k] for k in z.files}


def correct(group, classes):
    return np.isin(group['prob_mean'].argmax(1), classes)


def a08_recall_by_seed():
    return {e: [float(correct(load(A08_FORWARD, name, f's{s}'), classes).mean()) for s in io.SEEDS] for e, (name, classes) in ENDPOINTS.items()}


def fallback_tables():
    kinds = {name: {str(k): vd.fallback_kind(g) for k, g in pickle.loads((data.GEOMETRY / f'{name}.pkl').read_bytes()).items()} for name in MEDIA}
    return kinds, {name: dict(Counter(rows.values())) for name, rows in kinds.items()}


def recall_by_fallback(kinds):
    systems = {'A08': (A08_FORWARD, 's0'), 'A08onH': (FORWARD, 'A08onH_s0'), 'H': (FORWARD, 'H_s0'), 'A08onZ': (FORWARD, 'A08onZ_s0'), 'Z': (FORWARD, 'Z_s0')}
    rows = []
    for endpoint, (name, classes) in ENDPOINTS.items():
        ids = [str(i) for i in load(A08_FORWARD, name, 's0')['ids']]
        labels = np.array([kinds[name][i] for i in ids])
        for kind in sorted(set(labels)):
            mask = labels == kind
            rows.append({'endpoint': endpoint, 'kind': kind, 'n': int(mask.sum()),
                         'recall': {s: float(correct(load(f, name, t), classes)[mask].mean()) for s, (f, t) in systems.items()}})
    return rows


def ev9v_provider_fallback():
    """Helena's own record for the allowed recordings; the test split is filtered out before any other field is read."""
    counts = Counter(str(row.get('fallback_source') or 'none') for row in data.provider_rows(data.allowed_rows()).values())
    return dict(counts)


def still_possible():
    dev = json.loads((data.PUBLIC / 'dev_trained_s0.json').read_text())['results']
    external = json.loads((data.PUBLIC / 'external_H_s0_Z_s0.json').read_text())['tags']
    out = {}
    for arm in data.ARMS:
        rows = {h: dev[h][f'{arm}|0'] for h in vd.HALVES}
        intervals = {h: rows[h]['against_a08'][METRIC]['ci95'] for h in vd.HALVES}
        floors = vd.floors_hold([{'half': h, 'boundary_recall': rows[h]['row']['boundary_recall_multi'], 'native_macro_f1': rows[h]['row']['native_macro_f1']} for h in vd.HALVES])
        tag = external[f'{arm}_s0']
        out[arm] = vd.still_possible_after_first_seed(intervals, floors, {e: v['ci95'] for e, v in tag['endpoints'].items()}, any(tag['usable_this_seed'].values()))
    return out


def resources():
    rows = [json.loads(line) for line in (data.RUN / 'ledger.jsonl').read_text().splitlines()]
    return {'gpu_hours_limit': data.GPU_HOURS_MAX, 'gpu_hours_limit_original': data.GPU_HOURS_ORIGINAL, 'gpu_hours_charged': sum(r.get('gpu_seconds') or 0 for r in rows) / 3600,
            'ledger': [{'what': r['what'], 'hours': (r.get('gpu_seconds') or 0) / 3600, 'ok': r.get('ok'), 'note': r.get('note', '')} for r in rows],
            'cpu_core_hours': 'not recorded', 'frame_cache_bytes': json.loads((data.PUBLIC / 'preparation.json').read_text())['cache_bytes']}


def main():
    kinds, counts = fallback_tables()
    value = {'a08_external_recall_by_seed': a08_recall_by_seed(), 'external_fallback_recordings': counts, 'recall_by_fallback_seed0': recall_by_fallback(kinds),
             'ev9v_provider_fallback_source': ev9v_provider_fallback(), 'still_possible_after_seed0': still_possible()}
    data.write_public('context_s0.json', value)
    data.write_public('resources.json', resources())
    print(json.dumps({k: value[k] for k in ('a08_external_recall_by_seed', 'external_fallback_recordings', 'ev9v_provider_fallback_source', 'still_possible_after_seed0')}, indent=1))


if __name__ == '__main__':
    main()
