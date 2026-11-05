#!/usr/bin/env python3
"""EX3 description written before Z's seeds 1 and 2 had any output. CPU, aggregates only, no reading rule depends on it.

A08's external recall moves a lot between its own seeds, and a shared seed number only means a shared manifest. So each
external recall is shown by seed, and every Z seed is compared with every A08 seed. Also: the development rows by seed,
and the GPU and CPU spent.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import json

import numpy as np

import ex3_context as context
import ex3_readout as ro
from exam_level import io
from helena_input import data, verdict as vd
from open_routing import data as ex2

METRIC = 'multi segment F1@50'
DEV_FILES = ('dev_trained_s0.json', 'dev_trained_Z_s1_s2.json')
CPU_LOG = data.RUN / 'cpu_z12.tsv'
A08_FORWARD = ex2.RUN / 'forward'


def correct_by_seed(folder, name, classes, tag_of, ids):
    """Each seed's correct calls, after checking that the recordings are the reference's, in the same order."""
    out = {}
    for seed in io.SEEDS:
        group = ro.load_group(folder, name, tag_of(seed))
        if group['ids'].tolist() != ids:
            raise SystemExit(f'{name} {tag_of(seed)}: the recordings differ from EX2')
        out[seed] = ro.correct(group, classes)
    return out


def recalls(values):
    by_seed = {seed: float(v.mean()) for seed, v in values.items()}
    numbers = list(by_seed.values())
    return {'by_seed': by_seed, 'mean': float(np.mean(numbers)), 'low': min(numbers), 'high': max(numbers)}


def endpoint_row(name, classes):
    reference = ro.load_group(A08_FORWARD, name, 's0')
    ids = reference['ids'].tolist()
    a08 = correct_by_seed(A08_FORWARD, name, classes, lambda s: f's{s}', ids)
    swap = correct_by_seed(ro.FORWARD, name, classes, lambda s: f'A08onZ_s{s}', ids)
    z = correct_by_seed(ro.FORWARD, name, classes, lambda s: f'Z_s{s}', ids)
    return {'n': len(ids), 'a08': recalls(a08), 'a08_on_z': recalls(swap), 'z': recalls(z), 'z_against_every_a08_seed': vd.all_seed_pairs(z, a08, ro.clusters_of(reference, name))}


def dev_row(stored):
    native = stored['native']['b3_majority']
    return {'f1': stored['row']['segment_f1_50_multi'], 'boundary_recall': stored['row']['boundary_recall_multi'], 'native_macro_f1': stored['row']['native_macro_f1'],
            'delta': stored['against_a08'][METRIC]['delta'], 'ci95': stored['against_a08'][METRIC]['ci95'], 'b3_status': stored['b3']['status'],
            'native_errors': native['errors'], 'a4c_recall': native['a4c_recall'], 'a5c_recall': native['a5c_recall'], 'real_order': stored['exam']['against_a08']}


def dev_by_seed():
    merged = vd.merge_dev([json.loads((data.PUBLIC / name).read_text()) for name in DEV_FILES])['results']
    return {half: {seed: dev_row(merged[half][f'Z|{seed}']) for seed in io.SEEDS} for half in vd.HALVES}


def cpu_of_extension():
    """Children are included. This summary's own line is written only after it ends, so it is not counted here."""
    return {**vd.parse_cpu_log(CPU_LOG.read_text()), 'note': 'this extension only, without this summary; the earlier stages were not recorded'}


def main():
    value = {'external': {endpoint: endpoint_row(name, classes) for endpoint, (name, classes) in ro.ENDPOINTS.items()}, 'dev': dev_by_seed(), 'cpu_extension': cpu_of_extension()}
    data.write_public('z_seed_context.json', value)
    data.write_public('resources.json', context.resources())
    print(json.dumps({e: {k: v[k] for k in ('a08', 'a08_on_z', 'z')} | {'above': v['z_against_every_a08_seed']['above_zero'], 'below': v['z_against_every_a08_seed']['below_zero']} for e, v in value['external'].items()}, indent=1))
    print(json.dumps(value['dev'], indent=1))
    print(json.dumps(value['cpu_extension'], indent=1))


if __name__ == '__main__':
    main()
