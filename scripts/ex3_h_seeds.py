#!/usr/bin/env python3
"""EX3 description of H's three seeds, written before H's seeds 1 and 2 had any output. CPU, stored outputs only, no reading rule depends on it.

The plan's section on H's seeds fixes what is described here:
1. each external recall by seed, H's every seed against A08's every seed, and H against Z by seed;
2. the CAMUS two-view checks that the recount added for Z: 2CH read as the target, views distinguished, 4CH-against-2CH AUROC;
3. EV9V at the rule's B3 setting, the highest-F1 setting and A08's window and quantile, and H against Z by seed.
Only counts and rates are written. Clip names and patient names stay on the server.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import json

import numpy as np

import ex3_after_recount as ar
import ex3_context as context
import ex3_evaluate as ev3
import ex3_readout as ro
import ex3_z_seeds as zs
from exam_level import io
from helena_input import data, verdict as vd
from scripts.ev9v_complementarity import decode
import scripts.report_segment_objective as rso
import scripts.report_video_switch_head as rsh

DEV_FILES = ('dev_trained_s0.json', 'dev_trained_H_s1_s2.json', 'dev_trained_Z_s1_s2.json')
CPU_LOG = data.RUN / 'cpu_h12.tsv'
OUTPUT = 'h_seed_context.json'
SEED0_TOLERANCE = 1e-9


def endpoint_row(name, classes):
    reference = ro.load_group(zs.A08_FORWARD, name, 's0')
    ids = reference['ids'].tolist()
    by_seed = lambda folder, tag_of: zs.correct_by_seed(folder, name, classes, tag_of, ids)
    a08, swap = by_seed(zs.A08_FORWARD, lambda s: f's{s}'), by_seed(ro.FORWARD, lambda s: f'A08onH_s{s}')
    h, z = by_seed(ro.FORWARD, lambda s: f'H_s{s}'), by_seed(ro.FORWARD, lambda s: f'Z_s{s}')
    clusters = ro.clusters_of(reference, name)
    same_seed = lambda a, b: {seed: vd.paired_recall_delta(a[seed], b[seed], clusters) for seed in io.SEEDS}
    return {'n': len(ids), 'a08': zs.recalls(a08), 'a08_on_h': zs.recalls(swap), 'h': zs.recalls(h), 'z': zs.recalls(z),
            'h_against_same_seed_a08': same_seed(h, a08), 'h_against_every_a08_seed': vd.all_seed_pairs(h, a08, clusters), 'h_against_same_seed_z': same_seed(h, z)}


def camus_section():
    reference = ar.view_group('camus_4ch', 'A08', 0)['groups'].tolist()
    if ar.view_group('camus_2ch', 'A08', 0)['groups'].tolist() != reference:
        raise SystemExit('the 2CH and 4CH patients do not line up')
    models = {(arm, seed): ar.camus_views(arm, seed, reference) for arm in ('A08', 'H', 'A08onH', 'Z') for seed in io.SEEDS}
    rows = {target: {f'H_against_{base}': {seed: ar.view_row(models[('H', seed)], models[(base, seed)], classes) for seed in io.SEEDS} for base in ('A08', 'A08onH', 'Z')}
            for target, classes in ar.TARGETS.items()}
    routes = lambda probs: np.bincount(probs.argmax(1), minlength=5).tolist()
    return {'patients': len(reference), 'rows': rows, 'routes': {f'H_s{seed}': {'4ch': routes(models[('H', seed)][0]), '2ch': routes(models[('H', seed)][1])} for seed in io.SEEDS}}


def merged_dev():
    return vd.merge_dev([json.loads((data.PUBLIC / name).read_text()) for name in DEV_FILES])['results']


def h_against_z(a8, halves, merged):
    """Each arm at its own rule setting, chosen once on calibration. Seed 0 must reproduce the stored H − Z row."""
    out = {half: {} for half in halves}
    for seed in io.SEEDS:
        params = {arm: merged['calibration'][f'{arm}|{seed}']['b3']['params'] for arm in ('H', 'Z')}
        for half, half_data in halves.items():
            rows = {arm: rso.row_of(a8.cfg, decode(params[arm], ar.stored_probs(half_data, seed, arm)), half_data.ev) for arm in ('H', 'Z')}
            paired = ev3.compare(rows['H'], rows['Z'])[ar.METRIC]
            out[half][seed] = {'delta': paired['delta'], 'ci95': paired['ci95']}
    for half in halves:
        stored = merged[half]['H - Z|0']['paired'][ar.METRIC]['delta']
        if abs(stored - out[half][0]['delta']) > SEED0_TOLERANCE:
            raise SystemExit(f'{half}: H − Z at seed 0 does not reproduce the stored row')
    return out


def main():
    a8 = rsh.a08_ctx()
    summary, choices, pins = ev3.a08_sources()
    halves = {half: ev3.HalfData(a8, half, list(io.SEEDS), summary, choices, pins) for half in ('calibration', 'selection')}
    merged = merged_dev()
    value = {'note': 'fixed before H seeds 1 and 2 had any output; no reading rule depends on it',
             'external': {endpoint: endpoint_row(name, classes) for endpoint, (name, classes) in ro.ENDPOINTS.items()},
             'camus': camus_section(),
             'dev': {half: {seed: zs.dev_row(merged[half][f'H|{seed}']) for seed in io.SEEDS} for half in vd.HALVES},
             'b3': ar.b3_section(a8, halves, choices, arm='H'),
             'h_against_z': h_against_z(a8, halves, merged),
             'cpu_extension': {**vd.parse_cpu_log(CPU_LOG.read_text()), 'note': 'this extension only, without this summary'}}
    data.write_public(OUTPUT, value)
    data.write_public('resources.json', context.resources())
    print(json.dumps(value, indent=1, default=str))


if __name__ == '__main__':
    main()
