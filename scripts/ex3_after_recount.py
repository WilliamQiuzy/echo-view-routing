#!/usr/bin/env python3
"""EX3 description added after the independent recount of Z's three seeds. CPU, stored outputs only, no reading rule depends on it.

The recount raised three questions after every result was known, so all of this is post hoc, on development and reused data.
1. CAMUS: does Z also read 2CH recordings as A4C more often, and does it separate one patient's two views any better?
2. EV9V: how much of Z's segment F1 loss comes with the B3 setting the rule chose?
3. The selection half's native row is identical across Z's seeds. Are those the same clips?
Only counts and rates are written. Clip names and patient names stay on the server.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import json

import numpy as np

import ex3_evaluate as ev3
import ex3_readout as ro
from echo_routing.evaluate.complementarity import sha256_file
from exam_level import io
from helena_input import data, verdict as vd
from open_routing import data as ex2
from scripts.ev9v_complementarity import decode
import scripts.report_center_supervision as rcs
import scripts.report_raw_video_context as rr
import scripts.report_segment_objective as rso
import scripts.report_video_switch_head as rsh

A4C, A5C = 2, 3
TARGETS = {'a4c': (A4C,), 'apical': (A4C, A5C)}
METRIC = 'multi segment F1@50'
OUTPUT = 'after_recount.json'


def view_group(name, arm, seed):
    """Stored CAMUS outputs of one model. Patients must sit in one order across models, as must be true for pairing."""
    folder, tag = (ex2.RUN / 'forward', f's{seed}') if arm == 'A08' else (ro.FORWARD, f'{arm}_s{seed}')
    return ro.load_group(folder, name, tag)


def camus_views(arm, seed, reference):
    four, two = view_group('camus_4ch', arm, seed), view_group('camus_2ch', arm, seed)
    if four['groups'].tolist() != reference or two['groups'].tolist() != reference:
        raise SystemExit(f'{arm} s{seed}: the patients are not in the reference order')
    return four['prob_mean'], two['prob_mean']


def view_row(system, base, classes):
    """system and base: (4CH, 2CH) probabilities of one model. Each measure is the system minus the base, by patient."""
    patients = np.arange(len(system[0]))
    read = lambda probs: np.isin(probs.argmax(1), classes).astype(float)
    score = lambda probs: probs[:, list(classes)].sum(1)
    return {'read_2ch_as_target': vd.paired_recall_delta(read(system[1]), read(base[1]), patients),
            'read_4ch_as_target': vd.paired_recall_delta(read(system[0]), read(base[0]), patients),
            'views_distinguished': vd.paired_recall_delta(vd.distinguished(read(system[0]), read(system[1])), vd.distinguished(read(base[0]), read(base[1])), patients),
            'auroc_4ch_against_2ch': vd.paired_auroc_delta(score(system[0]), score(system[1]), score(base[0]), score(base[1]))}


def camus_section():
    reference = view_group('camus_4ch', 'A08', 0)['groups'].tolist()
    if view_group('camus_2ch', 'A08', 0)['groups'].tolist() != reference:
        raise SystemExit('the 2CH and 4CH patients do not line up')
    models = {(arm, seed): camus_views(arm, seed, reference) for arm in ('A08', 'Z', 'A08onZ') for seed in io.SEEDS}
    models[('H', 0)] = camus_views('H', 0, reference)
    rows = {target: {'Z_against_A08': {seed: view_row(models[('Z', seed)], models[('A08', seed)], classes) for seed in io.SEEDS},
                     'Z_against_A08onZ': {seed: view_row(models[('Z', seed)], models[('A08onZ', seed)], classes) for seed in io.SEEDS},
                     'H_against_A08': {0: view_row(models[('H', 0)], models[('A08', 0)], classes)}}
            for target, classes in TARGETS.items()}
    routes = lambda probs: np.bincount(probs.argmax(1), minlength=5).tolist()
    labels = {arm: [models[(arm, seed)][0].argmax(1) for seed in io.SEEDS] for arm in ('A08', 'Z')}
    agreement = [[float((z == a).mean()) for a in labels['A08']] for z in labels['Z']]
    within = {arm: [float((labels[arm][i] == labels[arm][j]).mean()) for i, j in ((0, 1), (0, 2), (1, 2))] for arm in labels}
    return {'patients': len(reference), 'rows': rows, 'routes_2ch': {f'{arm}_s{seed}': routes(probs[1]) for (arm, seed), probs in models.items()},
            'agreement_on_4ch': {'z_by_a08_seed': agreement, 'within_arm_pairs_01_02_12': within}}


def stored_probs(half_data, seed):
    path = data.RUN / 'evaluation' / half_data.half / f'Z_s{seed}.npz'
    identity = f'Z|{seed}|{sha256_file(data.final_weights("Z", seed))}|{data.input_fingerprint("Z")}'
    logits, _ = rr.cached_logits(path, identity, half_data.W, half_data.orders, half_data.seqs)
    if logits is None:
        raise SystemExit(f'Z s{seed}: stored outputs do not match their identity')
    return rr.per_sample(half_data.W, logits, half_data.seqs)


def against_a08(a8, half_data, seed, params, probs):
    system = rso.row_of(a8.cfg, decode(params, probs), half_data.ev)
    reference = rso.row_of(a8.cfg, decode(half_data.a08[seed]['params'], half_data.a08[seed]['probs']), half_data.ev)
    paired = ev3.compare(system, reference)
    return {'f1': system['row']['segment_f1_50_multi'], 'a08_f1': reference['row']['segment_f1_50_multi'], 'boundary_recall': system['row']['boundary_recall_multi'], 'native_macro_f1': system['row']['native_macro_f1'],
            'delta': paired[METRIC]['delta'], 'ci95': paired[METRIC]['ci95'], 'false_segments_delta': paired['multi FP segments']['delta'], 'false_segments_ci95': paired['multi FP segments']['ci95']}


def shape(params):
    return int(params['half_window']), int(params['nms_radius']), float(params['threshold_quantile'])


def named_settings(pick, a08_params):
    """The rule's setting, the highest F1 ignoring the floors, and the grid setting with A08's window, radius and quantile."""
    grid = pick['grid']
    same = [row for row in grid if shape(row['params']) == shape(a08_params)]
    if len(same) != 1:
        raise SystemExit('the grid should hold exactly one setting with the shape of A08')
    return {'rule': pick['params'], 'highest_f1': max(grid, key=lambda r: (r['segment_f1_50_multi'], -r['order']))['params'], 'a08_shape': same[0]['params']}


def native_errors(half_data, preds):
    wrong = set()
    for q in half_data.seqs:
        if q['kind'] == 'native' and int(np.bincount(np.asarray(preds[q['id']], int), minlength=5).argmax()) != int(np.asarray(q['labels'])[0]):
            wrong.add(q['id'])
    return wrong


def overlap(sets):
    every = set.intersection(*sets)
    return {'by_seed': [len(s) for s in sets], 'in_all_seeds': len(every)}


def b3_section(a8, halves, choices):
    settings, floors, grid_rows, errors = {}, {}, {}, {half: {} for half in halves}
    probs = {half: {seed: stored_probs(halves[half], seed) for seed in io.SEEDS} for half in halves}
    out = {half: {} for half in halves}
    for seed in io.SEEDS:
        a08_params = choices[f'all_s{seed}']['params']
        pick = rcs.b3_choose(a8.cfg, rso.DIAG, probs['calibration'][seed], halves['calibration'].ev)
        settings[seed] = {**named_settings(pick, a08_params), 'a08': a08_params}
        floors[seed] = {'status': pick['status'], 'feasible': pick['n_feasible']}
        grid_rows[seed] = [{'shape': shape(r['params']), 'f1': r['segment_f1_50_multi'], 'native_macro_f1': r['native_macro_f1'], 'boundary_recall': r['boundary_recall_multi']} for r in pick['grid']]
        for half, half_data in halves.items():
            out[half][seed] = {name: against_a08(a8, half_data, seed, params, probs[half][seed]) for name, params in settings[seed].items() if name != 'a08'}
            errors[half][seed] = {'Z': native_errors(half_data, rso.row_of(a8.cfg, decode(settings[seed]['rule'], probs[half][seed]), half_data.ev)['preds']),
                                  'A08': native_errors(half_data, rso.row_of(a8.cfg, decode(halves[half].a08[seed]['params'], halves[half].a08[seed]['probs']), half_data.ev)['preds'])}
    native = {half: {'Z': overlap([errors[half][s]['Z'] for s in io.SEEDS]), 'Z_errors_also_wrong_in_any_a08_seed': len(set.intersection(*[errors[half][s]['Z'] for s in io.SEEDS]) & set.union(*[errors[half][s]['A08'] for s in io.SEEDS])), 'A08': overlap([errors[half][s]['A08'] for s in io.SEEDS])} for half in halves}
    return {'settings': settings, 'floors': floors, 'calibration_grid': grid_rows, 'against_a08': out, 'native_errors': native}


def main():
    a8 = rsh.a08_ctx()
    summary, choices, pins = ev3.a08_sources()
    halves = {half: ev3.HalfData(a8, half, list(io.SEEDS), summary, choices, pins) for half in ('calibration', 'selection')}
    value = {'note': 'post hoc after the independent recount; no reading rule depends on it', 'camus': camus_section(), 'b3': b3_section(a8, halves, choices)}
    data.write_public(OUTPUT, value)
    print(json.dumps(value, indent=1, default=str))


if __name__ == '__main__':
    main()
