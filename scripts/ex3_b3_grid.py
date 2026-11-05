#!/usr/bin/env python3
"""EX3 check added after reading seed 0. CPU, stored outputs only, no reading rule depends on it.

H's B3 search found no feasible setting, so its reported row uses the fixed diagnostic setting. This asks how much of
H's segment F1 loss that setting explains: the highest-F1 setting of the same 40-setting grid is taken on calibration,
ignoring the feasibility floors, and applied once to selection. The calibration row is optimistic by construction.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import json

import ex3_evaluate as ev3
from helena_input import data
from scripts.ev9v_complementarity import decode
import scripts.report_center_supervision as rcs
import scripts.report_raw_video_context as rr
import scripts.report_segment_objective as rso
import scripts.report_video_switch_head as rsh
from echo_routing.evaluate.complementarity import sha256_file

SEED = 0
METRIC = 'multi segment F1@50'


def stored_probs(half_data, arm):
    path = data.RUN / 'evaluation' / half_data.half / f'{arm}_s{SEED}.npz'
    ident = f'{arm}|{SEED}|{sha256_file(data.final_weights(arm, SEED))}|{data.input_fingerprint(arm)}'
    logits, _ = rr.cached_logits(path, ident, half_data.W, half_data.orders, half_data.seqs)
    if logits is None:
        raise SystemExit(f'{arm}: stored outputs do not match their identity')
    return rr.per_sample(half_data.W, logits, half_data.seqs)


def against_a08(a8, half_data, params, probs):
    system = rso.row_of(a8.cfg, decode(params, probs), half_data.ev)
    reference = rso.row_of(a8.cfg, decode(half_data.a08[SEED]['params'], half_data.a08[SEED]['probs']), half_data.ev)
    paired = ev3.compare(system, reference)
    return {'f1': system['row']['segment_f1_50_multi'], 'boundary_recall': system['row']['boundary_recall_multi'], 'native_macro_f1': system['row']['native_macro_f1'],
            'delta': paired[METRIC]['delta'], 'ci95': paired[METRIC]['ci95'], 'false_segments_delta': paired['multi FP segments']['delta'], 'false_segments_ci95': paired['multi FP segments']['ci95']}


def main():
    a8 = rsh.a08_ctx()
    summary, choices, pins = ev3.a08_sources()
    calibration = ev3.HalfData(a8, 'calibration', [SEED], summary, choices, pins)
    selection = ev3.HalfData(a8, 'selection', [SEED], summary, choices, pins)
    out = {}
    for arm in data.ARMS:
        probs = stored_probs(calibration, arm)
        pick = rcs.b3_choose(a8.cfg, rso.DIAG, probs, calibration.ev)
        grid = pick['grid']
        best = max(grid, key=lambda r: (r['segment_f1_50_multi'], -r['order']))
        out[arm] = {'settings': len(grid), 'status': pick['status'], 'highest_native_macro_f1': max(r['native_macro_f1'] for r in grid),
                    'native_floor_met': sum(r['native_macro_f1'] >= .980 for r in grid), 'boundary_floor_met': sum((r['boundary_recall_multi'] or 0) >= .95 for r in grid),
                    'reported_params': pick['params'], 'best_f1_params': best['params'],
                    'best_f1_calibration': against_a08(a8, calibration, best['params'], probs),
                    'best_f1_selection': against_a08(a8, selection, best['params'], stored_probs(selection, arm))}
    data.write_public('b3_grid_s0.json', out)
    print(json.dumps(out, indent=1, default=str))


if __name__ == '__main__':
    main()
