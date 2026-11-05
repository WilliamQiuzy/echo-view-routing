#!/usr/bin/env python3
"""EX3 development evaluation. Calibration is read first and fixes each model's B3 setting; selection follows.

  --stage swap                frozen A08 on H and Z pixels, at A08's own B3 setting
  --stage trained --seeds 0   the trained arms of those seeds
  --stage trained --arms Z --seeds 1 2   one arm only
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch

import scripts.prepare_raw_video_context as prv
import scripts.report_center_supervision as rcs
import scripts.report_raw_video_context as rr
import scripts.report_segment_objective as rso
import scripts.report_video_switch_head as rsh
import scripts.report_video_within_view as rw
import scripts.segment_objective_tables as st
import scripts.train_video_switch_head as vsh
import scripts.train_video_within_view as twv
import scripts.trained_token_readout as ttr
from echo_routing.config import load_paths
from echo_routing.evaluate.complementarity import sha256_file
from exam_level import io, metrics as mx, streams as streams_mod
from exam_level.decode import decode_b3
from helena_input import data, forward as fw, pixels as px
from scripts._common import load_cfg
from scripts.ev9v_complementarity import decode
from scripts.train_joint_visual_temporal import FramePaths

A4C, A5C = 2, 3
SWAP = ('H', 'Z')
EXAM_PARITY_STREAMS = 3
EXAM_PARITY_TOLERANCE = 1e-3


def a08_sources():
    path = 'configs/experiments/video_switch_head_v1.yaml'
    a10 = vsh.Ctx(load_cfg(SimpleNamespace(config=path, set=[])), path)
    summary = json.loads((io.ROOT / 'docs/results/2026-10-16_video_within_view/summary.json').read_text())
    choices = json.loads(vsh.pinned(a10.src['a08_choices']).read_text())['choices']
    return summary, choices, rsh.a08_output_pins()


def split_probs(values, counts):
    edges = np.cumsum([0, *counts])
    return [values[a:b] for a, b in zip(edges, edges[1:])]


def exam_bank(half):
    return np.load(io.RUN / 'streams' / f'exam_{half}.npy', allow_pickle=True).tolist()


def exam_a08(half, seed):
    with np.load(io.RUN / 'forward' / f'exam_{half}_s{seed}.npz') as z:
        if z['position_counts'].tolist() != [len(q['refs']) for q in exam_bank(half)]:
            raise SystemExit('EX1 outputs do not align with the real-order bank')
        return split_probs(z['probs'], z['position_counts'])


def native_counts(seqs, positions, window_means):
    """Confusions of whole recordings under the two rules. Labels come from the sequence, not from any model."""
    b3, mean = np.zeros((5, 5), int), np.zeros((5, 5), int)
    for q in seqs:
        if q['kind'] != 'native':
            continue
        truth = int(np.asarray(q['labels'])[0])
        b3[truth, int(np.bincount(np.asarray(positions[q['id']], int), minlength=5).argmax())] += 1
        mean[truth, int(window_means[q['id']].argmax())] += 1
    describe = lambda c: {'confusion': c.tolist(), 'errors': int(c.sum() - np.trace(c)), 'macro_f1': float(mx.macro_f1(c)), 'a4c_recall': float(c[A4C, A4C] / max(1, c[A4C].sum())),
                          'a5c_recall': float(c[A5C, A5C] / max(1, c[A5C].sum())), 'a4c_to_a5c': int(c[A4C, A5C]), 'a5c_to_a4c': int(c[A5C, A4C])}
    return {'b3_majority': describe(b3), 'window_mean': describe(mean)}


class HalfData:
    def __init__(self, a8, half, seeds, summary, choices, pins):
        self.half = half
        self.ev, self.seqs = a8.half(half, rso.DevAccess({half}))
        self.refs = rw.half_refs(self.seqs)
        self.W = rr.half_windows(self.seqs, {r: i for i, r in enumerate(self.refs)}, 5)
        self.orders = np.broadcast_to(np.arange(16, dtype=np.int64), self.W['s'].shape).copy()
        self.fp = FramePaths(prv.frame_dirs(load_paths(), a8.rv, {r[0] for r in self.refs}))
        self.a08 = {}
        for seed in seeds:
            path = io.ROOT / 'runs/video_within_view_v1/evaluation' / half / f'all_s{seed}.npz'
            if sha256_file(path) != pins[half][seed]:
                raise SystemExit('A08 saved outputs differ from their pin')
            logits, _ = rr.cached_logits(path, summary['runs'][f'all_s{seed}']['identity'], self.W, self.orders, self.seqs)
            if logits is None:
                raise SystemExit('A08 saved outputs do not match these windows')
            probs, params = rr.per_sample(self.W, logits, self.seqs), choices[f'all_s{seed}']['params']
            if any(float(params[k]) != float(io.b3_params(half, seed)[k]) for k in ('half_window', 'threshold', 'nms_radius')):
                raise SystemExit('A08 B3 settings differ between the two records')
            ttr.check_stored(rso.row_of(a8.cfg, decode(params, probs), self.ev)['row'], summary['systems'][f'A + B3 s{seed}'][half], f'A08 s{seed} {half}')
            self.a08[seed] = {'probs': probs, 'params': params, 'logits': logits}
        self.pixels = {}

    def frames(self, arm):
        if arm not in self.pixels:
            self.pixels = {arm: px.frames_for_refs(self.refs, data.pixels_for(arm, self.fp))}
        return self.pixels[arm]

    def window_means(self, logits):
        return {q['id']: io.softmax_mean(logits[self.W['seq'] == i]) for i, q in enumerate(self.seqs)}


def model_outputs(half_data, name, seed, arm, state_path, device, gpu):
    """Logits of one model on one input, reused only under the same identity."""
    path = data.RUN / 'evaluation' / half_data.half / f'{name}_s{seed}.npz'
    ident = f'{name}|{seed}|{sha256_file(state_path)}|{data.input_fingerprint(arm)}'
    logits, _ = rr.cached_logits(path, ident, half_data.W, half_data.orders, half_data.seqs)
    if logits is None:
        model = fw.build_model(torch.load(state_path, map_location='cpu', weights_only=True), device)
        logits, seconds = rr.model_logits(model, half_data.W, half_data.orders, half_data.frames(arm), device)
        gpu.append(float(seconds))
        rr.save_outputs(path, ident, half_data.W, half_data.orders, logits, rr.per_sample(half_data.W, logits, half_data.seqs), half_data.seqs, seconds)
        del model
        torch.cuda.empty_cache()
    return logits


def exam_rows(half, seed, arm, state_path, params, a08_params, device, pixels_fp):
    """Real-order streams: the arm against A08 at each system's own B3 setting, whole timestamps resampled."""
    bank = exam_bank(half)
    model = fw.build_model(torch.load(state_path, map_location='cpu', weights_only=True), device)
    pixels = data.pixels_for(arm, pixels_fp)
    probs = [fw.stream_positions(model, q, pixels, device) for q in bank]
    del model
    torch.cuda.empty_cache()
    rows = mx.as_rows([mx.stream_counts(q['labels'], decode_b3(p, params)) for q, p in zip(bank, probs)])
    base = mx.as_rows([mx.stream_counts(q['labels'], decode_b3(p, a08_params)) for q, p in zip(bank, exam_a08(half, seed))])
    clusters = [streams_mod.timestamp_of(q['clips'][0][0]) for q in bank]
    frame = mx.FIELDS.index('correct')
    public = {'system': mx.pooled(rows), 'a08': mx.pooled(base), 'against_a08': mx.paired_delta(rows, base, clusters, mx.f1_statistic), 'frame_errors_change': int(base[:, frame].sum() - rows[:, frame].sum())}
    return public, rows, clusters


def exam_parity(half, seed, device, fp):
    """A08 on its own pixels must reproduce EX1's real-order cache through this path."""
    bank = exam_bank(half)[:EXAM_PARITY_STREAMS]
    model = fw.build_model(torch.load(io.a08_weight_path(seed), map_location='cpu', weights_only=True), device)
    pixels = data.pixels_for('A08', fp)
    worst = max(float(np.abs(fw.stream_positions(model, q, pixels, device) - cached).max()) for q, cached in zip(bank, exam_a08(half, seed)))
    del model
    torch.cuda.empty_cache()
    if worst > EXAM_PARITY_TOLERANCE:
        raise SystemExit(f'the real-order path does not reproduce EX1: {worst}')
    return worst


def compare(system, reference):
    return st.paired(st.system_draws(system['tables']), st.system_draws(reference['tables']))


def evaluate_half(a8, half_data, jobs, chosen, device, gpu):
    """jobs: (system name, seed, input arm, weights path). chosen: system|seed -> B3 params, filled on calibration."""
    out, systems, exams = {}, {}, {}
    for name, seed, arm, weights in jobs:
        logits = model_outputs(half_data, name, seed, arm, weights, device, gpu)
        probs = rr.per_sample(half_data.W, logits, half_data.seqs)
        key = f'{name}|{seed}'
        if key not in chosen:
            if name.startswith('A08 on'):
                chosen[key] = {'status': 'a08_setting', 'params': half_data.a08[seed]['params'], 'n_feasible': None}
            else:
                pick = rcs.b3_choose(a8.cfg, rso.DIAG, probs, half_data.ev)
                chosen[key] = {x: pick[x] for x in ('status', 'params', 'n_feasible')}
        params = chosen[key]['params']
        system = rso.row_of(a8.cfg, decode(params, probs), half_data.ev)
        reference = rso.row_of(a8.cfg, decode(half_data.a08[seed]['params'], half_data.a08[seed]['probs']), half_data.ev)
        exam_public, exam_table, clusters = exam_rows(half_data.half, seed, arm, weights, params, half_data.a08[seed]['params'], device, half_data.fp)
        systems[key], exams[key] = system, (exam_table, clusters)
        out[key] = {'row': system['row'], 'counts': rsh.counts_of(half_data.seqs, system), 'against_a08': compare(system, reference), 'b3': chosen[key],
                    'native': native_counts(half_data.seqs, system['preds'], half_data.window_means(logits)), 'accounting': st.segment_accounting(half_data.seqs, system['preds'], reference['preds'])[0],
                    'exam': exam_public}
        print(half_data.half, key, 'multi F1', round(system['row']['segment_f1_50_multi'], 4), 'against A08', round(out[key]['against_a08']['multi segment F1@50']['delta'], 4), flush=True)
    for first, second in (('H', 'Z'), ('A08 on H', 'A08 on Z')):
        for seed in half_data.a08:
            a, b = f'{first}|{seed}', f'{second}|{seed}'
            if a in systems and b in systems:
                out[f'{first} - {second}|{seed}'] = {'paired': compare(systems[a], systems[b]), 'exam': mx.paired_delta(exams[a][0], exams[b][0], exams[a][1], mx.f1_statistic)}
    a08_rows = {seed: {'row': rso.row_of(a8.cfg, decode(v['params'], v['probs']), half_data.ev)['row'],
                       'native': native_counts(half_data.seqs, decode(v['params'], v['probs']), half_data.window_means(v['logits']))} for seed, v in half_data.a08.items()}
    return out, a08_rows


def jobs_for(stage, seeds, arms=data.ARMS):
    if stage == 'swap':
        return [(f'A08 on {arm}', seed, arm, io.a08_weight_path(seed)) for arm in SWAP for seed in seeds]
    return [(arm, seed, arm, data.final_weights(arm, seed)) for arm in arms for seed in seeds]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('swap', 'trained'), required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=list(io.SEEDS))
    parser.add_argument('--arms', nargs='+', choices=data.ARMS, default=list(data.ARMS))
    args = parser.parse_args()
    if args.stage == 'swap' and set(args.arms) != set(data.ARMS):
        parser.error('--arms applies to the trained stage only')
    began = time.time()
    device = fw.determinism(data.GPU_FRACTION)
    twv.determinism()
    a8 = rsh.a08_ctx()
    rw.check_grid(a8)
    summary, choices, pins = a08_sources()
    chosen, results, gpu, parity, half_rows = {}, {}, [], {}, {}
    jobs = jobs_for(args.stage, args.seeds, args.arms)
    for half in ('calibration', 'selection'):
        half_data = HalfData(a8, half, args.seeds, summary, choices, pins)
        parity[half] = {f's{s}': exam_parity(half, s, device, half_data.fp) for s in args.seeds}
        results[half], half_rows[half] = evaluate_half(a8, half_data, jobs, chosen, device, gpu)
        del half_data
    out = {'stage': args.stage, 'seeds': args.seeds, 'results': results, 'a08': half_rows, 'exam_parity_worst': parity, 'chosen': chosen, 'wall_seconds': time.time() - began, 'gpu_seconds': sum(gpu)}
    flat = json.loads(json.dumps(out, default=str))
    name = data.dev_result_name(args.stage, args.arms, args.seeds)
    data.write_public(name, flat)
    with (data.RUN / 'ledger.jsonl').open('a') as handle:
        handle.write(json.dumps({'what': f'evaluate {name}', 'gpu_seconds': time.time() - began, 'ok': True, 'note': 'whole wall time charged'}) + '\n')
    print('written', name, flush=True)


if __name__ == '__main__':
    main()
