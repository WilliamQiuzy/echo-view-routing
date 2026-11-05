#!/usr/bin/env python3
"""EX2 X1: A60's frozen heads and policies on real-order streams. CPU. Weights, thresholds and policies are unchanged."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
import torch

from echo_routing.temporal.baselines.registry import decode_with_prob
from echo_routing.temporal.proposal_features import targets
from echo_routing.temporal.proposal_runtime import infer
from echo_routing.temporal.proposal_training import ProposalRegressor, optimizer, restore
from exam_level import io, metrics as mx, streams as st
from exam_level.decode import decode_b3
from open_routing import action_budget as ab, data
from scripts.proposal_common import CFG, PUB, RUN as A60_RUN, ROOT, read, sha, verify_pin
from scripts.train_jepa_proposals import role_values

POLICIES = ('O', 'E', 'R')
MAIN_ARMS = ('J-E', 'J-R')
BANKS = ('exam', 'rewired')


def split(values, counts):
    edges = np.cumsum([0, *counts])
    return [values[a:b] for a, b in zip(edges, edges[1:])]


def load_heads():
    manifest, inputs = read(PUB / 'models_manifest.json'), read(PUB / 'training_inputs.json')
    scales = np.asarray(read(verify_pin(inputs['target_scales']))['scales'])
    heads = []
    for row in manifest['models']:
        path, norm = ROOT / row['final_path'], read(verify_pin(row['normalizer']))
        if sha(path) != row['final_sha256']:
            raise SystemExit('A60 head weights changed')
        model = ProposalRegressor(row['arm'], row['seed'])
        if restore(model, optimizer(model), torch.load(path, map_location='cpu', weights_only=True), row['identity'], norm, scales) != 6000:
            raise SystemExit('A60 head is not the frozen 6000-update state')
        heads.append({'seed': row['seed'], 'arm': row['arm'], 'model': model.eval(), 'norm': norm})
    return heads, scales, inputs['F_fit']


def check_a60_parity(heads, scales):
    """This pipeline must reproduce A60's saved predictions exactly on its own confirmation bank."""
    saved = np.load(A60_RUN / 'readout_predictions.private.npy', allow_pickle=True).item()['a60_confirmation']['selection']
    qs, a, j, b, _ = role_values('a60_confirmation_selection')
    checked = 0
    for head in (h for h in heads if h['arm'] == 'J'):
        name = next(k for k in saved if str(head['seed']) in k and 'J-R' in k)
        for q, x, y, z in zip(qs, a, j, b):
            prediction = infer(head['model'], head['norm'], scales, x, y, z, 'J')['R']['prediction']
            if q['id'] in saved[name] and not np.array_equal(np.asarray(prediction), np.asarray(saved[name][q['id']])):
                raise SystemExit('this pipeline does not reproduce the saved A60 prediction')
            checked += q['id'] in saved[name]
    if checked != 2 * len(qs):
        raise SystemExit(f'expected every J-R stream of both heads to be compared, got {checked}')
    return checked


def load_bank(bank, half):
    qs = np.load(data.EX1_RUN / 'streams' / f'{bank}_{half}.npy', allow_pickle=True).tolist()
    with np.load(data.EX1_RUN / 'forward' / f'{bank}_{half}_s0.npz') as z:
        a = split(z['probs'], z['position_counts'])
    with np.load(data.RUN / 'forward' / f'jepa_{bank}_{half}.npz') as z:
        j = split(z['probs'], z['position_counts'])
    if [len(x) for x in a] != [len(q['refs']) for q in qs] or [len(x) for x in j] != [len(q['refs']) for q in qs]:
        raise SystemExit('forward outputs do not align with the stream bank')
    params = read(CFG)['source']['a08']['original_B3']
    if any(float(params[k]) != float(io.b3_params(half, 0)[k]) for k in ('half_window', 'threshold', 'nms_radius')):
        raise SystemExit('A60 Base parameters differ from the original A08 parameters')
    base = [decode_with_prob('b3', p, {'decode': {'b3': params}})[0].astype(np.int64) for p in a]
    if any(not np.array_equal(x, decode_b3(p, params)) for x, p in zip(base, a)):
        raise SystemExit('A60 Base differs from the EX1 decoder')
    return qs, a, j, base


def action_effects(results, qs, b, policy, fit_f1):
    """True and predicted effects of the accepted action in each stream."""
    rows = {'accepted': 0, 'true_frame_change': 0, 'true_frame_harm_actions': 0, 'true_utility_positive': 0,
            'predicted_frame_change': 0., 'predicted_utility_positive': 0, 'predicted_safe_but_harmful': 0}
    for q, z, result in zip(qs, b, results):
        chosen = result[policy]
        if chosen['accepted'] is None:
            continue
        value, predicted = targets(z, chosen['accepted'], q['labels'], fit_f1), result['outputs'][chosen['accepted_index']]
        rows['accepted'] += 1
        rows['true_frame_change'] += int(value[1])
        rows['true_frame_harm_actions'] += int(value[1] > 0)
        rows['true_utility_positive'] += int(value[0] > 0)
        rows['predicted_frame_change'] += float(predicted[1])
        rows['predicted_utility_positive'] += int(predicted[0] > 0)
        rows['predicted_safe_but_harmful'] += int(value[1] > 0 and predicted[1] <= 0)
    return rows


def chunked_prediction(head, scales, q, x, y, z, policy):
    """One action per whole-clip chunk, with the same frozen head and policy. Also returns how many actions were accepted."""
    pieces, accepted = [], 0
    for start, end in ab.chunk_edges(q['clips']):
        chosen = infer(head['model'], head['norm'], scales, x[start:end], y[start:end], z[start:end], head['arm'])[policy]
        pieces.append(chosen['prediction'])
        accepted += chosen['accepted'] is not None
    return np.concatenate(pieces), accepted


def counted(qs, predictions):
    return mx.as_rows([mx.stream_counts(q['labels'], p) for q, p in zip(qs, predictions)])


def merged(qs, predictions):
    return mx.as_rows([mx.stream_counts(mx.merge_apical(q['labels']), mx.merge_apical(p)) for q, p in zip(qs, predictions)])


def contrast(rows, base_rows, clusters):
    frame = mx.FIELDS.index('correct')
    out = {'frame_errors_change': int(base_rows[:, frame].sum() - rows[:, frame].sum())}
    if clusters is None:
        out['segment_f1'] = {'delta': float(mx.f1_statistic(rows.sum(0)) - mx.f1_statistic(base_rows.sum(0))), 'ci95': None}
    else:
        out['segment_f1'] = mx.paired_delta(rows, base_rows, clusters, mx.f1_statistic)
    return out


def evaluate_bank(bank, half, heads, scales, fit_f1):
    qs, a, j, b = load_bank(bank, half)
    base_rows = counted(qs, b)
    clusters = [st.timestamp_of(q['clips'][0][0]) for q in qs] if bank == 'exam' else None
    records = [{'bank': bank, 'half': half, 'system': 'Base', 'budget': 'one_per_stream', **mx.pooled(base_rows), 'merged_segment_f1': mx.pooled(merged(qs, b))['segment_f1']}]
    for head in heads:
        results = [infer(head['model'], head['norm'], scales, x, y, z, head['arm']) for x, y, z in zip(a, j, b)]
        for policy in POLICIES:
            predictions = [r[policy]['prediction'] for r in results]
            rows = counted(qs, predictions)
            name = f"{head['arm']}-{policy}"
            records.append({'bank': bank, 'half': half, 'system': name, 'seed': head['seed'], 'budget': 'one_per_stream', **mx.pooled(rows),
                            'merged_segment_f1': mx.pooled(merged(qs, predictions))['segment_f1'], 'against_base': contrast(rows, base_rows, clusters),
                            'actions': action_effects(results, qs, b, policy, fit_f1)})
            if bank == 'exam' and name in MAIN_ARMS:
                pairs = [chunked_prediction(head, scales, q, x, y, z, policy) for q, x, y, z in zip(qs, a, j, b)]
                chunked = [p[0] for p in pairs]
                rows = counted(qs, chunked)
                records.append({'bank': bank, 'half': half, 'system': name, 'seed': head['seed'], 'budget': 'one_per_chunk', **mx.pooled(rows),
                                'merged_segment_f1': mx.pooled(merged(qs, chunked))['segment_f1'], 'against_base': contrast(rows, base_rows, clusters),
                                'chunks': sum(len(ab.chunk_edges(q['clips'])) for q in qs), 'accepted_actions': sum(p[1] for p in pairs)})
    return records


def rule(records, budget):
    verdict = {}
    for name in MAIN_ARMS:
        cells = [{'delta': r['against_base']['segment_f1']['delta'], 'ci_low': r['against_base']['segment_f1']['ci95'][0],
                  'frame_errors': r['against_base']['frame_errors_change']}
                 for r in records if r['bank'] == 'exam' and r['system'] == name and r['budget'] == budget]
        if len(cells) != 4:
            raise SystemExit('four cells required for the real-order rule')
        verdict[name] = {'survives': ab.survives(cells), 'cells': cells}
    return verdict


def main():
    heads, scales, fit_f1 = load_heads()
    parity = check_a60_parity(heads, scales)
    records = [r for bank in BANKS for half in io.HALVES for r in evaluate_bank(bank, half, heads, scales, fit_f1)]
    io.write_json(data.RUN / 'public/x1_a60_real_order.json', {'records': records, 'a60_predictions_reproduced_exactly': parity,
                                                               'rule_one_per_stream': rule(records, 'one_per_stream'), 'rule_one_per_chunk_secondary': rule(records, 'one_per_chunk')})
    print('x1 written', len(records), parity)


if __name__ == '__main__':
    main()
