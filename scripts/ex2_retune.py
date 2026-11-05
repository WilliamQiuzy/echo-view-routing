#!/usr/bin/env python3
"""EX2 X4: tune B3 on one half's real-order streams and measure on the other half. A diagnostic only. CPU."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np

from echo_routing.temporal.baselines.b3_js_divergence import boundary_scores, detect_boundaries, labels_from_boundaries
from exam_level import io, metrics as mx, streams as st
from exam_level.decode import decode_b3
from open_routing import data, retune as rt

N_CLASSES = 5
OTHER = {'selection': 'calibration', 'calibration': 'selection'}


def exam_bank(half, seed):
    qs = np.load(data.EX1_RUN / 'streams' / f'exam_{half}.npy', allow_pickle=True).tolist()
    with np.load(data.EX1_RUN / 'forward' / f'exam_{half}_s{seed}.npz') as z:
        edges = np.cumsum([0, *z['position_counts']])
        probs = [z['probs'][a:b] for a, b in zip(edges, edges[1:])]
    return qs, probs


def decode(prob, setting, threshold):
    cuts = detect_boundaries(prob, setting['half_window'], threshold, setting['nms_radius'])
    return labels_from_boundaries(prob, cuts)


def native_macro_f1(clips, probs, setting, threshold):
    """Macro-F1 of the majority decoded label per original recording."""
    confusion = np.zeros((N_CLASSES, N_CLASSES))
    for q, p in zip(clips, probs):
        confusion[int(q['labels'][0]), int(np.bincount(decode(p, setting, threshold), minlength=N_CLASSES).argmax())] += 1
    return float(mx.macro_f1(confusion))


def fit(half, seed):
    """The grid in its declared order on one half; thresholds are quantiles of its original recordings' scores."""
    cache = io.A08Cache(half, seed)
    clips = io.natives(half)
    native_probs = [cache.positions(q['id']) for q in clips]
    qs, probs = exam_bank(half, seed)
    scores = {h: np.concatenate([boundary_scores(p, h) for p in native_probs]) for h in rt.HALF_WINDOWS}
    rows = []
    for index, setting in enumerate(rt.settings()):
        threshold = float(np.quantile(scores[setting['half_window']], setting['quantile']))
        pooled = mx.pooled(mx.as_rows([mx.stream_counts(q['labels'], decode(p, setting, threshold)) for q, p in zip(qs, probs)]))
        rows.append({'setting': index, **setting, 'threshold': threshold, 'segment_f1': pooled['segment_f1'], 'change_recall': pooled['change_recall'],
                     'native_macro_f1': native_macro_f1(clips, native_probs, setting, threshold)})
    return rt.select(rows)


def compare(half, seed, chosen):
    """Retuned minus original B3 on a half that played no part in the choice."""
    qs, probs = exam_bank(half, seed)
    original = mx.as_rows([mx.stream_counts(q['labels'], decode_b3(p, io.b3_params(half, seed))) for q, p in zip(qs, probs)])
    retuned = mx.as_rows([mx.stream_counts(q['labels'], decode(p, chosen, chosen['threshold'])) for q, p in zip(qs, probs)])
    clusters = [st.timestamp_of(q['clips'][0][0]) for q in qs]
    return {'test_half': half, 'seed': seed, 'chosen': chosen, 'original_params': io.b3_params(half, seed), 'original': mx.pooled(original), 'retuned': mx.pooled(retuned),
            'segment_f1': mx.paired_delta(retuned, original, clusters, mx.f1_statistic),
            'frame_errors_change': int(original[:, mx.FIELDS.index('correct')].sum() - retuned[:, mx.FIELDS.index('correct')].sum())}


def consistent(rows, half):
    """Project convention: all three seeds' intervals exclude zero on the same side."""
    intervals = [r['segment_f1']['ci95'] for r in rows if r['test_half'] == half]
    return {'better': all(c[0] > 0 for c in intervals), 'worse': all(c[1] < 0 for c in intervals)}


def main():
    rows = [compare(half, seed, fit(OTHER[half], seed)) for half in io.HALVES for seed in io.SEEDS]
    io.write_json(data.RUN / 'public/x4_b3_retune.json', {'rows': rows, 'consistent': {h: consistent(rows, h) for h in io.HALVES}})
    print('x4 written', len(rows))


if __name__ == '__main__':
    main()
