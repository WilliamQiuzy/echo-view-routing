#!/usr/bin/env python3
"""EX1 stage 1: build exam streams and prove the metric code reproduces the published A08 numbers. CPU only."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np

from exam_level import io, metrics as mx, streams as st
from exam_level.decode import decode_b3
from exam_level.metrics import macro_f1

PUBLISHED_S0 = {'selection': {'multi_f1': .956349, 'native_macro': .978591},
                'calibration': {'multi_f1': .982692, 'native_macro': .992942}}
TOLERANCE = 1e-5
CLASSES = ('PLAX', 'PSAX', 'A4C', 'A5C', 'SC4C')


def reference(half, seed):
    """Multi-stream segment F1 and native macro-F1 of A08 + B3 from the original cache."""
    cache, params = io.A08Cache(half, seed), io.b3_params(half, seed)
    rows, conf = [], np.zeros((5, 5), int)
    for q in io.role_sequences(half):
        if q['kind'] == 'multi':
            rows.append(mx.stream_counts(q['labels'], decode_b3(cache.positions(q['id']), params)))
        elif q['kind'] == 'native':
            vote = np.bincount(decode_b3(cache.positions(q['id']), params), minlength=5).argmax()
            conf[int(q['labels'][0]), vote] += 1
    return {'multi': mx.pooled(mx.as_rows(rows)), 'native_macro': float(macro_f1(conf))}


def describe(sequences):
    """Aggregate structure of one bank."""
    labels = np.concatenate([q['labels'] for q in sequences])
    joins = [(q['labels'][j - 1], q['labels'][j]) for q in sequences for j in q['joins']]
    return {'streams': len(sequences), 'positions': int(labels.size),
            'position_share': dict(zip(CLASSES, np.round(np.bincount(labels, minlength=5) / labels.size, 4).tolist())),
            'joins': len(joins), 'same_view_joins': sum(a == b for a, b in joins),
            'length_p10_p50_p90': np.percentile([len(q['labels']) for q in sequences], [10, 50, 90]).tolist(),
            'true_segments': sum(len(np.flatnonzero(np.diff(q['labels']))) + 1 for q in sequences)}


def main():
    out = {'references': {}, 'banks': {}}
    for half in io.HALVES:
        out['references'][half] = {f's{s}': reference(half, s) for s in io.SEEDS}
        got, want = out['references'][half]['s0'], PUBLISHED_S0[half]
        if abs(got['multi']['segment_f1'] - want['multi_f1']) > TOLERANCE or abs(got['native_macro'] - want['native_macro']) > TOLERANCE:
            raise SystemExit(f'metric code does not reproduce published A08 values on {half}: {got}')
        natives = io.natives(half)
        out['banks'][half] = {'constructed_multi': describe([q for q in io.role_sequences(half) if q['kind'] == 'multi'])}
        for bank, (mode, seed) in st.BANKS.items():
            streams = st.exam_streams(natives, mode, half, seed=seed)
            used = sorted(c[0] for s in streams for c in s['clips'])
            if used != sorted(q['id'] for q in natives):
                raise SystemExit('stream bank does not use each native clip exactly once')
            path = io.RUN / 'streams' / f'{bank}_{half}.npy'
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, np.array(streams, dtype=object), allow_pickle=True)
            out['banks'][half][bank] = describe(streams)
    io.write_json(io.RUN / 'public/prepare.json', out)
    print('reference parity passed and banks written')


if __name__ == '__main__':
    main()
