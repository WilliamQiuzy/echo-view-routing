#!/usr/bin/env python3
"""EX1 E1 readout: A08 raw and B3 on exam, shuffled and rewired streams against the constructed bank. CPU only."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import csv

import numpy as np

from exam_level import io, metrics as mx, streams as st
from exam_level.decode import decode_b3

DECODERS = ('raw', 'b3')
STATISTICS = {'segment_f1': mx.f1_statistic, 'change_recall': mx.ratio_statistic('change_matched', 'change_true')}


def split_positions(values, counts):
    edges = np.cumsum([0, *counts])
    return [values[a:b] for a, b in zip(edges, edges[1:])]


def decode_rows(sequences, probs, params):
    """Per-decoder rows for five-class and merged-apical labels."""
    out = {}
    for decoder in DECODERS:
        preds = [p.argmax(1) if decoder == 'raw' else decode_b3(p, params) for p in probs]
        out[decoder] = mx.as_rows([mx.stream_counts(q['labels'], x) for q, x in zip(sequences, preds)])
        out[decoder + '_merged'] = mx.as_rows([mx.stream_counts(mx.merge_apical(q['labels']), mx.merge_apical(x))
                                               for q, x in zip(sequences, preds)])
    return out


def exam_cluster(q, bank):
    """Exam and shuffled streams resample by timestamp. Constructed streams resample by stream, as in A59."""
    return st.timestamp_of(q['clips'][0][0]) if bank in ('exam', 'shuffled') else q['id']


def bank_rows(half, seed):
    params = io.b3_params(half, seed)
    banks = {}
    for mode in st.BANKS:
        if mode.startswith('rewired_d') and seed != 0:
            continue
        sequences = np.load(io.RUN / 'streams' / f'{mode}_{half}.npy', allow_pickle=True).tolist()
        with np.load(io.RUN / 'forward' / f'{mode}_{half}_s{seed}.npz') as z:
            if z['position_counts'].tolist() != [len(q['refs']) for q in sequences]:
                raise SystemExit('forward output does not align with the stream bank')
            probs = split_positions(z['probs'], z['position_counts'])
        banks[mode] = {'rows': decode_rows(sequences, probs, params), 'clusters': [exam_cluster(q, mode) for q in sequences]}
    cache = io.A08Cache(half, seed)
    constructed = [q for q in io.role_sequences(half) if q['kind'] == 'multi']
    banks['constructed'] = {'rows': decode_rows(constructed, [cache.positions(q['id']) for q in constructed], params),
                            'clusters': [q['id'] for q in constructed]}
    return banks


def cell_records(half, seed, banks):
    records = []
    for bank, value in banks.items():
        for decoder, rows in value['rows'].items():
            summary = mx.pooled(rows)
            # Rewired banks scatter each session over many streams, so a stream bootstrap would be too narrow.
            ci = [None, None] if bank.startswith('rewired') else mx.unpaired_ci(rows, value['clusters'], mx.f1_statistic)
            records.append({'half': half, 'seed': seed, 'bank': bank, 'decoder': decoder, **summary,
                            'segment_f1_ci_low': ci[0], 'segment_f1_ci_high': ci[1]})
    return records


def comparisons(half, seed, banks):
    """Paired only where streams truly pair: same bank across decoders, and exam against shuffled by session."""
    out = []
    exam, shuffled = banks['exam'], banks['shuffled']

    def add(name, a, b, clusters):
        for stat, function in STATISTICS.items():
            out.append({'half': half, 'seed': seed, 'comparison': name, 'statistic': stat, **mx.paired_delta(a, b, clusters, function)})

    for bank in ('exam', 'shuffled', 'constructed'):
        add(f'{bank}: b3 - raw', banks[bank]['rows']['b3'], banks[bank]['rows']['raw'], banks[bank]['clusters'])
    if exam['clusters'] != [c for c in shuffled['clusters']]:
        raise SystemExit('exam and shuffled banks are not paired by session')
    add('b3: exam - shuffled', exam['rows']['b3'], shuffled['rows']['b3'], exam['clusters'])
    add('raw: exam - shuffled', exam['rows']['raw'], shuffled['rows']['raw'], exam['clusters'])
    return out


def main():
    records, compared = [], []
    for half in io.HALVES:
        for seed in io.SEEDS:
            banks = bank_rows(half, seed)
            records += cell_records(half, seed, banks)
            compared += comparisons(half, seed, banks)
    io.write_json(io.RUN / 'public/benchmark.json', {'cells': records, 'comparisons': compared})
    with (io.RUN / 'public/benchmark.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print('benchmark readout written', len(records), len(compared))


if __name__ == '__main__':
    main()
