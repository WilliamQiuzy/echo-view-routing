#!/usr/bin/env python3
"""EX2 post-hoc: cut development clips to CAMUS length and apply the same thresholds. Separates short input from a new dataset."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch

import ex2_forward as fw
import ex2_openset as xs
from echo_routing.temporal.dense_readout import ReadoutPixels
from exam_level import io
from open_routing import data, openset as os_

os.environ.pop('CUDA_VISIBLE_DEVICES', None)  # ex2_openset hides the GPU when imported; this job needs it.
POSITIONS = 4


def central(q):
    """The middle positions of one recording, as a recording of their own."""
    n = len(q['refs'])
    start = max(0, n // 2 - POSITIONS // 2)
    return {**q, 'refs': q['refs'][start:start + POSITIONS]}


def main():
    if fw.other_gpu_processes():
        raise SystemExit('GPU busy; not starting')
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(fw.GPU_FRACTION)
    device = torch.device('cuda')
    clips = [central(q) for half in io.HALVES for q in io.natives(half)]
    full = io.read_json(data.RUN / 'public/x3_open_set.json')['rows']
    out = []
    for seed in io.SEEDS:
        model = fw.build_a08(seed, device)
        pixels = ReadoutPixels(clips)
        pairs = [fw.embed_batches(model, device, fw.native_batches(q, pixels)) for q in clips]
        group = fw.summary([q['id'] for q in clips], [p[0] for p in pairs], [p[1] for p in pairs])
        reference = xs.load('train_reference', seed)
        labels = np.array([int(q['labels'][0]) for q in clips])
        for method, score in xs.scorers(reference).items():
            row = next(r for r in full if r['seed'] == seed and r['method'] == method)
            values = score(group)
            camus = score(xs.load('camus_4ch', seed))
            out.append({'seed': seed, 'method': method, 'windows_per_clip': float(group['window_counts'].mean()), 'clips': len(clips),
                        'short_dev_kept_at_full_threshold': os_.accept_share(values, row['threshold']), 'camus_4ch_kept_at_full_threshold': row['camus_4ch_retain'],
                        'median_score_short_dev': float(np.median(values)), 'median_score_camus_4ch': float(np.median(camus)),
                        'auroc_short_dev_vs_camus_4ch': os_.auroc(values, camus),
                        'short_dev_five_class_accuracy': float((group['prob_mean'].argmax(1) == labels).mean())})
        del model
        torch.cuda.empty_cache()
    io.write_json(data.RUN / 'public/posthoc_short_input.json', out)
    print('short input written', len(out))


if __name__ == '__main__':
    main()
