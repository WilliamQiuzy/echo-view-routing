#!/usr/bin/env python3
"""EX2 GPU job: JEPA on real-order streams, and A08 embeddings for the open-set test. Only frozen models; nothing is trained."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import subprocess
import time

import numpy as np
import torch

from echo_routing.evaluate.external_native import window_plan
from echo_routing.evaluate.window_target import window_frames
from echo_routing.temporal.dense_readout import ReadoutPixels
from echo_routing.temporal.structured_inputs import windows
from echo_routing.vit import target_mvit as tm
from exam_level import io
from open_routing import data
from scipy.special import logsumexp
from scripts.cache_jepa_proposals import build_source, jepa_sequence, reader
from scripts.external_original_tests import load_frames
from scripts.proposal_common import CFG, read, sequences

BANKS = ('exam', 'rewired')
BATCH = 32
GPU_FRACTION = .4
PARITY_TOLERANCE = 1e-3
PARITY_CLIPS = 12
JEPA_PARITY_CLIPS = 6
WAIT_LIMIT_SECONDS = 1800
WALL_LIMIT_SECONDS = 3300
EMBED_GROUPS = ('dev_selection', 'dev_calibration', 'train_reference', 'out_of_scope', 'dynamic', 'lvh', 'camus_2ch', 'camus_4ch')


def other_gpu_processes():
    text = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], capture_output=True, text=True, check=True).stdout
    return [line.strip() for line in text.splitlines() if line.strip() and int(line) != os.getpid()]


def wait_for_gpu():
    waited = 0
    while other_gpu_processes():
        if waited >= WAIT_LIMIT_SECONDS:
            raise SystemExit('GPU stayed busy for 30 minutes; not starting')
        time.sleep(60)
        waited += 60
    return waited


def jepa_parity(model, device):
    """The new forward must reproduce A60's cached JEPA probabilities on original recordings."""
    worst = 0.
    for half in io.HALVES:
        qs, cache = sequences(half), reader('jepa', half, readonly=True)
        natives = [i for i, q in enumerate(qs) if q['kind'] == 'native'][:JEPA_PARITY_CLIPS]
        pixels = ReadoutPixels([qs[i] for i in natives])
        for i in natives:
            worst = max(worst, float(np.abs(jepa_sequence(model, qs[i], pixels, device) - cache.load(i)).max()))
    if worst > PARITY_TOLERANCE:
        raise SystemExit(f'JEPA forward does not reproduce the A60 cache: {worst}')
    return worst


def run_jepa(device, deadline):
    model = build_source('jepa', read(CFG), device)
    receipt = {'parity_worst': jepa_parity(model, device), 'units': 0}
    for bank in BANKS:
        for half in io.HALVES:
            path = data.RUN / 'forward' / f'jepa_{bank}_{half}.npz'
            if path.exists() or time.time() > deadline:
                continue
            qs = np.load(data.EX1_RUN / 'streams' / f'{bank}_{half}.npy', allow_pickle=True).tolist()
            pixels = ReadoutPixels(qs)
            probs = [jepa_sequence(model, q, pixels, device) for q in qs]
            data.save_group(path, {'probs': np.concatenate(probs), 'position_counts': np.array([len(p) for p in probs])})
            receipt['units'] += sum((len(p) + 31) // 32 for p in probs)
    del model
    torch.cuda.empty_cache()
    return receipt


def build_a08(seed, device):
    state = torch.load(io.a08_weight_path(seed), map_location='cpu', weights_only=True, mmap=True)
    if len(state) != 400:
        raise SystemExit('A08 state key count changed')
    model = tm.build_model(None, tm.initial_new_state(0))
    model.load_state_dict(state, strict=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model.eval().to(device)


@torch.no_grad()
def embed_batches(model, device, batches):
    """batches yields (uint8 frames, marker slots). Returns window logits and embeddings."""
    logits, embeddings = [], []
    for frames, slots in batches:
        rgb = tm.to_rgb(torch.from_numpy(frames).to(device))
        z = model.embed(rgb, tm.marker_planes(torch.as_tensor(slots, device=device)))
        logits.append(model.classifier(z).float().cpu().numpy())
        embeddings.append(z.float().cpu().numpy())
    return np.concatenate(logits), np.concatenate(embeddings)


def native_batches(q, pixels):
    n = len(q['refs'])
    centres = windows(n, 5)['centre']
    for begin in range(0, len(centres), BATCH):
        cs = centres[begin:begin + BATCH]
        samples = [window_frames(int(c), n, 16, 16) for c in cs]
        frames = np.stack([[pixels.frame(*q['refs'][int(i)]) for i in row] for row in samples])
        yield frames, [tm.marker_slot(row, int(c)) for row, c in zip(samples, cs)]


def media_batches(frames):
    _, window_index, slots = window_plan(len(frames))
    for begin in range(0, len(window_index), BATCH):
        yield frames[window_index[begin:begin + BATCH]], slots[begin:begin + BATCH]


def summary(ids, logits_list, embeddings_list, extra=None):
    return {'ids': np.array(ids), 'window_counts': np.array([len(x) for x in logits_list]),
            'prob_mean': np.stack([io.softmax_mean(x) for x in logits_list]),
            'logit_mean': np.stack([x.mean(0) for x in logits_list]),
            'energy': np.array([float(logsumexp(x.astype(float), axis=1).mean()) for x in logits_list]),
            'embed_mean': np.stack([x.mean(0) for x in embeddings_list]), **(extra or {})}


def native_group(model, device, sequences_list):
    pixels = ReadoutPixels(sequences_list)
    pairs = [embed_batches(model, device, native_batches(q, pixels)) for q in sequences_list]
    return summary([q['id'] for q in sequences_list], [p[0] for p in pairs], [p[1] for p in pairs])


def media_group(model, device, rows):
    pairs = [embed_batches(model, device, media_batches(load_frames(q)[0])) for q in rows]
    return summary([q['id'] for q in rows], [p[0] for p in pairs], [p[1] for p in pairs], {'groups': np.array([q['group'] for q in rows])})


def group_builder(name):
    """How to obtain one embedding group. The private manifest rows carry only media and timing fields here."""
    if name == 'dev_selection':
        return 'native', io.natives('selection')
    if name == 'dev_calibration':
        return 'native', io.natives('calibration')
    if name == 'train_reference':
        return 'native', [data.native_from_row(r) for r in data.train_reference_rows()]
    if name == 'out_of_scope':
        return 'native', [data.native_from_row(r) for r in data.out_of_scope_rows()]
    if name == 'dynamic':
        return 'media', [r for _, r in data.dynamic_sample()]
    if name == 'lvh':
        return 'media', data.external_rows('lvh')
    if name == 'camus_2ch':
        return 'media', [data.camus_row(i, '2CH') for i in range(1, data.CAMUS_PATIENTS + 1)]
    if name == 'camus_4ch':
        return 'media', [data.camus_row(i, '4CH') for i in range(1, data.CAMUS_PATIENTS + 1)]
    raise ValueError('declared embedding group required')


def a08_parity(model, device, seed):
    """Embedding forward must reproduce original window probabilities on development and external recordings."""
    worst = 0.
    for half in io.HALVES:
        cache = io.A08Cache(half, seed)
        clips = io.natives(half)[:PARITY_CLIPS]
        pixels = ReadoutPixels(clips)
        for q in clips:
            logits, _ = embed_batches(model, device, native_batches(q, pixels))
            worst = max(worst, float(np.abs(io.softmax_mean(logits) - cache.window_mean(q['id'])).max()))
    for dataset in io.EXTERNAL_DATASETS:
        records, rows = io.external_records(dataset)[:PARITY_CLIPS], data.external_rows(dataset)[:PARITY_CLIPS]
        for record, row in zip(records, rows):
            logits, _ = embed_batches(model, device, media_batches(load_frames(row)[0]))
            worst = max(worst, float(np.abs(io.softmax_mean(logits) - io.softmax_mean(record[f'logits_s{seed}'])).max()))
    if worst > PARITY_TOLERANCE:
        raise SystemExit(f'embedding forward does not reproduce cached A08 outputs: {worst}')
    return worst


def main():
    waited = wait_for_gpu()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(GPU_FRACTION)
    device = torch.device('cuda')
    start, cpu_start = time.time(), time.process_time()
    deadline = start + WALL_LIMIT_SECONDS
    receipt = {'gpu_wait_seconds': waited, 'jepa': run_jepa(device, deadline), 'a08_parity_worst': {}, 'windows': 0, 'stopped_at_wall_limit': False}
    for seed in io.SEEDS:
        model = build_a08(seed, device)
        receipt['a08_parity_worst'][f's{seed}'] = a08_parity(model, device, seed)
        for name in EMBED_GROUPS:
            path = data.RUN / 'forward' / f'embed_{name}_s{seed}.npz'
            if path.exists():
                continue
            if time.time() > deadline:
                receipt['stopped_at_wall_limit'] = True
                continue
            kind, members = group_builder(name)
            values = native_group(model, device, members) if kind == 'native' else media_group(model, device, members)
            data.save_group(path, values)
            receipt['windows'] += int(values['window_counts'].sum())
            io.write_json(data.RUN / 'forward_progress.json', {'done': path.name, 'wall': time.time() - start})
        del model
        torch.cuda.empty_cache()
    receipt.update({'wall_seconds': time.time() - start, 'process_cpu_seconds': time.process_time() - cpu_start,
                    'max_gpu_bytes': int(torch.cuda.max_memory_allocated())})
    io.write_json(data.RUN / f'public/forward_receipt_{int(start)}.json', receipt)
    print('forward complete', receipt)


if __name__ == '__main__':
    main()
