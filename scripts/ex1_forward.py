#!/usr/bin/env python3
"""EX1 stage 2: frozen A08 forward across real joins of the exam banks, plus out-of-scope clips. Short GPU job."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
import subprocess
import time

import numpy as np
import torch

from echo_routing.evaluate.window_target import nearest_centre, window_frames
from echo_routing.features.sampling import stride_indices
from echo_routing.temporal.dense_readout import ReadoutPixels
from echo_routing.temporal.structured_inputs import windows
from echo_routing.vit import target_mvit as tm
from exam_level import io, streams as st

PARITY_TOLERANCE = 1e-3
PARITY_CLIPS = 12
BATCH = 32
GPU_FRACTION = .4
OUT_OF_SCOPE = 'PMPALA'


WAIT_LIMIT_SECONDS = 1800
WALL_LIMIT_SECONDS = 1800


def wait_for_gpu():
    waited = 0
    while other_gpu_processes():
        if waited >= WAIT_LIMIT_SECONDS:
            raise SystemExit('GPU stayed busy for 30 minutes; not starting')
        time.sleep(60)
        waited += 60
    return waited


def other_gpu_processes():
    text = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'],
                          capture_output=True, text=True, check=True).stdout
    return [line.strip() for line in text.splitlines() if line.strip() and int(line) != os.getpid()]


def build(seed, device):
    path = io.a08_weight_path(seed)
    state = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    if len(state) != 400:
        raise SystemExit('A08 state key count changed')
    model = tm.build_model(None, tm.initial_new_state(0))
    model.load_state_dict(state, strict=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model.eval().to(device)


@torch.no_grad()
def predict(model, q, pixels, device):
    """Window logits with the original A08 sampling; windows may read across a join because refs are the whole stream."""
    n = len(q['refs'])
    centres = windows(n, 5)['centre']
    out = []
    for begin in range(0, len(centres), BATCH):
        cs = centres[begin:begin + BATCH]
        samples = [window_frames(int(c), n, 16, 16) for c in cs]
        frames = np.stack([[pixels.frame(*q['refs'][int(i)]) for i in row] for row in samples])
        markers = torch.tensor([tm.marker_slot(row, int(c)) for row, c in zip(samples, cs)], device=device)
        out.append(model(tm.to_rgb(torch.from_numpy(frames).to(device)), tm.marker_planes(markers)).float().cpu().numpy())
    logits = np.concatenate(out)
    prob = torch.softmax(torch.from_numpy(logits), 1).numpy()[nearest_centre(n, centres)]
    return logits, centres, prob


def save_bank(path, sequences, model, pixels, device):
    temporary = path.with_suffix('.tmp.npz')
    logits, centres, probs, lengths = [], [], [], []
    for q in sequences:
        a, b, c = predict(model, q, pixels, device)
        logits.append(a), centres.append(b), probs.append(c), lengths.append(len(a))
    np.savez(temporary, logits=np.concatenate(logits), centres=np.concatenate(centres), probs=np.concatenate(probs),
             window_counts=np.array(lengths), position_counts=np.array([len(q['refs']) for q in sequences]))
    temporary.replace(path)
    return sum(lengths)


def out_of_scope_clips():
    rows = [r for r in io.manifest_rows('validation') if r['raw_label'] == OUT_OF_SCOPE]
    return [{'id': r['video_id'], 'kind': 'native', 'labels': np.zeros(0),
             'refs': [(r['video_id'], int(i), 1.0) for i in stride_indices(int(r['n_frames']), float(r['fps_playback']), 10.)]}
            for r in rows]


def check_parity(model, device, seed):
    """The new forward must reproduce original cached positions, and the reference rule must rebuild the natives."""
    worst = 0.
    rows = {r['video_id']: r for r in io.manifest_rows('validation')}
    for half in io.HALVES:
        natives, cache = io.natives(half)[:PARITY_CLIPS], io.A08Cache(half, seed)
        for q in natives:
            rebuilt = stride_indices(int(rows[q['id']]['n_frames']), float(rows[q['id']]['fps_playback']), 10.)
            if [int(r[1]) for r in q['refs']] != rebuilt.tolist():
                raise SystemExit('stride rule does not rebuild native references')
        pixels = ReadoutPixels(natives)
        for q in natives:
            worst = max(worst, float(np.abs(predict(model, q, pixels, device)[2] - cache.positions(q['id'])).max()))
    if worst > PARITY_TOLERANCE:
        raise SystemExit(f'forward does not reproduce cached A08 probabilities: {worst}')
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
    receipt = {'windows': 0, 'parity_worst_probability_difference': {}, 'gpu_wait_seconds': waited, 'stopped_at_wall_limit': False}
    folder = io.RUN / 'forward'
    folder.mkdir(parents=True, exist_ok=True)
    oos = out_of_scope_clips()
    for seed in io.SEEDS:
        model = build(seed, device)
        receipt['parity_worst_probability_difference'][f's{seed}'] = check_parity(model, device, seed)
        for half in io.HALVES:
            for bank in st.BANKS:
                path = folder / f'{bank}_{half}_s{seed}.npz'
                if path.exists() or (bank.startswith('rewired_d') and seed != 0):
                    continue
                if time.time() - start > WALL_LIMIT_SECONDS:
                    receipt['stopped_at_wall_limit'] = True
                    continue
                sequences = np.load(io.RUN / 'streams' / f'{bank}_{half}.npy', allow_pickle=True).tolist()
                receipt['windows'] += save_bank(path, sequences, model, ReadoutPixels(sequences), device)
                io.write_json(io.RUN / 'forward_progress.json', {'done': path.name, 'wall': time.time() - start})
        path = folder / f'out_of_scope_s{seed}.npz'
        if not path.exists():
            receipt['windows'] += save_bank(path, oos, model, ReadoutPixels(oos), device)
        del model
        torch.cuda.empty_cache()
    receipt.update({'wall_seconds': time.time() - start, 'process_cpu_seconds': time.process_time() - cpu_start,
                    'max_gpu_bytes': int(torch.cuda.max_memory_allocated()), 'out_of_scope_clips': len(oos)})
    io.write_json(io.RUN / f'public/forward_receipt_{int(start)}.json', receipt)
    print('forward complete', receipt)


if __name__ == '__main__':
    main()
