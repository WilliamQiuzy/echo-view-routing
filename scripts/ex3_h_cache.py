#!/usr/bin/env python3
"""EX3 second extension, CPU only: rebuild Helena's frame cache for H's seeds 1 and 2, then prove it is H seed 0's cache.

Only the cache and its index.json are written. Z's geometry index and the stored public files are left alone, so this
is ex3_prepare.py's cache step on its own. Two checks must pass before any training:
  1. H seed 0's training identity of both stages reproduces with the new index.json;
  2. the first epoch's pixel digest of both stages equals H seed 0's training record.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import hashlib
import json
import time
from multiprocessing import Pool

import ex3_prepare as prep
import ex3_train as tr
import scripts.processed_input as pi
from helena_input import data

STAGES = ('v01', 'a08')
SEED0 = data.RUN / 'train' / data.run_key('H', 0)


def epoch_digest(window_digests):
    """The training loop's per-epoch pixel digest: sha256 over each update's window digests, in update order."""
    digest = hashlib.sha256()
    for windows in window_digests:
        digest.update(windows.numpy().tobytes())
    return digest.hexdigest()


def rebuild():
    """ex3_prepare.main's cache step, unchanged: the same recordings, frames, decoding, file names and index order."""
    if data.CACHE.exists():
        raise SystemExit('a cache folder already exists; remove it explicitly before a rebuild')
    v01, a08 = prep.contexts()
    allowed = data.allowed_rows()
    need = prep.needed_frames(v01, a08)
    if set(need) - set(allowed):
        raise SystemExit('a needed recording is outside the train and validation splits')
    rows = data.provider_rows(allowed)
    statuses = {v: data.provider_status(rows.get(v), allowed[v]['n_frames']) for v in need}
    usable = [v for v in need if statuses[v] is None]
    frames_total = sum(len(need[v]) for v in usable)
    if frames_total * prep.FRAME_BYTES > data.CACHE_BYTES_MAX:
        raise SystemExit(f'the cache would exceed its storage limit: {frames_total * prep.FRAME_BYTES / 1e9:.1f} GB')
    data.CACHE.mkdir(parents=True)
    jobs = [(v, rows[v]['output_path'], sorted(need[v]), int(allowed[v]['n_frames']), str(data.CACHE / f'{hashlib.sha256(v.encode()).hexdigest()[:20]}.npy')) for v in sorted(usable)]
    with Pool(prep.WORKERS) as pool:
        decoded = pool.map(pi._decode, jobs, chunksize=4)
    index = {'videos': {}}
    for (video, _, frames, n, _), record in zip(jobs, decoded):
        if record['fallback'] is not None:
            statuses[video] = record['fallback']
        else:
            index['videos'][video] = {'npy': record['npy'], 'frames': frames, 'n_frames': n, 'fallback': None}
    for video in need:
        if statuses[video] is not None:
            index['videos'][video] = {'fallback': statuses[video]}
    (data.CACHE / 'index.json').write_text(json.dumps(index))
    stored = sum(p.stat().st_size for p in data.CACHE.glob('*.npy'))
    return {'recordings_needed': len(need), 'frames_cached': frames_total, 'cache_bytes': int(stored)}


def stored_manifest(stage):
    return json.loads((SEED0 / stage / 'run_manifest.json').read_text())


def identity_reproduces(ctxs, stage):
    _, manifest_sha = ctxs[stage].manifest(0)
    start_sha = ctxs['v01'].init(0)[1] if stage == 'v01' else stored_manifest('v01')['final_sha256']
    return tr.identity('H', 0, stage, manifest_sha, start_sha) == stored_manifest(stage)['identity']


def first_epoch_reproduces(ctxs, stage):
    m, _ = ctxs[stage].manifest(0)
    pixels = tr.frame_source(ctxs, 'H', m)
    recorded = stored_manifest(stage)['epochs'][0]
    if recorded['updates'] != tr.EPOCH:
        raise SystemExit(f'{stage}: the first stored record is not the first epoch')
    return epoch_digest(tr.window_inputs(m, pixels, u)[3] for u in range(tr.EPOCH)) == recorded['pixels_digest']


def main():
    began = time.time()
    result = rebuild()
    ctxs = tr.contexts()
    result['identity_reproduces'] = {stage: identity_reproduces(ctxs, stage) for stage in STAGES}
    result['first_epoch_pixels_reproduce'] = {stage: first_epoch_reproduces(ctxs, stage) for stage in STAGES}
    result['ok'] = all(result['identity_reproduces'].values()) and all(result['first_epoch_pixels_reproduce'].values())
    result['wall_seconds'] = time.time() - began
    data.write_public('h_cache_rebuild.json', result)
    print(json.dumps(result), flush=True)
    if not result['ok']:
        raise SystemExit("the rebuilt cache is not H seed 0's cache; nothing trains")


if __name__ == '__main__':
    main()
