#!/usr/bin/env python3
"""EX3 preparation, CPU only: the frames each input needs, Helena's frame cache, the Z geometry index, the side ratio and the alignment check."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import collections
import hashlib
import json
import pickle
import time
from multiprocessing import Pool
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image

import scripts.prepare_raw_video_context as prv
import scripts.processed_input as pi
import scripts.report_segment_objective as rso
import scripts.report_video_switch_head as rsh
import scripts.report_video_within_view as rw
import scripts.train_raw_video_context as rvc
import scripts.train_video_within_view as twv
from echo_routing.config import load_paths
from exam_level import io
from helena_input import data, geometry as gm
from open_routing import data as ex2
from scripts._common import load_cfg
from scripts.train_joint_visual_temporal import FramePaths

WORKERS = 6
SPOT_VIDEOS = 20
SPOT_SEED = 2026110402
FRAME_BYTES = 224 * 224 * 3


def contexts():
    v01 = rvc.Ctx(load_cfg(SimpleNamespace(config=data.V01_CONFIG, set=[])), data.V01_CONFIG)
    a08 = twv.Ctx(load_cfg(SimpleNamespace(config=data.A08_CONFIG, set=[])), data.A08_CONFIG)
    return v01, a08


def needed_frames(v01, a08):
    """Every (recording, frame) any training, development or reference forward can ask for."""
    need = collections.defaultdict(set)
    for seed in io.SEEDS:
        for ctx in (v01, a08):
            m, _ = ctx.manifest(seed)
            for video, frame in zip(m['videos'][m['vid']].ravel().tolist(), m['frame'].ravel().tolist()):
                need[str(video)].add(int(frame))
    ctx8 = rsh.a08_ctx()
    for half in io.HALVES:
        _, sequences = ctx8.half(half, rso.DevAccess({half}))
        for video, frame, _ in rw.half_refs(sequences):
            need[str(video)].add(int(frame))
        for q in io.natives(half):
            for video, frame, _ in q['refs']:
                need[str(video)].add(int(frame))
        for q in np.load(io.RUN / 'streams' / f'exam_{half}.npy', allow_pickle=True).tolist():
            for video, frame, _ in q['refs']:
                need[str(video)].add(int(frame))
    for row in [*ex2.train_reference_rows(), *ex2.out_of_scope_rows()]:
        for video, frame, _ in ex2.native_from_row(row)['refs']:
            need[str(video)].add(int(frame))
    return need


def z_entry(row):
    """One recording's centre and layout segments from Helena's stored analysis."""
    meta = json.loads((data.HELENA / 'metadata' / f"{row['video_id']}.json").read_text())
    segments = []
    for sid, segment in enumerate(meta['segments']):
        path = data.HELENA / 'masks' / row['video_id'] / f'segment_{sid:03d}' / 'final.png'
        with Image.open(path) as opened:
            mask = np.asarray(opened.convert('L')) > 127
        if mask.shape != (int(row['height']), int(row['width'])) or not mask.any():
            raise ValueError('unexpected stored mask')
        segments.append({'start': int(segment['start_frame']), 'end': int(segment['end_frame_exclusive']), 'mask': gm.pack_mask(mask)})
    return {'cx': float(meta['crop']['crop_center_x']), 'cy': float(meta['crop']['crop_center_y']), 'segments': segments}


def spot_check(rows, statuses, frame_paths, allowed):
    """Helena's frames must align with ours at offset zero."""
    candidates = sorted(v for v, s in statuses.items() if s is None and allowed[v]['split'] == 'train')
    picks = np.random.default_rng(SPOT_SEED).choice(len(candidates), min(SPOT_VIDEOS, len(candidates)), replace=False)
    checks = []
    for video in (candidates[int(i)] for i in picks):
        capture = cv2.VideoCapture(rows[video]['output_path'])
        decoded = []
        while True:
            ok, image = capture.read()
            if not ok:
                break
            decoded.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        capture.release()
        paths, n = frame_paths(video), len(decoded)
        for frame in (2, n // 2, n - 3):
            with Image.open(paths[frame]) as opened:
                original = pi.raw_in_pro_geometry(np.asarray(opened.convert('RGB')), rows[video])
            scores = {d: pi._ncc(original, decoded[frame + d]) for d in range(-2, 3) if 0 <= frame + d < n}
            checks.append(max(scores, key=scores.get) == 0 or scores[0] >= max(scores.values()) - 1e-3)
    return {'frames': len(checks), 'peak_at_zero': int(sum(checks))}


def main():
    started = time.time()
    data.PUBLIC.mkdir(parents=True, exist_ok=True)
    v01, a08 = contexts()
    allowed = data.allowed_rows()
    need = needed_frames(v01, a08)
    if set(need) - set(allowed):
        raise SystemExit('a needed recording is outside the train and validation splits')
    rows = data.provider_rows(allowed)
    statuses = {v: data.provider_status(rows.get(v), allowed[v]['n_frames']) for v in need}
    usable = [v for v in need if statuses[v] is None]
    frames_total = sum(len(need[v]) for v in usable)
    if frames_total * FRAME_BYTES > data.CACHE_BYTES_MAX:
        raise SystemExit(f'the cache would exceed its storage limit: {frames_total * FRAME_BYTES / 1e9:.1f} GB')
    data.CACHE.mkdir(parents=True, exist_ok=True)
    jobs = [(v, rows[v]['output_path'], sorted(need[v]), int(allowed[v]['n_frames']), str(data.CACHE / f'{hashlib.sha256(v.encode()).hexdigest()[:20]}.npy')) for v in sorted(usable)]
    with Pool(WORKERS) as pool:
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
    fov = {v: (z_entry(rows[v]) if statuses[v] is None else {'fallback': statuses[v]}) for v in need}
    with data.FOV_INDEX.open('wb') as handle:
        pickle.dump(fov, handle, protocol=pickle.HIGHEST_PROTOCOL)
    sides = [float(r['crop_size_original_pixels']) for r in rows.values() if r['crop_size_original_pixels']]
    ratio = gm.side_ratio(gm.a08_side(320, 240), sides)
    data.write_public(data.RATIO.name, {'ratio': ratio, 'a08_source_side': gm.a08_side(320, 240), 'median_helena_side': float(np.median(sides)), 'recordings': len(sides)})
    frame_paths = FramePaths(prv.frame_dirs(load_paths(), v01.rv, set(need)))
    spot = spot_check(rows, statuses, frame_paths, allowed)
    if spot['peak_at_zero'] != spot['frames']:
        raise SystemExit(f'alignment check failed: {spot}')
    counts = collections.Counter(f"{allowed[v]['split']}|{statuses[v] or 'output'}" for v in need)
    stored = sum(p.stat().st_size for p in Path(data.CACHE).glob('*.npy'))
    data.write_public('preparation.json', {'recordings_needed': len(need), 'status_by_split': dict(sorted(counts.items())), 'frames_cached': frames_total,
                                           'cache_bytes': int(stored), 'spot_check': spot, 'side_ratio': ratio, 'wall_seconds': time.time() - started})
    print('prepared', dict(counts), frames_total, stored / 1e9, spot, ratio)


if __name__ == '__main__':
    main()
