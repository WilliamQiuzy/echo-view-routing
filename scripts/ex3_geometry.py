#!/usr/bin/env python3
"""EX3 external geometry: a read-only copy of Helena's committed code, run in this round's folder.

  --parity            30 EV9V training recordings must reproduce Helena's stored crop and mask
  --group NAME        analyse one external group (dynamic, lvh, camus_2ch, camus_4ch)
"""
import argparse
import json
import pickle
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from exam_level import io
from helena_input import data, external as ex, geometry as gm
from open_routing import data as ex2

PARITY_VIDEOS = 30
PARITY_SEED = 2026110404
MIN_SAME_CROP = 28
MIN_MEDIAN_IOU = .99
MIN_H_NCC = .98
EARLY_RECORDINGS = 25
EARLY_FAILURES_MAX = 20
CHECKPOINT = data.RUN / 'helena_copy/FastSAM-s.pt'
CHECKPOINT_SHA256 = 'c9f78716a81c7aff0d608ccc73e1b82ab3aaad86005049f6a92106a0be6d0844'


def config():
    cfg = json.loads(data.HELENA_CONFIG.read_text())
    if io.sha256(CHECKPOINT) != CHECKPOINT_SHA256:
        raise SystemExit('the FastSAM weights differ from Helena\'s recorded provenance')
    return {**cfg, 'checkpoint': str(CHECKPOINT)}


def generator(cfg):
    from echo_fov.ev9v_oct16.candidates import CandidateGenerator
    return CandidateGenerator(cfg)


def groups():
    return {'dynamic': [r for _, r in ex2.dynamic_sample()], 'lvh': ex2.external_rows('lvh'),
            'camus_2ch': [ex2.camus_row(i, '2CH') for i in range(1, ex2.CAMUS_PATIENTS + 1)],
            'camus_4ch': [ex2.camus_row(i, '4CH') for i in range(1, ex2.CAMUS_PATIENTS + 1)]}


def h_frame_match(frames, geometry, rendered_path):
    """My external H input on one frame against the frame Helena rendered: only the video compression in Helena's render may separate them."""
    middle = len(frames) // 2
    capture = cv2.VideoCapture(rendered_path)
    capture.set(cv2.CAP_PROP_POS_FRAMES, middle)
    ok, image = capture.read()
    capture.release()
    if not ok:
        raise SystemExit('a rendered frame could not be decoded')
    mine = ex.h_input(frames[middle], geometry, middle, None).astype(np.float64).ravel()
    theirs = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float64).ravel()
    mine, theirs = mine - mine.mean(), theirs - theirs.mean()
    return float((mine * theirs).sum() / (np.sqrt((mine * mine).sum() * (theirs * theirs).sum()) + 1e-12))


def parity(cfg, make):
    allowed = data.allowed_rows()
    train = {v: r for v, r in allowed.items() if r['split'] == 'train'}
    rows = data.provider_rows(train)
    usable = sorted(v for v, r in rows.items() if data.provider_status(r, train[v]['n_frames']) is None)
    picks = [usable[int(i)] for i in np.random.default_rng(PARITY_SEED).choice(len(usable), PARITY_VIDEOS, replace=False)]
    same, ious, matches = 0, [], []
    from echo_fov.ev9v_oct16.sampling import sample_indices
    for video in picks:
        row = rows[video]
        frames = ex.read_recording({'kind': 'video', 'path': row['input_path']})
        if sample_indices(len(frames)) != json.loads(row['sampled_frame_indices']):
            raise SystemExit('the sampled frames differ from Helena\'s record')
        mine = ex.geometry_of(frames, cfg, make)
        if mine.get('failed'):
            raise SystemExit(f"the analysis failed on a recording Helena processed: {mine['failed']}")
        matches.append(h_frame_match(frames, mine, row['output_path']))
        stored = json.loads((data.HELENA / 'metadata' / f"{row['video_id']}.json").read_text())
        same += all(float(mine['plan'][k]) == float(stored['crop'][k]) for k in ('crop_left', 'crop_top', 'crop_right', 'crop_bottom'))
        with Image.open(data.HELENA / 'masks' / row['video_id'] / 'segment_000' / 'final.png') as opened:
            reference = np.asarray(opened.convert('L')) > 127
        first = gm.unpack_mask(mine['segments'][0]['mask'])
        ious.append(float((first & reference).sum() / max(1, (first | reference).sum())))
    result = {'videos': len(picks), 'same_crop': int(same), 'median_iou': float(np.median(ious)), 'min_iou': float(min(ious)), 'median_h_frame_ncc': float(np.median(matches)), 'min_h_frame_ncc': float(min(matches))}
    data.write_public('geometry_parity.json', result)
    print(result, flush=True)
    if same < MIN_SAME_CROP or result['median_iou'] < MIN_MEDIAN_IOU or result['median_h_frame_ncc'] < MIN_H_NCC:
        raise SystemExit('the copied analysis does not reproduce Helena\'s stored geometry')


def analyse_group(name, cfg, make):
    began = time.time()
    rows = groups()[name]
    out_path = data.GEOMETRY / f'{name}.pkl'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fingerprint = ex.analysis_fingerprint(cfg)
    fingerprint_path = out_path.with_suffix('.fingerprint')
    done = {}
    if out_path.is_file():
        if not fingerprint_path.is_file() or fingerprint_path.read_text() != fingerprint:
            raise SystemExit(f'{name}: the stored geometry came from other code or settings; remove it explicitly')
        with out_path.open('rb') as handle:
            done = pickle.load(handle)
    fingerprint_path.write_text(fingerprint)

    def save():
        temporary = out_path.with_suffix('.tmp')
        with temporary.open('wb') as handle:
            pickle.dump(done, handle, protocol=pickle.HIGHEST_PROTOCOL)
        temporary.replace(out_path)

    for i, q in enumerate(rows):
        if q['id'] in done:
            continue
        done[q['id']] = ex.geometry_of(ex.read_recording(q), cfg, make)
        if i + 1 == EARLY_RECORDINGS:
            failures = [g['failed'] for g in done.values() if g.get('failed')]
            if len(failures) > EARLY_FAILURES_MAX:
                raise SystemExit(f'{name}: {len(failures)} of the first {EARLY_RECORDINGS} recordings failed, for example: {failures[0]}')
        if (i + 1) % 50 == 0:
            save()
            print(name, i + 1, len(rows), round(time.time() - began), flush=True)
    save()
    sizes = [g['size'] for g in done.values() if not g.get('failed')]
    fallbacks = sum(any(s['fallback'] for s in g['segments']) for g in done.values() if not g.get('failed'))
    summary = {'group': name, 'recordings': len(done), 'failed': sum(bool(g.get('failed')) for g in done.values()), 'any_fallback': int(fallbacks),
               'median_crop_side': float(np.median(sizes)) if sizes else None, 'wall_seconds': time.time() - began}
    data.write_public(f'geometry_{name}.json', summary)
    with (data.RUN / 'ledger.jsonl').open('a') as handle:
        handle.write(json.dumps({'what': f'geometry {name}', 'gpu_seconds': time.time() - began, 'ok': True, 'note': 'whole wall time charged'}) + '\n')
    print(summary, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--parity', action='store_true')
    group.add_argument('--group', choices=('dynamic', 'lvh', 'camus_2ch', 'camus_4ch'))
    args = parser.parse_args()
    import torch
    torch.cuda.set_per_process_memory_fraction(.15)
    cfg = config()
    make = generator(cfg)
    if args.parity:
        parity(cfg, make)
    else:
        analyse_group(args.group, cfg, make)


if __name__ == '__main__':
    main()
