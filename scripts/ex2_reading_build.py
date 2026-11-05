#!/usr/bin/env python3
"""EX2 D2: build the blinded reading set. Images carry no text; the key stays on the server. CPU."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import collections
import shutil

import numpy as np
from PIL import Image

import ex1_clip_context as cc
from echo_routing.ingest.frames import list_frame_paths
from echo_routing.temporal.joint_visual import crop_uint8
from exam_level import clip_models as cm, io
from open_routing import data, reading as rd

CONTROLS_PER_CLASS = 30
RANDOM_DEV_A4C = 15
REPEAT_SHARE = .10
SEED = 2026110403
TILE = (320, 320)
QUALITY = 88
REPEAT_QUALITY = 84
FIXED_TIME = 1000000000
READING = data.RUN / 'private/reading'
FOLDER = READING / 'images'


def d1_strata(rng):
    """Training recordings by label and by the held-out B0 position vote."""
    groups = collections.defaultdict(list)
    for sessions in cc.load_d1().values():
        for s in sessions:
            for video_id, y, vote in zip(s['ids'], s['y'].tolist(), s['vote'].tolist()):
                groups[(y, vote)].append(video_id)
    wrong_five, wrong_four = sorted(groups[(cm.A5C, cm.A4C)]), sorted(groups[(cm.A4C, cm.A5C)])
    if len(wrong_five) != cc.PUBLISHED_D1['a5c_voted_a4c'] or len(wrong_four) != cc.PUBLISHED_D1['a4c_voted_a5c']:
        raise SystemExit('training strata do not match the published library')
    five, four = sorted(groups[(cm.A5C, cm.A5C)]), sorted(groups[(cm.A4C, cm.A4C)])
    return {'train_a5c_voted_a4c': [(v, 'A5C') for v in wrong_five], 'train_a4c_voted_a5c': [(v, 'A4C') for v in wrong_four],
            'train_a5c_control': [(five[int(i)], 'A5C') for i in sorted(rng.choice(len(five), CONTROLS_PER_CLASS, replace=False))],
            'train_a4c_control': [(four[int(i)], 'A4C') for i in sorted(rng.choice(len(four), CONTROLS_PER_CLASS, replace=False))]}


def dev_strata(rng):
    """All development A5C, A4C wrong under any A08 seed, and a fixed random sample of the other A4C."""
    wrong = set()
    labels = {}
    for half in io.HALVES:
        for seed in io.SEEDS:
            for s in cc.dev_sessions(half, seed):
                base = cc.base_labels(s)
                for video_id, y, b in zip(s['ids'], s['y'].tolist(), base.tolist()):
                    labels[video_id] = y
                    if y == cm.A4C and b != y:
                        wrong.add(video_id)
    fives = sorted(v for v, y in labels.items() if y == cm.A5C)
    rest = sorted(v for v, y in labels.items() if y == cm.A4C and v not in wrong)
    return {'dev_a5c_all': [(v, 'A5C') for v in fives], 'dev_a4c_wrong': [(v, 'A4C') for v in sorted(wrong)],
            'dev_a4c_random': [(rest[int(i)], 'A4C') for i in sorted(rng.choice(len(rest), RANDOM_DEV_A4C, replace=False))]}


def frame_dirs():
    return {r['video_id']: r['frames_dir'] for s in io.ALLOWED_SPLITS for r in io.manifest_rows(s)}


def montage(frames_dir):
    """Four frames on a two-by-two grid, with A08's own centre crop. Readers see the field A08 saw; border overlays are cut."""
    path = frames_dir if str(frames_dir).startswith('/') else io.ROOT / frames_dir
    files = list_frame_paths(path)
    canvas = Image.new('RGB', (2 * TILE[0], 2 * TILE[1]))
    for k, i in enumerate(rd.frame_picks(len(files))):
        tile = Image.fromarray(crop_uint8(Image.open(files[i]).convert('RGB'))).resize(TILE, Image.BICUBIC)
        canvas.paste(tile, ((k % 2) * TILE[0], (k // 2) * TILE[1]))
    return canvas


def main():
    rng = np.random.default_rng(SEED)
    strata = {**d1_strata(rng), **dev_strata(rng)}
    members = [{'video_id': v, 'label': y, 'stratum': name, 'repeat_of': None} for name, rows in strata.items() for v, y in rows]
    repeats = rng.choice(len(members), int(round(REPEAT_SHARE * len(members))), replace=False)
    entries = members + [{**members[int(i)], 'repeat_of': int(i)} for i in repeats.tolist()]
    ids = rd.blind_ids(len(entries), SEED)
    if FOLDER.exists():
        shutil.rmtree(FOLDER)
    FOLDER.mkdir(parents=True)
    dirs = frame_dirs()
    key = {}
    for index in rng.permutation(len(entries)).tolist():
        blind, entry = ids[index], entries[index]
        path = FOLDER / f'{blind}.jpg'
        montage(dirs[entry['video_id']]).save(path, quality=REPEAT_QUALITY if entry['repeat_of'] is not None else QUALITY, optimize=True)
        os.utime(path, (FIXED_TIME, FIXED_TIME))
        key[blind] = entry
    io.write_json(READING / 'key.json', key)
    counts = collections.Counter(e['stratum'] for e in members)
    io.write_json(data.RUN / 'public/d2_reading_set.json', {'strata': dict(counts), 'originals': len(members), 'repeats': len(repeats), 'images': len(entries)})
    print('reading set built', dict(counts), len(entries))


if __name__ == '__main__':
    main()
