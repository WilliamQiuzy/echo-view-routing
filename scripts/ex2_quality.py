#!/usr/bin/env python3
"""EX2 D1: mechanical audit of EV9V train and validation. Exact files, near duplicates and quality flags. CPU, no labels changed."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import collections
import hashlib

import numpy as np
from PIL import Image

from echo_routing.ingest.frames import list_frame_paths
from exam_level import io, streams as st
from open_routing import data, quality as ql

HASH_FRAMES = 5
FLAG_PAIRS = 6
COLOUR_FRAMES = 4
BLANK_STD = 5.
STRICT_NCC = .99


def half_of():
    return {q['id']: half for half in io.HALVES for q in io.natives(half)}


def picks(n, count):
    return np.unique(np.linspace(0, n - 1, count).round().astype(int)).tolist()


def audit_clip(frames_dir):
    """Five hashes with their small images, and the three flags. The static flag reads original-size neighbouring frames."""
    files = list_frame_paths(frames_dir)
    n = len(files)
    hashes, smalls = [], []
    for i in picks(n, HASH_FRAMES):
        gray = np.asarray(Image.open(files[i]).convert('L'))
        if gray.std() < BLANK_STD:
            continue
        hashes.append(ql.dhash(gray))
        smalls.append(ql.small_gray(gray).astype(np.uint8))
    differences = [np.abs(np.asarray(Image.open(files[i]).convert('L'), float) - np.asarray(Image.open(files[min(i + 1, n - 1)]).convert('L'), float)).mean()
                   for i in picks(max(n - 1, 1), FLAG_PAIRS)]
    colour = ql.colour_share(np.stack([np.asarray(Image.open(files[i]).convert('RGB')) for i in picks(n, COLOUR_FRAMES)]))
    return {'hashes': hashes, 'smalls': smalls, 'static': ql.static_from_differences(differences), 'colour': ql.is_colour(colour), 'short': ql.is_short(n)}


def md5(path):
    digest = hashlib.md5()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 22), b''):
            digest.update(block)
    return digest.hexdigest()


def resolve(path):
    return str(path) if str(path).startswith('/') else str(io.ROOT / path)


def main():
    rows = [r for s in io.ALLOWED_SPLITS for r in io.manifest_rows(s)]
    halves = half_of()
    owners = [{'id': r['video_id'], 'session': st.session_of(r['video_id']), 'stamp': st.timestamp_of(r['video_id']), 'split': r['split'],
               'half': halves.get(r['video_id']), 'code': r['raw_label'], 'family': r['family5_label'] or None} for r in rows]
    audits = [audit_clip(resolve(r['frames_dir'])) for r in rows]
    digests = collections.defaultdict(list)
    for i, r in enumerate(rows):
        digests[md5(resolve(r['video_path']))].append(i)
    exact_pairs = [(v[0], other) for v in digests.values() for other in v[1:]]
    values = np.array([h for a in audits for h in a['hashes']], np.uint64)
    who = np.array([i for i, a in enumerate(audits) for _ in a['hashes']])
    smalls = np.stack([s for a in audits for s in a['smalls']])
    found = ql.near_duplicate_candidates(values, who, np.array([o['session'] for o in owners]))
    confirmed = []
    for (a, b), frame_pairs in found.items():
        best = max(ql.ncc(smalls[r], smalls[c]) for r, c in frame_pairs)
        if best >= ql.MIN_NCC:
            confirmed.append((a, b, best))
    flags = {}
    for name in ('static', 'colour', 'short'):
        table = collections.Counter((owners[i]['split'], owners[i]['code']) for i, a in enumerate(audits) if a[name])
        flags[name] = {'total': sum(table.values()), 'by_split_and_code': {f'{k[0]}/{k[1]}': v for k, v in sorted(table.items())}}
    strict = [p for p in confirmed if p[2] >= STRICT_NCC]
    io.write_json(data.RUN / 'public/d1_mechanical_audit.json', {
        'recordings': len(rows), 'hashed_frames': len(values), 'blank_or_flat_frames_skipped': sum(HASH_FRAMES - len(a['hashes']) for a in audits),
        'exact_duplicate_files': ql.describe_pairs(exact_pairs, owners), 'near_duplicate_candidate_pairs': len(found),
        'near_duplicates_ncc_at_least_95': ql.describe_pairs(confirmed, owners), 'near_duplicates_ncc_at_least_99': ql.describe_pairs(strict, owners), 'flags': flags})
    conflicts = sorted({owners[i]['id'] for a, b, *_ in confirmed if ql.family_conflict(owners[a], owners[b]) for i in (a, b)})
    io.write_json(data.RUN / 'private/d1_private.json', {'label_conflict_recordings': conflicts,
                                                         'flagged': {name: [owners[i]['id'] for i, a in enumerate(audits) if a[name]] for name in ('static', 'colour', 'short')},
                                                         'pairs': [[owners[a]['id'], owners[b]['id'], float(s)] for a, b, s in confirmed]})
    print('d1 written', len(rows), len(found), len(confirmed))


if __name__ == '__main__':
    main()
