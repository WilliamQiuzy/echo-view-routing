#!/usr/bin/env python3
"""EX2 post-hoc, added after the fact-check so its new numbers are reproducible. CPU, aggregates only."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
from PIL import Image

import ex1_clip_context as cc
import ex2_openset as xs
import ex2_quality as q
from echo_routing.ingest.frames import list_frame_paths
from echo_routing.temporal.baselines.b3_js_divergence import detect_boundaries, labels_from_boundaries
from exam_level import io
from open_routing import data, openset as os_, quality as ql, reading as rd

N_CLASSES = 5
NAMES = ('PLAX', 'PSAX', 'A4C', 'A5C', 'SC4C')


def small_frames(frames_dir):
    files = list_frame_paths(q.resolve(frames_dir))
    return [ql.small_gray(np.asarray(Image.open(files[i]).convert('L'))) for i in q.picks(len(files), q.HASH_FRAMES)]


def components(pairs):
    """Sizes of the groups of recordings linked by at least one pair."""
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b, *_ in pairs:
        parent[find(a)] = find(b)
    sizes = {}
    for x in list(parent):
        sizes[find(x)] = sizes.get(find(x), 0) + 1
    return sorted(sizes.values(), reverse=True)


def pair_detail():
    """How many of the five sampled frames match across each confirmed pair, and how the other frames correlate."""
    pairs = io.read_json(data.RUN / 'private/d1_private.json')['pairs']
    dirs = {r['video_id']: r['frames_dir'] for s in io.ALLOWED_SPLITS for r in io.manifest_rows(s)}
    matched, others = [], []
    for a, b, _ in pairs:
        x, y = small_frames(dirs[a]), small_frames(dirs[b])
        best = [max(ql.ncc(u, v) for v in y) for u in x]
        matched.append(sum(v >= ql.MIN_NCC for v in best))
        others += [v for v in best if v < ql.MIN_NCC]
    return {'pairs': len(pairs), 'matching_frames_of_five_per_pair': sorted(matched), 'median_best_ncc_of_other_frames': float(np.median(others)) if others else None,
            'recording_group_sizes': components(pairs)}


def rule_threshold():
    """The smallest count out of 90 that makes the pre-written signal fire, given the observed control of 2 of 30."""
    return next(k for k in range(91) if rd.label_signal(k, 90, 2, 30))


def dev_clip_errors():
    """Development recording errors under three rules on the same cached outputs."""
    out = []
    for half in io.HALVES:
        clips = io.natives(half)
        y = np.array([int(c['labels'][0]) for c in clips])
        for seed in io.SEEDS:
            cache = io.A08Cache(half, seed)
            params = io.b3_params(half, seed)
            positions = [cache.positions(c['id']) for c in clips]
            window = np.array([cache.window_mean(c['id']).argmax() for c in clips])
            vote = np.array([np.bincount(p.argmax(1), minlength=N_CLASSES).argmax() for p in positions])
            b3 = np.array([np.bincount(labels_from_boundaries(p, detect_boundaries(p, int(params['half_window']), float(params['threshold']), int(params['nms_radius']))), minlength=N_CLASSES).argmax() for p in positions])
            out.append({'half': half, 'seed': seed, 'clips': len(y), 'errors_window_mean': int((window != y).sum()), 'errors_position_vote': int((vote != y).sum()), 'errors_b3_majority': int((b3 != y).sum())})
    return out


def open_set_detail():
    """Retention by development class at the stored threshold, and acceptance at the lowest development score."""
    full = io.read_json(data.RUN / 'public/x3_open_set.json')['rows']
    rows = []
    for seed in io.SEEDS:
        groups = {n: xs.load(n, seed) for n in ('dev_selection', 'dev_calibration', 'train_reference', 'camus_2ch', 'camus_4ch', 'dynamic', 'lvh', 'out_of_scope')}
        labels = np.array([int(c['labels'][0]) for half in io.HALVES for c in io.natives(half)])
        for method, score in xs.scorers(groups['train_reference']).items():
            row = next(r for r in full if r['seed'] == seed and r['method'] == method)
            dev = np.concatenate([score(groups['dev_selection']), score(groups['dev_calibration'])])
            lowest = float(dev.min())
            rows.append({'seed': seed, 'method': method, 'dev_kept_by_class': {NAMES[c]: float((dev[labels == c] >= row['threshold']).mean()) for c in range(N_CLASSES)},
                         'accept_at_lowest_dev_score': {n: os_.accept_share(score(groups[n]), lowest) for n in ('camus_2ch', 'camus_4ch', 'dynamic', 'lvh', 'out_of_scope')}})
    return rows


def main():
    io.write_json(data.RUN / 'public/posthoc_wording_checks.json', {'near_duplicate_pairs': pair_detail(), 'smallest_count_of_90_firing_the_label_rule': rule_threshold(),
                                                                    'dev_clip_errors_by_rule': dev_clip_errors(), 'open_set_detail': open_set_detail()})
    print('wording checks written')


if __name__ == '__main__':
    main()
