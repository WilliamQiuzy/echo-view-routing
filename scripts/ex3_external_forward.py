#!/usr/bin/env python3
"""EX3 embeddings for the external and open-set readouts, one model on one input per tag.

  --check              A08 on its own input through this path must reproduce EX2's stored external groups
  --tags H_s0 Z_s0     trained arms on their own inputs
  --tags A08onH_s0 ... frozen A08 on another input
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import json
import re
import time
from types import SimpleNamespace

import numpy as np
import torch

import scripts.prepare_raw_video_context as prv
from echo_routing.config import load_paths
from exam_level import io
from helena_input import data, external as ex, forward as fw
from open_routing import data as ex2
from ex3_geometry import groups as media_groups
from scripts.train_joint_visual_temporal import FramePaths

MEDIA = ('dynamic', 'lvh', 'camus_2ch', 'camus_4ch')
PARITY_RECORDINGS = 8
PARITY_TOLERANCE = 1e-3
TAG = re.compile(r'^(?:(H|Z)|A08on(H|Z))_s([012])$')
FORWARD = data.RUN / 'forward'


def parse(tag):
    match = TAG.match(tag)
    if not match:
        raise SystemExit(f'undeclared tag {tag}')
    seed = int(match.group(3))
    if match.group(1):
        return match.group(1), data.final_weights(match.group(1), seed), seed
    return match.group(2), io.a08_weight_path(seed), seed


def native_groups():
    return {'dev_selection': io.natives('selection'), 'dev_calibration': io.natives('calibration'),
            'train_reference': [ex2.native_from_row(r) for r in ex2.train_reference_rows()],
            'out_of_scope': [ex2.native_from_row(r) for r in ex2.out_of_scope_rows()]}


def frame_paths_for(natives):
    from scripts.train_raw_video_context import Ctx
    from scripts._common import load_cfg
    videos = {r[0] for members in natives.values() for q in members for r in q['refs']}
    v01 = Ctx(load_cfg(SimpleNamespace(config=data.V01_CONFIG, set=[])), data.V01_CONFIG)
    return FramePaths(prv.frame_dirs(load_paths(), v01.rv, videos))


def media_group(model, device, rows, arm, geometry, ratio):
    pairs, groups = [], []
    for q in rows:
        frames = ex.media_frames(q, arm, geometry[q['id']] if geometry else {}, ratio)
        pairs.append(fw.embed_batches(model, device, fw.media_batches(frames)))
        groups.append(q['group'])
    return fw.summary([q['id'] for q in rows], [p[0] for p in pairs], [p[1] for p in pairs], {'groups': np.array(groups)})


def native_group(model, device, members, pixels):
    pairs = [fw.embed_batches(model, device, fw.sequence_batches(q, pixels)) for q in members]
    return fw.summary([q['id'] for q in members], [p[0] for p in pairs], [p[1] for p in pairs])


def check(device):
    """Same recordings, same weights: this path must give EX2's stored window means."""
    worst = {}
    media = media_groups()
    for seed in io.SEEDS:
        model = fw.build_model(torch.load(io.a08_weight_path(seed), map_location='cpu', weights_only=True), device)
        for name in MEDIA:
            rows = media[name][:PARITY_RECORDINGS]
            with np.load(ex2.RUN / 'forward' / f'embed_{name}_s{seed}.npz') as z:
                stored = z['prob_mean'][:PARITY_RECORDINGS]
                if z['ids'][:PARITY_RECORDINGS].tolist() != [q['id'] for q in rows]:
                    raise SystemExit('EX2 group order differs')
            got = media_group(model, device, rows, 'A08', None, None)['prob_mean']
            worst[f'{name}_s{seed}'] = float(np.abs(got - stored).max())
        del model
        torch.cuda.empty_cache()
    data.write_public('external_parity.json', worst)
    print(worst, flush=True)
    if max(worst.values()) > PARITY_TOLERANCE:
        raise SystemExit('the external path does not reproduce EX2')


def stored_identity(path):
    with np.load(path) as z:
        return str(z['identity']) if 'identity' in z.files else None


def save_with_identity(path, values, identity):
    ex2.save_group(path, {**values, 'identity': np.array(identity)})


def run_tag(tag, device):
    arm, weights, seed = parse(tag)
    if max(json.loads((data.PUBLIC / 'external_parity.json').read_text()).values()) > PARITY_TOLERANCE:
        raise SystemExit('the external path has not reproduced EX2')
    began = time.time()
    identity = f'{tag}|{io.sha256(weights)}|{data.input_fingerprint(arm)}'
    model = fw.build_model(torch.load(weights, map_location='cpu', weights_only=True), device)
    natives = native_groups()
    pixels = data.pixels_for(arm, frame_paths_for(natives))
    ratio = data.side_ratio()
    media = media_groups()
    for name, members in natives.items():
        path = FORWARD / f'embed_{name}_{tag}.npz'
        if not path.exists() or stored_identity(path) != identity:
            save_with_identity(path, native_group(model, device, members, pixels), identity)
    for name in MEDIA:
        path = FORWARD / f'embed_{name}_{tag}.npz'
        media_identity = f"{identity}|{io.sha256(data.GEOMETRY / f'{name}.pkl')[:16]}"
        if not path.exists() or stored_identity(path) != media_identity:
            save_with_identity(path, media_group(model, device, media[name], arm, data.external_geometry(name), ratio), media_identity)
    del model
    torch.cuda.empty_cache()
    with (data.RUN / 'ledger.jsonl').open('a') as handle:
        handle.write(json.dumps({'what': f'external forward {tag}', 'gpu_seconds': time.time() - began, 'ok': True, 'note': f'seed {seed}'}) + '\n')
    print('forward', tag, round(time.time() - began), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--check', action='store_true')
    group.add_argument('--tags', nargs='+')
    args = parser.parse_args()
    FORWARD.mkdir(parents=True, exist_ok=True)
    device = fw.determinism(data.GPU_FRACTION)
    if args.check:
        check(device)
        return
    for tag in args.tags:
        run_tag(tag, device)


if __name__ == '__main__':
    main()
