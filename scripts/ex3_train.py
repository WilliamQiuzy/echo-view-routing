#!/usr/bin/env python3
"""EX3 training: V01's recipe, then A08 A's recipe from the arm's own V01 end point. Only the frame source changes; no teacher.

  --check            the original pixels must reproduce both recipes' first epoch; short smoke runs and timing for H and Z
  --queue H_s0 Z_s0  full chains, in order
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import echo_routing.vit.target_mvit as tm
import scripts.prepare_raw_video_context as prv
import scripts.train_raw_video_context as rvc
import scripts.train_video_within_view as twv
from echo_routing.config import load_paths
from echo_routing.evaluate.complementarity import sha256_file
from exam_level import io
from helena_input import data
from scripts._common import load_cfg
from scripts.train_joint_visual_temporal import FramePaths

WINDOWS = 8
EPOCH = 375
GROUP = twv.GROUP_NAME
REPRODUCE_TOLERANCE = 1e-6
GRAD_TOLERANCE = 1e-4
SMOKE_UPDATES = 40
UNFREEZE_UPDATES = 12
ORIGINAL_LOGS = {'v01': 'runs/raw_video_context_v1/train/ordered_s0/train_log.jsonl', 'a08': 'runs/video_within_view_v1/train/all_s0/train_log.jsonl'}
CODE = (Path(__file__), *(Path(data.__file__).parent / name for name in data.PIXEL_CODE))


def contexts():
    return {'v01': rvc.Ctx(load_cfg(SimpleNamespace(config=data.V01_CONFIG, set=[])), data.V01_CONFIG),
            'a08': twv.Ctx(load_cfg(SimpleNamespace(config=data.A08_CONFIG, set=[])), data.A08_CONFIG)}


def window_inputs(m, pixels, u):
    """Update u's eight windows from any frame source: frames, marker slots, targets and pixel digests."""
    order = m['order_ordered']
    frames = np.empty((WINDOWS, tm.N_SLOTS, 224, 224, 3), np.uint8)
    digests = np.empty((WINDOWS, 32), np.uint8)
    memo = {}
    for w in range(WINDOWS):
        hashes = []
        for slot in range(tm.N_SLOTS):
            j = int(order[u, w, slot])
            ref = (int(m['vid'][u, w, j]), int(m['frame'][u, w, j]), round(float(m['gamma'][u, w, j]), 6))
            if ref not in memo:
                image = pixels.frame(str(m['videos'][ref[0]]), ref[1], ref[2])
                memo[ref] = (image, tm.frame_hash(image))
            frames[w, slot], digest = memo[ref]
            hashes.append(digest)
        digests[w] = np.frombuffer(tm.window_digest(hashes), np.uint8)
    return (torch.from_numpy(frames), torch.from_numpy(m['marker'][u].astype(np.int64)), torch.from_numpy(m['target'][u].astype(np.int64)), torch.from_numpy(digests))


class UpdateInputs(Dataset):
    def __init__(self, m, pixels, updates):
        self.m, self.pixels, self.updates = m, pixels, list(updates)

    def __len__(self):
        return len(self.updates)

    def __getitem__(self, i):
        u = self.updates[i]
        return (u, *window_inputs(self.m, self.pixels, u))


def loader(m, pixels, updates, workers):
    return DataLoader(UpdateInputs(m, pixels, updates), batch_size=None, num_workers=workers, prefetch_factor=2 if workers else None, pin_memory=True)


def frame_source(ctxs, arm, m):
    videos = {str(v) for v in m['videos'].tolist()}
    if videos - set(prv.allowlist(ctxs['v01'].rv, io.ROOT)):
        raise SystemExit('a manifest recording is not a training clip')
    return data.pixels_for(arm, FramePaths(prv.frame_dirs(load_paths(), ctxs['v01'].rv, videos)))


def start_of(ctxs, stage, seed, device, own_start=None):
    """The model and optimiser exactly as the original trainers build them, then the original seeding.
    own_start is (state, sha256) of the arm's own V01 end point; without it the A08 stage starts from frozen V01, as A08 did."""
    ctx = ctxs[stage]
    if stage == 'v01':
        init, sha = ctx.init(seed)
        model = rvc.make_model(ctx, init, device)
    else:
        state, sha = own_start if own_start is not None else ctx.source_state(seed)
        model = twv.make_model(state, device)
    opt = rvc.make_optimizer(model, ctx.o)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return ctx, model, opt, sha


def one_step(stage, model, opt, batch, lrs, o, device):
    if stage == 'v01':
        return rvc.train_step(model, opt, batch, lrs, o, device)
    return twv.train_step(model, opt, batch, 'all', lrs, o, device)


def check_batch(m, batch, compare_digests):
    u = int(batch[0])
    if not np.array_equal(batch[3].numpy(), m['target'][u]) or not np.array_equal(batch[2].numpy(), m['marker'][u]):
        raise SystemExit(f'update {u}: targets or markers differ from the manifest')
    if compare_digests:
        rvc.check_digests(m, GROUP, u, batch[4])


def code_digest():
    digest = hashlib.sha256()
    for path in CODE:
        digest.update(Path(path).read_bytes())
    return digest.hexdigest()


def identity(arm, seed, stage, manifest_sha, start_sha):
    source = data.CACHE / 'index.json' if arm == 'H' else data.FOV_INDEX
    payload = {'experiment': 'ex3_helena_input_v1', 'arm': arm, 'seed': seed, 'stage': stage, 'manifest': manifest_sha, 'start': start_sha, 'plan': sha256_file(data.RUN / 'plan.md'),
               'source': sha256_file(source), 'code': code_digest(), 'environment': prv.environment()}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def ledger(name, seconds, ok, note=''):
    with (data.RUN / 'ledger.jsonl').open('a') as handle:
        handle.write(json.dumps({'what': name, 'gpu_seconds': seconds, 'ok': ok, 'note': note, 'finished': time.time()}) + '\n')


def spent_hours():
    path = data.RUN / 'ledger.jsonl'
    return sum(json.loads(line)['gpu_seconds'] for line in path.read_text().splitlines()) / 3600 if path.is_file() else 0.


def train_stage(ctxs, arm, seed, stage, device, workers, own_start=None):
    m, msha = ctxs[stage].manifest(seed)
    ctx, model, opt, ssha = start_of(ctxs, stage, seed, device, own_start)
    ident = identity(arm, seed, stage, msha, ssha)
    key = data.run_key(arm, seed)
    run_dir, ckpt = data.RUN / 'train' / key / stage, data.RUN / 'checkpoints' / key / stage
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / 'run_manifest.json'
    if rvc.finished(manifest_path, ident):
        print(f'[{key} {stage}] already finished', flush=True)
        return
    done, log = rvc.load_resume(ckpt / 'resume.pt', model, opt, ident) if (ckpt / 'resume.pt').is_file() else (0, [])
    pixels = frame_source(ctxs, arm, m)
    print(f'[{key} {stage}] identity {ident[:12]}; start at {done}/{ctx.total}', flush=True)
    began, acc, epoch_hash = time.time(), [], hashlib.sha256()
    torch.cuda.reset_peak_memory_stats(device)
    for batch in loader(m, pixels, range(done, ctx.total), workers):
        check_batch(m, batch, False)
        u = int(batch[0])
        epoch_hash.update(batch[4].numpy().tobytes())
        lrs = rvc.apply_schedule(model, opt, u, ctx.warmup, ctx.total, ctx.o)
        acc.append(one_step(stage, model, opt, batch, lrs, ctx.o, device))
        done = u + 1
        if done % 25 == 0:
            io.write_json(run_dir / 'progress.json', {'updates': done, 'total': ctx.total, 'seconds': time.time() - began, 'loss': acc[-1]['loss']})
        if done % EPOCH == 0:
            record = {'updates': done, 'loss': float(np.mean([a['loss'] for a in acc])), 'train_accuracy': sum(a['correct'] for a in acc) / sum(a['windows'] for a in acc),
                      'grad_norm_mean': float(np.mean([a['grad_norm'] for a in acc])), 'seconds_this_process': time.time() - began,
                      'peak_gpu_mb': torch.cuda.max_memory_allocated(device) / 2**20, 'pixels_digest': epoch_hash.hexdigest()}
            log.append(record)
            rvc.save_resume(ckpt / 'resume.pt', model, opt, done, ident, log)
            print(f"[{key} {stage}] {done}/{ctx.total} loss {record['loss']:.4f} acc {record['train_accuracy']:.3f} {record['seconds_this_process']:.0f}s", flush=True)
            acc, epoch_hash = [], hashlib.sha256()
    if done != ctx.total:
        raise SystemExit(f'stopped at update {done} of {ctx.total}')
    final = ckpt / 'final.pt'
    torch.save(model.state_dict(), final)
    io.write_json(manifest_path, {'identity': ident, 'arm': arm, 'seed': seed, 'stage': stage, 'updates': done, 'epochs': log, 'final_checkpoint': str(final), 'final_sha256': sha256_file(final), 'manifest_sha256': msha})
    (ckpt / 'resume.pt').unlink(missing_ok=True)
    del model, opt
    torch.cuda.empty_cache()


def chain(ctxs, arm, seed, device, workers):
    train_stage(ctxs, arm, seed, 'v01', device, workers)
    path = data.final_weights(arm, seed, 'v01')
    train_stage(ctxs, arm, seed, 'a08', device, workers, own_start=(torch.load(path, map_location='cpu', weights_only=True), sha256_file(path)))


def reproduce(ctxs, stage, device, workers):
    """Original pixels through this trainer: the first epoch is before the encoder unfreezes, so its loss must match the record."""
    ctx = ctxs[stage]
    m, _ = ctx.manifest(0)
    ctx, model, opt, _ = start_of(ctxs, stage, 0, device)
    pixels = frame_source(ctxs, 'A08', m)
    steps = []
    for batch in loader(m, pixels, range(EPOCH), workers):
        check_batch(m, batch, True)
        lrs = rvc.apply_schedule(model, opt, int(batch[0]), ctx.warmup, ctx.total, ctx.o)
        steps.append(one_step(stage, model, opt, batch, lrs, ctx.o, device))
    del model, opt
    torch.cuda.empty_cache()
    recorded = json.loads((io.ROOT / ORIGINAL_LOGS[stage]).read_text().splitlines()[0])
    if recorded['updates'] != EPOCH:
        raise SystemExit('the original log does not start with the first epoch')
    mean_loss = float(np.mean([s['loss'] for s in steps]))
    accuracy = sum(s['correct'] for s in steps) / sum(s['windows'] for s in steps)
    grad = float(np.mean([s['grad_norm'] for s in steps]))
    return {'stage': stage, 'mean_loss': mean_loss, 'recorded': recorded['loss'], 'difference': abs(mean_loss - recorded['loss']), 'accuracy_difference': abs(accuracy - recorded['train_accuracy']),
            'grad_difference': abs(grad - recorded['grad_norm_mean'])}


def smoke(ctxs, arm, stage, device, workers):
    """A few updates with the encoder trainable from the third, so the real memory peak and step time of the long phase show up."""
    ctx = ctxs[stage]
    m, _ = ctx.manifest(0)
    ctx, model, opt, _ = start_of(ctxs, stage, 0, device, own_start=ctx.source_state(0) if stage == 'a08' else None)
    pixels = frame_source(ctxs, arm, m)
    losses, times, black = [], [], []
    torch.cuda.reset_peak_memory_stats(device)
    for batch in loader(m, pixels, range(UNFREEZE_UPDATES), workers):
        began = time.time()
        lrs = rvc.apply_schedule(model, opt, int(batch[0]), 2, ctx.total, ctx.o)
        losses.append(one_step(stage, model, opt, batch, lrs, ctx.o, device)['loss'])
        times.append(time.time() - began)
        black.append(float((batch[1].numpy().max(-1) == 0).mean()))
    if not np.isfinite(losses).all():
        raise SystemExit(f'{arm} {stage}: non-finite loss')
    peak = torch.cuda.max_memory_allocated(device) / 2**20
    del model, opt
    torch.cuda.empty_cache()
    return {'arm': arm, 'stage': stage, 'first_loss': losses[0], 'last_loss': losses[-1], 'black_pixel_share': float(np.mean(black)), 'unfrozen_step_seconds': float(np.mean(times[4:])), 'peak_gpu_mb': peak}


def check(device, workers):
    began = time.time()
    ctxs = contexts()
    result = {'reproduce': [reproduce(ctxs, stage, device, workers) for stage in ('v01', 'a08')]}
    worst = max(r['difference'] for r in result['reproduce'])
    result['ok'] = bool(worst <= REPRODUCE_TOLERANCE and all(r['accuracy_difference'] == 0 and r['grad_difference'] <= GRAD_TOLERANCE for r in result['reproduce']))
    if not result['ok']:
        data.write_public('train_check.json', result)
        ledger('check', time.time() - began, False, f'worst {worst}')
        raise SystemExit(f'the original pixels did not reproduce the recorded first epoch: {result}')
    result['smoke'] = [smoke(ctxs, arm, stage, device, workers) for arm in data.ARMS for stage in ('v01', 'a08')]
    m, msha = ctxs['v01'].manifest(0)
    result['identity_sample'] = {arm: identity(arm, 0, 'v01', msha, 'check')[:12] for arm in data.ARMS}
    started = time.time()
    list(loader(m, frame_source(ctxs, 'Z', m), range(50, 80), workers))
    result['z_loader_seconds_per_update'] = (time.time() - started) / 30
    data.write_public('train_check.json', result)
    ledger('check', time.time() - began, True)
    print(json.dumps(result, indent=1), flush=True)


def queue(keys, device, workers):
    ctxs = contexts()
    for key in keys:
        arm, seed = key.split('_s')
        if spent_hours() + 1. > data.GPU_HOURS_MAX:
            raise SystemExit('the GPU limit would be exceeded; nothing new starts')
        began, ok, note = time.time(), False, ''
        try:
            chain(ctxs, arm, int(seed), device, workers)
            ok = True
        except BaseException as error:
            note = f'{type(error).__name__}: {str(error)[:200]}'
            raise
        finally:
            ledger(f'train {key}', time.time() - began, ok, note)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=4)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--check', action='store_true')
    group.add_argument('--queue', nargs='+')
    args = parser.parse_args()
    twv.determinism()
    torch.cuda.set_per_process_memory_fraction(data.GPU_FRACTION)
    device = torch.device('cuda')
    if args.check:
        check(device, args.workers)
    else:
        queue(args.queue, device, args.workers)


if __name__ == '__main__':
    main()
