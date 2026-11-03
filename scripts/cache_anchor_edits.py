#!/usr/bin/env python3
"""A59 pinned original5 A08 and frozen JEPA; all new joins truly forwarded."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import sys
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from scripts.anchor_common import RUN, PUB, CFG, read, write, sha, verify_pin, verify_freeze, sequences, digest
from echo_routing.temporal.anchor_cache import ProbabilityCache
from echo_routing.temporal.anchor_features import probabilities
from echo_routing.temporal.anchor_inputs import native_reuse_index
from echo_routing.temporal.structured_inputs import windows
from echo_routing.evaluate.window_target import nearest_centre, window_frames
from echo_routing.vit import target_mvit as tm
from echo_routing.temporal.dense_sequence import DenseSequence, chunk_starts, chunk_positions, stitch_probabilities
from echo_routing.temporal import jepa21_source
from echo_routing.temporal.dense_readout import ReadoutPixels


def determinism():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_source(name, cfg, device='cpu'):
    pin = cfg['source'][name]
    path = verify_pin(pin)
    if name == 'a08':
        state = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        model = tm.build_model(None, tm.initial_new_state(0))
        expected = 400
    elif name == 'jepa':
        source = ROOT / pin['source_manifest']
        if sha(source) != pin['source_manifest_sha256']:
            raise ValueError('JEPA source manifest changed')
        source = read(source)
        for path, expected_sha in source['loaded_model_subset_SHA'].items():
            if sha(jepa21_source.UPSTREAM / path) != expected_sha:
                raise ValueError('pinned official JEPA source subset changed')
        model = DenseSequence(jepa21_source.build_encoder(), 'J-high', seed=2026110257,
                              inactive_names=source['inactive_video_names'])
        state = torch.load(verify_pin(pin), map_location='cpu', weights_only=True, mmap=True)['model']
        expected = 214
    else:
        raise ValueError('declared frozen producer required')
    if len(state) != expected:
        raise ValueError('complete producer state-key count changed')
    model.load_state_dict(state, strict=True)
    if not all(torch.equal(model.state_dict()[k], value) for k, value in state.items()):
        raise ValueError('strict loaded source differs from pinned weights')
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model.eval().to(device)


@torch.no_grad()
def a08_windows(model, q, pixels, centres, device='cuda', batch=32):
    out = []
    centres = np.asarray(centres, int)
    for begin in range(0, len(centres), batch):
        cs = centres[begin:begin + batch]
        samples = [window_frames(int(c), len(q['refs']), 16, 16) for c in cs]
        frames = np.stack([[pixels.frame(*q['refs'][int(i)]) for i in row] for row in samples])
        markers = torch.tensor([tm.marker_slot(row, int(c)) for row, c in zip(samples, cs)], device=device)
        out.append(model(tm.to_rgb(torch.from_numpy(frames).to(device)), tm.marker_planes(markers)).float().cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def a08_sequence(model, q, pixels, device='cuda', batch=32):
    n = len(q['refs'])
    grid = windows(n, 5)
    logits = a08_windows(model, q, pixels, grid['centre'], device, batch)
    result = torch.softmax(torch.from_numpy(logits), 1).numpy()[nearest_centre(n, grid['centre'])]
    probabilities(result)
    return result


@torch.no_grad()
def jepa_sequence(model, q, pixels, device='cuda'):
    n = len(q['refs'])
    starts, chunks = chunk_starts(n), []
    for begin in range(0, len(starts), 2):
        frames, masks = [], []
        for start in starts[begin:begin + 2]:
            indices, mask = chunk_positions(n, start)
            frames.append(np.stack([pixels.frame(*q['refs'][int(i)]) for i in indices]))
            masks.append(mask)
        chunks.extend(model.probabilities(torch.from_numpy(np.stack(frames)).to(device),
                                           torch.from_numpy(np.stack(masks)).to(device)).cpu().numpy())
    result = stitch_probabilities(chunks, n).astype(np.float32)
    probabilities(result)
    return result


def original_a08(half, qs, seed, cfg):
    path = ROOT / cfg['data']['outer_inputs']
    if sha(path) != cfg['data']['outer_inputs_sha256']:
        raise ValueError('original A08 input/cache pins changed')
    pin = read(path)['A08_cache_pins'][half][str(seed)]
    path = verify_pin(pin)
    values = []
    with np.load(path) as value:
        if value['sequence_ids'].tolist() != [q['id'] for q in qs] or str(value['identity']) != pin['identity']:
            raise ValueError('original A08 full sequence order/identity changed')
        prob = torch.softmax(torch.from_numpy(value['logits']), 1).numpy()
        for index, q in enumerate(qs):
            rows = np.flatnonzero(value['seq'] == index)
            if not np.array_equal(value['centre'][rows], windows(len(q['refs']), 5)['centre']):
                raise ValueError('original A08 centre axis changed')
            p = prob[rows][nearest_centre(len(q['refs']), value['centre'][rows])]
            saved = value['sample_probs'][value['sample_offsets'][index]:value['sample_offsets'][index + 1]]
            if p.shape != (len(q['refs']), 5) or np.max(np.abs(p - saved)) > 1e-7:
                raise ValueError('original A08 raw-logit/probability assignment parity failed')
            probabilities(p)
            values.append(p)
    return values, pin['original_params']


def old_cache(source, role, cfg):
    key = 'cache_fit_manifest.json' if role in ['fit', 'inner_cal'] else 'cache_outer_manifest.json'
    old_manifest = read(verify_pin(cfg['data']['old_A58_pins']['input_manifest.json']))
    row = next(r for r in read(verify_pin(cfg['data']['old_A58_pins'][key]))['records']
               if r['producer'] == source and r['role'] == role)
    qs = np.load(verify_pin(old_manifest['roles'][role]), allow_pickle=True).tolist()
    folder = ROOT / 'runs/a58_structured_evidence_v1/cache' / source / role
    if sha(folder / 'committed.jsonl') != row['journal_sha256']:
        raise ValueError('prior frozen producer journal changed')
    cache = ProbabilityCache(folder, row['identity'], [len(q['refs']) for q in qs], readonly=True)
    return cache, qs


def reader(source, role, readonly=False):
    freeze = verify_freeze('producer_freeze.json')
    manifest, cfg = read(PUB / 'input_manifest.json'), read(CFG)
    identity = digest({'source': source, 'source_contract': cfg['source'][source],
                       'input': manifest['roles'][role]['sha256'], 'frozen_objects': freeze['objects']})
    qs = sequences(role)
    return ProbabilityCache(RUN / 'cache' / source / role, identity, [len(q['refs']) for q in qs], readonly=readonly)


def cache(mode):
    freeze = verify_freeze('producer_freeze.json')
    cfg = read(CFG)
    admission = read(PUB / 'cache_admission.json')
    if not admission['all_caps_fit']:
        raise ValueError('complete remaining measured forecast did not admit cache work')
    if mode == 'fit':
        roles = ['fit', 'inner_cal']
    elif mode == 'outer':
        verify_freeze('outer_freeze.json')
        models = read(PUB / 'models_manifest.json')
        if not models['passed'] or models['formal_optimizer_updates'] != 24000 or not models['all_models_and12_inner_records_frozen']:
            raise ValueError('all four heads must freeze before fresh outer readout')
        roles = ['selection', 'calibration', 'old_stress_selection', 'old_stress_calibration',
                 'confirmation_selection', 'confirmation_calibration']
    else:
        raise ValueError('explicit FIT or frozen outer stage required')
    determinism()
    records = []
    for source in ['a08', 'jepa']:
        model = None
        for role in roles:
            qs, target = sequences(role), reader(source, role)
            initial, new_units, reused = target.cursor, 0, 0
            old, mapping, values = None, {}, None
            if mode == 'fit' and source == 'jepa':
                old, old_qs = old_cache('jepa', role, cfg)
                mapping = native_reuse_index(old_qs, qs)
            elif mode == 'outer' and role in ['selection', 'calibration'] and source == 'a08':
                values, params = original_a08(role, qs, 0, cfg)
                if params != cfg['source']['a08']['original_B3']:
                    raise ValueError('fixed original B3 parameters changed')
            elif mode == 'outer' and (role.startswith('old_stress_') or source == 'jepa' and role in ['selection', 'calibration']):
                old_role = role.replace('old_stress_', 'stress_')
                old, old_qs = old_cache('a08_original' if source == 'a08' else 'jepa', old_role, cfg)
                if [(q['id'], list(map(tuple, q['refs']))) for q in old_qs] != [(q['id'], list(map(tuple, q['refs']))) for q in qs]:
                    raise ValueError('old complete context differs from new readout identity')
                mapping = dict(enumerate(range(len(qs))))
            pixels, start = None, time.time()
            for index in range(initial, len(qs)):
                q = qs[index]
                if values is not None:
                    target.run_one(index, lambda index=index: values[index])
                    reused += 1
                elif index in mapping:
                    target.run_one(index, lambda index=index: old.load(mapping[index]))
                    reused += 1
                else:
                    if model is None:
                        model = build_source(source, cfg, 'cuda')
                    if pixels is None:
                        pixels = ReadoutPixels(qs)
                    predict = a08_sequence if source == 'a08' else jepa_sequence
                    target.run_one(index, lambda q=q: predict(model, q, pixels))
                    new_units += len(windows(len(q['refs']), 5)['centre']) if source == 'a08' else (len(q['refs']) + 31) // 32
                if (index + 1) % 25 == 0:
                    write(RUN / 'runtimecoordination.json', {'phase': 'frozen_source_cache_' + mode,
                          'source': source, 'role': role, 'sequences_done': index + 1, 'sequences': len(qs),
                          'new_neural_units_this_role': new_units, 'reused_sequences_this_role': reused,
                          'model_pid': os.getpid(), 'epoch': time.time(),
                          'next_action': 'finish FIT cache then fixed head graph/admission' if mode == 'fit' else 'all heads already frozen; finish sources then fixed readout'})
            records.append({'source': source, 'role': role, 'identity': target.identity,
                            'committed_sequences': target.cursor, 'input_sha256': read(PUB / 'input_manifest.json')['roles'][role]['sha256'],
                            'journal_sha256': sha(target.journal), 'new_neural_units': new_units,
                            'reused_sequences': reused, 'prior_committed_sequences': initial,
                            'wall_seconds_this_process': time.time() - start, 'complete': target.cursor == len(qs)})
        if model is not None:
            del model
            torch.cuda.empty_cache()
    write(PUB / f'cache_{mode}_manifest.json', {'passed': True, 'records': records,
          'producer_freeze_git': freeze['git_commit'], 'input_manifest_sha256': sha(PUB / 'input_manifest.json'),
          'formal_optimizer_updates_during_cache': 0, 'new_protected_media_forwards': 0})


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise ValueError('explicit fit/outer mode required')
    cache(sys.argv[1])
