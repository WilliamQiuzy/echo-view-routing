#!/usr/bin/env python3
"""A59 bounded FIT-only source/graph/timing admission; never outer selection."""
import os
import sys
if len(sys.argv) > 1 and sys.argv[1] != 'gpu':
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import copy
import math
from pathlib import Path
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from scripts.anchor_common import RUN, PUB, CFG, read, write, sha, digest, verify_pin, verify_freeze, sequences, save_array, array_pin
from scripts.cache_anchor_edits import build_source, determinism, a08_windows, a08_sequence, jepa_sequence, old_cache
from echo_routing.temporal.anchor_cache import ProbabilityCache
from echo_routing.temporal.anchor_inputs import native_reuse_index
from echo_routing.temporal.structured_inputs import windows
from echo_routing.temporal.dense_readout import ReadoutPixels
from echo_routing.temporal.baselines.registry import decode_with_prob
from echo_routing.temporal.anchor_runtime import prepare_training, infer
from echo_routing.temporal.anchor_features import fit_normalizer, normalize
from echo_routing.temporal.anchor_training import UtilityRegressor, optimizer, step, checkpoint, restore, atomic_save


def same(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b


def plan():
    verify_freeze('profile_plan_freeze.json')
    fit, cfg = sequences('fit'), read(CFG)
    selected = []
    for native in [True, False]:
        ordered = sorted((i for i, q in enumerate(fit) if (q['kind'] == 'native') == native),
                         key=lambda i: (len(fit[i]['refs']), i))
        selected.extend([ordered[0], ordered[len(ordered) // 2], ordered[-1]])
    selected = list(dict.fromkeys(selected))
    old_manifest = read(verify_pin(cfg['data']['old_A58_pins']['input_manifest.json']))
    old_panel = ROOT / 'runs/a58_structured_evidence_v1/a08_FIT_parity.private.json'
    if sha(old_panel) != old_manifest['parity_private_sha256']:
        raise ValueError('verified original5 FIT raw-logit parity panel changed')
    old_fit = np.load(verify_pin(cfg['data']['roles']['fit']), allow_pickle=True).tolist()
    by_id = {q['id']: i for i, q in enumerate(fit) if q['kind'] == 'native'}
    panel = []
    for row in read(old_panel):
        source = old_fit[row['sequence']]
        index = by_id[source['id']]
        if source['kind'] != 'native' or list(map(tuple, source['refs'])) != list(map(tuple, fit[index]['refs'])):
            raise ValueError('old original5 FIT parity footprint changed')
        panel.append({**row, 'sequence': index})
    reuse_counts, workload = {}, {}
    for role in ['fit', 'inner_cal']:
        qs = sequences(role)
        old, old_qs = old_cache('jepa', role, cfg)
        mapping = native_reuse_index(old_qs, qs)
        if len(mapping) != sum(q['kind'] == 'native' for q in qs):
            raise ValueError('complete JEPA native context reuse could not be qualified')
        reuse_counts[role] = {'native_sequences': len(mapping), 'old_cache_identity': old.identity,
                             'old_journal_sha256': sha(old.journal)}
        workload[role] = {'new_A08_windows': sum(len(windows(len(q['refs']), 5)['centre']) for q in qs),
                          'new_JEPA_chunks': sum((len(q['refs']) + 31) // 32 for i, q in enumerate(qs) if i not in mapping),
                          'new_JEPA_sequences': sum(i not in mapping for i in range(len(qs))),
                          'all_cache_sequences_per_source': len(qs)}
    private = {'selected_sequence_indices': selected, 'a08_raw_logit_panel': panel}
    path = RUN / 'FIT_profile_inputs.private.json'
    write(path, private)
    write(PUB / 'profile_input_identity.json', {'passed': True, 'private_input': array_pin(path),
          'FIT_sequence_sha256': read(PUB / 'input_manifest.json')['roles']['fit']['sha256'],
          'selection_rule': 'each kind shortest,median,longest by(length,index); no outcomes',
          'panel_sequences': len(selected), 'selected_length_summary': sorted(len(fit[i]['refs']) for i in selected),
          'raw_A08_parity_windows': len(panel), 'JEPA_native_reuse': reuse_counts,
          'FIT_inner_physical_workload': workload, 'new_protected_media_forwards': 0,
          'GPU_model_forwards': 0, 'formal_updates': 0})


def gpu():
    freeze = verify_freeze('FIT_graph_freeze.json')
    pin = read(PUB / 'profile_input_identity.json')
    private = read(verify_pin(pin['private_input']))
    fit, cfg = sequences('fit'), read(CFG)
    selected = private['selected_sequence_indices']
    qs = [fit[i] for i in selected]
    determinism()
    pixels = ReadoutPixels([*qs, *[fit[r['sequence']] for r in private['a08_raw_logit_panel']]])
    values, timings, loads = {}, {}, {}
    errors = {'A08_raw_pre_softmax_five_logit_max_error': 0., 'JEPA_native_probability_max_error': 0.}
    counts = {'A08_windows': 0, 'JEPA_chunks': 0}
    for source in ['a08', 'jepa']:
        start = time.perf_counter()
        model = build_source(source, cfg, 'cuda')
        torch.cuda.synchronize()
        loads[source] = time.perf_counter() - start
        values[source], timings[source] = [], []
        if source == 'a08':
            for row in private['a08_raw_logit_panel']:
                q = fit[row['sequence']]
                grid = windows(len(q['refs']), 5)
                location = np.flatnonzero(grid['centre'] == row['centre'])
                if len(location) != 1 or int(grid['marker'][location[0]]) != row['marker']:
                    raise ValueError('original native centre/marker identity changed')
                logits = a08_windows(model, q, pixels, [row['centre']])[0]
                errors['A08_raw_pre_softmax_five_logit_max_error'] = max(errors['A08_raw_pre_softmax_five_logit_max_error'], float(np.max(np.abs(logits - row['expected_logits']))))
                counts['A08_windows'] += 1
        old, old_qs = old_cache('jepa', 'fit', cfg) if source == 'jepa' else (None, None)
        mapping = native_reuse_index(old_qs, qs) if old is not None else {}
        for i, q in enumerate(qs):
            torch.cuda.synchronize()
            start = time.perf_counter()
            value = (a08_sequence if source == 'a08' else jepa_sequence)(model, q, pixels)
            torch.cuda.synchronize()
            wall = time.perf_counter() - start
            units = len(windows(len(q['refs']), 5)['centre']) if source == 'a08' else (len(q['refs']) + 31) // 32
            counts['A08_windows' if source == 'a08' else 'JEPA_chunks'] += units
            timings[source].append({'original_positions': len(q['refs']), 'neural_units': units, 'wall_seconds': wall})
            values[source].append(value)
            if i in mapping:
                error = float(np.max(np.abs(value - old.load(mapping[i]))))
                errors['JEPA_native_probability_max_error'] = max(errors['JEPA_native_probability_max_error'], error)
            write(RUN / 'runtimecoordination.json', {'phase': 'FIT_source_GPU_graph_profile',
                  'source': source, 'profile_sequences_done': i + 1, 'profile_sequences': len(qs),
                  'model_pid': os.getpid(), 'formal_optimizer_updates': 0, 'epoch': time.time(),
                  'next_action': 'finish pinned FIT parity/timing; CPU graph and complete forecast before cache'})
        if any(p.requires_grad for p in model.parameters()):
            raise ValueError('source parameter unexpectedly trainable')
        del model
        torch.cuda.empty_cache()
    if errors['A08_raw_pre_softmax_five_logit_max_error'] > 1e-4 or errors['JEPA_native_probability_max_error'] > 1e-6:
        raise ValueError('pinned FIT source parity tolerance exceeded')
    path = RUN / 'FIT_profile_posteriors.private.npy'
    save_array(path, {'selected_sequence_indices': selected, 'probabilities': values})
    IO = []
    for source in ['a08', 'jepa']:
        cache = ProbabilityCache(RUN / 'engineering' / ('source_IO_' + source), source + '|' + sha(path), [len(q['refs']) for q in qs])
        for i, value in enumerate(values[source]):
            start = time.perf_counter()
            cache.run_one(i, lambda value=value: value)
            cache.load(i)
            IO.append(time.perf_counter() - start)
    write(PUB / 'FIT_GPU_checks.json', {'passed': True, 'source_load_seconds': loads, 'source_timings': timings,
          'parity': errors, 'physical_neural_units': counts, 'actual_sources': 2,
          'source_parameters_all_frozen': True, 'original5_source_contract': cfg['source']['a08'],
          'posteriors': array_pin(path), 'profile_input_identity_sha256': sha(PUB / 'profile_input_identity.json'),
          'FIT_graph_freeze_git': freeze['git_commit'], 'maximum_durable_cache_IO_seconds_per_sequence': max(IO),
          'formal_optimizer_updates': 0, 'new_protected_media_forwards': 0})


def cpu():
    freeze = verify_freeze('CPU_graph_freeze.json')
    gpu_result = read(PUB / 'FIT_GPU_checks.json')
    if not gpu_result['passed']:
        raise ValueError('real source graph checks are incomplete')
    profile = np.load(verify_pin(gpu_result['posteriors']), allow_pickle=True).item()
    fit, cfg = sequences('fit'), read(CFG)
    qs = [fit[i] for i in profile['selected_sequence_indices']]
    a, j = profile['probabilities']['a08'], profile['probabilities']['jepa']
    base = [decode_with_prob('b3', p, {'decode': {'b3': cfg['source']['a08']['original_B3']}})[0] for p in a]
    data = prepare_training(qs, a, j, base)
    if not len(data['targets']):
        raise ValueError('fixed FIT profile produced no candidate graph; bounded engineering decision required')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    physical_updates, model_count, checks = 0, 0, {}
    identity = {key: digest({'profile': sha(PUB / 'FIT_GPU_checks.json'), 'field': key})
                for key in ['code', 'config', 'sources', 'input', 'draw_plan', 'normalizer']}
    for arm in ['A', 'J']:
        norm = fit_normalizer(data['features'][arm])
        x = torch.from_numpy(normalize(data['features'][arm], norm))
        y = torch.from_numpy(data['targets'])
        batches, substituted = [], []
        for kind in ['native', 'constructed']:
            ids = data['kind_indices'][kind]
            if not len(ids):
                ids = np.arange(len(y))
                substituted.append(kind)
            picked = ids[np.arange(32) % len(ids)]
            batches.append((x[picked], y[picked]))
        def run(stop, value=None):
            nonlocal physical_updates, model_count
            model_count += 1
            model = UtilityRegressor(arm, 2026110359)
            opt = optimizer(model)
            start = 0 if value is None else restore(model, opt, value, identity, norm)
            rows = []
            for u in range(start, stop):
                rows.append(step(model, opt, batches, u))
                physical_updates += 1
            return checkpoint(model, opt, stop, identity, norm), rows
        expected, rows = run(4)
        repeat, _ = run(4)
        first, _ = run(2)
        file = RUN / 'engineering' / f'{arm}_resume.pt'
        atomic_save(first, file)
        resumed, _ = run(4, torch.load(file, map_location='cpu', weights_only=True))
        if not same(expected['model'], repeat['model']) or not same(expected['optimizer'], repeat['optimizer']) or not same(expected['model'], resumed['model']) or not same(expected['optimizer'], resumed['optimizer']):
            raise ValueError('real FIT-derived feature repeat or disk resume mismatch')
        live = {name: any(row['nonzero_gradient_tensors'][name] for row in rows) for name in rows[0]['nonzero_gradient_tensors']}
        if not all(live.values()):
            raise ValueError('later first-layer/tensor gradient liveness failed on fixed FIT graph')
        model_count += 1
        model = UtilityRegressor(arm, 2026110359)
        opt = optimizer(model)
        times = []
        for u in range(20):
            start = time.perf_counter()
            step(model, opt, batches, u)
            times.append(time.perf_counter() - start)
            physical_updates += 1
        inference = []
        for q, pa, pj, b in zip(qs, a, j, base):
            inference.append(infer(model, norm, pa, pj, b, arm)['cached_feature_head_policy_seconds'])
        checks[arm] = {'later_tensor_liveness': live, 'repeat_and_2plus2_disk_resume_exact': True,
                       '64candidate_update_median_seconds': float(np.median(times)),
                       '64candidate_update_max_seconds': max(times),
                       'cached_head_feature_policy_max_seconds_per_sequence': max(inference),
                       'engineering_only_missing_kind_substitution': substituted,
                       'formal_plan_or_normalizer_or_Ffit_not_selected_here': True}
    if torch.cuda.is_initialized():
        raise ValueError('CPU head graph unexpectedly initialized CUDA')
    write(PUB / 'FIT_CPU_checks.json', {'passed': True, 'heads': checks,
          'engineering_optimizer_updates': physical_updates, 'engineering_mutable_models': model_count,
          'FIT_profile_candidates': len(data['targets']), 'FIT_profile_kind_candidates': {k: len(v) for k, v in data['kind_indices'].items()},
          'actual64candidate_tensor_graph': True, 'wholeFIT_fixed_plan_check_required_after_cache': True,
          'CUDA_initialized': False, 'formal_optimizer_updates': 0,
          'GPU_checks_sha256': sha(PUB / 'FIT_GPU_checks.json'), 'FIT_graph_freeze_git': freeze['git_commit']})


if __name__ == '__main__':
    modes = {'plan': plan, 'gpu': gpu, 'cpu': cpu}
    if len(sys.argv) != 2 or sys.argv[1] not in modes:
        raise ValueError('explicit plan/gpu/cpu FIT engineering stage required')
    modes[sys.argv[1]]()
