#!/usr/bin/env python3
"""A59 CPU graph v2: real-panel timing plus explicit gradient-control fixture.

A panel fitted to one/constant candidate cannot establish first-layer weight
liveness. Report that data condition and test the computational path separately;
neither fixture nor its normalizer is used by the formal FIT trainer.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import sys
from pathlib import Path
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from scripts.anchor_common import RUN, PUB, CFG, read, write, sha, digest, verify_pin, verify_freeze, sequences
from echo_routing.temporal.anchor_features import fit_normalizer, normalize
from echo_routing.temporal.anchor_training import UtilityRegressor, optimizer, step, checkpoint, restore, atomic_save
from echo_routing.temporal.anchor_runtime import prepare_training, infer
from echo_routing.temporal.baselines.registry import decode_with_prob


def same(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b


def probe_head(arm, features, targets, kind_indices, folder, input_sha, profile_axes=None):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    norm = fit_normalizer(features)
    x, y = torch.from_numpy(normalize(features, norm)), torch.from_numpy(targets)
    batches, substituted = [], []
    for kind in ['native', 'constructed']:
        ids = kind_indices[kind]
        if not len(ids):
            ids = np.arange(len(y))
            substituted.append(kind)
        picked = ids[np.arange(32) % len(ids)]
        batches.append((x[picked], y[picked]))
    identity = {key: digest({'input': input_sha, 'field': key, 'engineering': 'head_runtime_v2'})
                for key in ['code', 'config', 'sources', 'input', 'draw_plan', 'normalizer']}
    updates, model_count = 0, 0
    def run(stop, value=None):
        nonlocal updates, model_count
        model_count += 1
        model = UtilityRegressor(arm, 2026110359)
        opt = optimizer(model)
        start = 0 if value is None else restore(model, opt, value, identity, norm)
        rows = []
        for u in range(start, stop):
            rows.append(step(model, opt, batches, u))
            updates += 1
        return checkpoint(model, opt, stop, identity, norm), rows
    expected, rows = run(4)
    repeat, _ = run(4)
    prefix, _ = run(2)
    file = Path(folder) / f'{arm}_head_runtime_v2_resume.pt'
    atomic_save(prefix, file)
    resumed, _ = run(4, torch.load(file, map_location='cpu', weights_only=True))
    if any(not same(expected[field], value[field]) for value in [repeat, resumed] for field in ['model', 'optimizer']):
        raise ValueError('real FIT-derived repeat or2plus2disk resume differs')
    actual_live = {name: any(row['nonzero_gradient_tensors'][name] for row in rows) for name in rows[0]['nonzero_gradient_tensors']}
    model_count += 1
    timed = UtilityRegressor(arm, 2026110359)
    opt, times = optimizer(timed), []
    for u in range(20):
        start = time.perf_counter()
        step(timed, opt, batches, u)
        times.append(time.perf_counter() - start)
        updates += 1
    # This deliberately nondegenerate fixture diagnoses the graph, not efficacy.
    rng = np.random.default_rng([2026110359, 59, 99])
    cx = torch.from_numpy(rng.normal(size=(64, 98)).astype(np.float32))
    cy = torch.linspace(-1, 1, 64) + .25
    control_batch = [(cx[:32], cy[:32]), (cx[32:], cy[32:])]
    model_count += 1
    control = UtilityRegressor(arm, 2026110359)
    opt, control_rows = optimizer(control), []
    for u in range(4):
        control_rows.append(step(control, opt, control_batch, u))
        updates += 1
    live = {name: any(row['nonzero_gradient_tensors'][name] for row in control_rows) for name, _ in control.named_parameters()}
    if not all(live.values()) or any(not bool(opt.state[p]['exp_avg_sq'].count_nonzero()) for p in control.parameters()):
        raise ValueError('nondegenerate control failed complete tensor gradient/moment liveness')
    latency = [infer(timed, norm, a, j, b, arm)['cached_feature_head_policy_seconds']
               for a, j, b in profile_axes] if profile_axes is not None else []
    return {'source_panel_nonconstant_normalized_features': int(np.count_nonzero(np.std(features, axis=0))),
            'actual_panel_later_tensor_liveness': actual_live,
            'nondegenerate_control_later_tensor_liveness': live,
            'repeat_and_2plus2_disk_resume_exact': True,
            'engineering_optimizer_updates': updates, 'engineering_mutable_models': model_count,
            '64candidate_update_median_seconds': float(np.median(times)),
            '64candidate_update_max_seconds': max(times),
            'cached_head_feature_policy_max_seconds_per_sequence': max(latency) if latency else None,
            'engineering_only_missing_kind_substitution': substituted,
            'gradient_control_used_as_study_training_data': False,
            'formal_plan_normalizer_or_Ffit_selected_here': False}


def main():
    freeze = verify_freeze('CPU_graph_v2_freeze.json')
    gpu = read(PUB / 'FIT_GPU_checks.json')
    profile = np.load(verify_pin(gpu['posteriors']), allow_pickle=True).item()
    fit, cfg = sequences('fit'), read(CFG)
    qs = [fit[i] for i in profile['selected_sequence_indices']]
    a, j = profile['probabilities']['a08'], profile['probabilities']['jepa']
    base = [decode_with_prob('b3', p, {'decode': {'b3': cfg['source']['a08']['original_B3']}})[0] for p in a]
    data = prepare_training(qs, a, j, base)
    if not len(data['targets']):
        raise ValueError('no actual FIT-panel candidates to benchmark')
    checks = {arm: probe_head(arm, data['features'][arm], data['targets'], data['kind_indices'],
                             RUN / 'engineering', sha(PUB / 'FIT_GPU_checks.json'), list(zip(a, j, base)))
              for arm in ['A', 'J']}
    if torch.cuda.is_initialized():
        raise ValueError('CPU graph unexpectedly initialized CUDA')
    write(PUB / 'FIT_CPU_checks.json', {'passed': True, 'heads': checks,
          'engineering_optimizer_updates': sum(v['engineering_optimizer_updates'] for v in checks.values()),
          'engineering_mutable_models': sum(v['engineering_mutable_models'] for v in checks.values()),
          'prior_failed_cpu_profile_updates_retained_separately': 12,
          'prior_failed_cpu_profile_mutable_models_retained_separately': 4,
          'FIT_profile_candidates': len(data['targets']),
          'FIT_profile_kind_candidates': {k: len(v) for k, v in data['kind_indices'].items()},
          'actual64candidate_tensor_graph': True, 'wholeFIT_fixed_plan_check_required_after_cache': True,
          'CUDA_initialized': False, 'formal_optimizer_updates': 0,
          'GPU_checks_sha256': sha(PUB / 'FIT_GPU_checks.json'), 'CPU_graph_freeze_git': freeze['git_commit'],
          'failed_source_frozen_producer_preserved': True})


if __name__ == '__main__':
    if sys.argv[1:] != ['run']:
        raise ValueError('explicit CPU graph engineering stage required')
    main()
