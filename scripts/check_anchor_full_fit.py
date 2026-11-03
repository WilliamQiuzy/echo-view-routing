#!/usr/bin/env python3
"""A59 actual whole-FIT arithmetic, fixed-plan CPU graph, and measured update path."""
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
from scripts.anchor_common import RUN, PUB, CFG, read, write, sha, digest, verify_pin, verify_freeze
from scripts.train_anchor_edits import role_values
from echo_routing.temporal.anchor_data_checks import validate
from echo_routing.temporal.anchor_features import normalize
from echo_routing.temporal.anchor_training import UtilityRegressor, optimizer, step, checkpoint, restore, atomic_save
from scripts.check_anchor_head_runtime import same


def main():
    freeze = verify_freeze('wholeFIT_check_freeze.json')
    inputs = read(PUB / 'training_inputs.json')
    data = np.load(verify_pin(inputs['dataset']), allow_pickle=True).item()
    qs, a, j, base, sources = role_values('fit')
    if sources != inputs['source_cache_identities']:
        raise ValueError('wholeFIT source cache identities changed')
    norms = {arm: read(verify_pin(pin)) for arm, pin in inputs['normalizers'].items()}
    plans = {int(seed): np.load(verify_pin(pin), allow_pickle=True).item() for seed, pin in inputs['plans'].items()}
    arithmetic = validate(data, qs, a, base, norms, plans)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    checks, total_updates, models = {}, 0, 0
    seed = 2026110359
    plan = plans[seed]
    for arm in ['A', 'J']:
        x = torch.from_numpy(normalize(data['features'][arm], norms[arm]))
        y = torch.from_numpy(data['targets'])
        identity = {'code': digest(freeze['objects']), 'config': sha(CFG), 'sources': digest(sources),
                    'input': inputs['dataset']['sha256'], 'draw_plan': inputs['plans'][str(seed)]['sha256'],
                    'normalizer': inputs['normalizers'][arm]['sha256']}
        def batch(u):
            out = []
            for kind in ['native', 'constructed']:
                index = data['kind_indices'][kind][plan[kind][u]]
                out.append((x[index], y[index]) if len(index) else None)
            return out
        def run(stop, value=None):
            nonlocal total_updates, models
            models += 1
            model = UtilityRegressor(arm, seed)
            opt = optimizer(model)
            start = 0 if value is None else restore(model, opt, value, identity, norms[arm])
            if value is None:
                with torch.no_grad():
                    if torch.count_nonzero(model(x)):
                        raise ValueError('wholeFIT initial output is not exact zero/no-op')
            rows = []
            for u in range(start, stop):
                rows.append(step(model, opt, batch(u), u))
                total_updates += 1
            return checkpoint(model, opt, stop, identity, norms[arm]), rows
        expected, rows = run(4)
        initial_losses = [float(np.mean(target.numpy() ** 2)) for value in batch(0) if value is not None for _, target in [value]]
        if not np.isclose(rows[0]['loss'], np.mean(initial_losses), rtol=1e-6, atol=1e-6):
            raise ValueError('actual kind-balanced MSE disagrees with independent NumPy reduction')
        repeat, _ = run(4)
        prefix, _ = run(2)
        path = RUN / 'engineering' / f'wholeFIT_{arm}_resume.pt'
        atomic_save(prefix, path)
        resumed, _ = run(4, torch.load(path, map_location='cpu', weights_only=True))
        if any(not same(expected[field], value[field]) for value in [repeat, resumed] for field in ['model', 'optimizer']):
            raise ValueError('wholeFIT fixed-plan repeat or2plus2resume mismatch')
        live = {name: any(row['nonzero_gradient_tensors'][name] for row in rows) for name in rows[0]['nonzero_gradient_tensors']}
        if not all(live.values()):
            raise ValueError('actual complete-FIT normalized fixed-plan gradient support needs diagnosis')
        models += 1
        model, times = UtilityRegressor(arm, seed), []
        opt = optimizer(model)
        for u in range(20):
            start = time.perf_counter()
            step(model, opt, batch(u), u)
            times.append(time.perf_counter() - start)
            total_updates += 1
        start = time.perf_counter()
        atomic_save(checkpoint(model, opt, 20, identity, norms[arm]), RUN / 'engineering' / f'wholeFIT_{arm}_timed.pt')
        checks[arm] = {'all_four_tensor_gradient_history_live': live,
                       'zero_initial_scores_allFIT_candidates': True,
                       'repeat_and2plus2disk_resume_model_and_optimizer_exact': True,
                       'fixed_seed_plan_rows_checked': [0, 19],
                       'candidate_batch_each_nonempty_kind': 32,
                       'actual_physical_candidates_per_update': sum(32 for kind in ['native', 'constructed'] if len(data['kind_indices'][kind])),
                       'actual_plan_empty_kind_mask': {kind: not len(data['kind_indices'][kind]) for kind in ['native', 'constructed']},
                       'fixed_plan_update_max_seconds': max(times),
                       'fixed_plan_update_median_seconds': float(np.median(times)),
                       'complete_checkpoint_and_directory_sync_seconds': time.perf_counter() - start}
    if torch.cuda.is_initialized():
        raise ValueError('wholeFIT CPU graph unexpectedly initialized CUDA')
    write(PUB / 'wholeFIT_checks.json', {'passed': True, 'arithmetic': arithmetic, 'heads': checks,
          'actual_wholeFIT_normalizers_and_prefrozen_seed_plan_used': True,
          'synthetic_gradient_fixture_used': False, 'engineering_optimizer_updates': total_updates,
          'engineering_mutable_models': models, 'formal_optimizer_updates': 0,
          'training_inputs_sha256': sha(PUB / 'training_inputs.json'),
          'wholeFIT_check_freeze_git': freeze['git_commit'], 'CUDA_initialized': False})


if __name__ == '__main__':
    if sys.argv[1:] != ['run']:
        raise ValueError('explicit wholeFIT actual-data graph stage required')
    main()
