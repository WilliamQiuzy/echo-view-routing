#!/usr/bin/env python3
"""A59 fixed four CPU heads; inner records are diagnostic, never selectors."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import json
import sys
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from scripts.anchor_common import RUN, PUB, CFG, read, write, sha, digest, verify_pin, verify_freeze, sequences, save_array, array_pin
from scripts.cache_anchor_edits import reader
from echo_routing.temporal.anchor_training import UtilityRegressor, optimizer, draw_plan, step, checkpoint, restore, atomic_save, UpdateJournal, fsync_directory
from echo_routing.temporal.anchor_features import fit_normalizer, normalize
from echo_routing.temporal.anchor_runtime import infer, prepare_training
from echo_routing.temporal.anchor_edits import majority
from echo_routing.temporal.baselines.registry import decode_with_prob
from echo_routing.temporal.dense_runtime import metrics
from echo_routing.temporal.dense_analysis import short_metrics
from echo_routing.temporal.anchor_exposure import summarize as summarize_exposure


def role_values(role):
    qs = sequences(role)
    caches = {source: reader(source, role, readonly=True) for source in ['a08', 'jepa']}
    values = {source: [cache.load(i) for i in range(len(qs))] for source, cache in caches.items()}
    cfg = read(CFG)
    base = [decode_with_prob('b3', p, {'decode': {'b3': cfg['source']['a08']['original_B3']}})[0] for p in values['a08']]
    return qs, values['a08'], values['jepa'], base, {k: v.identity for k, v in caches.items()}


def prepare():
    verify_freeze('training_prepare_freeze.json')
    verify_freeze('producer_freeze.json')
    if not read(PUB / 'cache_fit_manifest.json')['passed'] or (PUB / 'training_inputs.json').exists():
        raise ValueError('complete cache and fresh training preparation required')
    qs, a, j, base, sources = role_values('fit')
    data = prepare_training(qs, a, j, base)
    if not len(data['targets']):
        write(PUB / 'no_action_domain.json', {'genuine_no_action_domain': True, 'formal_optimizer_updates': 0,
              'FIT_sources_forwarded_are_not_head_optimizer_exposures': True})
        return
    normalizers = {arm: fit_normalizer(data['features'][arm]) for arm in ['A', 'J']}
    for key in ['mean', 'std']:
        if not np.array_equal(normalizers['A'][key][:62], normalizers['J'][key][:62]):
            raise ValueError('matched common/A08 FIT normalizer blocks differ')
    path = RUN / 'training_dataset.private.npy'
    save_array(path, data)
    norms = {}
    for arm, norm in normalizers.items():
        target = RUN / f'normalizer_{arm}.private.json'
        write(target, {k: v.tolist() for k, v in norm.items()})
        norms[arm] = array_pin(target)
    plans = {}
    for seed in read(CFG)['training']['seeds']:
        plan = draw_plan(len(data['kind_indices']['native']), len(data['kind_indices']['constructed']), seed)
        target = RUN / f'draw_plan_seed{seed}.private.npy'
        save_array(target, plan)
        plans[str(seed)] = array_pin(target)
    eligible_sequences = {row['sequence_index'] for row in data['candidate_rows']}
    eligibility = summarize_exposure(qs, data['candidate_rows'])
    summary = {'passed': True, 'dataset': array_pin(path), 'normalizers': norms, 'plans': plans,
               'source_cache_identities': sources, 'FIT_sequence_sha256': read(PUB / 'input_manifest.json')['roles']['fit']['sha256'],
               'F_fit': data['F_fit'], 'FIT_operating_counts': data['FIT_operating_counts'],
               'candidates': len(data['targets']), 'kind_candidates': {k: len(v) for k, v in data['kind_indices'].items()},
               'eligible_sequences': len(eligible_sequences), 'ineligible_sequences': len(qs) - len(eligible_sequences),
               'features_each': 98, 'matched_common62_normalizer_equal': True,
               'normalizer_scope': 'all eligible FIT candidates only; population std floor1e-6',
               'native_utility_is_count_stability_proxy_not_macroF1_surrogate': True,
               'forwarded_sources_are_not_all_optimizer_exposed': True,
               'candidate_eligibility': {k: v for k, v in eligibility.items() if not k.startswith('actual_')},
               'inner_or_outer_used_for_targets_normalizer_or_plan': False, 'formal_optimizer_updates': 0}
    write(PUB / 'training_inputs.json', summary)


def train_one(model, opt, data, plan, normalizer, identity, folder, stop=6000,
              latest_every=100, milestones=(2000, 4000, 6000), on_inner=None, on_progress=None):
    if not 0 <= stop <= 6000 or latest_every < 1:
        raise ValueError('bounded fixed-schedule update cursor required')
    journal = UpdateJournal(folder, identity)
    folder = Path(folder)
    inner_pending = folder / 'inner_pending.json'
    if inner_pending.exists():
        raise ValueError('ambiguous pending inner readout; no automatic replay')
    latest = folder / 'latest.pt'
    if not latest.exists() and journal.cursor == 0:
        atomic_save(checkpoint(model, opt, 0, identity, normalizer), latest)
    start = journal.resume_cursor()
    restore(model, opt, torch.load(latest, map_location='cpu', weights_only=True), identity, normalizer)
    if start > stop:
        raise ValueError('cannot move the committed optimizer cursor backward')
    feature = torch.from_numpy(normalize(data['features'][model.arm], normalizer))
    target = torch.from_numpy(data['targets'])
    def inner(cursor):
        path = folder / f'step{cursor}.pt'
        marker = folder / f'inner_step{cursor}.json'
        expected = {'identity': identity, 'weight_sha256': sha(path), 'step': cursor}
        if marker.exists():
            value = read(marker)
            if any(value[k] != v for k, v in expected.items()):
                raise ValueError('completed inner/checkpoint binding changed')
            result = value['callback_result']
            if isinstance(result, dict) and 'path' in result and 'sha256' in result:
                verify_pin(result)
            return
        if inner_pending.exists():
            raise ValueError('ambiguous pending inner readout')
        with inner_pending.open('x') as handle:
            json.dump(expected, handle, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        fsync_directory(folder)
        result = on_inner(model, cursor, path) if on_inner is not None else None
        write(marker, {**expected, 'callback_result': result,
                       'engineering_helper_without_inner_requested': on_inner is None})
        inner_pending.unlink()
        fsync_directory(folder)
    if start in milestones:
        inner(start)
    for update in range(start, stop):
        batches, indices = [], []
        for kind in ['native', 'constructed']:
            picked = data['kind_indices'][kind][plan[kind][update]]
            indices.extend(picked.tolist())
            batches.append((feature[picked], target[picked]) if len(picked) else None)
        journal.begin(update)
        t0 = time.perf_counter()
        row = step(model, opt, batches, update)
        row.update({'candidate_indices': indices,
                    'physical_feature_positions': sum(data['candidate_rows'][i]['feature_positions'] for i in indices),
                    'optimizer_update_wall_seconds': time.perf_counter() - t0})
        cursor = update + 1
        payload = checkpoint(model, opt, cursor, identity, normalizer) if cursor % latest_every == 0 or cursor in milestones else None
        if cursor in milestones:
            atomic_save(payload, folder / f'step{cursor}.pt')
        journal.finish(row, payload)
        if cursor in milestones:
            inner(cursor)
        if on_progress is not None and cursor % 25 == 0:
            on_progress(cursor, row)
    if torch.cuda.is_initialized():
        raise ValueError('CPU head unexpectedly initialized CUDA; resource correction required')
    rows = [json.loads(line) for line in journal.trace.read_text().splitlines()] if journal.trace.exists() else []
    actual = [float(opt.state[p]['step']) for p in model.parameters()] if stop else []
    if actual and any(value != stop for value in actual):
        raise ValueError('actual Adam counters differ from committed cursor')
    return {'committed_physical_updates': len(rows), 'actual_Adam_update': stop,
            'new_updates_this_process': stop - start,
            'physical_candidate_exposures': sum(row['physical_candidates'] for row in rows),
            'physical_feature_position_exposures': sum(row['physical_feature_positions'] for row in rows),
            'all_tensor_gradient_liveness_over_history': {name: any(row['nonzero_gradient_tensors'][name] for row in rows)
                                                         for name, _ in model.named_parameters()},
            'journal_sha256': sha(journal.trace) if journal.trace.exists() else None,
            'CUDA_initialized': False}


def formal():
    freeze = verify_freeze('formal_freeze.json')
    if not read(PUB / 'formal_admission.json')['all_caps_fit']:
        raise ValueError('complete measured remaining matrix forecast is not admitted')
    cfg, inputs = read(CFG), read(PUB / 'training_inputs.json')
    if not inputs['passed']:
        raise ValueError('FIT preparation is incomplete')
    data = np.load(verify_pin(inputs['dataset']), allow_pickle=True).item()
    inner_qs, inner_a, inner_j, inner_b, inner_sources = role_values('inner_cal')
    records, progress = [], []
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    for seed in cfg['training']['seeds']:
        plan_pin = inputs['plans'][str(seed)]
        plan = np.load(verify_pin(plan_pin), allow_pickle=True).item()
        for arm in cfg['training']['arms']:
            normalizer = read(verify_pin(inputs['normalizers'][arm]))
            identity = {'code': digest(freeze['objects']), 'config': sha(CFG),
                        'sources': digest(inputs['source_cache_identities']), 'input': inputs['dataset']['sha256'],
                        'draw_plan': plan_pin['sha256'], 'normalizer': inputs['normalizers'][arm]['sha256']}
            model = UtilityRegressor(arm, seed)
            opt = optimizer(model)
            folder = RUN / 'models' / f'seed{seed}' / arm
            def record_inner(current, cursor, path):
                target = PUB / f'inner_seed{seed}_{arm}_step{cursor}.json'
                if target.exists():
                    old = read(target)
                    if old['weight_sha256'] != sha(path) or old['identity'] != identity:
                        raise ValueError('existing inner diagnostic record binding changed')
                    verify_pin(old['private_predictions'])
                    return {'path': str(target.relative_to(ROOT)), 'sha256': sha(target)}
                predictions = {policy: {} for policy in ['U', 'G']}
                guard_matches, candidates, accepted = 0, 0, {'U': 0, 'G': 0}
                current.eval()
                for q, a, j, b in zip(inner_qs, inner_a, inner_j, inner_b):
                    result = infer(current, normalizer, a, j, b, arm)
                    candidates += len(result['spans'])
                    for policy in ['U', 'G']:
                        predictions[policy][q['id']] = result[policy]['prediction']
                        accepted[policy] += result[policy]['accepted'] is not None
                    guard_matches += majority(result['G']['prediction']) == majority(b)
                if guard_matches != len(inner_qs):
                    raise ValueError('global majority protection failed on complete inner axes')
                private_labels = folder / f'inner_step{cursor}_labels.private.npy'
                save_array(private_labels, predictions)
                value = {'seed': seed, 'arm': arm, 'step': cursor, 'weight_sha256': sha(path),
                         'identity': identity, 'source_cache_identities': inner_sources,
                         'private_predictions': array_pin(private_labels),
                         'policies': {policy: {**metrics(inner_qs, pred)[0], **short_metrics(inner_qs, pred, 16)}
                                      for policy, pred in predictions.items()},
                         'candidate_count': candidates, 'accepted_actions': accepted,
                         'all_sequence_guard_majorities_equal_Base': True,
                         'diagnostic_only_no_checkpoint_or_threshold_selection': True}
                write(target, value)
                current.train()
                return {'path': str(target.relative_to(ROOT)), 'sha256': sha(target)}
            def update_progress(cursor, row):
                write(RUN / 'runtimecoordination.json', {'phase': 'formal_utility_heads', 'seed': seed,
                      'arm': arm, 'committed_updates': cursor, 'actual_Adam_updates': cursor,
                      'prior_completed_model_updates': sum(r['committed_physical_updates'] for r in progress),
                      'model_pid': os.getpid(), 'epoch': time.time(),
                      'next_action': 'complete all4x6000 and12 diagnostic inner records; freeze before outer'})
            result = train_one(model, opt, data, plan, normalizer, identity, folder,
                               on_inner=record_inner, on_progress=update_progress)
            actual_rows = [json.loads(line) for line in (folder / 'physical_updates.jsonl').read_text().splitlines()]
            result['actual_candidate_and_context_exposure'] = summarize_exposure(
                sequences('fit'), data['candidate_rows'], [i for row in actual_rows for i in row['candidate_indices']])
            progress.append(result)
            records.append({'seed': seed, 'arm': arm, 'result': result,
                            'final_path': str((folder / 'step6000.pt').relative_to(ROOT)),
                            'final_sha256': sha(folder / 'step6000.pt'), 'identity': identity})
    write(PUB / 'formal_completion.json', {'passed': True, 'models': records,
          'physical_models': 4, 'formal_optimizer_updates': sum(r['result']['committed_physical_updates'] for r in records),
          'CUDA_initialized': torch.cuda.is_initialized(), 'inner_used_for_model_selection': False})


if __name__ == '__main__':
    if sys.argv[1:] == ['prepare']:
        prepare()
    elif sys.argv[1:] == ['formal']:
        formal()
    else:
        raise ValueError('explicit prepare/formal stage required')
