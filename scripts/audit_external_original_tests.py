#!/usr/bin/env python3
"""Independent, CPU-only recount of the frozen A50 saved predictions.

Run on echo-l40s only AFTER both inference queues and their reports are complete::

    python scripts/audit_external_original_tests.py

Only NumPy and the standard library are imported at startup. OpenCV is imported
only to recheck video container frame counts/FPS. No model or production evaluator
is imported, no GPU operation occurs, and private case identifiers never enter
stdout or public output. --root permits an explicitly located server checkout.
The only writes are verification.json and uncertainty.json in the result folder.
A failed audit exits nonzero and writes an aggregate-only failed verification.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import gzip
import hashlib
import itertools
import json
import math
import re
import struct
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULT_REL = 'docs/results/2026-10-30_external_original_tests'
RUN_REL = 'runs/external_original_test_v1'
CLASSES = ('PLAX', 'PSAX', 'A4C', 'A5C', 'SC4C')
RAW9 = ('PLHLA', 'PMASA', 'PMVLSA', 'PASA', 'A4C', 'A5C', 'PMPALA', 'PPMLSA', 'SC4C')
RAW_TO_FIVE = (0, 1, 1, 1, 2, 3, None, 1, 4)
EXPECTED = {'ev9v': 800, 'dynamic': 1277, 'lvh': 343, 'camus': 50}
SEEDS = (0, 1, 2)
CONDITIONS = ('legacy_per_key_prior', 'corrected_prior_once_same_old_weights')
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 20261030
# These hashes were read from the already-frozen aggregate preparation receipts,
# before this independent script was written. They are independent trust anchors:
# editing protocol AND its local mirror cannot silently bless a different run.
ANCHORS = {
    'preparation.json': '9cfe576a90e88702e2b2750d50bbd1f102f39a9dcde1136c245c8b3f62b66c63',
    'preparation_v2.json': '08a34797b505303d32fd9171df1edbb34157f3c5b749e09853254a3af00f298d',
    'baseline_preparation.json': 'fdc75c818a7a5247bde073b70fc9414b75ffddfd3361f2dbd9b81addcbfd1a1f',
    'input_audit.json': '322133d4c220f5ea53ef82ddbd92e4325e6286e0a1bf0b2e9905e989e1dd4bc2',
    'timing_amendment.json': '2b78be70ed67f0ab239ba50a9026571176aa11b998a789608d50af63dadb54ef',
    'checks_v1.json': '4d3125fd87817752c27c81675e482cc1e71c6247778912c7135b076501818105',
    'checks.json': '9ad85f53ef65fa1c62de796e03b1335521302d3451f1b7b867f47ffc080cd727',
}
BOUND_PROTOCOLS = {'ev9v': 'protocol_v1.json', 'dynamic': 'protocol_v1.json',
                   'lvh': 'protocol.json', 'camus': 'protocol.json'}
BINDING_SHA = 'e937041c5ac8d5c0bac292783381f5cf251719602d57f14b3c1ca31ab478416a'
QUALIFICATION_SHA = '3b8457ab4208d9c7e6fc162c1890101a71188fcb4fa809d26abfd7d8130a2ae3'
HISTORY_REL = 'docs/results/2026-10-23_frozen_external_benchmark/evaluation_v1.json'
HISTORY_SHA = 'e7702bc66b3b2bc018263a9d60bad5b382c1d197124d15e90c19aa1b95d34a8f'
CHOICES_REL = 'docs/results/2026-10-16_video_within_view/choices.json'
CHOICES_SHA = 'b2a731e5f4393187ee9954f710821ac04938652f0a7d00e0381ad0b8b21cf27e'
LIMITS = [
    'EV9V is the historically accessed STFM source test, not external or held-out confirmation.',
    'Dynamic A4C and LVH PLAX are source-code recall checks, not EF or wall-thickness endpoints.',
    'CAMUS official 4CH source codes are not expert adjudication of A4C versus A5C.',
    'CAMUS half-cycle clips span approximately 0.18 to 0.51 seconds; short input support limits interpretation.',
    'EV9V clusters are sessions, not confirmed patients; Dynamic/LVH patient identities are unknown.',
    'CAMUS clusters are official patient identities; unknown pretraining/source overlap remains unknown.',
    'Clip labels are weak supervision; no clinical natural-transition or local-anatomy result is established.',
    'No source pixel reload or model forward is performed. Cached input-pixel digests are checked for form only.',
    'Media size/mtime, source-metadata hashes, video container timing and CAMUS header length are rechecked; these are not raw-media byte hashes.',
    'The prospective LVH timing correction retains v1 EV9V/Dynamic caches; LVH/CAMUS use the explicitly bound v2 protocol. Cache chronology uses filesystem timestamps, not recorded model-forward start times.',
    'Protocol pins only its declared source modules; unpinned transitive/library/environment dependencies cannot be retroactively frozen.',
    'Baseline caches have no per-case preprocessing/sampling/timing receipt; only logits shape, identity, manifest duration and declared input recipe can be audited.',
    'Per-case GPU timing is not stored in prediction caches; inference resource accounting requires the separate terminal/resource receipts.',
    'STFM whole-file classification emits no internal cuts by construction; zero cuts is not a stability result.',
    'Bootstrap intervals describe this fixed evaluation and grouping; they are not a clinical or finite-sample guarantee.',
]


class AuditError(RuntimeError):
    """A controlled failure whose message contains no private case details."""


def require(condition, message):
    if not condition:
        raise AuditError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def require_hash(path, expected, label):
    require(Path(path).is_file(), label + ': required hash input missing')
    actual = sha(path)
    require(actual == expected, label + ': immutable hash mismatch')
    return actual


def public_source_hash_summary(verified_mapping):
    """Keep per-source paths private after individually verifying every hash."""
    canonical = json.dumps(verified_mapping, sort_keys=True, separators=(',', ':')).encode()
    return {'file_count': len(verified_mapping),
            'aggregate_sha256': hashlib.sha256(canonical).hexdigest()}


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def compare(actual, expected, label):
    """Compare all aggregate fields, including unsupported columns and nulls."""
    if isinstance(actual, dict):
        require(isinstance(expected, dict) and actual.keys() == expected.keys(), label + ': aggregate keys mismatch')
        for key, value in actual.items():
            compare(value, expected[key], label)
    elif isinstance(actual, list):
        require(isinstance(expected, list) and len(actual) == len(expected), label + ': aggregate length mismatch')
        for a, b in zip(actual, expected):
            compare(a, b, label)
    elif isinstance(actual, float):
        require(isinstance(expected, (float, int)) and not isinstance(expected, bool)
                and math.isfinite(expected) and math.isclose(actual, expected, rel_tol=0, abs_tol=1e-12),
                label + ': aggregate number mismatch')
    else:
        require(type(actual) is type(expected) and actual == expected, label + ': aggregate value mismatch')


def sample_indices(n_frames, fps):
    require(isinstance(n_frames, (int, np.integer)) and n_frames > 0
            and math.isfinite(fps) and fps > 0, 'invalid source timing')
    count = math.floor((n_frames - 1) * 10 / fps + 1e-9) + 1
    return np.minimum(np.rint(np.arange(count, dtype=float) * fps / 10).astype(np.int64), n_frames - 1)


def window_centres(n_samples):
    require(n_samples > 0, 'empty sample sequence')
    centres = list(range(2, n_samples, 5))
    if not centres or centres[-1] < n_samples - 3:
        centres.append(n_samples - 1)
    return np.asarray(centres, dtype=np.int64)


def decode(logits, centres, n_samples, params):
    """Independent softmax, nearest-centre assignment and frozen JSD decoder.

    Stable score ties keep the earlier sample; equal nearest-centre distances
    use the earlier centre. B3 requires different left/right dominant labels.
    """
    x = np.asarray(logits, dtype=np.float64)
    c = np.asarray(centres, dtype=np.int64)
    require(x.ndim == 2 and x.shape == (len(c), 5) and np.isfinite(x).all(), 'finite logits/shape required')
    exponent = np.exp(x - np.max(x, axis=1)[:, None])
    posterior = exponent / np.sum(exponent, axis=1)[:, None]
    # searchsorted makes the nearest-neighbour rule independent of the runner's
    # full distance-matrix implementation, including its first-index tie rule.
    right = np.searchsorted(c, np.arange(n_samples), side='left').clip(0, len(c)-1)
    left = np.maximum(right - 1, 0)
    nearest = np.where(np.abs(np.arange(n_samples)-c[left]) <= np.abs(c[right]-np.arange(n_samples)), left, right)
    p = posterior[nearest]
    argmax = np.argmax(p, axis=1)
    h = int(params['half_window'])
    threshold = float(params['threshold'])
    radius = int(params['nms_radius'])
    require(h >= 1 and radius >= 0 and math.isfinite(threshold), 'invalid frozen B3 parameters')
    prefix = np.vstack([np.zeros(5), np.cumsum(p, axis=0)])
    scores = np.zeros(n_samples, dtype=float)
    for cut in range(1, n_samples):
        lo, hi = max(0, cut-h), min(n_samples, cut+h)
        l = (prefix[cut]-prefix[lo]) / (cut-lo)
        r = (prefix[hi]-prefix[cut]) / (hi-cut)
        middle = (l+r)/2
        scores[cut] = .5 * sum(l * np.log((l+1e-12)/(middle+1e-12))) + .5 * sum(r * np.log((r+1e-12)/(middle+1e-12)))
    candidates = sorted(np.flatnonzero(scores >= threshold), key=lambda i: (-scores[i], i))
    selected = []
    for cut in candidates:
        if all(abs(int(cut)-previous) > radius for previous in selected):
            selected.append(int(cut))
    boundaries = []
    for cut in sorted(selected):
        # A zero threshold may select sample zero, but the empty left side
        # cannot establish a label change and is therefore not a boundary.
        if cut > 0:
            l = p[max(0, cut-h):cut].mean(axis=0).argmax()
            r = p[cut:min(n_samples, cut+h)].mean(axis=0).argmax()
            if l != r:
                boundaries.append(cut)
    b3 = np.empty(n_samples, dtype=np.int64)
    points = [0, *boundaries, n_samples]
    for start, end in itertools.pairwise(points):
        b3[start:end] = p[start:end].mean(axis=0).argmax()
    return argmax, b3


def majority(labels):
    lab = np.asarray(labels)
    if lab.ndim != 1 or not len(lab) or not np.issubdtype(lab.dtype, np.integer) or np.any((lab < -1) | (lab > 4)):
        raise ValueError('nonempty integer labels from -1 through 4 required')
    counts = [int(np.count_nonzero(lab == value)) for value in (0, 1, 2, 3, 4, -1)]
    winner = counts.index(max(counts))
    return -1 if winner == 5 else winner


def metrics(rows):
    require(bool(rows), 'empty evaluation')
    matrix = np.zeros((5, 6), dtype=np.int64)
    cuts, samples = 0, 0
    for row in rows:
        y, pred = row['true'], row['pred']
        require(y in range(5) and pred in range(-1, 5), 'unsupported reference or invalid prediction')
        matrix[y, 5 if pred == -1 else pred] += 1
        cuts += int(row['cuts'])
        samples += int(row['n_samples'])
    # Fractional source durations need an accurate reduction. A naive += loop
    # loses ulps over hundreds of clips and differs from Python 3.12's improved
    # sum. fsum is an independent compensated calculation, not a looser check.
    duration = math.fsum(float(row['duration_s']) for row in rows)
    support = matrix.sum(axis=1)
    correct = np.diag(matrix[:, :5])
    precision_recall_denominator = support + matrix[:, :5].sum(axis=0)
    f1 = np.divide(2*correct, precision_recall_denominator, out=np.zeros(5), where=precision_recall_denominator != 0)
    return {'n_clips': len(rows), 'accuracy': float(correct.sum()/len(rows)),
                'class_support': dict(zip(CLASSES, support.tolist())),
                'class_recall': {name: float(correct[i]/support[i]) if support[i] else None for i, name in enumerate(CLASSES)},
                'five_class_macro_f1': float(np.mean(f1)) if np.all(support > 0) else None,
                'confusion': matrix.tolist(), 'confusion_columns': [*CLASSES, 'unsupported'],
                'internal_cuts': cuts, 'internal_cuts_per_clip': cuts/len(rows), 'duration_s': duration,
                'internal_cuts_per_minute': cuts*60/duration if duration else None, 'n_samples': samples}


def stfm_probability(logits, prior_once):
    x = np.asarray(logits, dtype=float)
    require(x.shape == (10, 9) and np.isfinite(x).all(), 'STFM finite logits/shape mismatch')
    # The authors' Softplus threshold is 20, not an unconstrained exp(x).
    evidence = np.empty_like(x)
    large = x > 20
    evidence[large] = x[large]
    evidence[~large] = np.log1p(np.exp(-np.abs(x[~large]))) + np.maximum(x[~large], 0)
    posterior = evidence.sum(axis=0) + (.8 if prior_once else 8.)
    return posterior/posterior.sum()


def group_stfm(probability):
    probability = np.asarray(probability, dtype=float)
    require(probability.shape == (9,) and np.isfinite(probability).all(), 'invalid STFM probability vector')
    out = np.zeros(6)
    for raw_index, family in enumerate(RAW_TO_FIVE):
        out[5 if family is None else family] += probability[raw_index]
    return out


def audit_case(path, q, protocol, key):
    with np.load(path, allow_pickle=False) as cache:
        expected_keys = {'key', 'id', 'receipt', 'centres'} | {
            f'{field}_s{s}' for s in SEEDS for field in ('logits', 'argmax', 'frozen_B3')}
        require(set(cache.files) == expected_keys, 'prediction cache fields mismatch')
        require(str(cache['key']) == key, 'prediction cache key mismatch')
        require(str(cache['id']) == q['id'], 'prediction cache identity mismatch')
        receipt = json.loads(str(cache['receipt']))
        n, fps = receipt['n_frames'], receipt['fps']
        require(type(n) is int and n > 0 and math.isfinite(fps) and fps > 0, 'invalid source timing receipt')
        require(q['n_frames'] is None or n == q['n_frames'], 'source frame count receipt mismatch')
        require(float(fps) == float(q['fps']), 'source FPS receipt mismatch')
        require(math.isclose(receipt['duration_s'], n/fps, abs_tol=1e-12, rel_tol=0), 'source duration receipt mismatch')
        indices = sample_indices(n, fps)
        require(receipt['sample_frame_indices'] == indices.tolist(), 'source sampling receipt mismatch')
        require(bool(re.fullmatch('[0-9a-f]{64}', receipt['input_pixels_sha256'])), 'pixel receipt hash malformed')
        centres = window_centres(len(indices))
        require(np.issubdtype(cache['centres'].dtype, np.integer) and np.array_equal(cache['centres'], centres),
                'window centres mismatch')
        rows = {}
        for seed in SEEDS:
            lg = cache[f'logits_s{seed}']
            require(lg.shape == (len(centres), 5) and np.issubdtype(lg.dtype, np.floating)
                    and np.isfinite(lg).all(), 'prediction finite logits/shape mismatch')
            raw, b3 = decode(lg, centres, len(indices), protocol['weights'][str(seed)]['b3'])
            for name, rebuilt in [('argmax', raw), ('frozen_B3', b3)]:
                field = f'{name}_s{seed}'
                stored = cache[field]
                require(np.issubdtype(stored.dtype, np.integer) and np.array_equal(stored, rebuilt),
                        'complete decoded labels mismatch')
                rows[field] = {'true': int(q['true']), 'pred': majority(rebuilt), 'n_samples': len(rebuilt),
                                   'cuts': int(np.count_nonzero(np.diff(rebuilt))), 'duration_s': float(receipt['duration_s'])}
        return rows, receipt


def ci95(values):
    return np.quantile(np.asarray(values, dtype=float), [.025, .975]).tolist()


def grouped_bootstrap(truth, groups, predictions, cuts=None, draws=BOOTSTRAP_DRAWS,
                      seed=BOOTSTRAP_SEED, classes=CLASSES):
    """Case-weighted metrics after sampling whole groups with replacement.

    One multiplicity matrix is shared by every model, every seed and paired
    contrast. Seed means average the three independently evaluated seed scores
    within each draw, keeping the original case count (never 3*n).
    """
    truth = np.asarray(truth, dtype=np.int64)
    groups = np.asarray(groups, dtype=str)
    require(len(truth) == len(groups) and len(truth) > 0, 'bootstrap membership mismatch')
    _, group_index = np.unique(groups, return_inverse=True)
    ngroups = int(group_index.max()+1)
    rng = np.random.default_rng(seed)
    multiplicity = rng.multinomial(ngroups, np.full(ngroups, 1/ngroups), size=draws)
    k = len(classes)
    full_support = bool(np.all(np.bincount(truth, minlength=k) > 0))
    cuts = {} if cuts is None else cuts
    result = {'groups': ngroups, 'n_cases': len(truth), 'models': {}}
    sampled = {}
    model_recalls = {}
    for name, predicted in predictions.items():
        predicted = np.asarray(predicted, dtype=np.int64)
        require(predicted.shape == truth.shape and np.all((predicted >= -1) & (predicted < k)),
                'bootstrap prediction mismatch')
        grouped = np.zeros((ngroups, k, k+1), dtype=np.int64)
        np.add.at(grouped, (group_index, truth, np.where(predicted == -1, k, predicted)), 1)
        resampled_cm = (multiplicity @ grouped.reshape(ngroups, -1)).reshape(draws, k, k+1)
        support = resampled_cm.sum(axis=2)
        tp = resampled_cm[:, np.arange(k), np.arange(k)]
        n = support.sum(axis=1)
        accuracy = tp.sum(axis=1)/n
        denominator = support + resampled_cm[:, :, :k].sum(axis=1)
        f1 = np.divide(2*tp, denominator, out=np.zeros((draws, k)), where=denominator > 0)
        macro = f1.mean(axis=1) if full_support else None
        source_cm = grouped.sum(axis=0)
        source_support = source_cm.sum(axis=1)
        source_tp = source_cm[np.arange(k), np.arange(k)]
        source_denominator = source_support + source_cm[:, :k].sum(axis=0)
        source_f1 = np.divide(2*source_tp, source_denominator, out=np.zeros(k), where=source_denominator > 0)
        cut_values = np.asarray(cuts.get(name, np.zeros(len(truth))), dtype=float)
        require(cut_values.shape == truth.shape and np.isfinite(cut_values).all(), 'bootstrap cut counts mismatch')
        group_cuts = np.bincount(group_index, weights=cut_values, minlength=ngroups)
        cut_draws = (multiplicity @ group_cuts)/n
        recall = np.divide(tp, support, out=np.full((draws, k), np.nan), where=support > 0)
        model_recalls[name] = recall
        model = {'correct': int(np.count_nonzero(truth == predicted)), 'accuracy': float(np.mean(truth == predicted)),
                     'accuracy_ci95': ci95(accuracy), 'macro_f1': float(source_f1.mean()) if full_support else None,
                     'macro_f1_ci95': ci95(macro) if full_support else None,
                     'class_recall': {c: float(source_tp[j]/source_support[j]) if source_support[j] else None for j, c in enumerate(classes)},
                     'cuts_per_clip': float(cut_values.mean()), 'cuts_per_clip_ci95': ci95(cut_draws)}
        result['models'][name] = model
        sampled[name] = {'accuracy': accuracy, 'macro_f1': macro, 'cuts_per_clip': cut_draws}
    b3 = [f'frozen_B3_s{s}' for s in SEEDS]
    if all(name in predictions for name in b3):
        accuracy = np.mean([sampled[name]['accuracy'] for name in b3], axis=0)
        macro = np.mean([sampled[name]['macro_f1'] for name in b3], axis=0) if full_support else None
        cut_draws = np.mean([sampled[name]['cuts_per_clip'] for name in b3], axis=0)
        recall_mean = np.mean([model_recalls[name] for name in b3], axis=0)
        model = {'accuracy': float(np.mean([result['models'][name]['accuracy'] for name in b3])),
                     'ci95': ci95(accuracy), 'macro_f1': float(np.mean([result['models'][name]['macro_f1'] for name in b3])) if full_support else None,
                     'macro_ci95': ci95(macro) if full_support else None,
                     'class_recall': {c: float(np.mean([result['models'][name]['class_recall'][c] for name in b3]))
                                   if result['models'][b3[0]]['class_recall'][c] is not None else None for c in classes},
                     'class_recall_ci95': {c: ci95(recall_mean[np.isfinite(recall_mean[:, j]), j])
                                         if np.isfinite(recall_mean[:, j]).any() else None for j, c in enumerate(classes)},
                     'cuts_per_clip': float(np.mean([result['models'][name]['cuts_per_clip'] for name in b3])),
                     'cuts_per_clip_ci95': ci95(cut_draws)}
        result['mean_B3'] = model
        sampled['mean_B3'] = {'accuracy': accuracy, 'macro_f1': macro, 'cuts_per_clip': cut_draws}
    return result, sampled


def csv_rows(path):
    with Path(path).open(newline='') as handle:
        return list(csv.DictReader(handle))


def validate_timing_revision(old_protocol, new_protocol, old_manifest, new_manifest, qualification):
    """Permit the prospectively frozen, PTS-qualified LVH timing repair only."""
    mutable_protocol = {'created_utc', 'manifest_sha256', 'code_sha256', 'timing_amendment'}
    compare({k: v for k, v in old_protocol.items() if k not in mutable_protocol},
            {k: v for k, v in new_protocol.items() if k not in mutable_protocol}, 'unchanged model/protocol')
    old_code = old_protocol['code_sha256']
    new_code = new_protocol['code_sha256']
    require(set(new_code) == set(old_code) | {'scripts/amend_external_lvh_timing.py'},
            'unchanged inference code set mismatch')
    compare(old_code, {k: v for k, v in new_code.items() if k in old_code}, 'unchanged inference code')
    compare({k: v for k, v in old_manifest.items() if k != 'datasets'},
            {k: v for k, v in new_manifest.items() if k not in {'datasets', 'timing_amendment'}},
            'unchanged source metadata')
    require(set(old_manifest['datasets']) == set(new_manifest['datasets']), 'unchanged dataset roster mismatch')
    for name, rows in old_manifest['datasets'].items():
        if name != 'lvh':
            compare(rows, new_manifest['datasets'][name], 'unchanged source membership')
    old_rows, new_rows = old_manifest['datasets']['lvh'], new_manifest['datasets']['lvh']
    qualified = qualification['rows']
    require(len(qualified) == len(old_rows) == len(new_rows), 'LVH qualification cardinality mismatch')
    at = {q['id']: q for q in qualified}
    require(len(at) == len(qualified) and set(at) == {q['id'] for q in old_rows},
            'LVH qualification exact membership mismatch')
    errors = []
    for old, new in zip(old_rows, new_rows):
        compare({k: v for k, v in old.items() if k != 'fps'},
                {k: v for k, v in new.items() if k not in {'fps', 'reported_csv_fps', 'fps_authority'}},
                'unchanged LVH fields')
        source = at[old['id']]
        require(set(source) == {'id', 'container_fps', 'csv_field_fps', 'decoded_frames', 'pts_delta_max_error'},
                'LVH qualification fields mismatch')
        require(source['csv_field_fps'] == old['fps'] == new['reported_csv_fps']
                and source['container_fps'] == new['fps'] and math.isfinite(new['fps']) and new['fps'] > 0
                and new['fps_authority'] == 'decoded_PTS_and_container_CFR', 'LVH FPS authority mismatch')
        require(source['decoded_frames'] == old['n_frames'] == new['n_frames'], 'LVH decoded-frame qualification mismatch')
        error = source['pts_delta_max_error']
        require(math.isfinite(error) and 0 <= error <= 1e-6, 'LVH PTS CFR qualification mismatch')
        errors.append(error)
    require(qualification['max_error'] == max(errors), 'LVH PTS maximum recount mismatch')
    require(math.isfinite(qualification['seconds']) and qualification['seconds'] > 0, 'LVH qualification resource receipt mismatch')
    return {'qualified_recordings': len(qualified), 'max_pts_delta_error_s': max(errors),
            'model_weights_B3_preprocess_windows_membership_unchanged': True}


def validate_memberships(root, datasets, baseline, source_files):
    """Independently reconstruct official test membership from pinned metadata."""
    expected_metadata = {'data/manifests/ev9v_native.csv', 'data/echonet/dynamic/EchoNet-Dynamic/FileList.csv',
                         'data/echonet/lvh/MeasurementsList.csv', 'data/camus/CAMUS_public/database_split/subgroup_testing.txt'}
    require(expected_metadata.issubset(source_files), 'source metadata set incomplete')
    ev = [q for q in csv_rows(root/'data/manifests/ev9v_native.csv') if q['split'] == 'test']
    require(len(ev) == 888 and len(baseline) == 888, 'original nine-code cardinality mismatch')
    base_ids = [q['id'] for q in baseline]
    require(base_ids == [q['video_id'] for q in ev] and len(set(base_ids)) == 888, 'baseline source membership mismatch')
    shared = []
    for source, q in zip(ev, baseline):
        raw = int(source['raw9_index'])
        require(0 <= raw < 9 and q['true9'] == raw, 'raw-nine truth mismatch')
        family = RAW_TO_FIVE[raw]
        declared = int(float(source['family5_index'])) if source['family5_index'] else None
        require(family == declared == q['true5'], 'five-family source mapping mismatch')
        require(q['n_frames'] == int(source['n_frames']) and q['path'] == source['frames_dir'], 'baseline source metadata mismatch')
        if family is not None:
            shared.append((source['video_id'], family, int(source['n_frames']), source['frames_dir']))
    ev_rows = datasets['ev9v']
    require([(q['id'], q['true'], q['n_frames'], q['path']) for q in ev_rows] == shared,
            'shared native source membership mismatch')
    require(all(q['fps'] == 30 and q['kind'] == 'jpeg' and q['group'] == q['id'].rsplit('_', 1)[0] for q in ev_rows),
            'EV9V timing/group definition mismatch')
    require(len({q['group'] for q in ev_rows}) == 125, 'EV9V session cardinality mismatch')
    dynamic = [q for q in csv_rows(root/'data/echonet/dynamic/EchoNet-Dynamic/FileList.csv') if q['Split'] == 'TEST']
    dy = datasets['dynamic']
    require([q['id'] for q in dy] == [q['FileName'] for q in dynamic], 'Dynamic official membership mismatch')
    for q, source in zip(dy, dynamic):
        require(q['group'] == q['id'] and q['kind'] == 'video' and q['true'] == 2
                and q['fps'] == float(source['FPS']) and q['n_frames'] == int(source['NumberOfFrames']),
                'Dynamic official timing/labels mismatch')
    all_lvh = csv_rows(root/'data/echonet/lvh/MeasurementsList.csv')
    split_sets = {}
    test_lvh = {}
    for source in all_lvh:
        split_sets.setdefault(source['HashedFileName'], set()).add(source['split'])
        if source['split'] == 'test':
            test_lvh.setdefault(source['HashedFileName'], source)
    require(all(len(split) == 1 for split in split_sets.values()), 'LVH recording crosses official splits')
    lvh = datasets['lvh']
    require([q['id'] for q in lvh] == list(test_lvh), 'LVH official membership mismatch')
    for q in lvh:
        source = test_lvh[q['id']]
        require(q['group'] == q['id'] and q['kind'] == 'video' and q['true'] == 0
                and q['reported_csv_fps'] == float(source['FPS']) and q['n_frames'] == int(source['Frames'])
                and q['fps_authority'] == 'decoded_PTS_and_container_CFR',
                'LVH official timing/labels mismatch')
    ids = [q.strip() for q in (root/'data/camus/CAMUS_public/database_split/subgroup_testing.txt').read_text().splitlines() if q.strip()]
    require([q['id'] for q in datasets['camus']] == ids, 'CAMUS official membership mismatch')
    for q in datasets['camus']:
        require(q['group'] == q['id'] and q['kind'] == 'nifti' and q['true'] == 2 and q['n_frames'] is None,
                'CAMUS official label/group mismatch')
        media = Path(q['path'])
        info = media.parent/(q['id']+'_Info_4CH.cfg')
        if not info.exists():
            info = media.parent/'Info_4CH.cfg'
        require(str(info.relative_to(root)) in source_files, 'CAMUS frame-rate source not pinned')
        fields = dict(line.split(':', 1) for line in info.read_text().splitlines() if ':' in line)
        rate = {key.strip(): value.strip() for key, value in fields.items()}['FrameRate']
        require(float(rate) == q['fps'], 'CAMUS must use Info FrameRate, not NIfTI z spacing')
    require(len(source_files) == 54, 'unexpected source metadata cardinality')


def audit_media(q):
    """Read container/header metadata only; never perform image preprocessing."""
    path = Path(q['path'])
    require(path.exists(), 'source media missing')
    if path.is_file():
        require(path.stat().st_size == q['source_size'] and path.stat().st_mtime_ns == q['source_mtime_ns'],
                'source media stat identity mismatch')
    if q['kind'] == 'jpeg':
        n = sum(p.suffix.lower() in {'.jpg', '.jpeg', '.png'} for p in path.iterdir())
        require(n == q['n_frames'], 'source JPEG frame count changed')
    elif q['kind'] == 'nifti':
        with gzip.open(path, 'rb') as handle:
            header = handle.read(348)
        require(len(header) == 348, 'CAMUS NIfTI header incomplete')
        endian = '<' if struct.unpack('<i', header[:4])[0] == 348 else '>'
        require(struct.unpack(endian+'i', header[:4])[0] == 348, 'CAMUS NIfTI-1 header mismatch')
        dims = struct.unpack(endian+'8h', header[40:56])
        require(dims[0] == 3 and all(v > 0 for v in dims[1:4]), 'CAMUS x/y/time shape mismatch')
        n = dims[3]
    else:
        import cv2
        capture = cv2.VideoCapture(str(path))
        try:
            require(capture.isOpened(), 'source video container unavailable')
            n = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = capture.get(cv2.CAP_PROP_FPS)
            require(n == q['n_frames'] and math.isfinite(fps)
                    and abs(fps-q['fps']) <= max(.02, .001*q['fps']), 'source video container timing mismatch')
        finally:
            capture.release()
    return n


def audit_baseline(private, protocol, shared_ids):
    manifest = read_json(private/'baseline_manifest.json')
    key = sha(private.parent/'baseline_protocol.json')
    predictions = {condition: {} for condition in CONDITIONS}
    raw_predictions = {condition: [] for condition in CONDITIONS}
    summaries = {}
    digest = hashlib.sha256()
    expected_files = {f'{i:05d}.npz' for i in range(888)}
    require({p.name for p in (private/'baseline_predictions').iterdir()} == expected_files, 'baseline cache cardinality/extras mismatch')
    for index, q in enumerate(manifest):
        path = private/'baseline_predictions'/f'{index:05d}.npz'
        digest.update(bytes.fromhex(sha(path)))
        with np.load(path, allow_pickle=False) as cache:
            require(set(cache.files) == {'key', 'id', 'logits'}, 'baseline cache fields mismatch')
            require(str(cache['key']) == key and str(cache['id']) == q['id'], 'baseline cache identity mismatch')
            for condition in CONDITIONS:
                probability = stfm_probability(cache['logits'], condition.startswith('corrected'))
                raw_predictions[condition].append(int(probability.argmax()))
                if q['true5'] is not None:
                    pred = int(group_stfm(probability).argmax())
                    predictions[condition][q['id']] = {'true': q['true5'], 'pred': -1 if pred == 5 else pred,
                                                         'n_samples': 1, 'cuts': 0, 'duration_s': q['n_frames']/30}
    truth9 = np.asarray([q['true9'] for q in manifest])
    for condition in CONDITIONS:
        cm = np.zeros((9, 9), dtype=np.int64)
        np.add.at(cm, (truth9, raw_predictions[condition]), 1)
        tp = cm.diagonal()
        denominator = cm.sum(axis=0)+cm.sum(axis=1)
        f1 = np.divide(2*tp, denominator, out=np.zeros(9), where=denominator > 0)
        require(set(predictions[condition]) == set(shared_ids), 'baseline shared membership mismatch')
        summaries[condition] = {
            'original_nine_code': {'n_clips': 888, 'accuracy': float(tp.sum()/888),
                                   'macro_f1': float(f1.mean()), 'confusion': cm.tolist()},
            'shared_five_family': metrics([predictions[condition][ident] for ident in shared_ids]),
            'cut_note': 'Whole-file classification emits no internal cut by construction; not a stability finding.'}
    return manifest, predictions, raw_predictions, summaries, digest.hexdigest()


def verify(root):
    start = time.monotonic()
    root = Path(root).resolve()
    result = root/RESULT_REL
    run = root/RUN_REL
    private = run/'private'
    hashes = {name: require_hash(result/name, value, name) for name, value in ANCHORS.items()}
    hashes['historical_A26'] = require_hash(root/HISTORY_REL, HISTORY_SHA, 'historical A26')
    hashes['choices'] = require_hash(root/CHOICES_REL, CHOICES_SHA, 'frozen B3 choices')
    protocol = read_json(run/'protocol.json')
    protocol_v1 = read_json(run/'protocol_v1.json')
    baseline_protocol = read_json(run/'baseline_protocol.json')
    compare(protocol_v1, read_json(result/'preparation.json'), 'v1 protocol mirror')
    compare(protocol, read_json(result/'preparation_v2.json'), 'v2 protocol mirror')
    compare(baseline_protocol, read_json(result/'baseline_preparation.json'), 'baseline protocol mirror')
    hashes['protocol'] = require_hash(run/'protocol.json', ANCHORS['preparation_v2.json'], 'native v2 protocol')
    hashes['protocol_v1'] = require_hash(run/'protocol_v1.json', ANCHORS['preparation.json'], 'native v1 protocol')
    hashes['protocol_bindings'] = require_hash(run/'protocol_bindings.json', BINDING_SHA, 'dataset protocol bindings')
    compare(read_json(run/'protocol_bindings.json'), BOUND_PROTOCOLS, 'dataset protocol bindings')
    bound_protocols = {'protocol.json': protocol, 'protocol_v1.json': protocol_v1}
    for filename in ('checks_v1.json', 'checks.json'):
        development = read_json(result/filename)
        require(development['protocol_sha256'] == sha(run/('protocol_v1.json' if filename == 'checks_v1.json' else 'protocol.json')),
                'development parity protocol mismatch')
        require(set(development['development_parity']) == {'0', '1', '2'}, 'development parity seed roster mismatch')
        for values in development['development_parity'].values():
            require(values['windows'] == 16 and values['argmax_equal']
                    and math.isfinite(values['max_logit_abs']) and values['max_logit_abs'] <= 1e-4,
                    'frozen development parity failed')
    hashes['baseline_protocol'] = require_hash(run/'baseline_protocol.json', ANCHORS['baseline_preparation.json'], 'baseline protocol')
    hashes['config'] = require_hash(root/'configs/experiments/external_original_test_v1.yaml', protocol['config_sha256'], 'frozen config')
    hashes['manifest'] = require_hash(private/'manifest.json', protocol['manifest_sha256'], 'native manifest')
    hashes['manifest_v1'] = require_hash(private/'manifest_v1.json', protocol_v1['manifest_sha256'], 'native v1 manifest')
    hashes['baseline_manifest'] = require_hash(private/'baseline_manifest.json', baseline_protocol['manifest_sha256'], 'baseline manifest')
    hashes['native_code'] = {name: require_hash(root/name, value, 'native frozen code') for name, value in protocol['code_sha256'].items()}
    hashes['baseline_code'] = require_hash(root/'scripts/external_native_baselines.py', baseline_protocol['code_sha256'], 'baseline code')
    hashes['baseline_mapping'] = require_hash(root/'echo_routing/evaluate/external_baselines.py', baseline_protocol['mapping_sha256'], 'baseline mapping')
    hashes['baseline_weights'] = require_hash(root/baseline_protocol['weights'], baseline_protocol['weights_sha256'], 'baseline weights')
    require(baseline_protocol['conditions'] == list(CONDITIONS), 'baseline conditions changed')
    choices = read_json(root/CHOICES_REL)['choices']
    hashes['native_weights'] = {}
    require(set(protocol['weights']) == {'0', '1', '2'}, 'three fixed model seeds required')
    for seed in SEEDS:
        item = protocol['weights'][str(seed)]
        hashes['native_weights'][str(seed)] = require_hash(root/item['path'], item['sha256'], 'A08 frozen weights')
        compare(item['b3'], choices[f'all_s{seed}']['params'], 'B3 frozen choices')
    manifest = read_json(private/'manifest.json')
    manifest_v1 = read_json(private/'manifest_v1.json')
    qualification_file = private/'lvh_timing_qualification.json'
    hashes['LVH_timing_qualification'] = require_hash(qualification_file, QUALIFICATION_SHA, 'LVH source timing qualification')
    timing_checks = validate_timing_revision(protocol_v1, protocol, manifest_v1, manifest, read_json(qualification_file))
    amendment = read_json(result/'timing_amendment.json')
    require(amendment['records'] == amendment['frames_match'] == timing_checks['qualified_recordings'] == 343
            and amendment['all_CFR'] and amendment['max_pts_delta_error_s'] == timing_checks['max_pts_delta_error_s']
            and amendment['predecessor_protocol_sha256'] == hashes['protocol_v1']
            and amendment['active_protocol_sha256'] == hashes['protocol']
            and amendment['qualification_sha256'] == hashes['LVH_timing_qualification']
            and amendment['binding_file_sha256'] == hashes['protocol_bindings'], 'prospective timing amendment receipt mismatch')
    require(protocol['timing_amendment']['all_before_first_LVH_forward']
            and protocol['timing_amendment']['qualification_sha256'] == hashes['LVH_timing_qualification']
            and manifest['timing_amendment']['qualification_sha256'] == hashes['LVH_timing_qualification'],
            'prospective timing qualification binding mismatch')
    amendment_time = datetime.datetime.fromisoformat(amendment['created_utc']).timestamp()
    require(qualification_file.stat().st_mtime <= amendment_time + 1,
            'LVH qualification timestamp follows prospective amendment')
    datasets = {name: (manifest_v1 if filename == 'protocol_v1.json' else manifest)['datasets'][name]
                for name, filename in BOUND_PROTOCOLS.items()}
    require(set(datasets) == set(EXPECTED), 'dataset roster mismatch')
    counts = {name: len(rows) for name, rows in datasets.items()}
    require(counts == EXPECTED == protocol['counts'], 'fixed dataset cardinalities mismatch')
    for rows in datasets.values():
        require(len({q['id'] for q in rows}) == len(rows), 'duplicate native recording')
    hashes['source_metadata'] = {name: require_hash(root/name, value, 'source metadata') for name, value in manifest['source_files'].items()}
    baseline_manifest = read_json(private/'baseline_manifest.json')
    validate_memberships(root, datasets, baseline_manifest, manifest['source_files'])
    source_audit = read_json(result/'input_audit.json')
    require(source_audit['camus']['all_checks_passed'] == 50 and source_audit['camus']['arrays_checked'] == 50
            and source_audit['camus']['uint8_cast_lossless'] and source_audit['camus']['axes_counts'] == {'L,P,S': 50},
            'independent CAMUS whole-source audit did not pass')
    summary, rows_by_dataset, native_digest = {}, {}, hashlib.sha256()
    n_cached_samples = 0
    for dataset, records in datasets.items():
        rows = {f'{decoder}_s{s}': [] for decoder in ('argmax', 'frozen_B3') for s in SEEDS}
        folder = private/'predictions'/dataset
        require({p.name for p in folder.iterdir()} == {f'{i:05d}.npz' for i in range(len(records))},
                'native cache cardinality/extras mismatch')
        for index, q in enumerate(records):
            n = audit_media(q)
            path = folder/f'{index:05d}.npz'
            if dataset in {'lvh', 'camus'}:
                require(path.stat().st_mtime >= amendment_time, 'new-target cache precedes timing amendment')
            native_digest.update(bytes.fromhex(sha(path)))
            selected_protocol = bound_protocols[BOUND_PROTOCOLS[dataset]]
            case, receipt = audit_case(path, q, selected_protocol, sha(run/BOUND_PROTOCOLS[dataset]))
            require(n == receipt['n_frames'], 'container/header versus receipt frame count mismatch')
            n_cached_samples += len(receipt['sample_frame_indices'])
            for name in rows:
                rows[name].append(case[name])
        summary[dataset] = {name: metrics(values) for name, values in rows.items()}
        rows_by_dataset[dataset] = rows
    require(set(read_json(result/'native_results.json')) == set(EXPECTED), 'native aggregate dataset roster mismatch')
    compare(summary, read_json(result/'native_results.json'), 'native aggregate')
    ids = [q['id'] for q in datasets['ev9v']]
    baseline_manifest, baseline_predictions, baseline_raw, baseline_summary, baseline_digest = audit_baseline(private, baseline_protocol, ids)
    compare(baseline_summary, read_json(result/'stfm_native_results.json'), 'STFM aggregate')
    history = read_json(root/HISTORY_REL)['values']
    historical_checks = {}
    for seed in (0, 2):
        native = summary['ev9v'][f'frozen_B3_s{seed}']
        old = history[f'A08_s{seed}']
        # Exact binary float equality is intentional for this same-case recount.
        require(native['accuracy'] == old['clip accuracy']['value']
                and native['five_class_macro_f1'] == old['clip macro-F1']['value'], 'historical A26 exact native recount mismatch')
        historical_checks[f's{seed}'] = {'accuracy': native['accuracy'], 'macro_f1': native['five_class_macro_f1'], 'exact': True}
    uncertainty = {'method': {'bootstrap_draws': BOOTSTRAP_DRAWS, 'bootstrap_rng_seed': BOOTSTRAP_SEED,
                              'interval': 'percentile 2.5%,97.5%; whole-group sampling with replacement',
                              'case_weighting': 'recordings within sampled groups retain their multiplicity',
                              'seed_mean': 'same groups/cases resampled for all three seeds; arithmetic mean of their metrics; no 3n pooling'},
                   'datasets': {}, 'baseline': {}, 'paired_original': {}}
    for dataset, records in datasets.items():
        rows = rows_by_dataset[dataset]
        truth = [q['true'] for q in records]
        groups = [q['group'] for q in records]
        predictions = {name: [q['pred'] for q in values] for name, values in rows.items()}
        cuts = {name: [q['cuts'] for q in values] for name, values in rows.items()}
        if dataset == 'ev9v':
            for condition in CONDITIONS:
                predictions[condition] = [baseline_predictions[condition][ident]['pred'] for ident in ids]
        boot, draws = grouped_bootstrap(truth, groups, predictions, cuts)
        boot['grouping'] = {'ev9v': 'session; patient identity unknown', 'dynamic': 'recording; patient identity unknown',
                            'lvh': 'recording; patient identity unknown', 'camus': 'patient'}[dataset]
        if dataset == 'ev9v':
            require(boot['groups'] == 125, 'bootstrap EV9V session denominator mismatch')
            for condition in CONDITIONS:
                uncertainty['baseline'][condition] = boot['models'].pop(condition)
                uncertainty['baseline'][condition].update(groups=125, grouping='session', n_cases=800)
                uncertainty['paired_original'][condition] = {}
                for name in [f'{decoder}_s{s}' for decoder in ('frozen_B3', 'argmax') for s in SEEDS]+['mean_B3']:
                    model = boot['mean_B3'] if name == 'mean_B3' else boot['models'][name]
                    base = uncertainty['baseline'][condition]
                    accuracy = model['accuracy'] - base['accuracy']
                    macro = model['macro_f1'] - base['macro_f1']
                    uncertainty['paired_original'][condition][name] = {
                        'n_cases': 800, 'groups': 125, 'grouping': 'paired sessions; identical case membership',
                        'accuracy_difference': accuracy,
                        'accuracy_difference_ci95': ci95(draws[name]['accuracy']-draws[condition]['accuracy']),
                        'macro_f1_difference': macro,
                        'macro_f1_difference_ci95': ci95(draws[name]['macro_f1']-draws[condition]['macro_f1'])}
        uncertainty['datasets'][dataset] = boot
    raw_boot, _ = grouped_bootstrap([q['true9'] for q in baseline_manifest],
                                    [q['id'].rsplit('_', 1)[0] for q in baseline_manifest], baseline_raw, classes=RAW9)
    for condition in CONDITIONS:
        uncertainty['baseline'][condition]['original_nine_code'] = {
            **raw_boot['models'][condition], 'groups': raw_boot['groups'], 'n_cases': 888, 'grouping': 'session'}
    hashes['native_prediction_aggregate_sha256'] = native_digest.hexdigest()
    hashes['baseline_prediction_aggregate_sha256'] = baseline_digest
    hashes['native_results'] = sha(result/'native_results.json')
    hashes['stfm_native_results'] = sha(result/'stfm_native_results.json')
    hashes['audit_script'] = sha(Path(__file__))
    # Source metadata includes CAMUS patient paths. Individual hashes have
    # already passed above; expose only a count and a digest of that mapping.
    # The complete mapping remains in the existing private source manifest.
    hashes['source_metadata'] = public_source_hash_summary(hashes['source_metadata'])
    uncertainty['limits'] = LIMITS
    verification = {
        'passed': True, 'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'checks': {'frozen_protocol_config_code_weights': True, 'source_metadata_hashes': True,
                   'official_membership_and_complete_truth': True, 'exact_cache_counts_keys_identities': True,
                   'finite_logits_shapes_centres_sampling_timing': True, 'complete_independently_decoded_labels': True,
                   'independent_native_majority_confusion_metrics_cuts': True, 'STFM_raw9_shared5_both_fusions': True,
                   'unsupported_predictions_retained_as_errors': True, 'PSAX_probability_mass_grouped': True,
                   'CAMUS_whole_source_CPU_audit_reference': True, 'grouped_bootstrap_same_seed_membership': True,
                   'prospective_LVH_timing_repair': timing_checks,
                   'dataset_protocol_bindings': BOUND_PROTOCOLS,
                   'A26_exact_native_recount': historical_checks},
        'counts': {**counts, 'ev9v_sessions': 125, 'STFM_original_raw9': 888, 'STFM_shared5': 800,
                   'native_cache_files': sum(counts.values()), 'native_10Hz_samples_per_seed': n_cached_samples,
                   'model_seeds': list(SEEDS), 'bootstrap_draws': BOOTSTRAP_DRAWS},
        'hashes': hashes, 'recounted_native_metrics': summary, 'recounted_STFM_metrics': baseline_summary,
        'limits': LIMITS, 'elapsed_wall_seconds': time.monotonic()-start,
        'execution': 'CPU only; no production decoder/metric imports; no model forward; aggregate-only artifacts'}
    # Write success only after all endpoint/identity/uncertainty checks complete.
    write_json(result/'uncertainty.json', uncertainty)
    verification['hashes']['uncertainty'] = sha(result/'uncertainty.json')
    write_json(result/'verification.json', verification)
    return verification


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT, help='server checkout containing private prediction caches')
    args = parser.parse_args()
    try:
        verification = verify(args.root)
    except Exception as error:  # noqa: BLE001 -- sanitize private exception payloads.
        # Avoid echoing file paths, identifiers, exception payloads or per-case
        # values from NumPy/OpenCV/JSON errors into shared logs/artifacts.
        reason = str(error) if isinstance(error, AuditError) else 'required input unavailable or malformed ('+type(error).__name__+')'
        failed = {'passed': False, 'reason': reason, 'limits': LIMITS,
                  'note': 'A failed/pending audit must not be interpreted as a verified result.'}
        write_json(args.root/RESULT_REL/'verification.json', failed)
        print(json.dumps({'passed': False, 'reason': reason}), flush=True)
        return 1
    print(json.dumps({'passed': True, 'counts': verification['counts'],
                      'protocol_sha256': verification['hashes']['protocol'],
                      'elapsed_wall_seconds': verification['elapsed_wall_seconds']}), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
