"""Independent external audit: hand-calculated endpoints and failure guards."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/audit_external_original_tests.py'


def audit_module():
    assert SCRIPT.exists(), 'independent audit implementation is not yet present'
    spec = importlib.util.spec_from_file_location('independent_external_audit', SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_decoding_uses_nearest_centre_first_tie_and_frozen_change_gate():
    a = audit_module()
    logits = np.array([[20., -20, -20, -20, -20], [-20, 20, -20, -20, -20]])
    raw, b3 = a.decode(logits, np.array([2, 7]), 10,
                       {'half_window': 2, 'threshold': .1, 'nms_radius': 1})
    assert raw.tolist() == [0] * 5 + [1] * 5
    assert np.array_equal(raw, b3)
    raw, b3 = a.decode(np.tile([3., 2, 1, 0, -1], (2, 1)), [2, 7], 10,
                       {'half_window': 2, 'threshold': 0, 'nms_radius': 1})
    assert b3.tolist() == [0] * 10


def test_short_sampling_window_and_half_sample_ties():
    a = audit_module()
    assert a.sample_indices(1, 54).tolist() == [0]
    assert a.sample_indices(11, 25).tolist() == [0, 2, 5, 8, 10]
    assert a.window_centres(1).tolist() == [0]
    assert a.window_centres(6).tolist() == [2, 5]
    assert a.window_centres(10).tolist() == [2, 7]


def test_unknown_predictions_stay_errors_and_single_class_macro_is_null():
    a = audit_module()
    rows = [{'true': 2, 'pred': 2, 'cuts': 1, 'n_samples': 10, 'duration_s': 1},
            {'true': 2, 'pred': -1, 'cuts': 0, 'n_samples': 4, 'duration_s': .4}]
    result = a.metrics(rows)
    assert result['accuracy'] == .5
    assert result['class_recall']['A4C'] == .5
    assert result['five_class_macro_f1'] is None
    assert result['confusion'][2] == [0, 0, 1, 0, 0, 1]
    assert result['internal_cuts_per_clip'] == .5
    assert a.majority([-1, 2]) == 2
    with pytest.raises(ValueError):
        a.majority([5])


def test_full_support_f1_counts_unknown_false_negatives():
    a = audit_module()
    rows = [{'true': i, 'pred': i, 'cuts': 0, 'n_samples': 1, 'duration_s': 1} for i in range(5)]
    rows.append({'true': 0, 'pred': -1, 'cuts': 0, 'n_samples': 1, 'duration_s': 1})
    result = a.metrics(rows)
    assert result['accuracy'] == 5 / 6
    assert result['five_class_macro_f1'] == pytest.approx((2 / 3 + 4) / 5)


def test_stfm_fusion_groups_probability_and_preserves_unsupported_mass():
    a = audit_module()
    probability = np.array([.2, .12, .12, .12, .18, .01, .12, .12, .01])
    grouped = a.group_stfm(probability)
    assert grouped.argmax() == 1
    assert grouped[1] == pytest.approx(.48)
    assert grouped[5] == .12
    x = np.zeros((10, 9))
    x[0, 6] = 1000
    legacy = a.stfm_probability(x, prior_once=False)
    corrected = a.stfm_probability(x, prior_once=True)
    assert legacy.argmax() == corrected.argmax() == 6
    assert a.group_stfm(corrected).argmax() == 5
    assert corrected[6] > legacy[6]
    assert np.isfinite(corrected).all() and corrected.sum() == pytest.approx(1)


def test_bootstrap_resamples_groups_and_seed_mean_not_three_n():
    a = audit_module()
    # Two groups with unequal case counts. Anti-correlated models make the
    # per-draw model mean exactly .5, whatever groups are resampled.
    truth = np.array([0, 0, 1, 2, 3, 4])
    p0 = truth.copy()
    p1 = (truth + 1) % 5
    pred = {'frozen_B3_s0': p0, 'frozen_B3_s1': p1,
            'frozen_B3_s2': p0, 'argmax_s0': p0}
    result, draws = a.grouped_bootstrap(truth, ['a', 'a', 'a', 'b', 'b', 'b'], pred,
                                        draws=100, seed=12)
    assert result['groups'] == 2
    assert result['models']['frozen_B3_s0']['correct'] == 6
    assert result['models']['frozen_B3_s1']['correct'] == 0
    assert result['mean_B3']['accuracy'] == pytest.approx(2 / 3)
    assert result['mean_B3']['ci95'] == pytest.approx([2 / 3, 2 / 3])
    assert np.array_equal(draws['frozen_B3_s0']['accuracy'], draws['frozen_B3_s2']['accuracy'])
    assert np.all(draws['frozen_B3_s0']['accuracy'] + draws['frozen_B3_s1']['accuracy'] == 1)


def test_cache_audit_reconstructs_labels_and_fails_on_wrong_key_or_nan(tmp_path):
    a = audit_module()
    q = {'id': 'private', 'n_frames': 16, 'fps': 30, 'true': 0}
    centres = np.array([2, 5])
    logits = np.tile([20., -20, -20, -20, -20], (2, 1))
    receipt = {'n_frames': 16, 'fps': 30, 'duration_s': 16/30,
                   'sample_frame_indices': [0, 3, 6, 9, 12, 15], 'input_pixels_sha256': 'a'*64}
    params = {'half_window': 7, 'threshold': .2, 'nms_radius': 5}
    protocol = {'weights': {str(i): {'b3': params} for i in range(3)}}
    arrays = {'key': np.asarray('key'), 'id': np.asarray('private'),
                  'receipt': np.asarray(json.dumps(receipt)), 'centres': centres}
    for seed in range(3):
        arrays[f'logits_s{seed}'] = logits
        arrays[f'argmax_s{seed}'] = np.zeros(6, int)
        arrays[f'frozen_B3_s{seed}'] = np.zeros(6, int)
    path = tmp_path/'case.npz'
    np.savez(path, **arrays)
    rows, _ = a.audit_case(path, q, protocol, 'key')
    assert len(rows) == 6 and rows['frozen_B3_s0']['pred'] == 0
    with pytest.raises(a.AuditError, match='cache key'):
        a.audit_case(path, q, protocol, 'wrong')
    arrays['logits_s0'] = np.full((2, 5), np.nan)
    np.savez(path, **arrays)
    with pytest.raises(a.AuditError, match='finite logits'):
        a.audit_case(path, q, protocol, 'key')


def test_cache_rejects_changed_labels_and_sampling_receipt(tmp_path):
    a = audit_module()
    q = {'id': 'private', 'n_frames': 1, 'fps': 30, 'true': 0}
    receipt = {'n_frames': 1, 'fps': 30, 'duration_s': 1/30,
                   'sample_frame_indices': [0], 'input_pixels_sha256': 'a'*64}
    params = {'half_window': 7, 'threshold': .2, 'nms_radius': 5}
    protocol = {'weights': {str(i): {'b3': params} for i in range(3)}}
    arrays = {'key': 'key', 'id': 'private', 'receipt': json.dumps(receipt), 'centres': [0]}
    for s in range(3):
        arrays[f'logits_s{s}'] = [[20., -20, -20, -20, -20]]
        arrays[f'argmax_s{s}'] = [0]
        arrays[f'frozen_B3_s{s}'] = [0]
    arrays['frozen_B3_s2'] = [1]
    path = tmp_path/'case.npz'
    np.savez(path, **arrays)
    with pytest.raises(a.AuditError, match='decoded labels'):
        a.audit_case(path, q, protocol, 'key')
    arrays['frozen_B3_s2'] = [0]
    receipt['sample_frame_indices'] = [1]
    arrays['receipt'] = json.dumps(receipt)
    np.savez(path, **arrays)
    with pytest.raises(a.AuditError, match='sampling'):
        a.audit_case(path, q, protocol, 'key')


def test_hash_and_aggregate_comparison_fail_closed(tmp_path):
    a = audit_module()
    path = tmp_path/'data'
    path.write_text('content')
    with pytest.raises(a.AuditError, match='hash'):
        a.require_hash(path, '0'*64, 'test metadata')
    with pytest.raises(a.AuditError, match='aggregate'):
        a.compare({'accuracy': .5, 'confusion': [[1, 1]]},
                  {'accuracy': .5, 'confusion': [[2, 0]]}, 'aggregate')


def test_cluster_bootstrap_retains_all_recordings_in_selected_session():
    a = audit_module()
    truth = [0, 0, 0, 0]
    pred = {'model': [0, 0, 0, -1]}
    result, draws = a.grouped_bootstrap(truth, ['large', 'large', 'large', 'small'], pred,
                                       draws=500, seed=20261030)
    assert result['models']['model']['accuracy'] == .75
    assert set(draws['model']['accuracy']) == {0., .75, 1.}
    assert result['models']['model']['accuracy_ci95'] == [0., 1.]
    assert result['models']['model']['macro_f1_ci95'] is None


def test_timing_revision_allows_only_qualified_lvh_rate_repair():
    import copy
    a = audit_module()
    old_protocol = {'created_utc': 'old', 'manifest_sha256': 'old', 'weights': {'0': {'b3': {'threshold': .2}}},
                    'config_sha256': 'fixed', 'counts': {'lvh': 1}, 'code_sha256': {'runner': 'fixed'}}
    new_protocol = copy.deepcopy(old_protocol)
    new_protocol.update(created_utc='new', manifest_sha256='new', timing_amendment={})
    new_protocol['code_sha256']['scripts/amend_external_lvh_timing.py'] = 'amend'
    old = {'created_utc': 'old', 'source_files': {'source': 'hash'}, 'datasets': {
        'ev9v': [], 'dynamic': [], 'camus': [],
        'lvh': [{'id': 'private', 'fps': 17.355, 'n_frames': 120, 'true': 0}]}}
    new = copy.deepcopy(old)
    new['timing_amendment'] = {}
    new['datasets']['lvh'][0].update(fps=58., reported_csv_fps=17.355,
                                    fps_authority='decoded_PTS_and_container_CFR')
    qualification = {'rows': [{'id': 'private', 'container_fps': 58., 'csv_field_fps': 17.355,
                              'decoded_frames': 120, 'pts_delta_max_error': 1e-15}],
                     'max_error': 1e-15, 'seconds': 1.}
    result = a.validate_timing_revision(old_protocol, new_protocol, old, new, qualification)
    assert result['qualified_recordings'] == 1
    damaged = copy.deepcopy(new)
    damaged['datasets']['lvh'][0]['true'] = 1
    with pytest.raises(a.AuditError, match='unchanged'):
        a.validate_timing_revision(old_protocol, new_protocol, old, damaged, qualification)
    damaged_protocol = copy.deepcopy(new_protocol)
    damaged_protocol['weights']['0']['b3']['threshold'] = .25
    with pytest.raises(a.AuditError, match='unchanged'):
        a.validate_timing_revision(old_protocol, damaged_protocol, old, new, qualification)
    qualification['rows'][0]['pts_delta_max_error'] = .1
    with pytest.raises(a.AuditError, match='PTS'):
        a.validate_timing_revision(old_protocol, new_protocol, old, new, qualification)


def test_duration_uses_accurate_fractional_sum_instead_of_naive_accumulation():
    a = audit_module()
    rows = [{'true': 0, 'pred': 0, 'cuts': 1, 'n_samples': 1, 'duration_s': .1}
            for _ in range(10)]
    result = a.metrics(rows)
    assert result['duration_s'] == 1.
    assert result['internal_cuts_per_minute'] == 600.


def test_public_source_hash_summary_contains_no_private_paths_or_case_ids():
    import hashlib
    a = audit_module()
    verified = {
        'data/camus/database_nifti/patient0001/patient0001_Info_4CH.cfg': 'a'*64,
        'data/camus/database_nifti/patient0002/patient0002_Info_4CH.cfg': 'b'*64,
    }
    public = a.public_source_hash_summary(verified)
    assert public == {
        'file_count': 2,
        'aggregate_sha256': hashlib.sha256(
            json.dumps(verified, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
    }
    assert public == a.public_source_hash_summary(dict(reversed(list(verified.items()))))
    serialized = json.dumps(public)
    for private_value in [*verified, 'patient0001', 'patient0002', 'database_nifti']:
        assert private_value not in serialized
    # The existing private mapping remains complete and untouched.
    assert len(verified) == 2 and verified[next(iter(verified))] == 'a'*64
