"""Check anonymized A61 posthoc case arithmetic; never reads server private data or runs NN."""
import hashlib
import json
import math
import resource
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PUB = HERE.parent / '2026-11-04_action_safety_ranking'
started = time.time()
usage = resource.getrusage(resource.RUSAGE_SELF)
raw = (HERE / 'case_evidence.anonymous.json').read_bytes()
assert hashlib.sha256(raw).hexdigest() == 'a441f5384101ee547f294cbe0ce8f0ca29be17b13a5299a9ad8c21532fcafe29'
evidence = json.loads(raw)
clarification = json.loads((HERE / 'case_clarification.anonymous.json').read_bytes())
summary = json.loads((PUB / 'summary.json').read_bytes())
for path, expected in evidence['input_sha256'].items():
    local = HERE.parents[2] / path
    if path.startswith(('docs/', 'configs/')):
        assert hashlib.sha256(local.read_bytes()).hexdigest() == expected, path
os_receipt = json.loads((HERE / 'case_extraction_OS_CPU.json').read_bytes())
assert os_receipt['observed_worker_OS_exit_code'] == 0
assert os_receipt['child_exit_code_directly_waited']
assert os_receipt['termination_reason'] is None and os_receipt['stderr'] == ''
assert evidence['new_NN_forwards'] == evidence['optimizer_updates'] == 0
assert not evidence['A62_started'] and not evidence['thresholds_changed']
assert not evidence['original_A61_files_modified']
assert evidence['Case_B_same_axis_across_seeds'] and evidence['Case_B_same_action_across_seeds']
assert len(evidence['cases']) == 3
fit_f1 = evidence['FIT_summary']['F_fit']
results = []
for case in evidence['cases']:
    selected = case['selected']
    assert selected['H'] + selected['R'] + selected['neutral_changed'] == selected['changed_K']
    assert selected['dE'] == selected['H'] - selected['R']
    assert math.isclose(selected['u'], 2 * selected['delta_M'] - fit_f1 * selected['delta_P'])
    assert math.isclose(selected['v'], -1 - abs(selected['u']))
    assert selected['v'] < 0 < selected['predicted_utility']
    assert selected['predicted_H'] < selected['predicted_R']
    assert selected['pFP'] < .5 and selected['r'] < selected['tau']
    assert math.isclose(selected['predicted_H'] + selected['predicted_R'] + selected['predicted_neutral'], selected['changed_K'], abs_tol=1e-5)
    # r=max(max_position_pH,pFP), so strict r>pFP identifies max_position_pH exactly.
    assert selected['r'] > selected['pFP']
    assert case['true_safe_candidates'] == case['true_safe_beneficial_u_positive_candidates'] == 0
    assert case['eligible_candidates'] == (2 if case['case'] == 'Case A' else 1)
    for arm in case['all8_matched_arms']:
        assert not arm['E_N_C']['E']['noop'] and not arm['E_N_C']['N']['noop']
        assert arm['true_arm_target_ranking_top']['noop']
        assert all(c['noop'] for c in arm['counterfactual_labels_posthoc_not_deployable'].values())
    table, half = ('a60_confirmation', 'selection') if case['case'] == 'Case A' else ('a61_confirmation', 'calibration')
    metrics = summary['systems'][table][half][f"seed{case['seed']}_S-L-C_fixed6000"]
    gross = metrics['gross_reference_position_effects']['multi']
    counts = metrics['counts']
    new_m = counts['multi_matched_segments'] - selected['delta_M']
    new_p = counts['multi_predicted_segments'] - selected['delta_P']
    truth = counts['multi_true_segments']
    results.append({'case': case['case'], 'seed': case['seed'],
                    'exact_max_position_pH_from_cached_r': selected['r'],
                    'actual_bank_H_R': [gross['H'], gross['R']],
                    'veto_only_this_bad_action_H_R': [gross['H'] - selected['H'], gross['R'] - selected['R']],
                    'actual_bank_frame_errors': metrics['multi_frame_errors'],
                    'veto_only_this_bad_action_frame_errors': metrics['multi_frame_errors'] - selected['dE'],
                    'actual_bank_short_FN16': metrics['short_true_unmatched_16samples'],
                    'veto_only_this_bad_action_short_FN16': metrics['short_true_unmatched_16samples'] - selected['dFN16'],
                    'actual_bank_F1': metrics['segment_f1_50_multi'],
                    'veto_only_this_bad_action_F1': 2 * new_m / (new_p + truth),
                    'counterfactual_scope': 'posthoc label-assisted accounting, not a deployable result or new heldout test'})
    details = next(c for c in clarification['cases'] if (c['case'], c['seed']) == (case['case'], case['seed']))
    assert len(details['added_boundaries']) - len(details['removed_boundaries']) == selected['delta_P']
    if case['case'] == 'Case B':
        assert details['lost_true_units'] == [{'class': 'A4C', 'length': 11}, {'class': 'A5C', 'length': 9}]
        assert (selected['H'], selected['R'], selected['delta_M'], selected['delta_P'], selected['dFN16']) == (9, 2, -2, -2, 2)
    else:
        assert (selected['H'], selected['R'], selected['delta_M'], selected['delta_P']) == (40, 0, 0, 1)
after = resource.getrusage(resource.RUSAGE_SELF)
report = {'passed': True, 'source_sealed_summary_sha256': hashlib.sha256((PUB / 'summary.json').read_bytes()).hexdigest(),
          'primary_accept_records': 3, 'distinct_case_axes': 2,
          'distinct_axis_arm_controls': 16, 'no_new_NN_or_optimizer_or_A62': True,
          'posthoc_bad_action_veto_accounting': results,
          'server_extraction_CPU_seconds_not_double_counted_with_child_self_sample': os_receipt['CPU_SELF_plus_waitedchildren_seconds'],
          'local_verification_SELF_CPU_seconds_through_sample': after.ru_utime + after.ru_stime - usage.ru_utime - usage.ru_stime,
          'wall_seconds': time.time() - started, 'write_exit_tool_tail_unmetered': True}
(HERE / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
