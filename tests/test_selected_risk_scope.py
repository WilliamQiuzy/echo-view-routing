"""Hand-built A63 matrix, CI and absolute-safety contracts; no NN/data jobs."""
import copy
import hashlib
import importlib
import numpy as np
import pytest

m = importlib.import_module('echo_routing.temporal.selected_risk_metrics')


def api(name):
    assert callable(getattr(m, name, None)), 'A63 pure API missing: ' + name
    return getattr(m, name)


def complete():
    plan = api('readout_plan')()
    rows = []
    for spec in plan['rows']:
        table, half, seed, policy = (spec[k] for k in ['table', 'half', 'seed', 'policy'])
        recovered = 0
        if policy == 'C1' and table == 'a61_confirmation':
            recovered = {('selection', 2026110461): 6, ('selection', 2026110462): 5,
                         ('calibration', 2026110461): 4, ('calibration', 2026110462): 3}[(half, seed)]
        if policy == 'C1' and table == 'a63_confirmation':
            recovered = 1
        zero = {'native': 0, 'multi': 0, 'pair': 0}
        rows.append({**spec, 'input_identity': hashlib.sha256((table+half).encode()).hexdigest(),
                     'native_confusion': np.diag([10]*5).tolist(), 'native_vote_changes': 0,
                     'H': dict(zero), 'R': dict(zero), 'dFP16': dict(zero), 'short16_lost': dict(zero),
                     'class_H': [0]*5, 'JEPA_short16_recovered': recovered,
                     'multi_f1': .92 if policy == 'C1' else .90})
    comparisons = []
    lookup = {(r['table'], r['half'], r['seed'], r['policy']): r for r in rows}
    for spec in plan['comparisons']:
        left = lookup[(spec['table'], spec['half'], spec['seed'], spec['left_policy'])]
        right = lookup[(spec['table'], spec['half'], None if spec['right_policy'] == 'Base' else spec['seed'], spec['right_policy'])]
        comparisons.append({**spec, 'delta95': {'segment_f1_50_multi': [.01, .03]},
                            'delta99_375': {'segment_f1_50_multi': [.01, .03]} if spec['primary_family'] else {},
                            'input_identity': left['input_identity'],
                            'observed_delta': {'segment_f1_50_multi': left['multi_f1'] - right['multi_f1']},
                            'undefined_bootstrap_replicates': {'segment_f1_50_multi': 0}})
    return rows, comparisons


def test_exact_endpoint_scope_and_primary_family_are_frozen():
    p = api('readout_plan')()
    assert len(p['rows']) == 238 and len(p['inner_rows']) == 34
    assert len(p['comparisons']) == 140 and sum(r['primary_family'] for r in p['comparisons']) == 8
    assert len(p['support_cells']) == 56 and sum(r['readonly_A62'] for r in p['support_cells']) == 48
    assert len(p['native_cells']) == 32 and len(p['base_native_references']) == 2
    assert len(p['regression_records']) == 32
    assert len({(r['table'], r['half'], r['seed'], r['policy']) for r in p['rows']}) == 238
    assert {tuple(r['primary_quantiles']) for r in p['comparisons'] if r['primary_family']} == {(.003125, .996875)}
    assert (p['auxiliary_models'], p['auxiliary_updates'], p['new_event_models'], p['formal_updates'], p['total_updates']) == (8, 48000, 4, 24000, 72000)


def test_intervals_share_exact5000_samples_and_keep_undefined():
    f = api('paired_intervals')
    result = f({'linear': np.linspace(-1, 1, 5000), 'partial': [float('nan')]*10 + [.25]*4990,
                'undefined': [float('nan')]*5000}, primary=True)
    assert result['delta95']['linear'] == pytest.approx([-.95, .95])
    assert result['delta99_375']['linear'] == pytest.approx([-.99375, .99375])
    assert result['delta99_375']['partial'] == [.25, .25]
    assert result['undefined_bootstrap_replicates']['partial'] == 10
    assert result['delta95']['undefined'] is None
    with pytest.raises(ValueError): f({'x': [0.]*4999}, primary=True)
    with pytest.raises(ValueError): f({'x': [float('inf')] + [0.]*4999})


def test_complete_safe_useful_matrix_supports_bounded_sampling_claim():
    rows, comps = complete()
    r = api('acceptance')(rows, comps)
    assert r['hard_protection_pass'] and r['joint_objective_pass'] and r['H63_claim_supported']
    assert r['C1_cells'] == 28 and len(r['failures']) == 0
    assert r['independent_clinical_safety_claim'] is False


def test_missing_or_duplicate_matrix_is_incomplete_not_a_negative_result():
    rows, comps = complete(); f = api('acceptance')
    with pytest.raises(ValueError): f(rows[:-1], comps)
    with pytest.raises(ValueError): f(rows[:-1] + [rows[0]], comps)
    with pytest.raises(ValueError): f(rows, comps[:-1])


def test_repairs_and_fewer_FP_cannot_cancel_gross_frame_harm():
    rows, comps = complete()
    r = next(x for x in rows if x['policy'] == 'C1' and x['table'] == 'main')
    r['H']['multi'] = 40; r['R']['multi'] = 100; r['dFP16']['multi'] = -20; r['class_H'][0] = 40
    out = api('acceptance')(rows, comps)
    assert not out['hard_protection_pass'] and not out['H63_claim_supported']
    assert any(f['criterion'] == 'gross_H_multi' for f in out['failures'])


def test_shortloss_and_native_vote_changes_are_separate_hard_failures():
    rows, comps = complete()
    r = next(x for x in rows if x['policy'] == 'C1' and x['table'] == 'main')
    r['short16_lost']['pair'] = 1; r['native_vote_changes'] = 1
    out = api('acceptance')(rows, comps)
    assert not out['hard_protection_pass']
    assert {'short16_lost_pair', 'native_vote_preservation'} <= {f['criterion'] for f in out['failures']}


def test_all_four_cells_and_both_seeds_required_no_recovery_by_one_seed():
    rows, comps = complete()
    for r in rows:
        if r['policy'] == 'C1' and r['seed'] == 2026110462 and r['table'] in ['a61_confirmation', 'a63_confirmation']:
            r['JEPA_short16_recovered'] = 0
    out = api('acceptance')(rows, comps)
    assert not out['joint_objective_pass'] and not out['H63_claim_supported']
    assert sum(f['criterion'] == 'retain_A61_JEPA_short16' for f in out['failures']) == 2
    assert sum(f['criterion'] == 'new_JEPA_short16' for f in out['failures']) == 2


def test_zero_lower_CI_keeps_absolute_success_separate_from_sampling_increment():
    rows, comps = complete()
    c = next(x for x in comps if x['primary_family'] and x['right_policy'] == 'C0')
    c['delta99_375']['segment_f1_50_multi'] = [0., .03]
    out = api('acceptance')(rows, comps)
    assert out['hard_protection_pass'] and out['joint_objective_pass'] and not out['H63_claim_supported']
    assert any(f['criterion'] == 'sampling_increment_CI' for f in out['failures'])


@pytest.mark.parametrize('kind', ['input', 'count_type', 'class_sum', 'family_flag', 'replicates', 'comparison_input', 'observed_delta', 'missing_observed'])
def test_inconsistent_or_fabricated_adapter_fields_rejected(kind):
    rows, comps = complete(); row = next(r for r in rows if r['policy'] == 'C1')
    if kind == 'input': row['input_identity'] = 'f'*64
    if kind == 'count_type': row['H']['native'] = True
    if kind == 'class_sum': row['class_H'][0] = 1
    if kind == 'family_flag': comps[0]['primary_family'] = not comps[0]['primary_family']
    if kind == 'replicates': comps[0]['bootstrap_replicates'] = 4999
    if kind == 'comparison_input': comps[0]['input_identity'] = 'f'*64
    if kind == 'observed_delta': comps[0]['observed_delta']['segment_f1_50_multi'] = -.02
    if kind == 'missing_observed': del comps[0]['observed_delta']
    with pytest.raises(ValueError): api('acceptance')(rows, comps)


def test_reverse_observed_effect_with_positive_interval_cannot_claim_benefit():
    rows, comps = complete()
    for r in rows:
        if r['table'] == 'a63_confirmation' and r['policy'] == 'C1':
            r['multi_f1'] = .80
    lookup = {(r['table'], r['half'], r['seed'], r['policy']): r for r in rows}
    for c in comps:
        left = lookup[(c['table'], c['half'], c['seed'], c['left_policy'])]
        right = lookup[(c['table'], c['half'], None if c['right_policy'] == 'Base' else c['seed'], c['right_policy'])]
        c['observed_delta']['segment_f1_50_multi'] = left['multi_f1'] - right['multi_f1']
    out = api('acceptance')(rows, comps)
    assert out['hard_protection_pass'] and not out['joint_objective_pass'] and not out['H63_claim_supported']


def test_tolerance_does_not_turn_canonical_zero_effect_into_positive_gain():
    rows, comps = complete()
    row = next(r for r in rows if r['table'] == 'a63_confirmation' and r['policy'] == 'C1')
    row['multi_f1'] = .90
    for c in comps:
        if (c['table'], c['half'], c['seed'], c['left_policy']) == (row['table'], row['half'], row['seed'], 'C1'):
            c['observed_delta']['segment_f1_50_multi'] = 1e-15
    out = api('acceptance')(rows, comps)
    assert not out['joint_objective_pass'] and not out['H63_claim_supported']


def test_F1_delta_intervals_outside_mathematical_range_are_invalid_audit_data():
    rows, comps = complete()
    c = next(c for c in comps if c['primary_family'])
    c['delta95']['segment_f1_50_multi'] = [1.1, 1.2]
    c['delta99_375']['segment_f1_50_multi'] = [1.1, 1.2]
    with pytest.raises(ValueError): api('acceptance')(rows, comps)
