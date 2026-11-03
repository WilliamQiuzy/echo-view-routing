import importlib


def test_workload_has_fresh_joins_native_reuse_four_models70rows96contrasts():
    m = importlib.import_module('echo_routing.temporal.anchor_budget')
    roles = {name: {'sequences': n, 'positions': p, 'a08_stride5_windows': a, 'jepa_chunks': j}
             for name, n, p, a, j in [('fit', 10, 100, 20, 8), ('inner_cal', 4, 40, 8, 4),
                                      ('selection', 3, 30, 6, 3), ('calibration', 2, 20, 4, 2),
                                      ('old_stress_selection', 2, 20, 4, 2), ('old_stress_calibration', 2, 20, 4, 2),
                                      ('confirmation_selection', 2, 24, 5, 2), ('confirmation_calibration', 2, 26, 6, 2)]}
    reused = {'fit': {'new_JEPA_chunks': 3, 'new_JEPA_sequences': 2},
              'inner_cal': {'new_JEPA_chunks': 2, 'new_JEPA_sequences': 1}}
    x = m.workload(roles, reused)
    assert x['new_A08_windows'] == 39 and x['new_JEPA_chunks'] == 9
    assert x['new_A08_sequences'] == 18 and x['new_JEPA_sequences'] == 7
    assert x['source_cache_sequence_writes'] == 54
    assert x['formal_models'] == 4 and x['formal_optimizer_updates'] == 24000
    assert x['inner_head_score_calls'] == 48 and x['outer_head_score_calls'] == 52
    assert x['metric_rows'] == 70 and x['paired_comparisons'] == 96
    assert x['G_U_share_scores_and_weights']


def test_bulk_rate_plus_cold_overhead_envelopes_all_measured_source_cases():
    m = importlib.import_module('echo_routing.temporal.anchor_budget')
    rows = [{'neural_units': 1, 'wall_seconds': .4}, {'neural_units': 32, 'wall_seconds': 1.},
            {'neural_units': 96, 'wall_seconds': 3.2}]
    rate, overhead = m.source_rate(rows, 32)
    for row in rows:
        assert rate * row['neural_units'] + overhead >= row['wall_seconds'] - 1e-12
    assert m.source_seconds(rows, 100, 10, 32) >= 100 * rate + 10 * overhead


def test_storage_forecast_is_explicit_components_with_atomic_coexistence():
    m = importlib.import_module('echo_routing.temporal.anchor_budget')
    roles = {name: {'sequences': 10, 'positions': 100, 'a08_stride5_windows': 20, 'jepa_chunks': 5}
             for name in m.ALL_ROLES}
    result = m.storage_forecast(roles, current_tracked_bytes=123456)
    assert result['incremental_tracked_path_peak_upper_bytes'] == sum(result['components'].values())
    assert result['components']['all12milestones_and4latest'] == 16 * 262144
    assert result['components']['atomic_checkpoint_coexistence'] == 262144
    assert result['components']['current_retained_tracked_upper'] == 123456
    assert result['scope'] == 'conservative named-path upper;not whole-filesystem peak'
