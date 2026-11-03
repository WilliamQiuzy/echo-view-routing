"""Prospective A59 physical workload and component-based remaining forecast."""
ALL_ROLES = ['fit', 'inner_cal', 'selection', 'calibration', 'old_stress_selection',
             'old_stress_calibration', 'confirmation_selection', 'confirmation_calibration']


def workload(roles, reuse):
    fresh = ['confirmation_selection', 'confirmation_calibration']
    inner = ['fit', 'inner_cal']
    outer = ALL_ROLES[2:]
    return {'formal_models': 4, 'formal_optimizer_updates': 24000,
            'new_A08_windows': sum(roles[k]['a08_stride5_windows'] for k in inner + fresh),
            'new_A08_sequences': sum(roles[k]['sequences'] for k in inner + fresh),
            'new_JEPA_chunks': sum(reuse[k]['new_JEPA_chunks'] for k in inner) + sum(roles[k]['jepa_chunks'] for k in fresh),
            'new_JEPA_sequences': sum(reuse[k]['new_JEPA_sequences'] for k in inner) + sum(roles[k]['sequences'] for k in fresh),
            'source_cache_sequence_writes': 2 * sum(roles[k]['sequences'] for k in ALL_ROLES),
            'inner_head_score_calls': 12 * roles['inner_cal']['sequences'],
            'outer_head_score_calls': 4 * sum(roles[k]['sequences'] for k in outer),
            'FIT_candidate_structural_upper': (roles['fit']['positions'] + roles['fit']['sequences']) // 2,
            'inner_candidate_structural_upper': (roles['inner_cal']['positions'] + roles['inner_cal']['sequences']) // 2,
            'outer_candidate_structural_upper': sum((roles[k]['positions'] + roles[k]['sequences']) // 2 for k in outer),
            'metric_rows': 70, 'policy_rows': 48, 'reference_rows': 22,
            'inner_physical_records': 12, 'inner_policy_metric_sets': 24,
            'paired_comparisons': 96, 'bootstrap_replicates_each': 5000,
            'oracle_streams_each_domain': 720, 'G_U_share_scores_and_weights': True}


def source_rate(rows, bulk_minimum):
    if not rows or any(r['neural_units'] < 1 or r['wall_seconds'] <= 0 for r in rows):
        raise ValueError('actual positive source timings required')
    bulk = [r for r in rows if r['neural_units'] >= bulk_minimum]
    if not bulk:
        raise ValueError('representative bulk source measurement missing')
    rate = max(r['wall_seconds'] / r['neural_units'] for r in bulk)
    overhead = max(0., max(r['wall_seconds'] - rate * r['neural_units'] for r in rows))
    return rate, overhead


def source_seconds(rows, units, sequences, bulk_minimum):
    rate, overhead = source_rate(rows, bulk_minimum)
    return units * rate + sequences * overhead


def storage_forecast(roles, current_tracked_bytes):
    candidate_upper = (roles['fit']['positions'] + roles['fit']['sequences']) // 2
    outer_positions = sum(roles[k]['positions'] for k in ALL_ROLES[2:])
    outer_sequences = sum(roles[k]['sequences'] for k in ALL_ROLES[2:])
    components = {
        'current_retained_tracked_upper': current_tracked_bytes,
        'all_source_FP32posterior_arrays': 2 * 20 * sum(roles[k]['positions'] for k in ALL_ROLES),
        'source_NPZ_headers': 2 * 512 * sum(roles[k]['sequences'] for k in ALL_ROLES),
        'two_arms_raw_FP64feature_matrices': 2 * candidate_upper * 98 * 8,
        'private_candidate_rows_targets_kind_indices_upper': candidate_upper * 544,
        'two_seed_two_kind_draw_plans': 2 * 2 * 6000 * 32 * 8,
        'all12milestones_and4latest': 16 * 262144,
        'atomic_checkpoint_coexistence': 262144,
        'physical_update_trace_upper': 24000 * 4096,
        'four_paired_policy_label_score_journals_upper': 4 * (2 * outer_positions + 40 * ((outer_positions + outer_sequences) // 2) + 2048 * outer_sequences),
        'readout_aggregate_private_tables_and_public_figures': 50000000,
        'additional_code_report_and_atomic_IO_margin': 100000000}
    return {'components': components, 'incremental_tracked_path_peak_upper_bytes': sum(components.values()),
            'scope': 'conservative named-path upper;not whole-filesystem peak',
            'source_weights_copied_or_new_media_persisted': False}


def remaining_components(roles, reuse, gpu_checks, cpu_checks):
    work = workload(roles, reuse)
    io = gpu_checks['maximum_durable_cache_IO_seconds_per_sequence']
    source = (source_seconds(gpu_checks['source_timings']['a08'], work['new_A08_windows'], work['new_A08_sequences'], 32)
              + source_seconds(gpu_checks['source_timings']['jepa'], work['new_JEPA_chunks'], work['new_JEPA_sequences'], 4)
              + work['source_cache_sequence_writes'] * io + 2 * sum(gpu_checks['source_load_seconds'].values()))
    source = 1.5 * source + 120
    update = max(v['64candidate_update_max_seconds'] for v in cpu_checks['heads'].values())
    feature = max(v['cached_head_feature_policy_max_seconds_per_sequence'] for v in cpu_checks['heads'].values())
    # Treat the worst complete panel sequence as a per-candidate cost, then use
    # structural candidate bounds. This deliberately overcounts repeated work.
    cpu = {'source_cache_worker_and_probes': 1.2 * source,
           'FIT_candidate_targets_features_normalizers_plans': 4 * work['FIT_candidate_structural_upper'] * feature + 120,
           'four_fixed6000_heads_checkpoint_and_trace': 2 * 24000 * update + 240 * io + 120,
           'all12inner_GU_records': 1.5 * 12 * work['inner_candidate_structural_upper'] * feature + 180,
           'all70outer_policy_reference_rows': 1.5 * 4 * work['outer_candidate_structural_upper'] * feature + work['outer_head_score_calls'] * io + 400,
           'independent70metric_recount': 400,
           'all96paired5000bootstrap': 3 * 41.32912 * 96 / 132,
           'exact_oracle_both_domains_all3tables': 1800,
           'model_input_audits_curves_figures': 1800,
           'resource_exit_and_public_package_closure': 1800}
    return {'workload': work, 'remaining_GPU_seconds': source,
            'remaining_CPU_components_seconds': cpu, 'remaining_CPU_seconds': sum(cpu.values()),
            'remaining_serial_wall_seconds': source + sum(v for k, v in cpu.items() if k != 'source_cache_worker_and_probes'),
            'timing_basis': 'actualFITsource+64candidateCPUmeasurements;1.5source/featuremargin;structuralcandidateupper;unchangedA58bootstrapratex3;explicitpackage/oracleclosureallowances',
            'measured_source_rates': {name: dict(zip(['seconds_per_neural_unit', 'cold_seconds_per_sequence'], source_rate(gpu_checks['source_timings'][name], 32 if name == 'a08' else 4))) for name in ['a08', 'jepa']}}
