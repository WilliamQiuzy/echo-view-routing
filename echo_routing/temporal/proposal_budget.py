"""A60 measured all-matrix forecast, independent of candidate performance."""

def complete(roles,geometry,checks,costs,training,old_source_records,current_bytes,formal_pending=True,source_pending=True):
    fresh=['a60_confirmation_selection','a60_confirmation_calibration'];outer=[k for k in roles if k not in ['fit','inner_cal']]
    rates={}
    for source in ['a08','jepa']:
        rows=[r for r in old_source_records if r['source']==source and r['role'].startswith('confirmation_') and r['new_neural_units']>0]
        if len(rows)!=2:raise ValueError('actual two prior fresh joined source timings required')
        rates[source]=max(r['wall_seconds_this_process']/r['new_neural_units'] for r in rows)
    units={'a08':sum(roles[r]['a08_stride5_windows'] for r in fresh),'jepa':sum(roles[r]['jepa_chunks'] for r in fresh)}
    source=3*(sum(units[s]*rates[s] for s in units)+120) if source_pending else 0.
    unit=4*max(h['maximum_seconds_per_1plus_all_candidate'] for h in costs['heads'].values());fit_T=geometry['fit']['max_length']
    def c(r):return geometry[r].get('all_candidates',geometry[r].get('all_candidates_upper'))
    def infer(r):return unit*geometry[r].get('proposal_units_1plusC',geometry[r].get('proposal_units_1plusC_upper'))*max(1,geometry[r]['max_length']/fit_T)
    target_seconds=training['full_FIT_prepare_wall_seconds']/max(1,training['candidates'])
    diagnostic_candidate_calls=24*sum(c(r) for r in outer)+36*c('selection')+36*c('calibration')
    cpu={'new_confirmation_source_worker_and_probes':1.2*source,
         'four6000_durable_heads':2*24000*max(v['conservative_full_update_timing_seconds'] for v in checks['heads'].values())+120 if formal_pending else 0.,
         'all12inner36policy_sets':12*infer('inner_cal')+180 if formal_pending else 0.,
         'all32joint_policy_cells_and_three_sameweight_decodes':4*sum(infer(r) for r in outer)+.02*4*sum(roles[r]['sequences'] for r in outer),
         'all_policy_candidate_and_accepted_fullaxis_harm_length_score_diagnosis':2*diagnostic_candidate_calls*target_seconds,
         'all124historical_metrics_independent_stored_label_recount':4*37.814352*124/70+180,
         'all144paired5000_bootstrap':4*63.168966*144/96,
         'all32exact_fractional_oracles_and_old12_reproduction':3600.,
         'model_trace_optimizer_inner_exposure_audit_curves_plots':1800.,
         'resource_exit_public_package_selective_integration':1800.,
         'inflight_admission_and_transition':120.}
    fresh_positions=sum(roles[r]['positions'] for r in fresh);positions=sum(roles[r]['positions'] for r in outer);seq=sum(roles[r]['sequences'] for r in outer)
    components={'current_A60_tracked_named_paths':current_bytes,'missing_fresh_two_source_FP32_probability_arrays':2*20*fresh_positions if source_pending else 0,'fresh_source_NPZ_headers_and_journals':2*4096*240 if source_pending else 0,'all16full_head_milestones_latest':16*262144,'atomic_checkpoint_coexistence':262144,'all24000durable_traces':24000*4096,'all32jointpolicy_labels_scores_variable_replacements_headers':4*(255*positions+4096*seq),'private_predictions_oracle_tables_bootstrap_and_public_figures':100000000,'additional_audit_code_and_IO_peak_margin':200000000}
    return {'remaining_GPU_seconds':source,'remaining_CPU_components_seconds':cpu,'remaining_CPU_seconds':sum(cpu.values()),'remaining_serial_wall_seconds':source+sum(v for k,v in cpu.items() if k!='new_confirmation_source_worker_and_probes'),'incremental_tracked_path_peak_upper_bytes':sum(components.values()),'storage_components':components,'new_source_neural_units':units if source_pending else {'a08':0,'jepa':0},'source_rate_basis_seconds_per_unit':rates,'cached_inference_seconds_per_1plusC_with4xmargin':unit,'complete_axis_target_diagnostic_candidate_calls_upper':diagnostic_candidate_calls,'target_diagnostic_seconds_per_candidate_basis':target_seconds,'scope':{'formal_models':4,'formal_updates':24000,'inner_physical_records':12,'inner_policy_sets':36,'system_metric_rows':124,'joint_policy_cache_cells':32,'paired_comparisons':144,'bootstrap_replicates_each':5000,'oracle_cells':32,'old_oracle_reproduction_cells':12},'forecast_not_actual_spend':True,'fresh_unknown_candidate_bound':'<=2T actions;total replacement intervals<=2T;actual old-role source-only geometry used;no performance input','source_cost_distinct_from_cached_head_cost':True}
