"""A57-MD cost-only tier admission and explicit incremental storage components."""

def storage_components(seed_count,measured):
    if seed_count not in [1,2] or any(measured[k]<=0 for k in ['full_model_bytes','head_only_bytes','atomic_full_checkpoint_bytes','frozen_latest_optimizer_bytes']):raise ValueError('one/two matched seed pairs and positive measured sizes required')
    # Each seed has one joint and one frozen model;12milestones each. Official
    # source/raw arrays are shared, so do not charge or copy them again here.
    return {'joint_full_milestones':seed_count*12*int(measured['full_model_bytes']),'frozen_head_only_milestones':seed_count*12*int(measured['head_only_bytes']),'all_joint_latest_optimizers':seed_count*int(measured['atomic_full_checkpoint_bytes']),'all_frozen_latest_optimizers':seed_count*int(measured['frozen_latest_optimizer_bytes']),'one_atomic_optimizer_coexistence':max(int(measured['atomic_full_checkpoint_bytes']),int(measured['frozen_latest_optimizer_bytes'])),'inner_probabilities_predictions_tables_and_sequence_cache':seed_count*24*5_000_000,'outer_probabilities_journals_and_all_decoding_tables':seed_count*6*4_000_000,'new_trace_graph_public_code_and_figures_allowance':128_000_000,'serialization_and_operating_headroom':1_073_741_824,'new_source_weight_copy':0}

def select_tier(forecasts):
    eligible=[r['seed_count'] for r in forecasts if r['GPU_total_including_reserve']<=24 and r['CPU_total']<=72 and r['peak_bytes']<=40_000_000_000 and r['wall_hours']<=48]
    return max(eligible,default=0)
