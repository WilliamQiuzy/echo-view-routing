"""A59 pre-result fixed endpoints:70 rows and96 paired contrasts, no selection."""
SEEDS = [2026110359, 2026110360]
TABLES = ['main', 'old_stress', 'fresh_confirmation']
HALVES = ['selection', 'calibration']
BASE = 'A08_s0_original_B3'
MAIN_REFERENCES = [BASE, 'A08_s1_original_B3', 'A08_s2_original_B3', 'A08_s0_original_raw', 'JEPA_core_fixed6000_raw']
STRESS_REFERENCES = [BASE, 'A08_s0_original_raw', 'JEPA_core_fixed6000_raw']
CONTRASTS = [('J-G', 'A-G'), ('J-U', 'A-U'), ('J-G', 'J-U'), ('A-G', 'A-U'),
             ('A-U', BASE), ('A-G', BASE), ('J-U', BASE), ('J-G', BASE)]


def system(seed, policy):
    return f'seed{seed}_{policy}_fixed6000'


def plan():
    rows, comparisons = [], []
    for table in TABLES:
        for half in HALVES:
            for name in MAIN_REFERENCES if table == 'main' else STRESS_REFERENCES:
                rows.append({'table': table, 'half': half, 'system': name, 'role': 'reference'})
            for seed in SEEDS:
                for arm in ['A', 'J']:
                    for policy in ['U', 'G']:
                        rows.append({'table': table, 'half': half, 'system': system(seed, arm + '-' + policy),
                                     'role': 'policy', 'seed': seed, 'arm': arm, 'policy': policy,
                                     'physical_model': f'seed{seed}_{arm}_fixed6000', 'step': 6000})
                for left, right in CONTRASTS:
                    comparisons.append({'table': table, 'half': half, 'seed': seed,
                                        'left': system(seed, left),
                                        'right': right if right == BASE else system(seed, right),
                                        'contrast': left + '-' + right})
    return {'rows': rows, 'comparisons': comparisons, 'metric_rows': 70, 'policy_rows': 48,
            'reference_rows': 22, 'paired_comparisons': 96, 'bootstrap_seed': 2026110359,
            'bootstrap_replicates': 5000, 'fixed_step': 6000, 'inner_selection': False,
            'G_U_share_model_and_score_forward': True, 'fresh_confirmation_native_pairs_reuse_main': True,
            'old_stress_native_pairs_reuse_main': True}
