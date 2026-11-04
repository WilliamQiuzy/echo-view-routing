"""Prospectively exact A60 124 systems,144 contrasts,32 multi-only oracles."""
SEEDS=[2026110360,2026110361]
TABLES=['main','a58_stress','a59_confirmation','a60_confirmation']
HALVES=['selection','calibration']
BASE='A08_s0_original_B3'
MAIN_REFERENCES=[BASE,'A08_s1_original_B3','A08_s2_original_B3','A08_s0_original_raw','JEPA_core_fixed6000_raw']
STRESS_REFERENCES=[BASE,'A08_s0_original_raw','JEPA_core_fixed6000_raw']
CONTRASTS=[(f'J-{p}',f'A-{p}') for p in ['O','E','R']]+[(f'{a}-E',f'{a}-O') for a in ['A','J']]+[(f'{a}-R',f'{a}-E') for a in ['A','J']]+[(f'{a}-R',BASE) for a in ['A','J']]

def system(seed,policy):return f'seed{seed}_{policy}_fixed6000'

def plan():
    rows=[];comparisons=[];oracles=[]
    for table in TABLES:
        for half in HALVES:
            for name in MAIN_REFERENCES if table=='main' else STRESS_REFERENCES:rows.append({'table':table,'half':half,'system':name,'role':'reference'})
            for seed in SEEDS:
                for arm in ['A','J']:
                    for p in ['O','E','R']:rows.append({'table':table,'half':half,'system':system(seed,arm+'-'+p),'role':'policy','seed':seed,'arm':arm,'policy':p,'physical_model':f'seed{seed}_{arm}_fixed6000','step':6000})
                for left,right in CONTRASTS:comparisons.append({'table':table,'half':half,'seed':seed,'left':system(seed,left),'right':right if right==BASE else system(seed,right),'contrast':left+'-'+right})
            for domain in ['old','expanded']:
                for guard in ['U','G']:oracles.append({'table':table,'half':half,'domain':domain,'guard':guard,'scope':'multi-only exact pooled IoU50 F1 single action + no-op'})
    return {'rows':rows,'comparisons':comparisons,'oracle_cells':oracles,'metric_rows':124,'policy_rows':96,'reference_rows':28,'paired_comparisons':144,'bootstrap_seed':2026110360,'bootstrap_replicates':5000,'fixed_step':6000,'inner_selection':False,'O_E_R_share_model_and_score_forward':True,'stress_native_pairs_reuse_main':True}
