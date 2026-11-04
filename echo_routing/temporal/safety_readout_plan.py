"""A61 exact fixed6000 endpoints and matched contrasts, no model selection."""
SEEDS=[2026110461,2026110462]
ARMS=['U-P','U-L','S-P','S-L']
TABLES=['main','a58_stress','a59_confirmation','a60_confirmation','a61_confirmation']
HALVES=['selection','calibration']
BASE='A08_s0_original_B3'
MAIN_REFERENCES=[BASE,'A08_s1_original_B3','A08_s2_original_B3','A08_s0_original_raw','JEPA_core_fixed6000_raw']
STRESS_REFERENCES=[BASE,'A08_s0_original_raw','JEPA_core_fixed6000_raw']
CONTRASTS=[('S-L-C',BASE),('S-L-C','A60-J-R'),('S-L-C','S-P-C'),('U-L-C','U-P-C'),('S-P-C','U-P-C'),('S-L-C','U-L-C'),('S-L-C','S-L-N'),('S-L-N','S-L-E'),('U-P-N','A60-J-R'),('S-L-E','U-P-E'),('S-P-E','U-P-E'),('S-L-E','S-P-E')]


def system(seed,name):return f'seed{seed}_{name}_fixed6000'


def historical(seed,policy):return f'A60_seed{2026110360+SEEDS.index(seed)}_J-{policy}_fixed6000'


def plan():
    rows=[];comparisons=[];oracles=[];native=[];inner=[]
    for table in TABLES:
        for half in HALVES:
            references=MAIN_REFERENCES if table=='main' else STRESS_REFERENCES
            rows.extend({'table':table,'half':half,'system':name,'role':'reference'} for name in references)
            for seed in SEEDS:
                for arm in ARMS:
                    for policy in ['E','N','C']:
                        row={'table':table,'half':half,'system':system(seed,arm+'-'+policy),'role':'policy','seed':seed,'arm':arm,'policy':policy,'physical_model':system(seed,arm),'step':6000}
                        rows.append(row)
                        if table=='main':native.append(dict(row))
                for policy in ['E','R']:
                    row={'table':table,'half':half,'system':historical(seed,policy),'role':'A60_historical','old_seed':2026110360+SEEDS.index(seed),'replicate_seed':seed,'policy':policy,'step':6000}
                    rows.append(row)
                    if table=='main':native.append(dict(row))
                def resolve(name):return BASE if name==BASE else historical(seed,'R') if name=='A60-J-R' else system(seed,name)
                comparisons.extend({'table':table,'half':half,'seed':seed,'left':resolve(left),'right':resolve(right),'contrast':left+'-'+right} for left,right in CONTRASTS)
            oracles.extend({'table':table,'half':half,'domain':domain,'safety':safe,'guard':'G','scope':'multi-only exact pooled IoU50 F1/maxone+noop'} for domain in ['old','expanded'] for safe in ['unrestricted','net-safe','gross-safe'])
    for seed in SEEDS:
        for arm in ARMS:
            for step in [2000,4000,6000]:
                inner.extend({'seed':seed,'arm':arm,'step':step,'policy':p,'calibration':False} for p in ['E','N'])
            inner.append({'seed':seed,'arm':arm,'step':6000,'policy':'C','calibration':True})
    return {'rows':rows,'comparisons':comparisons,'oracle_cells':oracles,'native_cells':native,'inner_policy_sets':inner,'metric_rows':314,'policy_rows':240,'reference_rows':74,'paired_comparisons':240,'bootstrap_replicates':5000,'bootstrap_seed':2026110461,'fixed_step':6000,'checkpoint_selection':False,'inner_records':24,'thresholds':8,'stress_native_pairs_reuse_main':True,'historical_replicate_mapping_not_equal_training_seeds':True}
