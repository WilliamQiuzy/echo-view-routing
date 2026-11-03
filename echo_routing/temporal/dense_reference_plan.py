"""Fixed MD paired identities; no scores or performance-led reference choice."""
def md_reference_pairs(seeds):
    if list(seeds) not in [[2026110257],[2026110257,2026110258]]:raise ValueError('only the cost-admitted prospective MD seeds allowed')
    result=[]
    for seed in seeds:
        for mode in ['raw','inner_B3']:
            left=f'seed{seed}_J-MD_fixed6000|{mode}'
            result.append({'left':left,'right':f'seed{seed}_F-MD_fixed6000|{mode}','kind':'original_within_depth_adaptation'})
            result.append({'left':left,'right':'A08_original_B3_s0','kind':'fixed_reference_reporting_addendum_before_MDouter'})
    return result
