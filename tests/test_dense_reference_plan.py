import importlib
import pytest

def test_MD_comparisons_keep_both_seeds_raw_and_B3_and_fixed_A08seed0():
    a=importlib.import_module('echo_routing.temporal.dense_reference_plan');pairs=a.md_reference_pairs([2026110257,2026110258]);assert len(pairs)==8 and len({(x['left'],x['right']) for x in pairs})==8
    for seed in [2026110257,2026110258]:
        for mode in ['raw','inner_B3']:
            left=f'seed{seed}_J-MD_fixed6000|{mode}'
            assert {p['right'] for p in pairs if p['left']==left}=={f'seed{seed}_F-MD_fixed6000|{mode}','A08_original_B3_s0'}
    with pytest.raises(ValueError):a.md_reference_pairs([123])
