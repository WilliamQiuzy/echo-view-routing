import importlib

def test_primary6000_readout_order_preserves_every_frozen_row_and_original_tail():
    a=importlib.import_module('echo_routing.temporal.dense_readout_order')
    rows=[{'row':'package','seed':2026110257,'arm':'J-high','kind':'inner_selected_package','step':6000}]
    rows += [{'row':f'{seed}_{arm}_{step}','seed':seed,'arm':arm,'kind':'fixed_step','step':step} for seed in [2026110257,2026110258] for arm in ['L','F','J-high'] for step in [2000,6000]]
    original=list(rows);ordered=a.primary_first(rows,'J-high')
    assert [(x['seed'],x['arm']) for x in ordered[:4]]==[(2026110257,'F'),(2026110257,'J-high'),(2026110258,'F'),(2026110258,'J-high')]
    assert rows==original and sorted(x['row'] for x in rows)==sorted(x['row'] for x in ordered)
    assert ordered[4:]==[x for x in rows if x not in ordered[:4]]
