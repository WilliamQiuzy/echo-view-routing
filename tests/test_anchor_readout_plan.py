import importlib


def test_exact70_rows96contrasts_fixed6000_and_same_model_GU_binding():
    m = importlib.import_module('echo_routing.temporal.anchor_readout_plan')
    p = m.plan()
    assert len(p['rows']) == 70 and len(p['comparisons']) == 96
    assert sum(r['role'] == 'policy' for r in p['rows']) == 48
    assert sum(r['role'] == 'reference' for r in p['rows']) == 22
    assert len({(r['table'], r['half'], r['system']) for r in p['rows']}) == 70
    for table in ['main', 'old_stress', 'fresh_confirmation']:
        for half in ['selection', 'calibration']:
            for seed in [2026110359, 2026110360]:
                for arm in ['A', 'J']:
                    rows = [r for r in p['rows'] if r['role'] == 'policy' and r['table'] == table
                            and r['half'] == half and r['seed'] == seed and r['arm'] == arm]
                    assert len(rows) == 2 and rows[0]['physical_model'] == rows[1]['physical_model']
                    assert {r['policy'] for r in rows} == {'G', 'U'}
    assert p['bootstrap_seed'] == 2026110359 and p['bootstrap_replicates'] == 5000
    assert p['fixed_step'] == 6000 and p['inner_selection'] is False
    assert p['fresh_confirmation_native_pairs_reuse_main']
