import importlib


def test_new_reader_aggregate_oracle_use_exact_plan_and_explicit_roles():
    r=importlib.import_module('scripts.readout_anchor_edits')
    assert r.role_for('main','selection')=='selection'
    assert r.role_for('old_stress','calibration')=='old_stress_calibration'
    assert r.role_for('fresh_confirmation','selection')=='confirmation_selection'
    from echo_routing.temporal.anchor_readout_plan import plan
    p=plan()
    assert len(p['rows'])==70 and len(p['comparisons'])==96
    for module in ['scripts.aggregate_anchor_edits','scripts.oracle_anchor_edits','scripts.audit_anchor_edits','scripts.check_anchor_full_fit','scripts.admit_anchor_formal','scripts.close_anchor_resources']:
        assert callable(importlib.import_module(module).main)
