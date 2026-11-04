from echo_routing.temporal.proposal_readout_plan import plan

def test_exact124_144_32_no_selection_and_common_models():
    p=plan();assert len(p['rows'])==124 and len(p['comparisons'])==144 and len(p['oracle_cells'])==32
    assert len({(r['table'],r['half'],r['system']) for r in p['rows']})==124
    assert len({r['physical_model'] for r in p['rows'] if r['role']=='policy'})==4
    assert not p['inner_selection'] and p['fixed_step']==6000
    assert {r['domain'] for r in p['oracle_cells']}=={'old','expanded'}
