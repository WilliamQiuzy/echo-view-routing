import importlib,copy
import pytest

def module():
    try:m=importlib.import_module('echo_routing.temporal.selected_risk_cost_units')
    except ModuleNotFoundError:m=None
    assert m is not None,'explicit nonnegative measured-profile cost allocation absent'
    return m

def test_fixed_group_projection_retains_setup_and_every_volume_with_domination():
    m=module();terms=('N','T','T2','C','C2','TC','K','KT','CR');rows=[]
    for name,g in [('empty',dict(N=1,T=10,T2=100,C=0,C2=0,TC=0,K=0,KT=0,CR=0)),('envelope',dict(N=1,T=10,T2=100,C=4,C2=16,TC=40,K=4,KT=40,CR=4)),('long_K',dict(N=1,T=10,T2=100,C=1,C2=1,TC=10,K=5,KT=50,CR=3)),('legal_RLE',dict(N=1,T=10,T2=100,C=3,C2=9,TC=30,K=4,KT=40,CR=12))]:rows.append({'name':name,'geometry':g,'backend_seconds':2.})
    p={'profiles':rows,'backend_coefficients':dict(N=2.,T=.2,T2=.02,C=.5,C2=.125,TC=.05,K=.4,KT=.04,CR=2/12)};original=copy.deepcopy(p);new,proof=m.project_coefficients(p)
    assert p==original and set(new)==set(terms) and new['N']==p['backend_coefficients']['N']
    assert all(v>=0 for v in new.values()) and len(proof['all_four_profile_recounts'])==4
    assert all(r['dominated_before_original_margin2'] for r in proof['all_four_profile_recounts'])
    assert proof['extra_uncertainty_factor']==2 and new['CR']==p['backend_coefficients']['CR']

def test_missing_or_nonfinite_measurements_fail_closed():
    m=module()
    with pytest.raises(ValueError):m.project_coefficients({'profiles':[],'backend_coefficients':{}})

def test_actual_four_original_public_profiles_all_recount_before_margin():
    import json
    from pathlib import Path
    p=json.loads(Path('docs/results/2026-11-05_selected_action_risk/cost_profile.json').read_text());m=module();co,proof=m.project_coefficients(p)
    assert {x['profile'] for x in proof['all_four_profile_recounts']}=={'empty','envelope','long_K','legal_RLE'}
    for row in proof['all_four_profile_recounts']:
        assert sum(row['nonnegative_component_seconds'].values())==pytest.approx(row['sum_before_margin'])
        assert row['sum_before_margin']>=row['actual_measured_backend_seconds']-1e-9
