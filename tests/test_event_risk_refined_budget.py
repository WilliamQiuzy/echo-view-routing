import copy
import pytest


def fixture():
    geom={'N':1,'T':100,'T2':10000,'C':100,'C2':10000,'TC':10000,'K':100,'KT':10000,'CR':4000}
    fields=['event_inference_seconds','fixed_inference_seconds','six_gate_summary_seconds','three_oracle_options_coverage_solve_seconds','raw_cache_init_write_read_seconds']
    profiles=[]
    for name in ['empty','envelope','legal_RLE','long_K']:
        g=dict(geom)
        if name=='empty':g.update(C=0,C2=0,TC=0,K=0,KT=0,CR=0)
        profiles.append({'name':name,'geometry':g,**{f:.01 if name=='empty' else 1. for f in fields}})
    return {'profiles':profiles}


def test_factorized_terms_monotone_and_every_profile_dominated():
    from echo_routing.temporal.event_risk_refined_budget import coefficients,estimate
    p=fixture();c=coefficients(p,'six_gate_summary_seconds')
    assert set(c)=={'N','T','T2','C','C2','TC','K','KT','CR'} and all(v>=0 for v in c.values())
    for row in p['profiles']:assert estimate(row['geometry'],c)>=2*row['six_gate_summary_seconds']
    g=dict(p['profiles'][1]['geometry']);before=estimate(g,c)
    for key in c:
        h=dict(g);h[key]+=1
        assert estimate(h,c)>=before
    # Increasing matching RLE does not silently raise the per-T coefficient.
    other=copy.deepcopy(p);other['profiles'][2]['six_gate_summary_seconds']*=20
    assert coefficients(other,'six_gate_summary_seconds')['T']==c['T']


def test_missing_profile_or_changed_original_geometry_rejects():
    from echo_routing.temporal.event_risk_refined_budget import coefficients,validate_geometry
    p=fixture();p['profiles'].pop()
    with pytest.raises(ValueError):coefficients(p,'six_gate_summary_seconds')
    old={'geometry':{'selection':{'N':1,'T':100,'C':2,'TC':200,'K':2,'input_sha256':'x'}}}
    new={'geometry':{'selection':{'N':1,'T':100,'C':3,'TC':300,'K':3,'input_sha256':'x'}}}
    with pytest.raises(ValueError):validate_geometry(old,new)


def test_complete_remaining_scope_and_floors_are_retained():
    import json
    from pathlib import Path
    from echo_routing.temporal.event_risk_refined_budget import remaining
    from echo_routing.temporal.event_risk_cost_bounds import aggregate_bounds
    from echo_routing.temporal.event_risk_data import ROLES
    pub=Path(__file__).resolve().parents[1]/'docs/results/2026-11-04_anchor_context_risk'
    original=json.loads((pub/'cost_profile.json').read_text());manifest=json.loads((pub/'input_manifest.json').read_text());refined=fixture();refined['geometry']={}
    for role,g in original['geometry'].items():
        maximum=manifest['roles'][role]['max_length']
        refined['geometry'][role]={**g,'T2':g['T']*maximum,'C2':g['C']*max(1,g['max_C']),'KT':g['K']*maximum,'CR':g['C']*(2*maximum+3)}
    refined['future_structural_bounds']={}
    for role in ROLES:
        r=manifest['roles'][role];q,remainder=divmod(r['positions']-r['max_length'],r['sequences']-1)
        lengths=[r['max_length']]+[q+1]*remainder+[q]*(r['sequences']-1-remainder)
        refined['future_structural_bounds'][role]=aggregate_bounds(lengths)
    refined['integer_solver']={'streams':150,'options_each_excluding_noop':100000,'seconds':.1}
    v=remaining(original,refined,manifest)
    assert v['scope']=={'formal_updates':24000,'engineering_updates':72,'outer_rows':156,'inner_rows':26,'comparisons':240,'replicates':5000,'new_full_oracles':6,'old_readonly_oracles':30,'support':48,'native':24,'base_native':2,'regression':16,'fullstates':16}
    assert v['phases']['resource_history_exit_collector_and_handoff']==3600
    assert v['phases']['model_and_calibration_independent_audits']==600
    assert v['phases']['six_figures_PDF_QA_report']==600
    assert v['factors']=={'synthetic_backends':2,'formal_update':1.5}
    assert v['original_caps_and_deadline_unchanged']
    assert v['measured_backend_units']=={'new_full_oracle_endpoints':6,'safety_domains_per_measured_unit':3,'role_passes_per_half_including_independent_recount':2,'Ntop_support_endpoints':48,'additional_pooled_solves':54,'group_summary_passes_for_support_native_regression':4}
