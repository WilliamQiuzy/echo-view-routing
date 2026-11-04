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
