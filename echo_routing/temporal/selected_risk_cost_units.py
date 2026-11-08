"""Explicit alternative empirical cost allocation, not a physical global bound.

N retains its FULL empty/setup/IO coefficient. T/T² share their variable empty
cost; C/C²/TC share one envelope cost; K/KT share one long-K cost; CR is intact.
Every term stays nonnegative, all four original actual profiles must still be
dominated BEFORE the unchanged extra factor2. No GT/head outcome interface.
"""
import copy,math
from echo_routing.temporal.selected_risk_forecast import TERMS,nonnegative
from echo_routing.temporal.selected_risk_refined_forecast import remaining as prior_remaining
DIVISORS={'N':1,'T':2,'T2':2,'C':3,'C2':3,'TC':3,'K':2,'KT':2,'CR':1}
SOURCE={'N':'empty','T':'empty','T2':'empty','C':'envelope','C2':'envelope','TC':'envelope','K':'long_K','KT':'long_K','CR':'legal_RLE'}

def volume(g):return {k:nonnegative(g.get('all_'+k,g[k])) for k in TERMS}

def project_coefficients(profile):
    rows={r['name']:r for r in profile['profiles']}
    if len(profile['profiles'])!=4 or set(rows)!={'empty','envelope','long_K','legal_RLE'} or set(profile['backend_coefficients'])!=set(TERMS):raise ValueError('all four original measured profiles and all nine terms required')
    old=profile['backend_coefficients'];new={}
    for k in TERMS:
        source=rows[SOURCE[k]];expected=nonnegative(source['backend_seconds'])/max(1.,volume(source['geometry'])[k])
        if not math.isclose(nonnegative(old[k]),expected,rel_tol=1e-12,abs_tol=1e-15):raise ValueError('original coefficient not exact frozen measured component assignment')
        new[k]=old[k]/DIVISORS[k]
    recounts=[]
    for name,row in rows.items():
        v=volume(row['geometry']);parts={k:v[k]*new[k] for k in TERMS};total=sum(parts.values());measured=nonnegative(row['backend_seconds']);ok=total+1e-9>=measured
        if not ok:raise ValueError('alternative projection does not dominate original actual profile before uncertainty: '+name)
        recounts.append({'profile':name,'all_original_volume_values':v,'actual_measured_backend_seconds':measured,'nonnegative_component_seconds':parts,'sum_before_margin':total,'sum_after_unchanged_margin2':2*total,'dominated_before_original_margin2':ok,'scope_includes_original_empty_setup_and_full_cache_constructor_write_read':True})
    return new,{'all_four_profile_recounts':recounts,'original_coefficients':old,'new_coefficients':new,'fixed_divisors':DIVISORS,'setup_IO_N_coefficient_unchanged':True,'extra_uncertainty_factor':2,'all_nine_terms_nonnegative_and_monotone':True,'empirical_projection_not_guarantee_or_hardware_lower_bound':True}

def remaining(profile,shape,manifest,**kwargs):
    coef,proof=project_coefficients(profile);p=copy.deepcopy(profile);p['backend_coefficients']=coef;out=prior_remaining(p,shape,manifest,**kwargs)
    out.update({'estimator':'cost_component_units_v3','actual_four_profile_projection_proof':proof,'original_wide_profile_and_failed_admission_bytes_retained':True,'original_scope_floors_density2_and_all_closing_reserves_preserved':True});return out
