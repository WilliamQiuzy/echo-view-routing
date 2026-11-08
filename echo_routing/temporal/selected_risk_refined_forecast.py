"""Prospective full-pass forecast for the reviewed lossless persistence layout.

Old broad profile/failed v1 admission remain untouched. Only newly measured
cached-kind training and explicitly encoded future storage change estimates.
Every original nontraining phase/floor/cap/endpoint and density2 is retained.
"""
import copy,math
from echo_routing.temporal.selected_risk_forecast import remaining as original_remaining
from echo_routing.temporal.selected_risk_record_storage import storage_bound


def refine_complete(base,profile,shape,sessions):
    out=copy.deepcopy(base);phases=out['phases'];t=shape['training'];ratio=shape['trace_ratio_forecast']
    if not isinstance(ratio,(int,float)) or isinstance(ratio,bool) or not math.isfinite(ratio) or ratio<.5 or ratio>1:raise ValueError('frozen synthetic archive ratio forecast >=.5 and <=1 required')
    if any(not isinstance(t[k],(int,float)) or isinstance(t[k],bool) or not math.isfinite(t[k]) or t[k]<0 for k in ['utility_update_seconds','risk_update_seconds']):raise ValueError('actual nonnegative cached full-batch timings required')
    phases['aux48000_updates']=2*24000*(t['utility_update_seconds']+t['risk_update_seconds'])
    phases['lossless_12_trace_archive_and_roundtrip']=2*12*max(r['archive_and_exact_roundtrip_seconds'] for r in shape['trace_profiles'])
    phases['typed_record_IO_independent_recounts']=60.
    cpu=sum(phases.values());closing=phases['resource_OS_independent_audit_publication']
    if closing<3600:raise ValueError('original closing CPU floor cannot be reduced')
    out['CPU_seconds']=cpu;out['remaining_wall_seconds']=cpu-closing+max(7200.,closing)
    native=profile['geometry']['native'];mixed=[out['projected_geometry'][r] for r in ['F0_mixed','F1_mixed']];g={k:native[k]+sum(r[k] for r in mixed) for k in ['N','T','C','K']};C,K,N,T=g['C'],g['K'],g['N'],g['T']
    # One raw concatenation/offsets per auxiliary fold; no redundant position
    # slice list or persistent normalized205. The held-score phase is after
    # eight final audited/sealed auxiliary models, so its cache cannot coexist
    # with an in-progress auxiliary-data atomic temporary.
    aux=math.ceil((8*109+8*4+1024)*C+(8*47+8)*K+128*N+4*T)
    # First seed stores raw109+raw96+its twoC6. Second seed stores its own C6
    # plus an immutable first-seed payload SHA; every original feature bit is
    # checked. Action SHA replaces redundant JSON; fixed NPZ overhead<=4096.
    joint=math.ceil(8*(217+12)*C+2*4096*N)
    formal_targets=math.ceil(37*C+65536) # key digest/index/target only;205 inRAM
    feature_peak=max(2*aux,aux+joint+formal_targets)
    c=out['storage_components'];oldtrace=c.pop('72000_trace_rows_empirical_index_aware_forecast');active=6000*(max(r['max_line_bytes'] for r in shape['trace_profiles'])+1024)
    c.pop('features_joint_cache_and_event_copies');c.pop('selected_digest_records_4096_bytes_each')
    c['minimal_aux_plus_compact_shared_cache_phase_peak']=feature_peak
    c['retained_archive_plus_one_active_raw_and_atomic_peak']=math.ceil(ratio*oldtrace+active)
    records=int(4*(sum(r['N'] for r in mixed)+native['C']));bound=storage_bound(records,sessions);c['selected_typed_records_atomic_upper']=bound['atomic_coexistence_upper_bytes']
    out['storage_upper_bytes']=math.ceil(sum(c.values()));out.update(estimator='lossless_layout_cached_kind_v2',original_empirical_trace_bytes=oldtrace,trace_archive_ratio_forecast=ratio,active_trace_upper_bytes=active,feature_layout_components={'minimal_aux_datasets':aux,'minimal_aux_atomic_peak':2*aux,'complete_two_seed_compact_cache':joint,'formal_targets_only_no_persistent205':formal_targets,'held_score_and_aux_prepare_temporaries_cannot_coexist':True},selected_record_bound=bound,wall_closing_CPU_is_inside_original_7200wall_not_an_extra_serial_hour=True,CPU_closing_3600_and_wall_closing_7200_preserved=True,CPU_core_seconds_and_elapsed_seconds_are_separately_forecast=True,wall_serial_single_thread_profiles_plus_IO_and_closing_envelope=True,original_backend_terms_and_complete_scope_retained=True,old_failed_admission_preserved=True,actual_teacher_geometry_and_full_predraw_before_aux_required=True,all_future_layouts_are_prospective_not_claimed_executed=True)
    return out


def remaining(profile,shape,manifest,*,old_source_wall,old_a08_units,old_jepa_units,current_upper,sessions):
    original=original_remaining(profile,manifest,old_source_wall=old_source_wall,old_a08_units=old_a08_units,old_jepa_units=old_jepa_units,current_upper=current_upper)
    return refine_complete(original,profile,shape,sessions)
