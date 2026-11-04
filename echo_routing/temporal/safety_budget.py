"""A61 prospective complete-workflow forecast; estimates are never actual charges."""
import math
import numpy as np


def structural_cases(T):
    if not isinstance(T,int) or T<4:raise ValueError('declared maximum future original-axis length required')
    def prob(y):
        p=np.full((len(y),5),.05);p[np.arange(len(y)),y]=.8;return p
    b=np.zeros(T,np.int64);a=np.zeros(T,np.int64);a[1::2]=1;j=2+np.arange(T)%2
    n=T-T%2;base=np.r_[np.arange(n)%2,np.full(T-n,2)];raw=base.copy();raw[:n]=1-base[:n]
    return {'dense_C':(prob(a),prob(j),b),'long_K':(prob(raw),prob(base),base)}


def geometry_bound(pin):
    T=int(pin['max_length']);positions=int(pin['positions']);n=int(pin['sequences'])
    # Old mismatch components <=ceil(T/2), J runs <=T; changed multiset <=2T.
    return {'sequences':n,'T_sum':positions,'C_sum':1.5*positions+n,'K_sum':2*positions,'TC_sum':1.5*positions*T+positions}


def estimated_inference(g,costs):
    dense=next(r for r in costs['structural_cases'] if r['name']=='dense_C');long=next(r for r in costs['structural_cases'] if r['name']=='long_K')
    return 2*(dense['max_seconds']*(g['C_sum']/dense['C']+g['TC_sum']/(dense['T']*dense['C']))+long['max_seconds']*(g['K_sum']/long['K']+g['T_sum']/long['T']))+.001*g['sequences']


def complete(roles,checks,costs,retained_bytes,formal_pending=True):
    utility=checks['timings']['utility']['seconds_per_update_including_plan_batch_forward_backward_trace_and_full_checkpoint_IO'];risk=checks['timings']['risk']['seconds_per_update_including_plan_batch_forward_backward_trace_and_full_checkpoint_IO']
    training=1.5*(48000*utility+12000*risk)+120 if formal_pending else 0.
    inner=max(3600.,24*estimated_inference(costs['role_geometry']['inner_cal'],costs)+8*928*.02) if formal_pending else 0.
    outerroles=[r for r in roles if r not in ['fit','inner_cal']]
    inference=sum(estimated_inference(costs['role_geometry'].get(r,geometry_bound(roles[r])),costs) for r in outerroles)
    readout=max(7200.,8*inference+1200.) # all8 utility/risk packages plus4historical controls/audits/IO
    fresh=[roles[r] for r in roles if r.startswith('a61_confirmation_')]
    source_GPU=120+2*(sum(r['a08_stride5_windows'] for r in fresh)*.026734+sum(r['jepa_chunks'] for r in fresh)*.073776)
    source_CPU=2*source_GPU
    phases={'risk_and_utility_training_including_durable_IO':training,'all24_inner_E_N_and8_calibration':inner,'fresh240_dualproducer_CPU':source_CPU,'all314_rows_and56_native_cubes_privatecache_IO':readout,'all240paired5000_and60oracles16reproductions_independent_recount':3600.,'plots_package_independent_audits_actualclosure_and_integration_room':3600.}
    source_payloads=2*sum(r['positions'] for r in fresh)*5*4+480*4096
    fullstates=40*costs['largest_full_checkpoint_bytes'];atomic=2*costs['largest_full_checkpoint_bytes']
    trace=math.ceil(1.5*costs['maximum_engineering_trace_row_bytes']*60000)
    geometry={r:costs['role_geometry'].get(r,geometry_bound(roles[r])) for r in outerroles};T=sum(v['T_sum'] for v in geometry.values());C=sum(v['C_sum'] for v in geometry.values());N=sum(v['sequences'] for v in geometry.values())
    outercache=math.ceil(2*8*(128*T+256*C+8192*N)+50_000_000)
    ig=costs['role_geometry']['inner_cal'];innercache=math.ceil(24*(64*ig['T_sum']+128*ig['C_sum']+8192*ig['sequences']))
    storage={'current_retained_tracked_paths':int(retained_bytes),'all30milestone_plus10latest_full_states':int(fullstates),'atomic_checkpoint_coexistence':int(atomic),'all60000physical_trace_rows':int(trace),'fresh_dualsource_payloads_journals_atomic':int(source_payloads),'all_joint_policy_private_scores_labels_historicalcontrols':outercache,'all24inner_fullscore_label_records_plus_calibration':innercache,'reports_figures_review_and_general_margin':500_000_000}
    return {'formal_updates_remaining':60000 if formal_pending else 0,'fullstate_checkpoint_files':40,'mandatory_remaining_matrix':{'metric_rows':314,'comparisons':240,'bootstrap_replicates_each':5000,'oracles':60,'old_G_reproductions':16,'native_cells':56},'remaining_CPU_seconds':sum(phases.values()),'remaining_GPU_seconds':source_GPU,'remaining_serial_wall_seconds':sum(phases.values())+source_GPU+1200,'phase_CPU_forecasts_seconds':phases,'incremental_tracked_path_peak_upper_bytes':sum(storage.values()),'storage_components_upper_bytes':storage,'all_source_NN_forward_counts':{'a08_stride5_windows':sum(r['a08_stride5_windows'] for r in fresh),'jepa_chunks':sum(r['jepa_chunks'] for r in fresh),'genuine_joined_sequences_each_producer':240},'basis':'actual worst FIT20step complete IO timings ×allupdates ×1.5; observedA60source perunit ×newexactunits ×2+startup; empirical structuralT/C/K/TC model ×2 and explicit allstage floors; conservative forecast,not proven hardware runtime bound or spent cost','future_risk_and_utility_cached_inference_is_not_fullsystem_latency':True}
