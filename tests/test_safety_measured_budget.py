import copy,importlib
import pytest


def test_measured_forecast_requires_complete_probe_and_keeps_fullworkload():
    m=importlib.import_module('echo_routing.temporal.safety_measured_budget');roles={r:{'sequences':120,'positions':2400,'max_length':20,'a08_stride5_windows':600,'jepa_chunks':120} for r in m.EXPECTED_ROLES};roles['selection'].update(sequences=541,native=284);roles['calibration'].update(sequences=472,native=237)
    g={'N':541,'T':2400,'C':200,'C2':2000,'TC':4000,'all_C':220,'all_TC':4400,'K':300,'native_N':284,'native_T':1000,'native_T2':20000};observed={r:{**g,'N':roles[r]['sequences'],'native_N':roles[r].get('native',0)} for r in m.OUTER_ROLES if not r.startswith('a61_confirmation_')}
    checks={'timings':{r:{'seconds_per_update_including_plan_batch_forward_backward_trace_and_full_checkpoint_IO':.01} for r in ['utility','risk']}};costs={'structural_cases':[{'name':'dense_C','T':701,'C':1051,'max_seconds':1.3},{'name':'long_K','T':701,'K':700,'max_seconds':.04}],'role_geometry':{r:{'sequences':roles[r]['sequences'],'T_sum':2400,'C_sum':200,'K_sum':300,'TC_sum':4000} for r in roles if r not in ['fit'] and not r.startswith('a61_confirmation_')},'largest_full_checkpoint_bytes':90000,'maximum_engineering_trace_row_bytes':5000}
    timings={}
    for T in [16,701]:
        timings['task_'+str(T)]={'T':T,'N':640,'evaluation_prepare_max_seconds':.1,'paired5000_seconds':.1,'native_sessions':320,'multi':160,'pairs':160};timings['dense_'+str(T)]={'T':T,'C':max(1,T),'all_C':max(1,T),'TC':T*T,'all_TC':T*T,'C2':T*T,'K':T,'descriptor_max_seconds':.01,'cache_write_seconds':.01,'cache_read_seconds':.01,'oracle_max_seconds':.01};timings['native_'+str(T)]={'T':T,'N':1,'seconds':.001}
    timings['long_701']={**timings['dense_701'],'C':1,'K':175};timings['empty_descriptor']={'seconds':.00001};timings['probability_IO']={'T':701,'N':16,'write_seconds':.01,'read_seconds':.01};timings['integer_solver']={'seconds':.1,'streams':120,'options_each':1052};timings.update({n:{'generation_IO_and_audit_seconds':.1} for n in ['trace_utility','trace_risk']});timings['state40']={'seconds':.1};timings['six_figures']={'seconds':1.}
    measured={'passed':True,'timings':timings,'old_observable_geometry':observed,'new_optimizer_updates':0,'new_neural_forwards':0,'A61_outer_head_outcomes_used':False,'outer_label_values_used':False};out=m.complete_measured(roles,checks,costs,measured,1_000_000)
    assert out['formal_updates_remaining']==60000 and out['mandatory_remaining_matrix']=={'metric_rows':314,'comparisons':240,'bootstrap_replicates_each':5000,'oracles':60,'old_G_reproductions':16,'native_cells':56}
    assert out['phase_CPU_forecasts_seconds']['all240paired5000']>=240*.1*2 and out['whole_workflow_closure_integration_wall_room_seconds']==1200
    changed=copy.deepcopy(measured);changed['timings']['dense_701']['descriptor_max_seconds']*=2
    assert m.complete_measured(roles,checks,costs,changed,1_000_000)['remaining_CPU_seconds']>out['remaining_CPU_seconds']
    bad=copy.deepcopy(measured);bad['passed']=False
    with pytest.raises(ValueError):m.complete_measured(roles,checks,costs,bad,1_000_000)
    bad=copy.deepcopy(measured);bad['timings']['task_701']['native_sessions']=10
    with pytest.raises(ValueError):m.complete_measured(roles,checks,costs,bad,1_000_000)
