import importlib

def test_forecast_counts_every_source_role_original_stress_and_all_head_packages():
    m=importlib.import_module('echo_routing.temporal.structured_budget');roles={k:{'positions':p,'sequences':n,'a08_stride2_windows':a,'a08_stride5_windows':s,'jepa_chunks':j} for k,p,n,a,s,j in [('fit',200,10,100,40,40),('inner_cal',60,3,30,12,20),('selection',40,2,20,8,10),('calibration',44,2,22,9,11),('stress_selection',20,2,10,4,4),('stress_calibration',24,2,12,5,5)]};r=m.workload(roles,[1000,1500]);assert r['a08_dense_windows']==194 and r['a08_original_stress_windows']==9 and r['new_JEPA_chunks']==49
    assert r['formal_optimizer_updates']==48000 and r['head_training_valid_positions']==10000 and r['inner_MAP_positions']==1440 and r['main_MAP_positions']==1344 and r['stress_MAP_positions']==352

def test_all_new_and_inherited_resource_caps_are_conjunctive():
    m=importlib.import_module('echo_routing.temporal.structured_budget');b={'new_GPUjob_h':16,'cumulative_GPUjob_h':24,'prior_GPUjob_h':14.3839654494,'new_CPUcore_h':48,'cumulative_CPUcore_h':72,'prior_summed_measured_CPUreceipt_h':14.2846472567,'prior_setup_budget_allowance_h':.25,'new_peak_bytes':20_000_000_000,'cumulative_peak_upper_bytes':40_000_000_000,'prior_tracked_storage_upper_bytes':32_827_925_533,'new_deadline':200,'cumulative_deadline':180}
    assert m.caps_fit(b,5,20,1_000_000_000,170)
    for gpu,cpu,storage,end in [(10,20,1,170),(5,49,1,170),(5,20,8_000_000_000,170),(5,20,1,181)]:assert not m.caps_fit(b,gpu,cpu,storage,end)
