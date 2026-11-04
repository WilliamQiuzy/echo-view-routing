import importlib
import pytest


def test_source_package_retains_all_attempt_costs_and_separates_committed_from_last_new_units():
    m=importlib.import_module('echo_routing.temporal.safety_package');records=[{'source':'a08','role':'fresh','complete':True,'committed_sequences':2,'new_neural_units':3}];geometry={('a08','fresh'):[2,3]};stages=[{'stage':'cache_outer','attempt':1,'receipt_id':'cache_outer:1','GPU_job':True,'CUDA_context_occupied':True,'full_process_wall_seconds':10.,'measured_CPU_core_seconds':8.,'observed_worker_exit_code':1,'child_exit_code_directly_waited':True},{'stage':'cache_outer','attempt':2,'receipt_id':'cache_outer:2','GPU_job':True,'CUDA_context_occupied':True,'full_process_wall_seconds':7.,'measured_CPU_core_seconds':6.,'observed_worker_exit_code':0,'child_exit_code_directly_waited':True}]
    r=m.source_costs(records,geometry,stages)
    assert r['committed_unique_neural_units']=={'a08':5,'jepa':0} and r['last_manifest_invocation_new_neural_units']['a08']==3
    assert r['source_cache_GPU_jobwall_seconds']==17 and r['source_cache_CPU_seconds']==14
    assert r['recorded_source_attempts']==2 and r['actual_total_source_model_forward_calls'] is None
    assert r['source_python_model_forward_calls_for_committed_sequences']['a08']==2
    with pytest.raises(ValueError):m.source_costs(records,geometry,stages+[stages[0]])
    whole=[{**records[0],'new_neural_units':5}];single=m.source_costs(whole,geometry,[stages[1]])
    assert single['physical_call_count_exact_for_recorded_attempts'] and single['actual_total_source_model_forward_calls']['a08']==2
