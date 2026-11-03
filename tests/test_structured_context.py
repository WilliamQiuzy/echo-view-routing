import importlib,copy

def test_CPU_optimizer_context_is_charged_once_without_rewriting_raw_or_double_CPU():
    m=importlib.import_module('echo_routing.temporal.structured_context');raw=[{'stage':'formal','model_pid':1,'measurement_host':'host','exit_code':0,'start_epoch':10.,'end_epoch':20.,'full_process_wall_seconds':10.,'process_CPU_seconds':8.,'GPU_job':False}];before=copy.deepcopy(raw);guard={'formal_model_pid':1,'formal_start_epoch':10.,'guardian_model_pid':2,'process_CPU_seconds':1.5};out=m.classify_context_records(raw,guard);assert raw==before and out[0]['GPU_job'] is True and out[0]['raw_stage_GPU_job'] is False and out[0]['process_CPU_seconds']==9.5 and m.classify_context_records(out,guard)==out
    costs=importlib.import_module('echo_routing.temporal.structured_resources').closed_costs(out);assert costs['GPU_job_wall_seconds']==10 and costs['summed_measured_CPU_receipt_seconds']==9.5 and costs['closed_receipts']==1
