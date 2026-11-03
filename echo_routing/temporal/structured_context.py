"""Conservative GPU-context occupation overlay; raw CPU-backend receipts stay intact."""
def classify_context_records(records,guard=None):
    out=[];merged=False
    if guard is not None and guard['guardian_model_pid'] in {r['model_pid'] for r in records}:raise ValueError('auxiliary guard PID collides with measured worker')
    for row in records:
        r=dict(row)
        if r['stage'] in ['head_check','formal']:
            r['raw_stage_GPU_job']=r.get('raw_stage_GPU_job',r['GPU_job']);r['GPU_job']=True;r['GPU_charge_basis']='conservativefullCPUAdamWprocesswall;acceleratorgraphcapturehealthcheckcreatesCUDAcontext;formal426MiBobserved'
        if guard is not None and r['stage']=='formal' and r['model_pid']==guard['formal_model_pid'] and r['start_epoch']==guard['formal_start_epoch']:
            if merged:raise ValueError('auxiliary context guard must bind exactly one formal PID')
            base=r.get('raw_measured_process_CPU_seconds',r['process_CPU_seconds']);r['raw_measured_process_CPU_seconds']=base;r['process_CPU_seconds']=base+guard['process_CPU_seconds'];r['additional_measured_context_guard_CPU_seconds']=guard['process_CPU_seconds'];r['additional_context_guard_PID']=guard['guardian_model_pid'];merged=True
        out.append(r)
    if guard is not None and not merged:raise ValueError('auxiliary context guard has no unique completed formal worker receipt')
    return out
