"""A62 explicit new window and incremental caps; inherited costs never reset."""
import math,copy
from echo_routing.temporal.safety_resources import ExclusiveQueue
PRIOR_GPU_HOURS=18.488818130029575
PRIOR_CPU_ACCOUNTING_HOURS=19.78341831
PRIOR_STORAGE_UPPER_BYTES=37904365798
T0=1793824361
DEADLINE_EPOCH=1793867561
GPU_CAP_SECONDS=1800
CPU_CAP_SECONDS=21600
STORAGE_CAP_BYTES=1250000000
GPU_CLOSURE_RESERVE_SECONDS=7200
CPU_CLOSURE_RESERVE_SECONDS=3600
CPU_UNCERTAINTY_BUFFER_SECONDS=900
STORAGE_RESERVE_BYTES=500000000


def checked(v):
    if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<0:raise ValueError('finite nonnegative resource quantity required')
    return v


def totals(rows):
    cpu=gpu=0.;seen=set();identities=set();intervals=[];fail=unknown=0
    for r in rows:
        start=checked(r['started_epoch']);end=checked(r['finished_epoch']);wall=checked(r['full_process_wall_seconds'])
        if end<start or not math.isclose(end-start,wall,abs_tol=.01,rel_tol=0):raise ValueError('ordered fully charged interval required')
        if r['receipt_id'] in seen:raise ValueError('duplicate receipt identity')
        seen.add(r['receipt_id']);identity=(r.get('measurement_host'),r.get('model_pid'),start)
        if identity[0] is not None and identity[1] is not None:
            if identity in identities:raise ValueError('duplicate host/PID/start cost')
            identities.add(identity)
        cpu+=checked(r['measured_CPU_core_seconds'])
        if r['GPU_job'] or r['CUDA_context_occupied']:gpu+=wall;intervals.append((start,end))
        unknown+=r['observed_worker_exit_code'] is None;fail+=r['observed_worker_exit_code'] not in [None,0] or bool(r.get('termination_reason'))
    intervals.sort()
    if any(a[1]>b[0]+.001 for a,b in zip(intervals,intervals[1:])):raise ValueError('overlapping occupied GPU jobs cannot be guessed/deduplicated')
    return {'new_GPU_wall_seconds':gpu,'new_measured_CPU_seconds':cpu,'cumulative_GPU_hours':PRIOR_GPU_HOURS+gpu/3600,'cumulative_CPU_accounting_hours':PRIOR_CPU_ACCOUNTING_HOURS+cpu/3600,'closed_receipts':len(rows),'failed_stages':fail,'unknown_observed_worker_OS_statuses':unknown,'receipt_records':copy.deepcopy(rows),'historical_CPU_allowance_overlap_and_unmetered_gaps_retained':True}


def forecast(spent,GPU_seconds,CPU_seconds,incremental_peak_bytes,now,remaining_wall_seconds):
    for v in [GPU_seconds,CPU_seconds,incremental_peak_bytes,now,remaining_wall_seconds]:checked(v)
    complete_CPU=max(CPU_seconds,CPU_CLOSURE_RESERVE_SECONDS);newgpu=spent['new_GPU_wall_seconds']+GPU_seconds;newcpu=spent['new_measured_CPU_seconds']+complete_CPU
    gpu=PRIOR_GPU_HOURS+(newgpu+GPU_CLOSURE_RESERVE_SECONDS)/3600;cpu=PRIOR_CPU_ACCOUNTING_HOURS+(newcpu+CPU_UNCERTAINTY_BUFFER_SECONDS)/3600;storage=PRIOR_STORAGE_UPPER_BYTES+incremental_peak_bytes+STORAGE_RESERVE_BYTES
    passed=(newgpu<=GPU_CAP_SECONDS and newcpu<=CPU_CAP_SECONDS and incremental_peak_bytes<=STORAGE_CAP_BYTES and gpu<=24 and cpu<=72 and storage<=40000000000 and now>=T0 and now+remaining_wall_seconds<=DEADLINE_EPOCH)
    return {'all_caps_fit':passed,'forecast_incremental_GPU_seconds':newgpu,'forecast_incremental_CPU_seconds_including_closure':newcpu,'forecast_cumulative_GPU_hours_including_reserve':gpu,'forecast_cumulative_CPU_accounting_hours_including_buffer':cpu,'forecast_cumulative_tracked_storage_upper_bytes_including_reserve':storage,'forecast_end_epoch':now+remaining_wall_seconds,'includes_GPU_reserve_seconds':GPU_CLOSURE_RESERVE_SECONDS,'includes_CPU_closure_reserve_seconds':CPU_CLOSURE_RESERVE_SECONDS,'separate_CPU_uncertainty_buffer_seconds':CPU_UNCERTAINTY_BUFFER_SECONDS,'storage_forecast_reserve_bytes':STORAGE_RESERVE_BYTES,'unused_forecast_reserves_are_actual_spend':False,'deadline_epoch':DEADLINE_EPOCH}


def remaining_limits(spent,now,potential_GPU_context=True,closing=False):
    checked(now);cpu=min(CPU_CAP_SECONDS-spent['new_measured_CPU_seconds'],(72-spent['cumulative_CPU_accounting_hours'])*3600)-(0 if closing else CPU_CLOSURE_RESERVE_SECONDS)
    gpu=min(GPU_CAP_SECONDS-spent['new_GPU_wall_seconds'],(24-spent['cumulative_GPU_hours'])*3600-GPU_CLOSURE_RESERVE_SECONDS)
    return {'stage_wall_seconds':max(0.,min(cpu,DEADLINE_EPOCH-now,gpu if potential_GPU_context else float('inf'))),'remaining_CPU_seconds':max(0.,cpu),'remaining_GPU_seconds_before_reserve':max(0.,gpu),'deadline_epoch':DEADLINE_EPOCH,'closing':closing}
