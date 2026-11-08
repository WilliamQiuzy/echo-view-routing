"""A63 inherited account plus disjoint new actual receipts; forecasts never spend."""
import math,copy
from echo_routing.temporal.safety_resources import ExclusiveQueue
PRIOR_GPU_HOURS=18.657838855
PRIOR_CPU_ACCOUNTING_HOURS=20.231205854833334
PRIOR_STORAGE_UPPER_BYTES=38650409858
T0=1793853840
DEADLINE_EPOCH=1793897040
GPU_CAP_SECONDS=5400
CPU_CAP_SECONDS=28800
STORAGE_CAP_BYTES=1000000000
GPU_CLOSURE_RESERVE_SECONDS=7200
CPU_CLOSURE_RESERVE_SECONDS=3600
STORAGE_RESERVE_BYTES=128000000
WALL_CLOSURE_RESERVE_SECONDS=7200
def checked(v):
    if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<0:raise ValueError('finite nonnegative resource quantity required')
    return v


def totals(rows):
    cpu=gpu=account=allowance=0.;seen=set();identities=set();intervals=[];fail=unknown=unknown_cpu=0
    for r in rows:
        start=checked(r['started_epoch']);end=checked(r['finished_epoch']);wall=checked(r['full_process_wall_seconds'])
        if end<start or not math.isclose(end-start,wall,abs_tol=.01,rel_tol=0):raise ValueError('ordered fully charged interval required')
        if r['receipt_id'] in seen:raise ValueError('duplicate receipt identity')
        seen.add(r['receipt_id']);identity=(r.get('measurement_host'),r.get('model_pid'),start)
        if identity[0] is not None and identity[1] is not None:
            if identity in identities:raise ValueError('duplicate host/PID/start cost')
            identities.add(identity)
        measured=r['measured_CPU_core_seconds']
        if measured is None:
            if r.get('measurement_kind')!='administrative_partial_receipt':raise ValueError('unknown actual stage CPU cannot be hidden as an allowance')
            extra=checked(r['CPU_accounting_allowance_seconds']);account+=extra;allowance+=extra;unknown_cpu+=1
        else:cpu+=checked(measured);account+=checked(measured)
        if r['GPU_job'] or r['CUDA_context_occupied']:gpu+=wall;intervals.append((start,end))
        unknown+=r['observed_worker_exit_code'] is None;fail+=r['observed_worker_exit_code'] not in [None,0] or bool(r.get('termination_reason'))
    intervals.sort()
    if any(a[1]>b[0]+.001 for a,b in zip(intervals,intervals[1:])):raise ValueError('overlapping occupied GPU jobs cannot be guessed/deduplicated')
    return {'new_GPU_wall_seconds':gpu,'new_measured_CPU_seconds':cpu,'new_CPU_accounting_seconds':account,'administrative_CPU_allowance_seconds':allowance,'unknown_CPU_receipts':unknown_cpu,'cumulative_GPU_hours':PRIOR_GPU_HOURS+gpu/3600,'cumulative_CPU_accounting_hours':PRIOR_CPU_ACCOUNTING_HOURS+account/3600,'closed_receipts':len(rows),'failed_stages':fail,'unknown_observed_worker_OS_statuses':unknown,'receipt_records':copy.deepcopy(rows),'historical_CPU_allowance_overlap_and_unmetered_gaps_retained':True}


def forecast(spent,GPU_seconds,CPU_seconds,incremental_peak_bytes,now,remaining_wall_seconds):
    for v in [GPU_seconds,CPU_seconds,incremental_peak_bytes,now,remaining_wall_seconds]:checked(v)
    newgpu=spent['new_GPU_wall_seconds']+GPU_seconds;newcpu=spent.get('new_CPU_accounting_seconds',spent['new_measured_CPU_seconds'])+max(CPU_seconds,CPU_CLOSURE_RESERVE_SECONDS)
    gpu=PRIOR_GPU_HOURS+(newgpu+GPU_CLOSURE_RESERVE_SECONDS)/3600;cpu=PRIOR_CPU_ACCOUNTING_HOURS+newcpu/3600;storage=PRIOR_STORAGE_UPPER_BYTES+incremental_peak_bytes
    return {'all_caps_fit':newgpu<=GPU_CAP_SECONDS and newcpu<=CPU_CAP_SECONDS and incremental_peak_bytes<=STORAGE_CAP_BYTES and gpu<=24 and cpu<=72 and storage<=40000000000 and now>=T0 and now+remaining_wall_seconds<=DEADLINE_EPOCH,'forecast_incremental_GPU_seconds':newgpu,'forecast_incremental_CPU_seconds_including_closure':newcpu,'forecast_cumulative_GPU_hours_including_reserve':gpu,'forecast_cumulative_CPU_accounting_hours':cpu,'forecast_cumulative_tracked_storage_upper_bytes':storage,'forecast_incremental_tracked_storage_upper_bytes':incremental_peak_bytes,'forecast_end_epoch':now+remaining_wall_seconds,'unused_forecast_reserves_are_actual_spend':False,'GPU_reserve_seconds':7200,'CPU_closure_seconds':3600,'storage_closure_reserve_bytes_inside_increment':128000000,'wall_closure_reserve_seconds_inside_window':7200,'deadline_epoch':DEADLINE_EPOCH}

def remaining_limits(spent,now,potential_GPU_context=True,closing=False):
    checked(now);reserve=0 if closing else CPU_CLOSURE_RESERVE_SECONDS
    cpu=min(CPU_CAP_SECONDS-spent.get('new_CPU_accounting_seconds',spent['new_measured_CPU_seconds']),(72-spent['cumulative_CPU_accounting_hours'])*3600)-reserve
    gpu=min(GPU_CAP_SECONDS-spent['new_GPU_wall_seconds'],(24-spent['cumulative_GPU_hours'])*3600-GPU_CLOSURE_RESERVE_SECONDS)
    return {'stage_wall_seconds':max(0.,min(cpu,DEADLINE_EPOCH-now-(0 if closing else WALL_CLOSURE_RESERVE_SECONDS),gpu if potential_GPU_context else float('inf'))),'remaining_CPU_seconds':max(0.,cpu),'remaining_GPU_seconds_before_reserve':max(0.,gpu),'deadline_epoch':DEADLINE_EPOCH,'closing':closing}
