"""A60 accounting within the unchanged original cumulative research envelope."""
import fcntl
import math
import copy
from pathlib import Path


PRIOR_GPU_HOURS = 18.3349858850903
PRIOR_MEASURED_CPU_HOURS = 18.253671176388888
PRIOR_CPU_ALLOWANCE_HOURS = .75
PRIOR_STORAGE_UPPER_BYTES = 34541577626
DEADLINE_EPOCH = 1793820844.1596537
CLOSURE_GPU_RESERVE_HOURS = 2.


def nonnegative(value, name):
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f'finite nonnegative {name} required')
    return value


def totals(receipts):
    cpu, gpu, failures, unknown = 0., 0., 0, 0
    intervals, receipt_ids, process_ids = [], set(), set()
    legacy = 0
    for row in receipts:
        start = nonnegative(row['started_epoch'], 'receipt start')
        finish = nonnegative(row['finished_epoch'], 'receipt finish')
        wall = nonnegative(row['full_process_wall_seconds'], 'full process wall')
        if finish < start or not math.isclose(wall, finish - start, rel_tol=0, abs_tol=.01):
            raise ValueError('unordered or undercharged full process interval')
        receipt_id = row['receipt_id']
        if receipt_id in receipt_ids:
            raise ValueError('duplicate receipt identity')
        receipt_ids.add(receipt_id)
        if row.get('measurement_host') and row.get('model_pid') is not None:
            process_id = (row['measurement_host'], row['model_pid'], start)
            if process_id in process_ids:
                raise ValueError('duplicate actual host/PID/start charge')
            process_ids.add(process_id)
        else:
            legacy += 1
        cpu += nonnegative(row['measured_CPU_core_seconds'], 'summed measured stage CPU')
        if row['GPU_job'] or row['CUDA_context_occupied']:
            intervals.append((start, finish))
            gpu += wall
        status = row['observed_worker_exit_code']
        unknown += status is None
        failures += status not in [None, 0] or bool(row.get('termination_reason'))
    intervals.sort()
    if any(left[1] > right[0] + .001 for left, right in zip(intervals, intervals[1:])):
        raise ValueError('GPU occupation intervals overlap; charges cannot be deduplicated by guessing')
    return {'new_GPU_wall_seconds': gpu, 'new_measured_CPU_seconds': cpu,
            'cumulative_GPU_hours': PRIOR_GPU_HOURS + gpu / 3600,
            'cumulative_measured_CPU_hours': PRIOR_MEASURED_CPU_HOURS + cpu / 3600,
            'prior_CPU_allowance_hours': PRIOR_CPU_ALLOWANCE_HOURS,
            'cumulative_CPU_accounting_hours': PRIOR_MEASURED_CPU_HOURS + PRIOR_CPU_ALLOWANCE_HOURS + cpu / 3600,
            'closed_receipts': len(receipts), 'failed_stages': failures,
            'unknown_observed_worker_OS_statuses': unknown, 'GPU_intervals_disjoint': True,
            'complete_host_PID_start_provenance': not legacy, 'receipt_based_identity_count': legacy,
            'receipt_records': copy.deepcopy(receipts),
            'prior_CPU_overlap_uncertainty_retained': True,
            'scope': 'sum of measured receipts plus separate completed historical allowances; not unique whole-host CPU'}


def forecast(spent, GPU_seconds, CPU_seconds, incremental_peak_bytes, now, remaining_wall_seconds):
    for value, name in [(GPU_seconds, 'remaining GPU forecast'), (CPU_seconds, 'remaining CPU forecast'),
                        (incremental_peak_bytes, 'new tracked-path storage upper'), (now, 'current epoch'),
                        (remaining_wall_seconds, 'remaining complete workflow wall forecast')]:
        nonnegative(value, name)
    gpu = spent['cumulative_GPU_hours'] + GPU_seconds / 3600 + CLOSURE_GPU_RESERVE_HOURS
    cpu = spent['cumulative_CPU_accounting_hours'] + CPU_seconds / 3600
    storage = PRIOR_STORAGE_UPPER_BYTES + incremental_peak_bytes
    # Complete remaining wall includes actual closure; GPU reserve is not a shorter CPU deadline.
    projected_end = now + remaining_wall_seconds
    return {'all_caps_fit': gpu <= 24 and cpu <= 72 and storage <= 40000000000 and projected_end <= DEADLINE_EPOCH,
            'forecast_cumulative_GPU_hours_including_reserve': gpu,
            'forecast_cumulative_CPU_accounting_hours': cpu,
            'forecast_cumulative_tracked_path_storage_upper_bytes': storage,
            'forecast_end_epoch_including_closure_room': projected_end,
            'closure_GPU_reserve_hours': CLOSURE_GPU_RESERVE_HOURS,
            'unused_reserve_is_spent_cost': False, 'inherited_deadline_epoch': DEADLINE_EPOCH}


def remaining_limits(spent, now, potential_GPU_context=True, closing=False):
    nonnegative(now, 'current epoch')
    reserve = 0. if closing else CLOSURE_GPU_RESERVE_HOURS
    gpu_seconds = (24 - spent['cumulative_GPU_hours'] - reserve) * 3600
    cpu_seconds = (72 - spent['cumulative_CPU_accounting_hours']) * 3600
    deadline_seconds = DEADLINE_EPOCH - now
    wall = min(cpu_seconds, deadline_seconds, gpu_seconds if potential_GPU_context else float('inf'))
    return {'stage_wall_seconds': max(0., wall), 'remaining_CPU_seconds': max(0., cpu_seconds),
            'remaining_GPU_seconds_before_reserve': max(0., gpu_seconds),
            'earlier_deadline_epoch': DEADLINE_EPOCH, 'closing': bool(closing)}


class ExclusiveQueue:
    """Use the existing shared lock path for the entire supervised stage/closure."""
    def __init__(self, path):
        self.path, self.handle = Path(path), None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open('a+')
        try:
            fcntl.flock(self.handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            self.handle.close()
            self.handle = None
            raise
        return self

    def __exit__(self, *exception):
        fcntl.flock(self.handle, fcntl.LOCK_UN)
        self.handle.close()
        self.handle = None
