#!/usr/bin/env python3
"""A60 sole queue: bounded children, actual OS waits, and fail-closed cleanup."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import shutil
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from echo_routing.temporal.proposal_resources import ExclusiveQueue, totals, remaining_limits
from echo_routing.temporal.structured_guard import process_sample, storage_bytes
from echo_routing.temporal.anchor_cache import durable_directory, fsync_directory


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as handle:
        json.dump(value, handle, sort_keys=True, allow_nan=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def GPU_processes():
    if not shutil.which('nvidia-smi'):
        return []
    output = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True)
    return [int(line.strip()) for line in output.splitlines() if line.strip()]


def stage_exit_status(record):
    return 0 if record['observed_worker_exit_code'] == 0 and not record.get('termination_reason') and not record.get('guardian_exception') else 1


def execute(stage, command, run, pub, spent, declared_wall_seconds, GPU_job,
            storage_probe=None, GPU_probe=None):
    run, pub = Path(run), Path(pub)
    durable_directory(run)
    durable_directory(pub)
    storage_probe = storage_probe or (lambda: storage_bytes([run, pub]))
    GPU_probe = GPU_probe or GPU_processes
    with ExclusiveQueue(run / 'exclusive_supervisor.lock'):
        active = run / 'active_process.json'
        if active.exists():
            raise RuntimeError('existing queue claim requires independent reconciliation')
        # Caller snapshots may predate another completed locked stage. Merge
        # actual receipt identities inside this lock, including external costs.
        if 'receipt_records' not in spent:
            raise ValueError('aggregate without constituent receipt identities cannot be safely merged')
        merged = {row['receipt_id']: row for row in spent['receipt_records']}
        ledger = run / 'processes.jsonl'
        locked_rows = [json.loads(line) for line in ledger.read_text().splitlines()] if ledger.exists() else []
        external = pub / 'local_cpu_receipts.json'
        if external.exists():
            locked_rows += json.loads(external.read_text())['receipts']
        for row in locked_rows:
            key = row['receipt_id']
            if key in merged and merged[key] != row:
                raise ValueError('same receipt identity has conflicting locked costs')
            merged[key] = row
        spent = totals(list(merged.values()))
        if GPU_probe():
            raise RuntimeError('an existing GPU owner prevents a second queue')
        bounds = remaining_limits(spent, time.time(), potential_GPU_context=True)
        effective = min(float(declared_wall_seconds), bounds['stage_wall_seconds'] - 10.)
        if not math.isfinite(effective) or effective <= 0:
            raise RuntimeError('no admitted shared resource room remains')
        attempt = 1
        while (run / f'process_{stage}_{attempt}.json').exists():
            attempt += 1
        child_before = resource.getrusage(resource.RUSAGE_CHILDREN)
        own_before = resource.getrusage(resource.RUSAGE_SELF)
        start = time.time()
        child, code, reason, exception, start_ticks = None, None, None, None, None
        context_occupied, max_storage = False, 0
        environment = {**os.environ, 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'}
        if not GPU_job:
            environment['CUDA_VISIBLE_DEVICES'] = ''
        def child_limit():
            limit = max(1, math.ceil(min(effective, bounds['remaining_CPU_seconds'] - 5)))
            resource.setrlimit(resource.RLIMIT_CPU, (limit, limit + 5))
        with (run / f'{stage}_{attempt}.stdout.log').open('w') as stdout, (run / f'{stage}_{attempt}.stderr.log').open('w') as stderr:
            try:
                write_json(active, {'stage': stage, 'attempt': attempt, 'state': 'durable_launch_intent',
                                    'model_pid': None, 'supervisor_pid': os.getpid(),
                                    'measurement_host': socket.gethostname(), 'started_epoch': start,
                                    'command_sha256': hashlib.sha256(json.dumps(command).encode()).hexdigest(),
                                    'GPU_job': bool(GPU_job)})
                child = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=stdout, stderr=stderr, preexec_fn=child_limit)
                sample = process_sample(child.pid)
                start_ticks = sample['start_ticks'] if sample else None
                write_json(active, {'stage': stage, 'attempt': attempt, 'model_pid': child.pid,
                                    'state': 'running_PID_bound',
                                    'supervisor_pid': os.getpid(), 'measurement_host': socket.gethostname(),
                                    'started_epoch': start, 'process_start_ticks': start_ticks,
                                    'GPU_job': bool(GPU_job), 'CUDA_visibility_hidden': not GPU_job})
                last_storage, last_GPU = 0., 0.
                while True:
                    try:
                        code = child.wait(timeout=.05)
                        break
                    except subprocess.TimeoutExpired:
                        pass
                    now = time.time()
                    sample = process_sample(child.pid)
                    if sample and start_ticks is not None and sample['start_ticks'] != start_ticks:
                        raise RuntimeError('worker PID start identity changed')
                    if now - last_GPU >= 3:
                        owners = GPU_probe()
                        context_occupied |= child.pid in owners
                        if any(pid != child.pid for pid in owners):
                            reason = 'unexpected_foreign_GPU_owner'
                        last_GPU = now
                    own = resource.getrusage(resource.RUSAGE_SELF)
                    waited = resource.getrusage(resource.RUSAGE_CHILDREN)
                    current_CPU = ((sample['CPU_seconds'] if sample else 0.)
                                   + own.ru_utime + own.ru_stime - own_before.ru_utime - own_before.ru_stime
                                   + waited.ru_utime + waited.ru_stime - child_before.ru_utime - child_before.ru_stime)
                    if now - start >= effective:
                        reason = 'effective_stage_or_shared_deadline_wall_limit'
                    if current_CPU >= bounds['remaining_CPU_seconds'] - 5:
                        reason = 'actual_worker_supervisor_CPU_limit'
                    if (GPU_job or context_occupied) and now - start >= bounds['remaining_GPU_seconds_before_reserve'] - 10:
                        reason = 'actual_GPU_occupation_limit_preserving_closure_reserve'
                    if now - last_storage >= 10:
                        max_storage = max(max_storage, storage_probe())
                        if max_storage >= 5458422374:
                            reason = 'inherited_tracked_path_storage_upper_limit'
                        last_storage = now
                    if reason:
                        break
            except BaseException as error:
                exception = repr(error)
                reason = 'guardian_exception_fail_closed'
            finally:
                if child is not None:
                    if child.poll() is None:
                        child.terminate()
                        try:
                            code = child.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            child.kill()
                            code = child.wait()
                    else:
                        code = child.wait()
        finish = time.time()
        own_after = resource.getrusage(resource.RUSAGE_SELF)
        child_after = resource.getrusage(resource.RUSAGE_CHILDREN)
        cc = child_after.ru_utime + child_after.ru_stime - child_before.ru_utime - child_before.ru_stime
        sc = own_after.ru_utime + own_after.ru_stime - own_before.ru_utime - own_before.ru_stime
        row = {'stage': stage, 'attempt': attempt, 'receipt_id': f'{stage}:{attempt}',
               'measurement_host': socket.gethostname(), 'model_pid': child.pid if child else None,
               'supervisor_pid': os.getpid(), 'process_start_ticks': start_ticks,
               'started_epoch': start, 'finished_epoch': finish, 'full_process_wall_seconds': finish - start,
               'measured_CPU_core_seconds': cc + sc, 'measured_child_and_waited_probe_CPU_seconds': cc,
               'measured_supervisor_CPU_seconds': sc, 'GPU_job': bool(GPU_job),
               'CUDA_context_occupied': bool(context_occupied), 'CUDA_visibility_hidden': not GPU_job,
               'observed_worker_exit_code': code, 'child_exit_code_directly_waited': child is not None,
               'OS_status_inferred_from_manifest': False, 'termination_reason': reason,
               'guardian_exception': exception, 'maximum_sampled_tracked_storage_upper_bytes': max_storage,
               'effective_stage_wall_limit_seconds': effective, 'limits_at_launch': bounds,
               'CPU_scope': 'SELF plus waited CHILDREN including worker and probes; post-snapshot JSON bookkeeping unmetered'}
        write_json(run / f'process_{stage}_{attempt}.json', row)
        with (run / 'processes.jsonl').open('a') as handle:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + '\n')
            handle.flush()
            os.fsync(handle.fileno())
        fsync_directory(run)
        write_json(pub / f'process_{stage}_{attempt}.json', row)
        if active.exists():
            active.unlink()
            fsync_directory(run)
        return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage')
    parser.add_argument('--seconds', type=float, required=True)
    parser.add_argument('--GPU', action='store_true')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    run, pub = ROOT / 'runs/a60_jepa_proposals_v1', ROOT / 'docs/results/2026-11-03_jepa_proposals'
    rows = [json.loads(line) for line in (run / 'processes.jsonl').read_text().splitlines()] if (run / 'processes.jsonl').exists() else []
    row = execute(args.stage, args.command, run, pub, totals(rows), args.seconds, args.GPU)
    print(json.dumps(row), flush=True)
    return stage_exit_status(row)


if __name__ == '__main__':
    raise SystemExit(main())
