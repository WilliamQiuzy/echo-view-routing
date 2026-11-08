#!/usr/bin/env python3
"""A63 sole queue: bounded children, actual OS waits, and fail-closed cleanup."""
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
import uuid,re

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from echo_routing.temporal.selected_risk_budget import ExclusiveQueue, totals, remaining_limits, PRIOR_STORAGE_UPPER_BYTES, STORAGE_CAP_BYTES
def process_sample(pid):
    p=Path('/proc')/str(int(pid))/'stat'
    try:
        raw=p.read_text();fields=raw[raw.rfind(')')+2:].split();hz=os.sysconf('SC_CLK_TCK')
        return {'start_ticks':int(fields[19]),'CPU_seconds':(int(fields[11])+int(fields[12]))/hz}
    except (FileNotFoundError,ProcessLookupError):return None

def storage_bytes(paths):
    seen=set();total=0
    for root in paths:
        root=Path(root)
        if not root.exists():continue
        for p in [root,*root.rglob('*')]:
            if p.is_symlink():raise ValueError('new tracked namespace symlink forbidden')
            if p.is_file():
                st=p.stat();key=(st.st_dev,st.st_ino)
                if key not in seen:seen.add(key);total+=st.st_size
    return total
from echo_routing.temporal.anchor_cache import durable_directory, fsync_directory


def validate_path(path):
    p=Path(os.path.abspath(path))
    for v in [p,*p.parents]:
        if v.is_symlink():raise ValueError('supervisor output ancestor/leaf symlink forbidden')
    return p

def validate_stage(stage):
    if not isinstance(stage,str) or re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]{0,63}',stage) is None:raise ValueError('plain safe A63 stage basename required')
    return stage

def write_json(path, value):
    path = validate_path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('x') as handle:
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
    validate_stage(stage);run,pub=validate_path(run),validate_path(pub)
    durable_directory(run)
    durable_directory(pub)
    storage_probe = storage_probe or (lambda: storage_bytes([run, pub])+300000000)
    GPU_probe = GPU_probe or GPU_processes
    validate_path(run/'exclusive_supervisor.lock');validate_path(run/'active_process.json')
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
        locked_rows += [json.loads(p.read_text()) for p in pub.glob('entry_CPU_*.json')]
        for row in locked_rows:
            key = row['receipt_id']
            if key in merged and merged[key] != row:
                raise ValueError('same receipt identity has conflicting locked costs')
            merged[key] = row
        spent = totals(list(merged.values()))
        initial_storage=storage_probe();storage_cap=min(STORAGE_CAP_BYTES,40000000000-PRIOR_STORAGE_UPPER_BYTES)
        if not math.isfinite(initial_storage) or initial_storage<0 or initial_storage>=storage_cap:raise RuntimeError('launch refused: existing tracked storage upper already exceeds cap')
        initial_GPU_owners = GPU_probe()
        if GPU_job and initial_GPU_owners:
            raise RuntimeError('an existing GPU owner prevents a second GPU queue')
        bounds = remaining_limits(spent, time.time(), potential_GPU_context=GPU_job)
        effective = min(float(declared_wall_seconds), bounds['stage_wall_seconds'] - 10.)
        if not math.isfinite(effective) or effective <= 0:
            raise RuntimeError('no admitted shared resource room remains')
        attempt = 1
        while (run / f'process_{stage}_{attempt}.json').exists():
            attempt += 1
        child_before = resource.getrusage(resource.RUSAGE_CHILDREN)
        own_before = resource.getrusage(resource.RUSAGE_SELF)
        start = time.time();nonce=uuid.uuid4().hex;boot=(Path('/proc/sys/kernel/random/boot_id').read_text().strip() if Path('/proc/sys/kernel/random/boot_id').exists() else 'local_fixture')
        child, code, reason, exception, start_ticks = None, None, None, None, None
        context_occupied, max_storage = False, initial_storage
        durable_directory(run/'tmp')
        environment = {**os.environ, 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1','PYTHONDONTWRITEBYTECODE':'1','TMPDIR':str((run/'tmp').resolve()),'MPLCONFIGDIR':str((run/'tmp/matplotlib').resolve())}
        if not GPU_job:
            environment['CUDA_VISIBLE_DEVICES'] = ''
        def child_limit():
            limit = max(1, math.ceil(min(effective, bounds['remaining_CPU_seconds'] - 5)))
            resource.setrlimit(resource.RLIMIT_CPU, (limit, limit + 5))
        stdout_path=validate_path(run/f'{stage}_{attempt}.stdout.log');stderr_path=validate_path(run/f'{stage}_{attempt}.stderr.log')
        if stdout_path.exists() or stderr_path.exists():raise RuntimeError('unreceipted existing attempt logs require explicit reconciliation; no overwrite')
        with stdout_path.open('x') as stdout,stderr_path.open('x') as stderr:
            try:
                write_json(active, {'stage': stage, 'attempt': attempt, 'nonce':nonce,'boot_id':boot,'state': 'durable_launch_intent',
                                    'model_pid': None, 'supervisor_pid': os.getpid(),
                                    'measurement_host': socket.gethostname(), 'started_epoch': start,
                                    'command_sha256': hashlib.sha256(json.dumps(command).encode()).hexdigest(),
                                    'GPU_job': bool(GPU_job)})
                child = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=stdout, stderr=stderr, preexec_fn=child_limit)
                sample = process_sample(child.pid)
                start_ticks = sample['start_ticks'] if sample else None
                write_json(active, {'stage': stage, 'attempt': attempt,'nonce':nonce,'boot_id':boot,'model_pid': child.pid,
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
                        if GPU_job and any(pid != child.pid for pid in owners):
                            reason = 'unexpected_foreign_GPU_owner'
                        if not GPU_job and child.pid in owners:
                            reason = 'unexpected_CPU_stage_CUDA_context'
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
                        if max_storage >= min(STORAGE_CAP_BYTES,40000000000 - PRIOR_STORAGE_UPPER_BYTES):
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
        try:max_storage=max(max_storage,storage_probe())
        except BaseException as error:exception=repr(error);reason='final_storage_probe_failed'
        if active.exists() and (json.loads(active.read_text()).get('nonce')!=nonce or json.loads(active.read_text()).get('supervisor_pid')!=os.getpid()):exception='active claim ownership changed';reason='claim_cleanup_refused'
        finish = time.time()
        own_after = resource.getrusage(resource.RUSAGE_SELF)
        child_after = resource.getrusage(resource.RUSAGE_CHILDREN)
        cc = child_after.ru_utime + child_after.ru_stime - child_before.ru_utime - child_before.ru_stime
        sc = own_after.ru_utime + own_after.ru_stime - own_before.ru_utime - own_before.ru_stime
        if max_storage>=storage_cap:reason='final_actual_tracked_storage_upper_limit'
        if cc+sc>=bounds['remaining_CPU_seconds']:reason='final_actual_worker_supervisor_CPU_limit'
        if finish-start>=effective or finish>=bounds['deadline_epoch']:reason='final_effective_stage_or_deadline_wall_limit'
        if (GPU_job or context_occupied) and finish-start>=bounds['remaining_GPU_seconds_before_reserve']:reason='final_actual_GPU_occupation_limit'
        row = {'stage': stage, 'attempt': attempt,'nonce':nonce,'boot_id':boot,'receipt_id': f'{stage}:{attempt}',
               'measurement_host': socket.gethostname(), 'model_pid': child.pid if child else None,
               'supervisor_pid': os.getpid(), 'process_start_ticks': start_ticks,
               'started_epoch': start, 'finished_epoch': finish, 'full_process_wall_seconds': finish - start,
               'measured_CPU_core_seconds': cc + sc, 'measured_child_and_waited_probe_CPU_seconds': cc,
               'measured_supervisor_CPU_seconds': sc, 'GPU_job': bool(GPU_job),
               'CUDA_context_occupied': bool(context_occupied), 'CUDA_visibility_hidden': not GPU_job,
               'preexisting_foreign_GPU_owners': initial_GPU_owners,
               'foreign_GPU_wall_not_charged_as_A63': True,
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
        if active.exists() and json.loads(active.read_text()).get('nonce')==nonce and json.loads(active.read_text()).get('supervisor_pid')==os.getpid():
            active.unlink()
            fsync_directory(run)
        return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage');parser.add_argument('--execute',action='store_true');parser.add_argument('--approved-plan-sha')
    parser.add_argument('--seconds', type=float, required=True)
    parser.add_argument('--GPU', action='store_true')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    from scripts.selected_risk_common import authorize
    if not authorize(args):return 0
    if args.command[:1]==['--']:args.command=args.command[1:]
    if not args.command:raise ValueError('explicit bounded child command required')
    validate_stage(args.stage)
    from echo_routing.temporal.selected_risk_contracts import stage_spec
    entries,_=stage_spec(args.stage)
    if args.stage.endswith('_terminal') or len(args.command)<2 or Path(args.command[1]).resolve() not in {(ROOT/p).resolve() for p in entries} or '--execute' not in args.command[2:]:raise ValueError('one exact authorized new stage entry required; collector is standalone, never nested under same queue')
    run, pub = ROOT / 'runs/a63_selected_action_risk_v1', ROOT / 'docs/results/2026-11-05_selected_action_risk'
    rows = [json.loads(line) for line in (run / 'processes.jsonl').read_text().splitlines()] if (run / 'processes.jsonl').exists() else []
    row = execute(args.stage, args.command, run, pub, totals(rows), args.seconds, args.GPU)
    print(json.dumps(row), flush=True)
    return stage_exit_status(row)


def meter_entry(callback, run, pub):
    """Meter prelaunch rejection and metadata outside execute's already charged region."""
    run,pub=validate_path(run),validate_path(pub)
    start=time.time();a=resource.getrusage(resource.RUSAGE_SELF);b=resource.getrusage(resource.RUSAGE_CHILDREN)
    ledger=run/'processes.jsonl'
    before=[json.loads(x) for x in ledger.read_text().splitlines()] if ledger.exists() else []
    known={r['receipt_id'] for r in before};code=1;error=None
    try:
        code=callback()
    except SystemExit as e:
        code=int(e.code) if isinstance(e.code,int) else 1
    except BaseException as e:
        error=repr(e);code=1
    end=time.time();aa=resource.getrusage(resource.RUSAGE_SELF);bb=resource.getrusage(resource.RUSAGE_CHILDREN)
    after=[json.loads(x) for x in ledger.read_text().splitlines()] if ledger.exists() else []
    fresh=[r for r in after if r['receipt_id'] not in known and r.get('measurement_host')==socket.gethostname() and r.get('supervisor_pid')==os.getpid()]
    cpu=aa.ru_utime+aa.ru_stime+bb.ru_utime+bb.ru_stime
    excluded=sum(r['measured_CPU_core_seconds'] for r in fresh)
    if cpu+1e-6<excluded:raise ValueError('entry CPU cannot undercut already measured stage CPU')
    durable_directory(pub)
    value={'receipt_id':f'entry:{socket.gethostname()}:{os.getpid()}:{start}',
       'stage':'supervisor_entry_metadata','measurement_host':socket.gethostname(),'model_pid':os.getpid(),
       'supervisor_pid':None,'started_epoch':start,'finished_epoch':end,'full_process_wall_seconds':end-start,
       'measured_CPU_core_seconds':max(0.,cpu-excluded),'total_entry_SELF_waited_CHILDREN_CPU_seconds':cpu,
       'already_charged_stage_CPU_seconds_subtracted':excluded,'GPU_job':False,'CUDA_context_occupied':False,
       'observed_worker_exit_code':None,'planned_CLI_exit_status':code,'entry_exception':error,
       'new_worker_started':any(r['model_pid'] is not None for r in fresh),
       'scope':'whole fresh supervisor SELF+waited CHILDREN through snapshot minus execute regions; includes pre-entry imports/startup; SELF actual OS null; receipt serialization/exit tool tail unknown'}
    write_json(pub/f'entry_CPU_{os.getpid()}_{int(start*1e6)}.json',value)
    return code

if __name__ == '__main__':
    if '--execute' not in sys.argv:raise SystemExit(main())
    raise SystemExit(meter_entry(main,ROOT/'runs/a63_selected_action_risk_v1',ROOT/'docs/results/2026-11-05_selected_action_risk'))
