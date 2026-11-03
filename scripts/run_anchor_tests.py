#!/usr/bin/env python3
"""A59 local focused checks with actual subprocess PID and SELF+CHILDREN CPU."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import socket
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage')
    parser.add_argument('tests', nargs='+')
    args = parser.parse_args()
    if not args.stage.replace('_', '').isalnum():
        raise ValueError('plain receipt stage required')
    root = Path(__file__).resolve().parents[1]
    environment = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'}
    command = [sys.executable, '-m', 'pytest', '-q', *args.tests]
    child_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    self_before = resource.getrusage(resource.RUSAGE_SELF)
    start = time.time()
    process = subprocess.Popen(command, cwd=root, env=environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    output = process.communicate()[0]
    finish = time.time()
    child_after = resource.getrusage(resource.RUSAGE_CHILDREN)
    self_after = resource.getrusage(resource.RUSAGE_SELF)
    child_cpu = child_after.ru_utime + child_after.ru_stime - child_before.ru_utime - child_before.ru_stime
    self_cpu = self_after.ru_utime + self_after.ru_stime - self_before.ru_utime - self_before.ru_stime
    value = {'stage': args.stage, 'command': command, 'measurement_host': socket.gethostname(),
             'model_pid': process.pid, 'supervisor_pid': os.getpid(), 'started_epoch': start,
             'finished_epoch': finish, 'full_process_wall_seconds': finish - start,
             'observed_worker_exit_code': process.returncode, 'GPU_job': False,
             'CUDA_context_occupied': False, 'CUDA_visibility_hidden_before_child_import': True,
             'measured_child_CPU_seconds': child_cpu, 'measured_supervisor_CPU_seconds': self_cpu,
             'measured_CPU_core_seconds': child_cpu + self_cpu,
             'scope': 'local CPU synthetic software checks; no media or formal experiment updates',
             'output': output, 'output_sha256': hashlib.sha256(output.encode()).hexdigest()}
    path = root / 'docs/results/2026-11-03_anchor_edits' / f'{args.stage}.json'
    if path.exists():
        raise ValueError('never overwrite an earlier test receipt')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    print(output)
    print(json.dumps({k: v for k, v in value.items() if k != 'output'}))
    return process.returncode


if __name__ == '__main__':
    raise SystemExit(main())
