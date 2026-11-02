#!/usr/bin/env python3
"""A52 single queue supervisor; one owned process."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import resource
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/a54_dense_covariance_v1"
COMMANDS = {
 "intake": ("scripts/train_dense_cooccurrence.py","intake",120,False),
 "extract": ("scripts/train_dense_cooccurrence.py","extract",180,True),
 "train": ("scripts/train_dense_cooccurrence.py","train",120,False),
 "readout": ("scripts/train_dense_cooccurrence.py","readout",120,False),
 "audit": ("scripts/audit_dense_cooccurrence.py","final",120,False),
 "checks": ("scripts/audit_dense_cooccurrence.py","checks",120,False),
}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("stage", choices=COMMANDS); args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    owner_lock = (OUT / "exclusive_supervisor.lock").open("a+")
    try: fcntl.flock(owner_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as e:
        owner_lock.close(); raise RuntimeError("another supervisor owns the queue") from e
    active = OUT / "active_process.json"
    if active.exists():
        old = json.loads(active.read_text())
        try:
            os.kill(old["model_pid"], 0)
        except ProcessLookupError:
            pass
        else:
            raise RuntimeError("existing owned worker alive; refuse second executor")
    attempt = 1
    while (OUT / f"process_{args.stage}_{attempt}.json").exists(): attempt += 1
    script, stage, limit, gpu = COMMANDS[args.stage]
    before = resource.getrusage(resource.RUSAGE_CHILDREN); start = time.time()
    def bound(): resource.setrlimit(resource.RLIMIT_CPU, (limit, limit + 5))
    with (OUT / f"{args.stage}_{attempt}.stdout.log").open("w") as stdout, (OUT / f"{args.stage}_{attempt}.stderr.log").open("w") as stderr:
        child = subprocess.Popen([str(ROOT / ".venv/bin/python"), script, stage], cwd=ROOT,
                                 stdout=stdout, stderr=stderr, preexec_fn=bound)
        active.write_text(json.dumps({"stage": args.stage, "attempt": attempt, "model_pid": child.pid,
            "supervisor_pid": os.getpid(), "start_epoch": start, "GPU_job": gpu}) + "\n")
        timeout = False
        try: code = child.wait(timeout=limit + 30)
        except subprocess.TimeoutExpired:
            timeout = True; child.terminate()
            try: code = child.wait(timeout=10)
            except subprocess.TimeoutExpired: child.kill(); code = child.wait()
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    receipt = {"stage": args.stage, "attempt": attempt, "model_pid": child.pid, "supervisor_pid": os.getpid(),
        "start_epoch": start, "end_epoch": time.time(), "exit_code": code, "timed_out": timeout,
        "full_process_wall_seconds": time.time() - start,
        "process_CPU_seconds": after.ru_utime + after.ru_stime - before.ru_utime - before.ru_stime,
        "GPU_job": gpu, "max_RSS_KB": after.ru_maxrss}
    (OUT / f"process_{args.stage}_{attempt}.json").write_text(json.dumps(receipt, indent=2) + "\n")
    with (OUT / "processes.jsonl").open("a") as f: f.write(json.dumps(receipt) + "\n")
    active.unlink(); print(json.dumps(receipt), flush=True)
    raise SystemExit(code)


if __name__ == "__main__": main()
