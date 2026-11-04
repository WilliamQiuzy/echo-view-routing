"""A60-only write paths and immutable source/input identities."""
import hashlib
import json
from pathlib import Path

import numpy as np

from echo_routing.temporal.anchor_cache import atomic_json, durable_directory, fsync_directory, sha

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/a60_jepa_proposals_v1'
PUB = ROOT / 'docs/results/2026-11-03_jepa_proposals'
CFG = ROOT / 'configs/experiments/jepa_proposals_v1.json'


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    atomic_json(path, value)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def verify_pin(pin):
    path = ROOT / pin['path']
    if sha(path) != pin['sha256']:
        raise ValueError('frozen source/input file changed: ' + pin['path'])
    return path


def verify_freeze(name):
    value = read(RUN / name)
    for path, expected in value['objects'].items():
        if sha(ROOT / path) != expected:
            raise ValueError('frozen A60 code/config/input changed: ' + path)
    return value


def sequences(role):
    manifest = read(PUB / 'input_manifest.json')
    return np.load(verify_pin(manifest['roles'][role]), allow_pickle=True).tolist()


def save_array(path, value):
    path = Path(path)
    durable_directory(path.parent)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('xb') as handle:
        np.save(handle, value, allow_pickle=True)
        handle.flush()
        import os
        os.fsync(handle.fileno())
    temporary.replace(path)
    fsync_directory(path.parent)


def array_pin(path):
    return {'path': str(Path(path).relative_to(ROOT)), 'sha256': sha(path)}


def actual_receipts():
    records = [json.loads(x) for x in (RUN/'processes.jsonl').read_text().splitlines()] if (RUN/'processes.jsonl').exists() else []
    external = read(PUB/'local_cpu_receipts.json')
    for path,expected in external['file_pins'].items():
        if sha(ROOT/path)!=expected:raise ValueError('raw external measured CPU receipt changed')
    records += external['receipts']
    records += [read(p) for p in PUB.glob('entry_CPU_*.json')]
    return records
