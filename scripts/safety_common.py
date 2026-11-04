"""A61-only write paths and immutable source/input identities."""
import hashlib
import json
from pathlib import Path

import numpy as np

from echo_routing.temporal.anchor_cache import atomic_json, durable_directory, fsync_directory, sha

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/a61_action_safety_ranking_v1'
PUB = ROOT / 'docs/results/2026-11-04_action_safety_ranking'
CFG = ROOT / 'configs/experiments/action_safety_ranking_v1.json'


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
            raise ValueError('frozen A61 code/config/input changed: ' + path)
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


def reader(source,role):
    from echo_routing.temporal.anchor_cache import ProbabilityCache
    manifest=read(PUB/'input_manifest.json');qs=sequences(role)
    if role not in manifest['source_cache_roles']:raise ValueError('only previously qualified read-only source role requested')
    row=manifest['source_cache_roles'][role][source];folder=ROOT/row['path']
    if sha(folder/'committed.jsonl')!=row['journal_sha256'] or row['input_sha256']!=manifest['roles'][role]['sha256']:raise ValueError('source cache/order/input identity changed')
    return ProbabilityCache(folder,row['identity'],[len(q['refs']) for q in qs],readonly=True)


def role_values(role):
    from echo_routing.temporal.baselines.registry import decode_with_prob
    qs=sequences(role);caches={s:reader(s,role) for s in ['a08','jepa']}
    values={s:[c.load(i) for i in range(len(qs))] for s,c in caches.items()}
    params=read(CFG)['source']['a08']['original_B3']
    baseline=[decode_with_prob('b3',p,{'decode':{'b3':params}})[0] for p in values['a08']]
    return qs,values['a08'],values['jepa'],baseline,{s:c.identity for s,c in caches.items()}
