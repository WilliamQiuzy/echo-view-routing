"""A58-only paths and immutable input helpers; prior namespaces are read-only."""
from pathlib import Path
import hashlib,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/a58_structured_evidence_v1'
PUB=ROOT/'docs/results/2026-11-03_structured_evidence'
CFG=ROOT/'configs/experiments/structured_evidence_v1.json'
def read(path):return json.loads(Path(path).read_text())
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(16*1024*1024),b''):h.update(b)
    return h.hexdigest()
def write(path,value):
    from echo_routing.temporal.dense_readout import atomic_json
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);atomic_json(path,value)
def verify_freeze(name):
    f=read(RUN/name)
    for p,h in f['objects'].items():
        if sha(ROOT/p)!=h:raise ValueError('frozen code/input changed: '+p)
    return f
def sequences(role):
    c=read(CFG)
    if role in c['data']['roles']:pin=c['data']['roles'][role]
    else:pin=read(PUB/'input_manifest.json')['roles'][role]
    path=ROOT/pin['path']
    if sha(path)!=pin['sha256']:raise ValueError('private sequence source digest changed')
    return np.load(path,allow_pickle=True).tolist()
