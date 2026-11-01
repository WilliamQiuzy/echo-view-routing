"""Durable once-only score cells and independent original-run references."""
import hashlib
import json
import os
from pathlib import Path
import numpy as np


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()


def write_json(path,obj):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_suffix(p.suffix+'.tmp')
    with t.open('w') as f:json.dump(obj,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    t.replace(p)


def validate_journal(log,checkpoint_step,inflight=False):
    if inflight or len(log)!=checkpoint_step or [q['step'] for q in log]!=list(range(1,checkpoint_step+1)):
        raise ValueError('journal differs from durable optimizer step; refuse repeated/unsaved updates')
    return checkpoint_step


def load_cell(root,key,identity):
    root=Path(root);receipt=json.loads((root/(key+'.json')).read_text())
    if receipt['identity']!=identity or receipt['sha256']!=sha(root/(key+'.npz')):raise ValueError('score cell identity/SHA mismatch')
    with np.load(root/(key+'.npz')) as x:rows,scores=x['rows'].copy(),x['scores'].copy()
    if len(rows)!=receipt['rows'] or rows.ndim!=1 or scores.shape!=rows.shape or not np.isfinite(scores).all():raise ValueError('incomplete raw score cell')
    return rows,scores


def claim_cell(root,key,identity):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    if (root/(key+'.json')).exists():load_cell(root,key,identity);return False
    p=root/(key+'.claim.json')
    try:
        with p.open('x') as f:json.dump(identity,f);f.flush();os.fsync(f.fileno())
    except FileExistsError as e:raise RuntimeError('interrupted/inflight score cell; no repeated inference') from e
    return True


def finish_cell(root,key,identity,rows,scores):
    root=Path(root);claim=json.loads((root/(key+'.claim.json')).read_text())
    if claim!=identity or (root/(key+'.json')).exists():raise ValueError('score cell already complete or rebound')
    p=root/(key+'.npz')
    with p.open('xb') as f:np.savez_compressed(f,rows=rows,scores=scores);f.flush();os.fsync(f.fileno())
    write_json(root/(key+'.json'),{'identity':identity,'rows':len(rows),'sha256':sha(p)})


def reference_labels(pk,rows,scores,off):
    """Independent explicit weighted loop; unchanged non-A4C labels, strict >0."""
    if len(np.unique(rows))!=len(rows) or len(rows)!=len(scores):raise ValueError('duplicate/missing score rows')
    lookup={int(row):float(s) for row,s in zip(rows,scores)}
    means=np.full(len(pk['base']),np.nan);pred=pk['base'].copy()
    for i,b in enumerate(pk['base']):
        if b!=2:continue
        a,z=pk['offsets'][i:i+2];total=0.;mass=0.
        for j in range(int(a),int(z)):
            row=int(pk['rows'][j])-int(off);weight=float(pk['weights'][j])
            if row not in lookup or not np.isfinite(lookup[row]):raise ValueError('eligible member score missing')
            total+=weight*lookup[row];mass+=weight
        if abs(mass-1)>1e-10:raise ValueError('membership weight mass differs from1')
        means[i]=total
        if total>0:pred[i]=3
    return means,pred
