"""Atomic checkpoint replacement and explicit replay accounting for A57."""
import os
from pathlib import Path
import torch

def atomic_torch_save(value,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix(path.suffix+'.tmp')
    try:
        with temp.open('wb') as f:torch.save(value,f);f.flush();os.fsync(f.fileno())
        temp.replace(path)
    finally:
        if temp.exists():temp.unlink()

def resume_accounting(rows,next_update):
    cursors={}
    for row in rows:
        attempt=int(row['attempt']);u=int(row['u'])
        if attempt in cursors and u!=cursors[attempt]+1:raise ValueError('physical trace has a logical gap')
        cursors[attempt]=u
    maximum=max((int(r['u']) for r in rows),default=-1)+1
    if not 0<=next_update<=maximum:raise ValueError('checkpoint cursor beyond observed physical trace')
    return {'committed_physical_updates':len(rows),'checkpoint_next_update':next_update,'invalidated_after_checkpoint':sum(int(r['u'])>=next_update for r in rows)}


def within_execution_limits(gpu_hours,cpu_hours,peak_bytes,wall_epoch):
    return gpu_hours<22 and cpu_hours<68 and peak_bytes<40_000_000_000 and wall_epoch<1793820844.1596537-7200
