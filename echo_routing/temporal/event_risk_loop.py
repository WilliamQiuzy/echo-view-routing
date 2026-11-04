"""A62 fixed candidate plan and durable CPU event-head updates; no CAL callbacks."""
import json,time
from pathlib import Path
import numpy as np
import torch
from echo_routing.temporal.event_risk_model import UpdateJournal,checkpoint,restore,step,atomic_save
from echo_routing.temporal.event_risk_data import validate_draw_plan,candidate_indices
from echo_routing.temporal.anchor_cache import sha


def train_one(model,opt,data,plan,identity,folder,stop=6000,latest_every=100,milestones=(2000,4000,6000),on_progress=None):
    if isinstance(stop,bool) or not isinstance(stop,int) or not 0<=stop<=6000 or latest_every<1:raise ValueError('fixed bounded6000 schedule required')
    validate_draw_plan(plan,len(data['kind_indices']['native']),len(data['kind_indices']['constructed']),model.seed)
    folder=Path(folder);journal=UpdateJournal(folder,identity);latest=folder/'latest.pt'
    if not latest.exists() and journal.cursor==0:atomic_save(latest,checkpoint(model,opt,0,identity,data['normalizers']))
    start=journal.resume_cursor();loaded=restore(model,opt,torch.load(latest,weights_only=True,map_location='cpu'),identity,data['normalizers'])
    if start!=loaded or start>stop:raise ValueError('physical trace/latest/cursor mismatch; no replay/rollback')
    x,y=data['x'][model.arm],data['y']
    for u in range(start,stop):
        idx=candidate_indices(plan,u,data['kind_indices']['native'],data['kind_indices']['constructed']);journal.begin(u);t=time.perf_counter();row=step(model,opt,{'x':x[idx],'y':y[idx]},u);row.update(candidate_indices=idx.tolist(),source_kind_blocks=plan['source_kinds'],optimizer_update_wall_seconds=time.perf_counter()-t)
        cursor=u+1;state=checkpoint(model,opt,cursor,identity,data['normalizers']) if cursor%latest_every==0 or cursor in milestones else None
        if cursor in milestones:atomic_save(folder/f'step{cursor}.pt',state)
        journal.finish(row,state)
        if on_progress is not None and cursor%25==0:on_progress(cursor,row)
    rows=[json.loads(x) for x in journal.trace.read_text().splitlines()] if journal.trace.exists() else []
    if torch.cuda.is_initialized() or (stop and any(float(opt.state[p]['step'])!=stop for p in model.parameters())):raise ValueError('CPU role or actual Adam cursor changed')
    return {'committed_physical_updates':len(rows),'actual_Adam_update':stop,'new_updates_this_process':stop-start,'physical_candidate_exposures':sum(r['physical_candidates'] for r in rows),'all_tensor_gradient_liveness_over_history':{n:any(r['nonzero_gradient_tensors'][n] for r in rows) for n,_ in model.named_parameters()},'journal_sha256':sha(journal.trace) if rows else None,'CUDA_initialized':False}
