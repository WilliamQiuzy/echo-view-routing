"""A63 complement-only auxiliary heads with unchanged S-L/risk losses/Adam.

All role pools must be nonempty: the predecessor fallback is never permitted.
The physical update journal forbids replay of pending/uncheckpointed work.
"""
import json,time
from pathlib import Path
import numpy as np
import torch
from echo_routing.temporal.selected_risk_supply import session_of
from echo_routing.temporal.safety_training import draw_plan,checkpoint,restore,step
from echo_routing.temporal.safety_data import tensor_data,batch
from echo_routing.temporal.proposal_training import UpdateJournal,atomic_save
from echo_routing.temporal.anchor_cache import sha


def validate_training_sources(sequences,train_sessions,held_sessions):
    train_sessions=set(train_sessions);held_sessions=set(held_sessions)
    if not train_sessions or not held_sessions or train_sessions&held_sessions:raise ValueError('disjoint nonempty complete train/held source sessions required')
    actual={session_of(v) for q in sequences for v,*_ in q['refs']}
    if not actual or actual-train_sessions or actual&held_sessions:raise ValueError('any auxiliary training reference crossing into held source forbidden')
    return True

def aux_plans(data,seed):
    for name in ['kind_groups','kind_indices']:
        if set(data[name])!={'native','constructed'} or any(len(data[name][k])==0 for k in ['native','constructed']):raise ValueError('complete nonempty native/mixed query and candidate pools; no fallback')
    plans={role:draw_plan(*(len(data['kind_groups' if role=='utility' else 'kind_indices'][k]) for k in ['native','constructed']),seed,role) for role in ['utility','risk']}
    if any(p['fallback'] or p['source_kinds']!=[0,1] for p in plans.values()):raise ValueError('auxiliary kind fallback forbidden')
    return plans

def train_aux_one(model,opt,data,plan,identity,folder,stop=6000,latest_every=100,milestones=(2000,4000,6000),on_progress=None):
    if torch.cuda.is_initialized():raise ValueError('unexpected CUDA before model/journal access')
    if isinstance(stop,bool) or not isinstance(stop,int) or not 0<=stop<=6000 or latest_every<1:raise ValueError('fixed 6000 auxiliary schedule required')
    expected=aux_plans(data,model.seed)[model.role]
    if set(plan)!=set(expected) or any(not np.array_equal(v,expected[k]) if isinstance(v,np.ndarray) else v!=expected[k] for k,v in plan.items()):raise ValueError('actual full query/candidate role draw changed')
    folder=Path(folder);journal=UpdateJournal(folder,identity);latest=folder/'latest.pt'
    if not latest.exists() and journal.cursor==0:atomic_save(checkpoint(model,opt,0,identity,data['normalizers'],data['scale']),latest)
    start=journal.resume_cursor();loaded=restore(model,opt,torch.load(latest,weights_only=True,map_location='cpu'),identity,data['normalizers'],data['scale'])
    if loaded!=start or start>stop:raise ValueError('physical trace/fullstate mismatch; no replay/rollback')
    tensors=tensor_data(data)
    for u in range(start,stop):
        if torch.cuda.is_initialized():raise ValueError('unexpected CUDA before physical auxiliary update')
        b,exposure=batch(data,tensors,plan,u,model.role,model.arm);journal.begin(u);begin=time.perf_counter();row=step(model,opt,b,u,model.role);row.update(exposure);row['optimizer_update_wall_seconds']=time.perf_counter()-begin;cursor=u+1
        state=checkpoint(model,opt,cursor,identity,data['normalizers'],data['scale']) if cursor%latest_every==0 or cursor in milestones else None
        if cursor in milestones:atomic_save(state,folder/f'step{cursor}.pt')
        journal.finish(row,state)
        if on_progress is not None and cursor%25==0:on_progress(cursor,row)
    rows=[json.loads(line) for line in journal.trace.read_text().splitlines()] if journal.trace.exists() else []
    if torch.cuda.is_initialized() or (stop and any(float(opt.state[p]['step'])!=stop for p in model.parameters())):raise ValueError('actual CPU Adam cursor differs')
    return {'committed_physical_updates':len(rows),'actual_Adam_update':stop,'new_updates_this_process':stop-start,'physical_candidate_exposures':sum(r['physical_candidates'] for r in rows),'physical_query_group_exposures':sum(r.get('physical_groups',0) for r in rows),'physical_changed_position_exposures':sum(r['changed_position_exposures'] for r in rows),'all_tensor_gradient_liveness':{n:any(r['nonzero_gradient_tensors'][n] for r in rows) for n,_ in model.named_parameters()},'journal_sha256':sha(journal.trace) if rows else None,'CUDA_initialized':False}
