"""A61 full-query versus full-changed-position fixed6000 durable lifecycle."""
import json,os,time
from pathlib import Path
import torch
from echo_routing.temporal.safety_training import UpdateJournal,checkpoint,restore,step,atomic_save
from echo_routing.temporal.safety_data import tensor_data,batch
from echo_routing.temporal.anchor_cache import sha,atomic_json,fsync_directory


def train_one(model,opt,data,plan,identity,folder,stop=6000,latest_every=100,milestones=(2000,4000,6000),on_inner=None,on_progress=None):
    if not 0<=stop<=6000 or latest_every<1 or plan['role']!=model.role or plan['seed']!=model.seed:raise ValueError('bounded fixed role/seed schedule required')
    folder=Path(folder);journal=UpdateJournal(folder,identity);pending=folder/'inner_pending.json';latest=folder/'latest.pt'
    if pending.exists():raise ValueError('ambiguous pending inner readout;no replay')
    if not latest.exists() and journal.cursor==0:atomic_save(checkpoint(model,opt,0,identity,data['normalizers'],data['scale']),latest)
    start=journal.resume_cursor();restore(model,opt,torch.load(latest,weights_only=True,map_location='cpu'),identity,data['normalizers'],data['scale'])
    if start>stop:raise ValueError('cannot roll committed physical cursor backward')
    tensors=tensor_data(data)
    def inner(cursor):
        if on_inner is None:return
        path=folder/f'step{cursor}.pt';marker=folder/f'inner_step{cursor}.json';expected={'identity':identity,'weight_sha256':sha(path),'step':cursor}
        if marker.exists():
            value=json.loads(marker.read_text())
            if any(value[k]!=v for k,v in expected.items()):raise ValueError('completed inner identity changed')
            pin=value['callback_result']
            if isinstance(pin,dict) and 'path' in pin and 'sha256' in pin:
                target=Path(__file__).resolve().parents[2]/pin['path']
                if sha(target)!=pin['sha256']:raise ValueError('committed callback output changed')
            return
        if pending.exists():raise ValueError('ambiguous inner callback')
        with pending.open('x') as handle:json.dump(expected,handle,sort_keys=True);handle.flush();os.fsync(handle.fileno())
        fsync_directory(folder);result=on_inner(model,cursor,path)
        atomic_json(marker,{**expected,'callback_result':result});pending.unlink();fsync_directory(folder)
    if start in milestones:inner(start)
    for u in range(start,stop):
        values,exposure=batch(data,tensors,plan,u,model.role,model.arm);journal.begin(u);t=time.perf_counter()
        row=step(model,opt,values,u,model.role);row.update(exposure);row['optimizer_update_wall_seconds']=time.perf_counter()-t
        cursor=u+1;payload=checkpoint(model,opt,cursor,identity,data['normalizers'],data['scale']) if cursor%latest_every==0 or cursor in milestones else None
        if cursor in milestones:atomic_save(payload,folder/f'step{cursor}.pt')
        journal.finish(row,payload)
        if cursor in milestones:inner(cursor)
        if on_progress is not None and cursor%25==0:on_progress(cursor,row)
    if torch.cuda.is_initialized():raise ValueError('CPU head initialized unexpected CUDA')
    rows=[json.loads(line) for line in journal.trace.read_text().splitlines()] if journal.trace.exists() else []
    if stop and any(float(opt.state[p]['step'])!=stop for p in model.parameters()):raise ValueError('actual Adam counters differ from physical cursor')
    return {'committed_physical_updates':len(rows),'actual_Adam_update':stop,'new_updates_this_process':stop-start,'physical_candidate_exposures':sum(r['physical_candidates'] for r in rows),'physical_query_group_exposures':sum(r.get('physical_groups',0) for r in rows),'physical_changed_position_exposures':sum(r['changed_position_exposures'] for r in rows),'physical_unequal_pair_exposures':sum(r.get('physical_unequal_pairs',0) for r in rows),'all_tensor_gradient_liveness_over_history':{n:any(r['nonzero_gradient_tensors'][n] for r in rows) for n,_ in model.named_parameters()},'journal_sha256':sha(journal.trace) if rows else None,'CUDA_initialized':False}
