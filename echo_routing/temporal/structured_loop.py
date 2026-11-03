"""A58 fixed-plan optimization with no silent replay of committed optimizer steps."""
from pathlib import Path
import json,time,os
import torch
from echo_routing.temporal import structured_training as train
from echo_routing.temporal.structured_lifecycle import audit_payload,state_digest
from echo_routing.temporal.dense_durable import atomic_torch_save
from echo_routing.temporal.dense_readout import atomic_json

def _equal(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a.cpu(),b.cpu())
    if isinstance(a,dict):return a.keys()==b.keys() and all(_equal(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return len(a)==len(b) and all(_equal(x,y) for x,y in zip(a,b))
    return a==b

def run_updates(model,opt,batch_for_row,plan,folder,identity,stop=6000,checkpoint_every=100,milestones=(2000,4000,6000),on_milestone=None,on_progress=None):
    """Formal caller fixes6000/100/(2000,4000,6000); smaller bounds are synthetic checks."""
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);pending=folder/'pending.json';trace=folder/'physical_updates.jsonl';latest=folder/'latest.pt'
    if pending.exists():raise RuntimeError('ambiguous pending head update: retain evidence and stop')
    rows=[json.loads(s) for s in trace.read_text().splitlines()] if trace.exists() else []
    if [r['u'] for r in rows]!=list(range(len(rows))) or any(r['identity']!=identity for r in rows):raise ValueError('physical optimizer journal has a gap or changed binding')
    cursor=0
    if latest.exists():
        payload=torch.load(latest,map_location='cpu',weights_only=True);audit_payload(model,payload,identity);cursor=train.restore(model,opt,payload,identity)
    if cursor!=len(rows):raise RuntimeError('committed update prefix differs from latest cursor; explicit independent accounting required, no silent optimizer replay')
    if cursor>stop:raise ValueError('checkpoint is beyond declared stop')
    def milestone(step,payload):
        path=folder/f'step{step}.pt'
        if path.exists():
            saved=torch.load(path,map_location='cpu',weights_only=True);audit_payload(model,saved,identity)
            if not _equal(saved,payload):raise ValueError('immutable full head/Adam/RNG checkpoint changed')
        else:atomic_torch_save(payload,path)
        if on_milestone is not None:
            rng=torch.get_rng_state();cuda=torch.cuda.get_rng_state_all() if next(model.parameters()).is_cuda else [];on_milestone(step,path);torch.set_rng_state(rng)
            if cuda:torch.cuda.set_rng_state_all(cuda)
    if cursor in milestones:milestone(cursor,train.checkpoint(model,opt,cursor,identity))
    for u in range(cursor,stop):
        indices=plan[u];batch=batch_for_row(indices);atomic_json(pending,{'u':u,'identity':identity,'epoch':time.time()});start=time.time();row=train.step(model,opt,batch,u);row.update({'identity':identity,'seed':model.seed,'arm':model.arm,'model_digest_after_update':state_digest(model.state_dict()),'plan_row':[int(i) for i in indices],'epoch':time.time(),'step_wall_seconds_before_journal':time.time()-start})
        with trace.open('a') as f:f.write(json.dumps(row)+'\n');f.flush();os.fsync(f.fileno())
        pending.unlink();rows.append(row)
        if (u+1)%checkpoint_every==0 or u+1 in milestones:
            payload=train.checkpoint(model,opt,u+1,identity);audit_payload(model,payload,identity);atomic_torch_save(payload,latest)
            if u+1 in milestones:milestone(u+1,payload)
        if on_progress is not None and (u+1)%25==0:on_progress(u+1,row)
    return {'logical_updates':stop,'physical_optimizer_updates':len(rows),'model_digest':state_digest(model.state_dict()),'no_optimizer_replay':True}
