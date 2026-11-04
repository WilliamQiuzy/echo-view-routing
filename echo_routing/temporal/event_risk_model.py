"""A62 matched205-dimensional event heads; CPU BCE and complete durable states."""
import copy,math,random
import numpy as np
import torch
from echo_routing.temporal.proposal_training import checked_identity,lr_factor,atomic_save as _atomic_save,UpdateJournal
from echo_routing.temporal.safety_training import seed_value
ARMS=('R1','R2')


class EventRisk(torch.nn.Module):
    def __init__(self,arm,seed):
        super().__init__();self.seed=seed_value(seed)
        if arm not in ARMS:raise ValueError('declared matched R1/R2 event arm required')
        self.arm=arm;self.role='candidate_unsafe_event'
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed);self.network=torch.nn.Sequential(torch.nn.Linear(205,32),torch.nn.GELU(),torch.nn.Linear(32,1));torch.nn.init.zeros_(self.network[2].weight);torch.nn.init.zeros_(self.network[2].bias)
    def forward(self,x):
        if x.ndim!=2 or x.shape[1]!=205 or not len(x):raise ValueError('nonempty normalized candidateN205 required; empty caller skips NN')
        return self.network(x).flatten()


def optimizer(model):
    named=list(model.named_parameters())
    return torch.optim.AdamW([{'params':[v for _,v in named],'param_names':[k for k,_ in named],'name':'event_risk','base_lr':.001,'lr':.001}],weight_decay=.01,betas=(.9,.999),eps=1e-8)


def recipe(g):
    return g.get('name')=='event_risk' and g.get('base_lr')==.001 and g.get('weight_decay')==.01 and tuple(g.get('betas',()))==(.9,.999) and g.get('eps')==1e-8 and not any(g.get(k,False) for k in ['amsgrad','maximize','differentiable']) and math.isfinite(g.get('lr',float('nan'))) and 0<=g['lr']<=.001


def binding(model,opt):
    named=list(model.named_parameters())
    return isinstance(opt,torch.optim.AdamW) and len(opt.param_groups)==1 and recipe(opt.param_groups[0]) and opt.param_groups[0].get('param_names')==[n for n,_ in named] and [id(x) for x in opt.param_groups[0]['params']]==[id(v) for _,v in named] and all(p.requires_grad and not p.is_cuda for _,p in named)


def validate_optimizer(model,payload,update):
    if isinstance(update,(bool,np.bool_)) or not isinstance(update,(int,np.integer)) or not 0<=update<=6000:raise ValueError('canonical exact fixed optimizer cursor required')
    groups=payload['param_groups'];named=list(model.named_parameters())
    if len(groups)!=1 or not recipe(groups[0]) or groups[0].get('param_names')!=[n for n,_ in named] or len(groups[0]['params'])!=len(named) or len(set(groups[0]['params']))!=len(named):raise ValueError('exact event AdamW parameter coverage/recipe required')
    g=groups[0];lr=.001 if update==0 else .001*lr_factor(update-1)
    if not math.isclose(g['lr'],lr,abs_tol=1e-15,rel_tol=1e-12) or set(payload['state'])!=(set(g['params']) if update else set()):raise ValueError('Adam cursor/state/LR mismatch')
    for key,(_,p) in zip(g['params'],named):
        if not update:continue
        s=payload['state'][key]
        if set(s)!={'step','exp_avg','exp_avg_sq'} or float(s['step'])!=update:raise ValueError('each actual Adam step must equal cursor')
        for name in ['exp_avg','exp_avg_sq']:
            v=s[name]
            if v.shape!=p.shape or v.dtype!=p.dtype or v.is_cuda or not bool(torch.isfinite(v).all()):raise ValueError('Adam moment shape/dtype/finite mismatch')
        if bool((s['exp_avg_sq']<0).any()):raise ValueError('negative Adam second moment')


def step(model,opt,batch,update):
    if not binding(model,opt):raise ValueError('exact live event optimizer required')
    validate_optimizer(model,opt.state_dict(),update);x,y=batch['x'],batch['y']
    if x.shape!=(64,205) or y.shape!=(64,) or any(v.dtype!=torch.float32 or v.is_cuda or v.requires_grad or not bool(torch.isfinite(v).all()) for v in [x,y]) or not bool(((y==0)|(y==1)).all()):raise ValueError('64 detached finite FP32 candidates and binary unsafe targets required')
    opt.param_groups[0]['lr']=.001*lr_factor(update);opt.zero_grad(set_to_none=True);loss=torch.nn.functional.binary_cross_entropy_with_logits(model(x),y);loss.backward();grads={n:p.grad for n,p in model.named_parameters()}
    if not bool(torch.isfinite(loss)) or any(g is None or not bool(torch.isfinite(g).all()) for g in grads.values()):raise ValueError('missing/nonfinite event gradient/loss')
    norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.))
    if not math.isfinite(norm):raise ValueError('nonfinite event gradient norm')
    opt.step()
    return {'u':int(update),'candidate_BCE':float(loss.detach()),'loss':float(loss.detach()),'lr':float(opt.param_groups[0]['lr']),'physical_candidates':64,'unsafe_targets':int(y.sum()),'finite_gradients':True,'gradient_norm':norm,'nonzero_gradient_tensors':{n:bool(g.count_nonzero()) for n,g in grads.items()}}


def norm_tensors(norms):
    if set(norms)!={'candidate','context'}:raise ValueError('original109/new96 FIT normalizers required')
    out={}
    for name,n in [('candidate',109),('context',96)]:
        if set(norms[name])!={'mean','std'}:raise ValueError('exact normalizer schema required')
        d={k:torch.as_tensor(np.asarray(norms[name][k])).clone().to(torch.float64) for k in ['mean','std']}
        if any(v.shape!=(n,) or not bool(torch.isfinite(v).all()) for v in d.values()) or bool((d['std']<1e-6).any()):raise ValueError('finite FIT normalizer shape/std contract required')
        out[name]=d
    return out


def checkpoint(model,opt,update,identity,norms):
    if not binding(model,opt):raise ValueError('live CPU event optimizer required')
    validate_optimizer(model,opt.state_dict(),update);rng=list(np.random.get_state());rng[1]=rng[1].tolist()
    return {'model':{k:v.detach().clone() for k,v in model.state_dict().items()},'optimizer':copy.deepcopy(opt.state_dict()),'update':int(update),'arm':model.arm,'seed':model.seed,'role':model.role,'schedule_total':6000,'identity':checked_identity(identity),'normalizers':norm_tensors(norms),'target_semantics':'unscaled y=int(H>0 or dFP16>0);target_scales identity slot binds this target contract, no numeric target scaler','rng':{'torch_CPU':torch.get_rng_state(),'numpy':rng,'python':random.getstate()}}


def restore(model,opt,value,identity,norms):
    if not binding(model,opt) or value['identity']!=checked_identity(identity) or (value['arm'],value['seed'],value['role'],value['schedule_total'])!=(model.arm,model.seed,model.role,6000) or value['target_semantics']!='unscaled y=int(H>0 or dFP16>0);target_scales identity slot binds this target contract, no numeric target scaler':raise ValueError('event model/source/arm/target identity mismatch')
    expected=norm_tensors(norms)
    if set(value['normalizers'])!=set(expected) or any(set(value['normalizers'][name])!=set(d) or any(not torch.equal(value['normalizers'][name][k],v) for k,v in d.items()) for name,d in expected.items()):raise ValueError('original/new FIT normalizers changed')
    current=model.state_dict();saved=value['model']
    if set(saved)!=set(current) or any(saved[k].shape!=v.shape or saved[k].dtype!=v.dtype or saved[k].is_cuda or not bool(torch.isfinite(saved[k]).all()) for k,v in current.items()):raise ValueError('complete finite event state required')
    validate_optimizer(model,value['optimizer'],value['update']);model.load_state_dict(saved,strict=True);opt.load_state_dict(value['optimizer']);torch.set_rng_state(value['rng']['torch_CPU']);rng=list(value['rng']['numpy']);rng[1]=np.asarray(rng[1],np.uint32);np.random.set_state(tuple(rng));random.setstate(value['rng']['python'])
    if not binding(model,opt):raise ValueError('restored optimizer binding changed')
    return value['update']


def atomic_save(path,value):_atomic_save(value,path)
