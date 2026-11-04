"""A61 matched utility/risk heads, complete-query losses and durable full states."""
import copy,math,random
import numpy as np
import torch
from echo_routing.temporal.proposal_training import (UpdateJournal,atomic_save,checked_identity,lr_factor)

SEEDS=(2026110461,2026110462)
ARMS=('U-P','U-L','S-P','S-L')


def seed_value(seed):
    if isinstance(seed,(bool,np.bool_)) or not isinstance(seed,(int,np.integer)) or int(seed) not in SEEDS:raise ValueError('canonical declared integer seed required')
    return int(seed)


def network(n,out):
    model=torch.nn.Sequential(torch.nn.Linear(n,32),torch.nn.GELU(),torch.nn.Linear(32,out))
    torch.nn.init.zeros_(model[2].weight);torch.nn.init.zeros_(model[2].bias)
    return model


class Utility(torch.nn.Module):
    def __init__(self,arm,seed):
        super().__init__();self.seed=seed_value(seed)
        if arm not in ARMS:raise ValueError('declared matched utility arm required')
        self.arm=arm;self.role='utility'
        with torch.random.fork_rng(devices=[]):torch.manual_seed(self.seed);self.network=network(109,1)
    def forward(self,x):return self.network(x)


class Risk(torch.nn.Module):
    def __init__(self,seed):
        super().__init__();self.seed=seed_value(seed);self.arm='risk';self.role='risk'
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed);self.position=network(47,3);self.candidate=network(109,1)
    def forward(self,x,positions):return self.position(positions),self.candidate(x).flatten()


def draw_plan(native,constructed,seed,role):
    seed=seed_value(seed)
    if role not in ['utility','risk'] or any(isinstance(n,bool) or not isinstance(n,(int,np.integer)) or n<0 for n in [native,constructed]):raise ValueError('complete nonnegative role pool counts required')
    counts=[int(native),int(constructed)]
    if not any(counts):raise ValueError('both candidate/group pools empty; structural no-action stop')
    source_kinds=[k if counts[k] else 1-k for k in [0,1]]
    arrays=[np.random.default_rng([seed,61,['utility','risk'].index(role),k]).integers(counts[source_kinds[k]],size=(6000,32),dtype=np.int64) for k in [0,1]]
    return {'indices':np.concatenate(arrays,axis=1),'source_kinds':source_kinds,'fallback':not all(counts),'seed':seed,'role':role}


def utility_losses(scores,targets,lengths,ranked):
    scores=scores.flatten();targets=targets.flatten()
    if sum(lengths)!=len(scores) or len(scores)!=len(targets) or not lengths or min(lengths)<1:raise ValueError('complete nonempty query groups required')
    point=[];ranks=[];pairs=0;offset=0
    for n in lengths:
        z=scores[offset:offset+n];t=targets[offset:offset+n];offset+=n
        point.append(((z-t.to(z.dtype))**2).mean())
        if ranked:
            z=torch.cat([z,z.new_zeros(1)]);t=torch.cat([t,t.new_zeros(1)])
            i,j=torch.triu_indices(n+1,n+1,offset=1);keep=t[i]!=t[j];i,j=i[keep],j[keep];pairs+=len(i)
            ranks.append(torch.nn.functional.softplus(-torch.sign(t[i]-t[j]).to(z.dtype)*(z[i]-z[j])).mean() if len(i) else z.sum()*0)
    p=torch.stack(point).mean();rank=torch.stack(ranks).mean() if ranks else p*0
    return p,rank,pairs


def risk_losses(position_logits,labels,fp_logits,fp_targets,lengths):
    if sum(lengths)!=len(labels) or len(fp_logits)!=len(lengths) or min(lengths)<1:raise ValueError('all changed positions per sampled action required')
    per_position=torch.nn.functional.cross_entropy(position_logits,labels,reduction='none')
    ids=torch.repeat_interleave(torch.arange(len(lengths)),torch.tensor(lengths))
    action=per_position.new_zeros(len(lengths)).scatter_add_(0,ids,per_position)/per_position.new_tensor(lengths)
    return action.mean(),torch.nn.functional.binary_cross_entropy_with_logits(fp_logits.flatten(),fp_targets.to(fp_logits.dtype))


def risk_summary(position_logits,fp_logit):
    if position_logits.ndim!=2 or position_logits.shape[1]!=3 or not len(position_logits) or not bool(torch.isfinite(position_logits).all()) or not bool(torch.isfinite(fp_logit).all()):raise ValueError('finite changed-position and candidate risk scores required')
    p=position_logits.detach().softmax(1).to(torch.float64);fp=float(fp_logit.detach().sigmoid())
    return {'predicted_H':float(p[:,0].sum()),'predicted_R':float(p[:,1].sum()),'predicted_neutral':float(p[:,2].sum()),'pFP':fp,'r':max(float(p[:,0].max()),fp)}


def optimizer(model):
    named=list(model.named_parameters())
    return torch.optim.AdamW([{'params':[p for _,p in named],'param_names':[n for n,_ in named],'name':'safety','base_lr':.001,'lr':.001}],weight_decay=.01,betas=(.9,.999),eps=1e-8)


def recipe(g):
    return g.get('name')=='safety' and g.get('base_lr')==.001 and g.get('weight_decay')==.01 and tuple(g.get('betas',()))==(.9,.999) and g.get('eps')==1e-8 and not any(g.get(k,False) for k in ['amsgrad','maximize','differentiable']) and math.isfinite(g.get('lr',float('nan'))) and 0<=g['lr']<=.001


def binding(model,opt):
    named=list(model.named_parameters())
    return isinstance(opt,torch.optim.AdamW) and len(opt.param_groups)==1 and recipe(opt.param_groups[0]) and opt.param_groups[0]['param_names']==[n for n,_ in named] and [id(x) for x in opt.param_groups[0]['params']]==[id(p) for _,p in named] and all(p.requires_grad and not p.is_cuda for _,p in named)


def validate_optimizer(model,payload,update):
    if isinstance(update,bool) or int(update)!=update or not 0<=update<=6000:raise ValueError('exact fixed optimizer cursor required')
    groups=payload['param_groups'];named=list(model.named_parameters())
    if len(groups)!=1 or not recipe(groups[0]) or groups[0].get('param_names')!=[n for n,_ in named] or len(groups[0]['params'])!=len(named) or len(set(groups[0]['params']))!=len(named):raise ValueError('optimizer recipe/named coverage changed')
    g=groups[0];states=payload['state'];lr=.001 if update==0 else .001*lr_factor(update-1)
    if not math.isclose(g['lr'],lr,rel_tol=1e-12,abs_tol=1e-15) or set(states)!=(set(g['params']) if update else set()):raise ValueError('optimizer exact state/cursor/schedule mismatch')
    for key,(_,p) in zip(g['params'],named):
        if not update:continue
        state=states[key]
        if set(state)!={'step','exp_avg','exp_avg_sq'} or float(state['step'])!=update:raise ValueError('Adam step differs from cursor')
        for k in ['exp_avg','exp_avg_sq']:
            t=state[k]
            if t.shape!=p.shape or t.dtype!=p.dtype or t.is_cuda or not bool(torch.isfinite(t).all()):raise ValueError('Adam moment shape/dtype/finite mismatch')
        if bool((state['exp_avg_sq']<0).any()):raise ValueError('negative Adam second moment')


def step(model,opt,batch,update,role):
    if role!=model.role or not binding(model,opt):raise ValueError('exact live optimizer and declared model role required')
    validate_optimizer(model,opt.state_dict(),update)
    x=batch['x']
    tensors=[v for v in batch.values() if isinstance(v,torch.Tensor)]
    if any(v.is_cuda or v.requires_grad or not bool(torch.isfinite(v).all()) for v in tensors) or x.dtype!=torch.float32 or x.shape[1]!=109:raise ValueError('detached finite CPU cached graph required')
    opt.param_groups[0]['lr']=.001*lr_factor(update);opt.zero_grad(set_to_none=True)
    if role=='utility':
        p,r,pairs=utility_losses(model(x).flatten(),batch['target'],batch['lengths'],model.arm.endswith('L'));loss=p+r
        details={'pointwise_MSE':float(p.detach()),'pairwise_logistic':float(r.detach()),'physical_groups':len(batch['lengths']),'physical_candidates':len(x),'physical_unequal_pairs':pairs}
    else:
        pos,fp=model(x,batch['positions']);ce,bce=risk_losses(pos,batch['labels'],fp,batch['fp'],batch['lengths']);loss=ce+bce
        details={'position_CE':float(ce.detach()),'candidate_BCE':float(bce.detach()),'physical_candidates':len(x),'physical_changed_positions':len(batch['positions'])}
    loss.backward();grads={n:p.grad for n,p in model.named_parameters()}
    if not bool(torch.isfinite(loss)) or any(g is None or not bool(torch.isfinite(g).all()) for g in grads.values()):raise ValueError('missing/nonfinite active gradient or loss')
    norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.))
    if not math.isfinite(norm):raise ValueError('nonfinite gradient norm')
    opt.step()
    return {'u':update,'loss':float(loss.detach()),'lr':opt.param_groups[0]['lr'],'gradient_norm':norm,'finite_gradients':True,'nonzero_gradient_tensors':{n:bool(g.count_nonzero()) for n,g in grads.items()},**details}


def norm_tensors(norms):
    if set(norms)!={'candidate','position'}:raise ValueError('both FIT normalizer identities required')
    out={}
    for name,n in [('candidate',109),('position',47)]:
        out[name]={k:torch.as_tensor(np.asarray(norms[name][k])).clone().to(torch.float64) for k in ['mean','std']}
        if any(v.shape!=(n,) or not bool(torch.isfinite(v).all()) for v in out[name].values()) or bool((out[name]['std']<1e-6).any()):raise ValueError('complete finite FIT normalizers required')
    return out


def checkpoint(model,opt,update,identity,norms,scale):
    if not binding(model,opt) or not math.isfinite(scale) or scale<1:raise ValueError('live CPU optimizer and common FIT RMS required')
    validate_optimizer(model,opt.state_dict(),update);rng=list(np.random.get_state());rng[1]=rng[1].tolist()
    return {'model':{k:v.detach().clone() for k,v in model.state_dict().items()},'optimizer':copy.deepcopy(opt.state_dict()),'update':int(update),'arm':model.arm,'role':model.role,'seed':model.seed,'schedule_total':6000,'identity':checked_identity(identity),'normalizers':norm_tensors(norms),'utility_scale':float(scale),'rng':{'torch_CPU':torch.get_rng_state(),'numpy':rng,'python':random.getstate()}}


def restore(model,opt,value,identity,norms,scale):
    if not binding(model,opt) or value['identity']!=checked_identity(identity) or (value['arm'],value['role'],value['seed'],value['schedule_total'],value['utility_scale'])!=(model.arm,model.role,model.seed,6000,float(scale)):raise ValueError('model/source/role/common scale identity changed')
    expected=norm_tensors(norms)
    if set(value['normalizers'])!=set(expected) or any(set(value['normalizers'][n])!=set(d) or any(not torch.equal(value['normalizers'][n][k],v) for k,v in d.items()) for n,d in expected.items()):raise ValueError('FIT normalizer identities changed')
    state=model.state_dict()
    if set(value['model'])!=set(state) or any(v.shape!=value['model'][k].shape or v.dtype!=value['model'][k].dtype or not bool(torch.isfinite(value['model'][k]).all()) for k,v in state.items()):raise ValueError('complete model keys/shape/dtype/finite contract changed')
    validate_optimizer(model,value['optimizer'],value['update']);model.load_state_dict(value['model'],strict=True);opt.load_state_dict(value['optimizer'])
    torch.set_rng_state(value['rng']['torch_CPU']);nr=list(value['rng']['numpy']);nr[1]=np.asarray(nr[1],np.uint32);np.random.set_state(tuple(nr));random.setstate(value['rng']['python'])
    if not binding(model,opt):raise ValueError('restored optimizer lost binding')
    return int(value['update'])
