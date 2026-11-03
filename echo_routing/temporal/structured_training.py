"""A58-only fixed head optimizer/update/checkpoint contract."""
import copy,math
import torch
from echo_routing.temporal.dev_train import lr_factor

def optimizer(model):
    named=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
    return torch.optim.AdamW([{'params':[p for _,p in named],'param_names':[n for n,_ in named],'name':'structured_head','base_lr':.001,'lr':.001}],betas=(.9,.999),eps=1e-8,weight_decay=.01)

def optimizer_binding(model,opt):
    expected=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
    return isinstance(opt,torch.optim.AdamW) and len(opt.param_groups)==1 and _recipe(opt.param_groups[0]) and opt.param_groups[0]['param_names']==[n for n,_ in expected] and [id(p) for p in opt.param_groups[0]['params']]==[id(p) for _,p in expected]

def _recipe(group):
    return group.get('name')=='structured_head' and group.get('base_lr')==.001 and group.get('weight_decay')==.01 and tuple(group.get('betas',()))==(.9,.999) and group.get('eps')==1e-8 and not group.get('amsgrad',False) and not group.get('maximize',False) and math.isfinite(group.get('lr',float('nan'))) and 0<=group['lr']<=.001

def _optimizer_cursor(model,payload,update):
    groups=payload['param_groups'];names=[n for n,p in model.named_parameters() if p.requires_grad]
    if len(groups)!=1 or not _recipe(groups[0]) or groups[0].get('param_names')!=names or len(groups[0]['params'])!=len(names) or len(set(groups[0]['params']))!=len(names):raise ValueError('checkpoint optimizer recipe or coverage changed')
    expected_lr=.001 if update==0 else .001*lr_factor(update-1,100,6000)
    if not math.isclose(groups[0]['lr'],expected_lr,rel_tol=1e-12,abs_tol=1e-15):raise ValueError('checkpoint optimizer schedule/cursor mismatch')
    states=payload['state'];expected=set(groups[0]['params']) if update else set()
    if set(states)!=expected or any('step' not in state or float(state['step'])!=update for state in states.values()):raise ValueError('checkpoint cursor disagrees with actual Adam counters')

def step(model,opt,batch,update):
    if len(batch)!=2 or not 0<=update<6000 or not optimizer_binding(model,opt):raise ValueError('fixed two complete sequences and optimizer binding required')
    for group in opt.param_groups:group['lr']=group['base_lr']*lr_factor(update,100,6000)
    opt.zero_grad(set_to_none=True);losses=[model.nll(a,b,y) for a,b,y in batch];loss=torch.stack(losses).mean();loss.backward();grad={n:p.grad for n,p in model.named_parameters() if p.requires_grad}
    if not bool(torch.isfinite(loss)) or any(g is None or not bool(torch.isfinite(g).all()) for g in grad.values()):raise ValueError('missing/nonfinite head loss or gradient')
    norm=float(torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.))
    if not math.isfinite(norm):raise ValueError('nonfinite unclipped gradient norm')
    opt.step();return {'u':update,'loss':float(loss.detach()),'native_NLL':float(losses[0].detach()),'constructed_NLL':float(losses[1].detach()),'gradient_norm':norm,'lr':opt.param_groups[0]['lr'],'finite_gradients':True,'nonzero_gradient_tensors':{n:bool(g.count_nonzero()) for n,g in grad.items()},'physical_sequences':2,'valid_original_positions':sum(len(y) for _,_,y in batch)}

def checkpoint(model,opt,update,identity):
    if not optimizer_binding(model,opt):raise ValueError('checkpoint optimizer binding changed')
    _optimizer_cursor(model,opt.state_dict(),update)
    return {'model':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},'optimizer':copy.deepcopy(opt.state_dict()),'update':int(update),'arm':model.arm,'seed':model.seed,'schedule_total':6000,'identity':identity,'cpu_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if next(model.parameters()).is_cuda else []}

def restore(model,opt,value,identity):
    if value['identity']!=identity or value['arm']!=model.arm or value['seed']!=model.seed or value['schedule_total']!=6000 or not 0<=value['update']<=6000:raise ValueError('head/source/plan/cursor identity changed')
    _optimizer_cursor(model,value['optimizer'],value['update'])
    model.load_state_dict(value['model'],strict=True);opt.load_state_dict(value['optimizer']);torch.set_rng_state(value['cpu_rng'])
    if value['cuda_rng']:torch.cuda.set_rng_state_all(value['cuda_rng'])
    if not optimizer_binding(model,opt):raise ValueError('restored optimizer lost exact parameter binding')
    return value['update']
