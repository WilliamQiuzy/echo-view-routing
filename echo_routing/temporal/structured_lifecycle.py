"""Independent A58 checkpoint checks and the fixed training-only selector."""
import hashlib,math
import torch
from echo_routing.temporal.dev_train import lr_factor

def state_digest(state):
    h=hashlib.sha256()
    for name,value in state.items():
        a=value.detach().cpu().contiguous();h.update(name.encode());h.update(str((tuple(a.shape),str(a.dtype))).encode());h.update(a.numpy().tobytes())
    return h.hexdigest()

def choose_inner(records):
    if len(records)!=3 or {r['step'] for r in records}!={2000,4000,6000}:raise ValueError('all three predeclared inner checkpoints required')
    best=None
    for row in records:
        if any(not math.isfinite(row[k]) for k in ['segment_f1_50_multi','native_macro_f1']) or row['short_spurious_segments_16samples']<0:raise ValueError('finite inner metrics and nonnegative shortFP required')
        if best is None:best=row;continue
        for key,direction,tolerance in [('segment_f1_50_multi',1,1e-12),('native_macro_f1',1,1e-12),('short_spurious_segments_16samples',-1,0),('step',-1,0)]:
            delta=(row[key]-best[key])*direction
            if abs(delta)>tolerance:
                if delta>0:best=row
                break
    return best

def audit_payload(model,value,identity):
    if value['identity']!=identity or value['arm']!=model.arm or value['seed']!=model.seed or value['schedule_total']!=6000 or not 0<=value['update']<=6000:raise ValueError('checkpoint source/arm/seed/plan/cursor binding differs')
    initial=model.state_dict();saved=value['model']
    if set(initial)!=set(saved) or any(saved[k].shape!=v.shape or saved[k].dtype!=v.dtype or not bool(torch.isfinite(saved[k]).all()) for k,v in initial.items()):raise ValueError('complete finite head key/shape/dtype binding required')
    expected=[(n,p) for n,p in model.named_parameters() if p.requires_grad];groups=value['optimizer']['param_groups'];states=value['optimizer']['state'];u=value['update']
    if len(groups)!=1:raise ValueError('exact single fixed Adam group required')
    group=groups[0];names=[n for n,_ in expected];ids=group['params'];lr=.001 if u==0 else .001*lr_factor(u-1,100,6000)
    if group.get('param_names')!=names or len(ids)!=len(names) or len(set(ids))!=len(ids) or group.get('name')!='structured_head' or group.get('base_lr')!=.001 or group.get('weight_decay')!=.01 or tuple(group.get('betas',()))!=(.9,.999) or group.get('eps')!=1e-8 or group.get('amsgrad',False) or group.get('maximize',False) or not math.isclose(group['lr'],lr,rel_tol=1e-12,abs_tol=1e-15):raise ValueError('optimizer names/recipe/schedule changed')
    if set(states)!=(set(ids) if u else set()):raise ValueError('optimizer tensor coverage differs')
    nonzero={}
    for identifier,(name,param) in zip(ids,expected):
        if not u:continue
        state=states[identifier]
        if set(state)!={'step','exp_avg','exp_avg_sq'} or float(state['step'])!=u:raise ValueError('Adam moment keys/cursor changed')
        for field in ['exp_avg','exp_avg_sq']:
            if state[field].shape!=param.shape or not bool(torch.isfinite(state[field]).all()):raise ValueError('Adam moment shape/finiteness changed')
        if bool((state['exp_avg_sq']<0).any()):raise ValueError('negative second moment')
        nonzero[name]=bool(state['exp_avg_sq'].count_nonzero())
    if model.arm=='D0' and not torch.equal(saved['duration'],torch.zeros_like(saved['duration'])):raise ValueError('duration-neutral g changed')
    return {'optimizer_tensors':len(ids),'optimizer_names':names,'complete_model_tensors':len(saved),'all_moments_finite_nonnegative_exact_shape':True,'Adam_steps':u,'nonzero_second_moment_tensors':nonzero,'model_digest':state_digest(saved)}
