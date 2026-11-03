"""Independent per-tensor state/optimizer audit for the four-depth video graph."""
import torch

def verify_optimizer(state,initial,optimizer,updates,joint,inactive_names,factor):
    head=[n for n in state if n.startswith('head.')]
    video=[n for n in state if n.startswith('encoder.') and n[8:] not in inactive_names] if joint else []
    groups=optimizer['param_groups'];assert len(groups)==(2 if joint else 1)
    binding=[]
    for group,names,lr,name in zip(groups,[head,video],[.001,3e-5],['head','active_video']):
        assert group['name']==name and len(group['params'])==len(names)
        assert group['base_lr']==lr and abs(group['lr']-lr*factor)<1e-15
        assert tuple(group['betas'])==(.9,.999) and group['eps']==1e-8 and group['weight_decay']==.05
        binding.extend(zip(group['params'],names))
    assert len({key for key,_ in binding})==len(binding)==len(head)+len(video)
    assert set(optimizer['state'])=={key for key,_ in binding}
    for key,name in binding:
        m=optimizer['state'][key];assert int(m['step'])==updates
        for k in ['exp_avg','exp_avg_sq']:assert m[k].shape==state[name].shape and torch.isfinite(m[k]).all()
        assert (m['exp_avg_sq']>=0).all() and m['exp_avg_sq'].sum()>0
    changed=[]
    for name,base in initial.items():
        key=name if name.startswith('encoder.') else 'encoder.'+name
        if key not in state:continue
        current=state[key];assert current.shape==base.shape and torch.isfinite(current).all()
        drift=not torch.equal(current,base)
        if not joint or key[8:] in inactive_names:assert not drift
        else:assert drift;changed.append(key[8:])
    assert len(changed)==len(video)
    return {'head_optimizer_tensors':len(head),'active_video_optimizer_tensors':len(video),'changed_video_tensors':len(changed),'frozen_image_tensors_unchanged':len(inactive_names),'exact_binding_moments_shapes_finite_steps_lr':True}
