#!/usr/bin/env python3
"""A57 post-training CPU audit; no new neural forward or development input."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import hashlib,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from echo_routing.temporal import dense_runtime as data
from echo_routing.temporal.dev_train import lr_factor
from scripts.train_dense_sequence import RUN,PUB,CFG,read,write,sha,make_plan

def audit():
 assert read(RUN/'runtimecoordination.json')['phase']=='formal_matrix_complete_pending_frozen_outer_readout'
 cfg=read(CFG);selected=read(PUB/'inner_lr_selection.json')['selected_joint_arm'];seed1=cfg['seed'];seed2=2026110258;expected=[(seed1,'L',6000),(seed1,'F',6000),(seed1,selected,6000),(seed1,'J-high' if selected=='J-low' else 'J-low',2000),(seed2,'F',6000),(seed2,selected,6000)]
 source=torch.load(ROOT/cfg['source_state'],map_location='cpu',weights_only=True,mmap=True);inactive=set(read(PUB/'source.json')['inactive_video_names']);models=[];entries=[];total=0
 for seed,arm,count in expected:
  folder=RUN/f'seed{seed}'/arm;assert not (folder/'pending.json').exists();rows=[json.loads(s) for s in (folder/'physical_updates.jsonl').read_text().splitlines()];logical=[r['u'] for r in rows];assert logical==list(range(count)),(arm,'uninterrupted trace expected; any replay must be independently accounted')
  assert all(np.isfinite(r['loss']) and np.isfinite(r['gradient_norm']) and r['groups']['head']>0 and (not arm.startswith('J') or r['groups']['active_video']>0) for r in rows)
  z=torch.load(folder/'latest.pt',map_location='cpu',weights_only=True,mmap=True);assert z['update']==count and z['seed']==seed and z['arm']==arm and z['schedule_total']==6000
  st=z['model'];head_names=[n for n in st if n.startswith('head.')];active_names=['encoder.'+n for n in source if n not in inactive] if arm.startswith('J') else [];names=head_names+active_names;groups=z['optimizer']['param_groups'];assert len(groups)==(2 if arm.startswith('J') else 1);binding=[]
  for group,nn,lr,name in zip(groups,[head_names,active_names],[.001,cfg['arms'][arm].get('encoder_lr',0)],['head','active_video']):
   assert group['name']==name and len(group['params'])==len(nn) and group['base_lr']==lr and abs(group['lr']-lr*lr_factor(count-1,200,6000))<1e-15 and group['betas']==(.9,.999) and group['eps']==1e-8 and group['weight_decay']==.05
   binding.extend(zip(group['params'],nn))
  assert len(binding)==len(names) and len({x[0] for x in binding})==len(names)
  for key,name in binding:
   moment=z['optimizer']['state'][key];assert int(moment['step'])==count
   for mk in ['exp_avg','exp_avg_sq']:assert moment[mk].shape==st[name].shape and torch.isfinite(moment[mk]).all()
   assert (moment['exp_avg_sq']>=0).all() and moment['exp_avg_sq'].sum()>0
  changed=[]
  for name,initial in source.items():
   current=st['encoder.'+name];assert current.shape==initial.shape and torch.isfinite(current).all();drift=not torch.equal(current,initial)
   if name in inactive or not arm.startswith('J'):assert not drift
   else:assert drift;changed.append(name)
  inner=[]
  for step in range(500,count+1,500):
   identity=read(folder/f'step{step}.identity.json');path=folder/f'step{step}.pt';assert identity['sha256']==sha(path) and identity['source']==read(PUB/'source.json')['target_encoder_sha256'];record=read(folder/f'inner_step{step}/metrics.json');assert record['step']==step and record['seed']==seed and record['arm']==arm;inner.append(record)
   fixed=step in ([2000,6000] if count==6000 else [2000])
   if fixed:entries.append({'row':f'seed{seed}_{arm}_fixed{step}','kind':'fixed_step','seed':seed,'arm':arm,'step':step,'path':str(path.relative_to(ROOT)),'sha256':identity['sha256']})
  best=data.choose_inner(inner);path=folder/f"step{best['step']}.pt";entries.append({'row':f'seed{seed}_{arm}_inner_package','kind':'inner_selected_package','seed':seed,'arm':arm,'step':best['step'],'path':str(path.relative_to(ROOT)),'sha256':sha(path)})
  models.append({'seed':seed,'arm':arm,'updates':count,'physical_rows':len(rows),'unique_logical_updates':len(set(logical)),'head_optimizer_tensors':len(head_names),'active_video_optimizer_tensors':len(active_names),'changed_video_tensors':len(changed),'inactive_video_tensors_unchanged':9,'finite_moments_exact_shapes_steps_lr_binding':True,'mean_first100_loss':float(np.mean([r['loss'] for r in rows[:100]])),'mean_last100_loss':float(np.mean([r['loss'] for r in rows[-100:]])),'latest_optimizer_sha256':sha(folder/'latest.pt'),'inner_chosen_step':best['step']});total+=len(rows);del z
 assert total==32000 and len(entries)==17
 record={'passed':True,'formal_physical_updates':total,'models':models,'outer_rows':entries,'all_formal_models_frozen_before_first_external_development_read':True,'new_external_development_forwards':0,'source_sha256':read(PUB/'source.json')['target_encoder_sha256'],'producer_git':read(RUN/'pipeline_freeze.json')['git_commit'],'epoch':time.time()};write(PUB/'models_manifest.json',record);print(json.dumps({'model_audit_passed':True,'updates':total,'rows':len(entries)}),flush=True)
if __name__=='__main__':audit()
