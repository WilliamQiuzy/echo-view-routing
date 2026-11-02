#!/usr/bin/env python3
"""CPU state and installed-graph audit; no neural forward/backward or weight changes."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
import sys,json,hashlib,inspect
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
import torch,torchvision
from torchvision.models.video import mvit as tv
from scripts import train_annotation_coverage as t
torch.set_num_threads(1)
O=R/'runs/a55_long_iteration_v1';P=R/'docs/results/2026-11-01_annotation_coverage';record=json.loads((P/'source_models_manifest.json').read_text());expected={'m.encoder.blocks.15.attn.rel_pos_h','m.encoder.blocks.15.attn.rel_pos_w','m.encoder.blocks.15.attn.rel_pos_t','m.encoder.blocks.15.attn.pool_q.pool.weight'}
assert 'x = x[:, 0]' in inspect.getsource(tv.MViT.forward) and 'attn[:, :, 1:, 1:] += rel_pos' in inspect.getsource(tv._add_rel_pos)
results={}
for arm in ['SF','SC','DF','DC']:
 bundle=torch.load(O/'source_train'/arm/'final.pt',map_location='cpu',weights_only=True);state=torch.load(O/'source_train'/arm/'optimizer_final.pt',map_location='cpu',weights_only=True);m,h=t.old.restore(bundle,'cpu');names={id(p):'m.'+n for n,p in m.named_parameters()};names.update({id(p):'h.'+n for n,p in h.named_parameters()});groups=t.old.t44.groups_of(m,h);zeros=[];selected=[]
 for group,(name,params) in zip(state['opt']['param_groups'],groups.items()):
  assert group['name']==name
  for i,param in zip(group['params'],params):
   n=names[id(param)];v=state['opt']['state'][i];assert int(v['step'])==3600;selected.append(n)
   if torch.count_nonzero(v['exp_avg'])==0 and torch.count_nonzero(v['exp_avg_sq'])==0:zeros.append(n)
 assert set(zeros)==expected and len(selected)==400 and not any(n.startswith('m.classifier.') for n in selected);drift=record['models'][arm]['drift_by_tensor'];assert all(drift[n]==0 for n in expected) and sum(drift[n]>0 for n in selected)==396
 pool_source=inspect.getsource(type(m.encoder.blocks[-1].attn.pool_q).forward);assert 'class_token' in pool_source and 'torch.cat' in pool_source
 results[arm]={'selected_optimizer_tensors':400,'optimizer_steps_each':3600,'changed_selected_tensors':396,'zero_moment_unchanged_selected_tensors':sorted(zeros),'old_five_class_head_excluded':True};del m,h,bundle,state
out={'passed':True,'models':results,'torch_version':torch.__version__,'torchvision_version':torchvision.__version__,'installed_mvit_module_sha256':hashlib.sha256(Path(inspect.getfile(tv)).read_bytes()).hexdigest(),'graph_explanation':'Final output reads CLS only; last relative positions modify nonCLS-query/nonCLS-key entries, and final query pooling convolution transforms nonCLS queries while CLS bypasses pooling. These four tensors have no path to final CLS loss. Layer norms and all other selected tensors remain live.','environment_scope':'Installed runtime snapshot at closure; source initial-state/production numerical checks were separately frozen before training.','new_neural_forwards':0,'new_neural_backwards':0,'new_target_readouts':0};(P/'independent_structural_zero.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
