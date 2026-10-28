"""Post-readout descriptive overlaps from immutable A46/A44 caches, no inference or tuning."""
import datetime
import hashlib
import json
import time
from pathlib import Path

import numpy as np

ROOT=Path.cwd();P=ROOT/'runs/video_local_motion_a5c_v1';R=ROOT/'docs/results/2026-10-28_local_motion_a5c'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def now():return datetime.datetime.now(datetime.UTC).isoformat()
started=now();old=ROOT/'runs/video_risk_constrained_a5c_v1/readout_cache_s0.npz';assert sha(old)=='526ab49bc0b69af56b23f8a1757921695127b834e642761080b9f3f8bcf7a5ef'
new=np.load(P/'private_readout.npz');previous=np.load(old);res={};counts={};allpairs={}
for h in [0,1]:
 b=new[f'pred_Base_h{h}'];y=new[f'label_h{h}'];assert np.array_equal(b,previous[f'run_pred_Base_heldout_f{h}']) and np.array_equal(y,previous[f'run_label_heldout_f{h}'])
 preds={k:new[f'pred_{k}_h{h}'] for k in ['F','S','Z_F','Z_S']};preds.update({f'A44_{k}':previous[f'run_pred_{k}_heldout_f{h}'] for k in ['C','N']});pairs={}
 for a,c in [('F','S'),('F','Z_F'),('S','Z_S'),('F','A44_C'),('F','A44_N')]:
  z={}
  for typ,mask in [('fix',(b==2)&(y==3)),('harm',(b==2)&(y==2))]:
   aa=mask&(preds[a]==3);cc=mask&(preds[c]==3);z[typ]={'shared':int((aa&cc).sum()),'first_only':int((aa&~cc).sum()),'second_only':int((cc&~aa).sum())}
  z['all_label_disagreements']=int((preds[a]!=preds[c]).sum());pairs[f'{a}_vs_{c}']=z
  for typ in ['fix','harm']:
   for k,v in z[typ].items():allpairs.setdefault(f'{a}_vs_{c}',{}).setdefault(typ,{}).setdefault(k,0);allpairs[f'{a}_vs_{c}'][typ][k]+=v
  allpairs[f'{a}_vs_{c}']['all_label_disagreements']=allpairs[f'{a}_vs_{c}'].get('all_label_disagreements',0)+z['all_label_disagreements']
 res[str(h)]=pairs
 counts[str(h)]={k:{'fix':int(((b==2)&(y==3)&(v==3)).sum()),'harm':int(((b==2)&(y==2)&(v==3)).sum())} for k,v in preds.items()}
r={'scope':'descriptive immutable cache overlaps, after once readout; no new classifier/threshold/model forward','folds':res,'oof_overlaps':allpairs,'counts':counts,'A44_cache_sha256':sha(old),'A46_cache_sha256':sha(P/'private_readout.npz'),'code_sha256':sha(Path(__file__)),'written':now()}
for p in [P/'cache_comparison.json',R/'cache_comparison.json']:p.write_text(json.dumps(r,indent=2)+'\n')
with (P/'ledger.jsonl').open('a') as f:f.write(json.dumps({'job':'A46 descriptive frozen-cache paired comparison actual process CPU incl imports','kind':'cpu','started':started,'finished':now(),'hours':time.process_time()/3600,'ok':True,'note':'no GPU, inference, variant, threshold choice or new source data'})+'\n')
print(json.dumps(allpairs,indent=1))
