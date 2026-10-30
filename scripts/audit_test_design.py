#!/usr/bin/env python3
"""Read-only CPU audit of the existing EV9V benchmark and saved A26 labels.

Does not generate recipes, run models or select settings. Public output contains
counts and metrics only; no recording/session IDs or per-case paths are exported.
"""
from __future__ import annotations
import hashlib,json,time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
REL='docs/results/2026-10-30_external_original_tests'
A26='docs/results/2026-10-23_frozen_external_benchmark'
CLASSES=['PLAX','PSAX','A4C','A5C','SC4C']
METRICS=['clip macro-F1','segment F1@50, pairs','segment F1@50, multi','native cuts per clip']
H09_TO_A26={'B5 ASFormer v2':'ASFormer_v2_frameorder','B6 STFM windowed, numeric-order model':'STFM_numeric_s100',
 'B7 EchoViewCLIP windowed':'EchoViewCLIP_stage1_windowed','B8 EchoPrime per frame':'EchoPrime_view_per_frame',
 'B8 + audited B3 (EchoPrime setting)':'EchoPrime_view_B3'}

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def quantiles(values):return np.quantile(values,[0,.25,.5,.75,1]).tolist()
def session(s):return s.rsplit('_',1)[0]

def main():
 start=time.monotonic();root=ROOT
 mp=root/'runs/frozen_external_benchmark_v1/benchmark/manifest.json';man=read(mp)
 summary=read(root/A26/'benchmark_summary.json');ev=read(root/A26/'evaluation_v1.json')
 digest=hashlib.sha256(json.dumps(man['items'],sort_keys=True,separators=(',',':'),default=str).encode()).hexdigest()
 assert digest==summary['manifest_digest']==ev['benchmark_digest'],'benchmark digest mismatch'
 items=man['items'];by_id={x['id']:x for x in items};stats={}
 for kind in ['native','pairs','multi']:
  group=[x for x in items if x['kind']==kind];cines=set();sess=set();frag_lengths=[];seg_lengths=[]
  class_samples=np.zeros(5,dtype=int);class_segments=np.zeros(5,dtype=int);class_fragments=np.zeros(5,dtype=int)
  transitions=np.zeros((5,5),dtype=int);class_cines=[set() for _ in range(5)]
  n_join=0;n_cross=0;n_gamma_join=0;n_within_repeat=0
  cells={}
  for x in group:
   y=np.array(x['labels']);assert len(y)==x['n'];class_samples+=np.bincount(y,minlength=5)
   edges=np.r_[0,np.flatnonzero(y[1:]!=y[:-1])+1,len(y)]
   assert edges[1:-1].tolist()==x['semantic']
   seg_lengths.extend(np.diff(edges).tolist());class_segments+=np.bincount(y[edges[:-1]],minlength=5)
   for edge in x['semantic']:transitions[y[edge-1],y[edge]]+=1
   fr=x['fragments'];n_within_repeat+=len(fr)-len({f['video_id'] for f in fr})
   for f in fr:
    cines.add(f['video_id']);class_cines[f['label']].add(f['video_id']);sess.add(session(f['video_id']));frag_lengths.append(f['end']-f['start']);class_fragments[f['label']]+=1
   for a,b in zip(fr,fr[1:]):
    n_join+=1;n_cross+=session(a['video_id'])!=session(b['video_id']);n_gamma_join+=a['variant']!=b['variant']
   if kind=='pairs':
    same=fr[0]['label']==fr[1]['label'];edit=fr[1]['variant']!='orig';key=('same' if same else 'different')+('_gamma' if edit else '_original');cells[key]=cells.get(key,0)+1
  st={'items':len(group),'samples':sum(x['n'] for x in group),'unique_recordings':len(cines),'sessions':len(sess),
   'fragments':len(frag_lengths),'true_segments':len(seg_lengths),'short_true_segments_le16':sum(v<=16 for v in seg_lengths),
   'physical_joins':n_join,'semantic_changes':sum(len(x['semantic']) for x in group),'cross_session_joins':n_cross,
   'gamma_variant_changes_at_joins':n_gamma_join,'repeated_recording_within_same_stream':n_within_repeat,
   'fragment_seconds_min_q25_median_q75_max':(np.array(quantiles(frag_lengths))/10).tolist(),
   'merged_true_segment_seconds_min_q25_median_q75_max':(np.array(quantiles(seg_lengths))/10).tolist(),
   'per_class_samples':dict(zip(CLASSES,map(int,class_samples))),
   'per_class_fragments':dict(zip(CLASSES,map(int,class_fragments))),
   'per_class_true_segments':dict(zip(CLASSES,map(int,class_segments))),
   'per_class_unique_recordings':dict(zip(CLASSES,map(len,class_cines))),
   'semantic_transition_counts':transitions.tolist(),
   'gamma_samples':{str(g):sum(r[2]==g for x in group for r in x['refs']) for g in [.9,1.,1.1]},'pair_cells':cells}
  for k,old in [('items','items'),('samples','samples'),('unique_recordings','clips'),('sessions','sessions'),('true_segments','true_segments'),('short_true_segments_le16','true_segments_le16'),('semantic_changes','true_changes')]:
   assert st[k]==summary['counts'][kind][old],'summary denominator mismatch'
  stats[kind]=st
 hashes={'manifest':sha(mp),'A26_evaluation':sha(root/A26/'evaluation_v1.json'),'summary':sha(root/A26/'benchmark_summary.json'),'audit_script':sha(Path(__file__))}
 diag={}
 for name,row in ev['values'].items():
  p=root/'runs/frozen_external_benchmark_v1/runs/v1/predictions'/f'{name}.npz';hashes[name+'_labels']=sha(p)
  with np.load(p,allow_pickle=False) as z:
   ids=z['ids'].tolist();offsets=z['offsets'];labels=z['labels']
  assert set(ids)==set(by_id) and len(ids)==1160 and len(offsets)==1161,'prediction membership mismatch'
  cm=np.zeros((5,5),int);bands={'near_semantic_change_le08s':[0,0],'outside_change_band':[0,0]};native_cuts=0
  for i,k in enumerate(ids):
   x=by_id[k];y=np.array(x['labels']);pred=labels[offsets[i]:offsets[i+1]]
   assert pred.shape==y.shape and np.isin(pred,np.arange(5)).all(),'prediction alignment mismatch'
   if x['kind']=='native':
    majority=int(np.bincount(pred,minlength=5).argmax());cm[y[0],majority]+=1;native_cuts+=np.count_nonzero(pred[1:]!=pred[:-1])
   elif x['kind']=='multi':
    near=np.zeros(len(y),bool)
    for edge in x['semantic']:near[max(0,edge-8):min(len(y),edge+8)]=True
    for key,mask in [('near_semantic_change_le08s',near),('outside_change_band',~near)]:
     bands[key][0]+=int(np.count_nonzero(pred[mask]==y[mask]));bands[key][1]+=int(np.count_nonzero(mask))
  den=cm.sum(0)+cm.sum(1);macro=float(np.mean(2*np.diag(cm)/den));acc=float(np.trace(cm)/cm.sum())
  assert abs(macro-row['clip macro-F1']['value'])<1e-12 and abs(acc-row['clip accuracy']['value'])<1e-12,'native recount mismatch'
  assert native_cuts/800==row['native cuts per clip']['value'],'cut count mismatch'
  diag[name]={'native_accuracy':acc,'native_macro_f1':macro,'native_confusion':cm.tolist(),'native_internal_cuts':int(native_cuts),
   'multi_accuracy_bands':{k:{'correct':v[0],'samples':v[1],'accuracy':v[0]/v[1]} for k,v in bands.items()}}
 hpath=root/'runs/_demo/true_order_joint_b5/joint.json';h09=read(hpath);hashes['H09_joint']=sha(hpath)
 for old,new in H09_TO_A26.items():
  for metric in METRICS:assert h09['endpoints'][old][metric]['value']==ev['values'][new][metric]['value'],'H09/A26 shared-row mismatch'
 h09rows=[{'name':name,**{metric:v[metric]['value'] for metric in METRICS}} for name,v in h09['endpoints'].items()]
 out={'passed':True,'kind':'read_only_frozen_benchmark_design_recount','created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
  'benchmark_digest':digest,'counts':stats,'A26_native_and_band_recount':diag,'H09_rows':h09rows,'H09_A26_shared_rows_exact':True,
  'hashes':hashes,'model_forwards':0,'GPU_seconds':0,'elapsed_wall_seconds':time.monotonic()-start,
  'limits':['Read-only saved-label audit; no model or raw pixels reloaded. No recipes regenerated.',
   'Near-change band is a fixed eight-sample half-window; differences are descriptive, confounded by class/duration and do not isolate causal effects of construction.',
   'Recording sessions are not verified patients; repeated sources across constructed streams are dependent.',
   'H09 rows are historical corrected same-bank whole-pipeline results, not original-paper task scores.']}
 p=root/REL/'test_design_audit.json';p.write_text(json.dumps(out,indent=2)+'\n')
 print(json.dumps({'passed':True,'benchmark_digest':digest,'A26_models':len(diag),'H09_models':len(h09rows),'counts':{k:{x:v[x] for x in ['items','fragments','true_segments','cross_session_joins','physical_joins']} for k,v in stats.items()},'elapsed_wall_seconds':out['elapsed_wall_seconds']}))

if __name__=='__main__':main()
