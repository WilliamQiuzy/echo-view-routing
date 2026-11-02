#!/usr/bin/env python3
"""New cache-only retrospective diagnostic; never calibrates/deploys an old model."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import hashlib,json,resource,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/post_a56_score_diagnostic_v1';PUB=ROOT/'docs/results/2026-11-02_post_a56_score_diagnostic'
CARD='docs/superpowers/plans/2026-11-02-post-a56-score-diagnostic.md'
BUDGETS=(0,2,5,10,20)


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,value):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
 with tmp.open('w') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 tmp.replace(p)


def validate(y,s):
 y=np.asarray(y,bool);s=np.asarray(s,float)
 if y.ndim!=1 or y.shape!=s.shape or not y.any() or y.all() or not np.isfinite(s).all():raise ValueError('finite scores and both fixed truth classes required')
 return y,s


def auc(y,s):
 y,s=validate(y,s);p,n=s[y],s[~y]
 return float(((p[:,None]>n).sum()+.5*(p[:,None]==n).sum())/(len(p)*len(n)))


def operating(y,s,threshold=0.):
 y,s=validate(y,s);p=s>threshold;tp=int((p&y).sum());fp=int((p&~y).sum());tn=int((~p&~y).sum());fn=int((~p&y).sum())
 return {'tp':tp,'fp':fp,'tn':tn,'fn':fn,'balanced_accuracy':.5*(tp/int(y.sum())+tn/int((~y).sum()))}


def roc_points(y,s):
 y,s=validate(y,s);thresholds=[float('inf'),*np.unique(s)[::-1].tolist(),-float('inf')]
 return [{'threshold':t,**operating(y,s,t)} for t in thresholds]


def budget_envelope(points,budgets):
 result=[]
 for budget in budgets:
  if int(budget)!=budget or budget<0:raise ValueError('nonnegative integer false-positive budget required')
  admitted=[p for p in points if p['fp']<=budget];best=max(p['tp'] for p in admitted);winning=[p for p in admitted if p['tp']==best]
  result.append({'FP_budget':int(budget),'max_tp':best,'attained_fp':min(p['fp'] for p in winning),'admissible_thresholds':len(admitted),'maximizing_threshold_count':len(winning)})
 return result


def main():
 start=time.time();OUT.mkdir(parents=True,exist_ok=True);PUB.mkdir(parents=True,exist_ok=True);freeze=read(OUT/'input_freeze.json');assert freeze['actual_git_objects_verified'] and freeze['private_ROC_arithmetic_before_freeze']==0
 for path,h in freeze['objects'].items():assert sha(ROOT/path)==h,path
 assert not (PUB/'summary.json').exists(),'retrospective readout already sealed; do not silently replace'
 old55=ROOT/'runs/a55_long_iteration_v1';old56=ROOT/'runs/a56_source_retention_v1';r55=read(ROOT/'docs/results/2026-11-01_annotation_coverage/source_readout.json');r56=read(ROOT/'docs/results/2026-11-02_source_retention/source_readout.json');m56=read(ROOT/'docs/results/2026-11-02_source_retention/input_manifest.json');assert sha(old55/'source_scores.private.npz')==r55['raw_score_cache_sha256'];assert sha(old56/'source_scores.private.npz')==r56['source_score_cache_sha256'];assert sha(old55/'bound.private.json')==m56['A55_bound_sha256'];assert sha(old56/'selection.private.json')==m56['selection_sha256'];b=read(old55/'bound.private.json')['diagnostic'];sel=read(old56/'selection.private.json');labels=read(old56/'source_labels.private.json');y=np.array([r['true_label']=='A5C' for r in b]);patients=np.array([r['patient'] for r in b]);videos=np.array([r['video'] for r in b]);assert len(b)==133 and int(y.sum())==39 and sel['source_diagnostic_ids']==[r['id'] for r in b];assert labels['y']==y.astype(int).tolist() and labels['patient']==patients.tolist() and labels['video']==videos.tolist()
 c55,c56=np.load(old55/'source_scores.private.npz'),np.load(old56/'source_scores.private.npz');scores={a:c55[a] for a in ('G-F','G-C','SF','SC','DF','DC')};scores.update({a:c56[a][:,3]-c56[a][:,2] for a in ('Z','R')});historical={};curves={}
 # First reproduce every old operating-point count and AUC; stop before envelopes on mismatch.
 for arm,s in scores.items():
  observed={**operating(y,s), 'AUC':auc(y,s)};reference=(r55 if arm in c55.files else r56)['metrics'][arm]
  assert all(abs(observed[k]-reference[k])<=1e-12 for k in observed),(arm,'historical mismatch')
  historical[arm]=observed
 summary={}
 for arm,s in scores.items():
  points=roc_points(y,s);envelope=budget_envelope(points,BUDGETS);pred=s>0;error={}
  for name,mask in [('false_positive',pred&~y),('false_negative',~pred&y)]:
   counts=sorted([int((patients[mask]==p).sum()) for p in set(patients[mask])],reverse=True);error[name]={'requests':int(mask.sum()),'patients':len(counts),'videos':len(set(videos[mask])),'patient_error_counts_descending_without_ids':counts}
  summary[arm]={'original_point':historical[arm],'empirical_envelope':envelope,'original_error_concentration':error,'unique_scores':len(np.unique(s)),'ROC_point_count_with_infinity_endpoints':len(points)};curves[arm+'_threshold']=np.array([p['threshold'] for p in points]);curves[arm+'_tp']=np.array([p['tp'] for p in points]);curves[arm+'_fp']=np.array([p['fp'] for p in points])
 np.savez(OUT/'ROC.private.npz',**curves)
 # Independent sklearn ROC trapezoid AUC and complete tie-group budget recount.
 from sklearn.metrics import roc_auc_score,roc_curve
 for arm,s in scores.items():
  assert abs(roc_auc_score(y,s)-historical[arm]['AUC'])<1e-12
  fpr,tpr,_=roc_curve(y,s,drop_intermediate=False);fps=np.rint(fpr*94).astype(int);tps=np.rint(tpr*39).astype(int)
  for budget,cell in zip(BUDGETS,summary[arm]['empirical_envelope']):assert int(tps[fps<=budget].max())==cell['max_tp']
 result={'diagnostic_id':'POST-A56-ROC','retrospective':True,'not_old_round_rescue':True,'deployment_calibrator_fitted':False,'historical_gates_and_outcomes_unchanged':True,'reference_assisted_optimistic_envelope':True,'arms':summary,'FP_budgets':list(BUDGETS),'A5C_total':39,'A4C_total':94,'patients':10,'videos':13,'class_group_denominators':{name:{'requests':int(mask.sum()),'patients':len(set(patients[mask])),'videos':len(set(videos[mask]))} for name,mask in [('A5C',y),('A4C',~y)]},'original_membership_order_and_historical_points_reproduced':True,'independent_sklearn_AUC_and_ROC_budget_recount_passed':True,'score_cache_sha256':{'A55':sha(old55/'source_scores.private.npz'),'A56':sha(old56/'source_scores.private.npz')},'source_readout_sha256':{'A55':sha(ROOT/'docs/results/2026-11-01_annotation_coverage/source_readout.json'),'A56':sha(ROOT/'docs/results/2026-11-02_source_retention/source_readout.json')},'private_ROC_sha256':sha(OUT/'ROC.private.npz'),'tie_rule':'strict score>threshold; whole tied groups; +inf,all distinct scores,-inf; minimum FP among TP-maximizers','public_export':'aggregates only; no scores, identifiers or threshold values','new_model_forwards':0,'optimizer_updates':0,'GPU_hours':0,'epoch':time.time()};write(PUB/'summary.json',result);u=resource.getrusage(resource.RUSAGE_SELF);verification={'passed':True,'freeze_git_commit':freeze['git_commit'],'summary_sha256':sha(PUB/'summary.json'),'historical_membership_points_AUC_reproduced':True,'independent_sklearn_budget_and_AUC_recount':True,'focused_tests':3,'new_model_forwards':0,'GPU_hours':0,'worker_CPU_core_seconds':u.ru_utime+u.ru_stime,'worker_wall_seconds':time.time()-start,'new_namespace_bytes':sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file()),'caps':{'CPU_core_seconds':1200,'new_bytes':100000000,'GPU_hours':0}};assert verification['worker_CPU_core_seconds']<1200 and verification['new_namespace_bytes']<100000000;write(PUB/'verification.json',verification);print(json.dumps({'historical_reproduction_passed':True,'independent_recount_passed':True,'maxTP_at_FP5':{a:v['empirical_envelope'][2]['max_tp'] for a,v in summary.items()},'CPU_core_seconds':verification['worker_CPU_core_seconds']}),flush=True)


if __name__=='__main__':main()
