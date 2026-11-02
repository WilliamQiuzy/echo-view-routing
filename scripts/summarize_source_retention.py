#!/usr/bin/env python3
"""A56 aggregate-only readout; no model construction or inference."""
import hashlib,json,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from echo_routing.temporal.direct_view_training import write_json
OUT=ROOT/'runs/a56_source_retention_v1';PUB=ROOT/'docs/results/2026-11-02_source_retention'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())


def auc(y,s):
 y=np.asarray(y,bool);s=np.asarray(s,float);pos=s[y];neg=s[~y]
 if not len(pos) or not len(neg):return float('nan')
 return float(((pos[:,None]>neg).sum()+.5*(pos[:,None]==neg).sum())/(len(pos)*len(neg)))


def metrics(y,s):
 y=np.asarray(y,bool);p=np.asarray(s)>0;tp=int((p&y).sum());fp=int((p&~y).sum());tn=int((~p&~y).sum());fn=int((~p&y).sum())
 return {'tp':tp,'fp':fp,'tn':tn,'fn':fn,'A5C_total':int(y.sum()),'A4C_total':int((~y).sum()),'TPR':tp/int(y.sum()),'FPR':fp/int((~y).sum()),'balanced_accuracy':.5*(tp/int(y.sum())+tn/int((~y).sum())),'AUC':auc(y,s)}


def main():
 receipt=read(PUB/'readout_receipt.json');manifest=read(PUB/'input_manifest.json');assert receipt['all_cells_complete'] and sha(OUT/'source_scores.private.npz')==receipt['private_score_sha256'];private=read(OUT/'source_labels.private.json');y=np.array(private['y'],bool);patients=np.array(private['patient']);videos=np.array(private['video']);scores=np.load(OUT/'source_scores.private.npz');oldread=read(ROOT/'docs/results/2026-11-01_annotation_coverage/source_readout.json');assert sha(ROOT/'docs/results/2026-11-01_annotation_coverage/source_readout.json')==manifest['frozen_source_reference_readout_sha256'];oldcache=ROOT/'runs/a55_long_iteration_v1/source_scores.private.npz';assert sha(oldcache)==oldread['raw_score_cache_sha256'];G=np.load(oldcache)['G-C'];margins={a:scores[a][:,3]-scores[a][:,2] for a in ('Z','R')};margins['G-C']=G;met={a:metrics(y,s) for a,s in margins.items()};paired={};confusion={}
 for a in ('Z','R'):
  q=scores[a];assert q.shape==(133,5) and np.isfinite(q).all();pred=q.argmax(1);table=np.zeros((2,5),int)
  for t in (0,1):
   for p in range(5):table[t,p]=int(((y==t)&(pred==p)).sum())
  confusion[a]={'rows_true_labels':['A4C','A5C'],'columns_original_labels':['PLAX','PSAX','A4C','A5C','SC4C'],'counts':table.tolist(),'five_class_correct':int((pred==y.astype(int)+2).sum()),'other_class_predictions':int((~np.isin(pred,[2,3])).sum()),'pairwise_positive_but_argmax_other_class':int(((margins[a]>0)&(~np.isin(pred,[2,3]))).sum()),'actual_A4C_to_A5C':int(((~y)&(pred==3)).sum()),'actual_A5C_to_A4C':int((y&(pred==2)).sum())}
 for left,right in [('R','Z'),('R','G-C'),('Z','G-C')]:
  lp,rp=margins[left]>0,margins[right]>0;parts={}
  for label,mask in [('A5C',y),('A4C',~y)]:
   discord=mask&(lp!=rp);parts[label]={'total':int(mask.sum()),'both_positive':int((mask&lp&rp).sum()),'left_only_positive':int((mask&lp&~rp).sum()),'right_only_positive':int((mask&~lp&rp).sum()),'both_negative':int((mask&~lp&~rp).sum()),'disagreement_patient_count':int(len(set(patients[discord]))),'disagreement_video_count':int(len(set(videos[discord])))}
  paired[left+'_minus_'+right]=parts
 r,z,g=met['R'],met['Z'],met['G-C'];absolute={'tp_at_least28':r['tp']>=28,'fp_at_most5':r['fp']<=5,'BA_at_least_point80':r['balanced_accuracy']>=.8,'AUC_at_least_point85':r['AUC']>=.85,'AUC_at_least_GC_plus_point03':r['AUC']>=g['AUC']+.03};contribution={'at_least4_fewer_FP':r['fp']<=z['fp']-4,'at_most2_fewer_TP':r['tp']>=z['tp']-2,'AUC_increment_at_least_point03':r['AUC']>=z['AUC']+.03}
 rng=np.random.default_rng(2026110256);clusters=np.unique(patients);draws=[]
 for k in range(2000):
  drawn=rng.choice(clusters,len(clusters),replace=True);idx=np.concatenate([np.flatnonzero(patients==p) for p in drawn]);yy=y[idx]
  if not yy.any() or yy.all():continue
  rr,zz=metrics(yy,margins['R'][idx]),metrics(yy,margins['Z'][idx]);draws.append([rr[x]-zz[x] for x in ('TPR','FPR','balanced_accuracy','AUC')])
 arr=np.asarray(draws);interval={name:{'point':r[name]-z[name],'percentile95':np.quantile(arr[:,i],[.025,.975]).tolist()} for i,name in enumerate(('TPR','FPR','balanced_accuracy','AUC'))};retention={};base=np.asarray(read(OUT/'initial_panel.private.json')['logits']);plan=np.load(OUT/'plan.private.npz');labels=np.repeat(np.arange(5),50)
 for a in ('Z','R'):
  fit=np.load(OUT/'models'/a/'fit_final.private.npz');q=fit['replay'];pred=q.argmax(1);bp=base.argmax(1);retention[a]={'role':'FIT_training_retention_only','rows':250,'class_totals':[50]*5,'correct':int((pred==labels).sum()),'Base_correct':int((bp==labels).sum()),'kept_Base_correct':int(((bp==labels)&(pred==labels)).sum()),'new_errors_among_Base_correct':int(((bp==labels)&(pred!=labels)).sum()),'new_correct_among_Base_wrong':int(((bp!=labels)&(pred==labels)).sum()),'correct_by_class':[int(((pred==labels)&(labels==c)).sum()) for c in range(5)],'Base_correct_by_class':[int(((bp==labels)&(labels==c)).sum()) for c in range(5)]}
 source_y=np.array([row['label']+2 for row in read(ROOT/'runs/a55_long_iteration_v1/bound.private.json')['parents']])[np.linspace(0,568,32,dtype=int)];source_fit={}
 for a in ('Z','R'):
  q=np.load(OUT/'models'/a/'fit_final.private.npz')['source'];pred=q.argmax(1);margin=q[:,3]-q[:,2];source_fit[a]={'role':'FIT_training_behavior_only','rows':32,'five_class_correct':int((pred==source_y).sum()),'A5C_total':int((source_y==3).sum()),'A4C_total':int((source_y==2).sum()),'A5C_pairwise_correct':int(((source_y==3)&(margin>0)).sum()),'A4C_pairwise_false_positive':int(((source_y==2)&(margin>0)).sum())}
 result={'run_id':'A56','scope':'Repeated10source-patient exploratory source diagnostic; no target OOF, clinical validation or publication-goal claim. FIT retention is training behavior only.','raw_logits_margin':'original_classifier_pre_softmax_q3_minus_q2; positive strictly>0','metrics':met,'actual_fiveclass_source':confusion,'paired':paired,'R_absolute_components':absolute,'R_absolute_passed':all(absolute.values()),'R_minus_Z_components':contribution,'R_minus_Z_passed':all(contribution.values()),'combined_passed':all(absolute.values()) and all(contribution.values()),'patient_cluster_bootstrap':{'draws':2000,'valid_draws':len(draws),'patients':len(clusters),'videos':len(set(videos)),'correlated_source_requests':133,'seed':2026110256,'R_minus_Z':interval,'interpretation':'descriptive intervals on repeatedly reused source patients, not fresh or clinical validation'},'FIT_retention':retention,'FIT_source_panel':source_fit,'source_patient_class_coverage':{label:{'patients':len(set(patients[mask])),'videos':len(set(videos[mask])),'requests':int(mask.sum())} for label,mask in [('A4C',~y),('A5C',y)]},'source_score_cache_sha256':sha(OUT/'source_scores.private.npz'),'new_source_diagnostic_forwards':receipt['new_source_model_forwards'],'target_readout':0,'optimizer_updates':2400,'training_forwards':38400,'training_backwards':38400,'timestamp_epoch':time.time()};write_json(PUB/'source_readout.json',result);print(json.dumps({'metrics':met,'absolute':all(absolute.values()),'contribution':all(contribution.values()),'combined':result['combined_passed'],'FIT_retention':retention}),flush=True)


if __name__=='__main__':main()
