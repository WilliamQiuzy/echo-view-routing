#!/usr/bin/env python3
"""Independent cached-score/state-membership recount for A56; zero inference."""
import hashlib,json,sys,time
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score,confusion_matrix
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from echo_routing.temporal.direct_view_training import write_json
OUT=ROOT/'runs/a56_source_retention_v1';PUB=ROOT/'docs/results/2026-11-02_source_retention'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
 r=read(PUB/'source_readout.json');rec=read(PUB/'readout_receipt.json');m=read(PUB/'models_manifest.json');inp=read(PUB/'input_manifest.json');labels=read(OUT/'source_labels.private.json');source=read(ROOT/'runs/a55_long_iteration_v1/bound.private.json')['diagnostic'];y=np.array([row['true_label']=='A5C' for row in source]);assert labels['y']==y.astype(int).tolist() and labels['patient']==[row['patient'] for row in source] and labels['video']==[row['video'] for row in source];scores=np.load(OUT/'source_scores.private.npz');assert sha(OUT/'source_scores.private.npz')==r['source_score_cache_sha256']==rec['private_score_sha256'];metric={};margins={};actual={}
 for a in ('Z','R'):
  q=scores[a];assert q.shape==(133,5) and np.isfinite(q).all();parts=[]
  for c in range(5):
   base=OUT/'source_cells'/(a+str(c));cell=read(str(base)+'.progress.json');complete=read(str(base)+'.complete.json');assert cell['done']==cell['n']==133 and cell['pending'] is None;assert cell['identity']==complete['identity'] and cell['identity']['model_sha256']==m['models'][a]['final_full_state_sha256'] and cell['identity']['source_manifest_sha256']==sha(PUB/'input_manifest.json') and cell['identity']['column']==c and complete['sha256']==sha(str(base)+'.npy');parts.append(np.load(str(base)+'.npy'))
  assert np.array_equal(q,np.stack(parts,axis=1));s=q[:,3]-q[:,2];margins[a]=s;pred=s>0;tn,fp,fn,tp=confusion_matrix(y,pred,labels=[False,True]).ravel();au=float(roc_auc_score(y,s));metric[a]={'tp':int(tp),'fp':int(fp),'tn':int(tn),'fn':int(fn),'balanced_accuracy':.5*(tp/(tp+fn)+tn/(tn+fp)),'AUC':au}
  for k,v in metric[a].items():assert abs(v-r['metrics'][a][k])<1e-12,(a,k)
  confusion=np.zeros((2,5),int)
  for truth,row in zip(y,q):confusion[int(truth),int(np.argmax(row))]+=1
  assert confusion.tolist()==r['actual_fiveclass_source'][a]['counts'] and int(confusion.sum())==133;actual[a]=confusion.tolist()
 oldpath=ROOT/'runs/a55_long_iteration_v1/source_scores.private.npz';oldread=read(ROOT/'docs/results/2026-11-01_annotation_coverage/source_readout.json');assert sha(oldpath)==oldread['raw_score_cache_sha256'] and sha(ROOT/'docs/results/2026-11-01_annotation_coverage/source_readout.json')==inp['frozen_source_reference_readout_sha256'];margins['G-C']=np.load(oldpath)['G-C'];assert abs(roc_auc_score(y,margins['G-C'])-r['metrics']['G-C']['AUC'])<1e-12
 for left,right in [('R','Z'),('R','G-C'),('Z','G-C')]:
  for label,truth in [('A4C',False),('A5C',True)]:
   p=r['paired'][left+'_minus_'+right][label];both=leftonly=rightonly=neither=0;patients=set();videos=set()
   for i,row in enumerate(source):
    if bool(y[i])!=truth:continue
    lp,rp=margins[left][i]>0,margins[right][i]>0
    if lp and rp:both+=1
    elif lp:leftonly+=1
    elif rp:rightonly+=1
    else:neither+=1
    if lp!=rp:patients.add(row['patient']);videos.add(row['video'])
   assert [both,leftonly,rightonly,neither,len(patients),len(videos)]==[p[k] for k in ['both_positive','left_only_positive','right_only_positive','both_negative','disagreement_patient_count','disagreement_video_count']]
 rr,zz=metric['R'],metric['Z'];gg=r['metrics']['G-C'];absolute=rr['tp']>=28 and rr['fp']<=5 and rr['balanced_accuracy']>=.8 and rr['AUC']>=.85 and rr['AUC']>=gg['AUC']+.03;contrib=rr['fp']<=zz['fp']-4 and rr['tp']>=zz['tp']-2 and rr['AUC']>=zz['AUC']+.03;assert r['R_absolute_passed']==absolute and r['R_minus_Z_passed']==contrib and r['combined_passed']==(absolute and contrib)
 selection=read(OUT/'selection.private.json');py=np.array([row['label'] for row in selection['panel_exact_windows']]);base=np.asarray(read(OUT/'initial_panel.private.json')['logits']).argmax(1)
 for a in ('Z','R'):
  q=np.load(OUT/'models'/a/'fit_final.private.npz')['replay'];p=q.argmax(1);v=r['FIT_retention'][a];assert len(p)==250 and np.bincount(py,minlength=5).tolist()==[50]*5;assert int((p==py).sum())==v['correct'] and int((base==py).sum())==v['Base_correct'] and int(((base==py)&(p!=py)).sum())==v['new_errors_among_Base_correct'];assert [int(((p==py)&(py==c)).sum()) for c in range(5)]==v['correct_by_class']
 assert rec['new_source_model_forwards']==r['new_source_diagnostic_forwards']==266 and r['target_readout']==0 and r['training_forwards']==r['training_backwards']==38400 and r['optimizer_updates']==2400;assert sha(ROOT/'models_frozen/video_within_view_v1/all_s0/final.pt')==inp['Base_A08_sha256'] and sha(ROOT/'runs/a55_long_iteration_v1/final_resource_receipt.json')==inp['inherited_A55_finalreceipt_sha256'];assert read(PUB/'independent_models.json')['passed'];write_json(PUB/'independent_final.json',{'passed':True,'readout_sha256':sha(PUB/'source_readout.json'),'score_cache_sha256':sha(OUT/'source_scores.private.npz'),'independent_sklearn_metrics':metric,'all10_cells_complete_identity_order_exact':True,'all133_source_labels_patients_videos_reconstructed':True,'full5class_confusion':actual,'paired_sets_counts_patient_video_reconstructed':True,'absolute_and_contribution_gates_independently_recomputed':True,'FIT_retention_independently_recounted':True,'Base_and_inherited_receipt_unchanged':True,'model_forwards':0,'target_readout':0,'epoch':time.time()});print(json.dumps({'independent_final_passed':True,'metrics':metric,'absolute':absolute,'contribution':contrib}),flush=True)


if __name__=='__main__':main()
