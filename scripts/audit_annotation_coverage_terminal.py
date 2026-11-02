#!/usr/bin/env python3
"""Additional cache-only closure audit; no neural imports or model execution."""
import hashlib,json,time,os,fcntl
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1];O=R/'runs/a55_long_iteration_v1';P=R/'docs/results/2026-11-01_annotation_coverage'
def read(p):return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def verify_cell(root,key,identity,expected):
 progress=read(root/(key+'.progress.json'));seal=read(root/(key+'.complete.json'));a=np.load(root/(key+'.npy'))
 assert progress=={'identity':identity,'n':133,'done':133,'pending':None}
 assert seal=={'identity':identity,'rows':133,'sha256':sha(root/(key+'.npy'))}
 assert a.shape==(133,) and np.isfinite(a).all() and np.array_equal(a,expected)
 with (root/(key+'.lock')).open('a+') as f:fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
 return 133
def main():
 m=read(P/'input_manifest.json');models=read(P/'source_models_manifest.json');fr=read(O/'models_freeze.json');final=read(P/'independent_source_final.json');b=read(O/'bound.private.json');z=np.load(O/'source_scores.private.npz');old=read(R/'runs/a51_verified_direct_view_pilot_v1/bound.private.json');original=[x for x in old['requests'] if x['role']=='diagnostic'];assert len(original)==len(b['diagnostic'])==133
 for a,c in zip(original,b['diagnostic']):
  for k in ['id','directory_label','true_label','centre','frames','role']:assert a[k]==c[k]
 assert final['passed'] and final['raw_score_cache_sha256']==sha(O/'source_scores.private.npz') and fr['source_models_manifest_sha256']==sha(P/'source_models_manifest.json')
 total=0
 for key in ['SF','SC','DF','DC','G-C']:
  h=m['Base_A08_sha256'] if key=='G-C' else models['models'][key]['bundle_sha256'];total+=verify_cell(O/'source_cells',key,{'source_model_sha256':h,'bound_sha256':m['bound_sha256'],'rows':133},z[key])
 assert total==665 and np.array_equal(z['G-F'],np.load(R/'runs/a51_verified_direct_view_pilot_v1/source_scores.private.npz')['G'])
 assert sha(R/'models_frozen/video_within_view_v1/all_s0/final.pt')==m['Base_A08_sha256']
 sparse={x['id']:x for x in b['sparse']}
 for g in ['F','C']:
  ctl=b['geometry'][g];x=np.array([sparse[r['id']]['geometry'][g] for r in b['diagnostic']]);s=((x-ctl['fit_mean'])/ctl['fit_scale'])@np.array(ctl['coef'])+ctl['intercept'];assert np.allclose(s,z['geometry-'+g],rtol=0,atol=1e-12)
 prior=b['geometry']['F']['constant_prior'];assert np.array_equal(z['constant'],np.full(133,np.log(prior/(1-prior))))
 train=read(O/'process_train_1.json');audit=read(O/'process_models_1.json');ro=read(O/'process_readout_1.json');assert train['end_epoch']<=audit['start_epoch']<=audit['end_epoch']<fr['freeze_epoch']<ro['start_epoch']<=ro['end_epoch'];assert all(v['exit_code']==0 for v in [train,audit,ro]);assert not (O/'active_process.json').exists()
 y=np.array([r['true_label']=='A5C' for r in original]);paired={}
 for base in ['G-F','G-C']:
  for arm in ['SF','SC','DF','DC']:
   pred=z[arm]>0;bp=z[base]>0;paired[arm+'-'+base]={'additional_A5C_correct':int((y&pred&~bp).sum()),'lost_original_A5C_correct':int((y&~pred&bp).sum()),'new_A4C_errors_among_binary_base_correct':int((~y&pred&~bp).sum()),'fixed_original_A4C_errors':int((~y&~pred&bp).sum()),'shared_A4C_errors':int((~y&pred&bp).sum())}
 out={'passed':True,'audit_epoch':time.time(),'source_score_cache_sha256':sha(O/'source_scores.private.npz'),'exact_source133membership_and_order':True,'five_durable_cells_complete_without_pending':True,'new_source_neural_forwards_verified':total,'old_G_F_scores_reused':133,'all_cached_geometry_and_constant_scores_reconstructed':True,'Base_A08_unchanged':True,'training_then_full_audit_then_actualGitfreeze_then_once_readout_order_verified':True,'no_active_owned_process':True,'paired_source_binary_baseline_comparisons':paired,'new_neural_forwards_in_this_audit':0,'new_target_readouts':0,'scope':'Paired source binary-view scores; binary baseline correctness is not full five-classA08 or target103/1413/fullF1 protection.'};(P/'independent_once_provenance.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
