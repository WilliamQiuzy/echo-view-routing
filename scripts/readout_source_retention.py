#!/usr/bin/env python3
"""A56 separately frozen corrected reader; trained producer remains immutable."""
import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES','0');os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,json,resource,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from scripts import train_source_retention as t
from echo_routing.temporal.source_retention_readout import score_rows
from echo_routing.temporal import direct_view_training as d


def readout():
 inputs=t.verify();assert torch.__version__==inputs['versions']['torch'] and np.__version__==inputs['versions']['numpy'];freeze=t.j(t.OUT/'models_freeze.json');reader=t.j(t.OUT/'readout_freeze.json');assert freeze['actual_git_objects_verified'] and freeze['models_manifest_sha256']==t.sha(t.PUB/'models_manifest.json');assert t.j(t.PUB/'independent_models.json')['passed'] and reader['actual_git_objects_verified']
 for p,h in reader['code'].items():assert t.sha(ROOT/p)==h,p
 assert t.j(t.PUB/'matched_inputs.json')['passed'] and t.j(t.PUB/'matched_inputs.json')['models_manifest_sha256']==t.sha(t.PUB/'models_manifest.json')
 if (t.PUB/'readout_receipt.json').exists():
  complete=t.j(t.PUB/'readout_receipt.json');assert complete['all_cells_complete'] and complete['models_freeze_git_commit']==freeze['git_commit'] and t.sha(t.OUT/'source_scores.private.npz')==complete['private_score_sha256'];print(json.dumps({'existing_complete_readout_preserved':True,'new_model_forwards':0}),flush=True);return
 man=t.j(t.PUB/'models_manifest.json');data=t.Inputs();allq={};new=0
 for arm in t.ARMS:
  entry=man['models'][arm];rd=t.OUT/'models'/arm;assert t.sha(rd/'final.pt')==entry['bundle_sha256'];m=t.build();m.load_state_dict(torch.load(rd/'final.pt',map_location='cpu',weights_only=True)['model']);cells=[]
  try:
   for c in range(5):cells.append(d.open_cell(t.OUT/'source_cells',arm+str(c),{'model_sha256':entry['final_full_state_sha256'],'column':c,'source_manifest_sha256':t.sha(t.PUB/'input_manifest.json')},133))
   def infer(i):
    with torch.no_grad():return t.logits(m,data.source.diagnostic(data.source.diag[i],'C'),8).cpu().numpy()[0]
   new+=score_rows(cells,infer);allq[arm]=np.stack([np.asarray(cell['scores']).copy() for cell in cells],axis=1)
  finally:
   for cell in cells:d.close_cell(cell)
  del m;torch.cuda.empty_cache()
 path=t.OUT/'source_scores.private.npz';tmp=path.with_suffix('.tmp')
 with tmp.open('wb') as f:np.savez(f,**allq);f.flush();os.fsync(f.fileno())
 tmp.replace(path);d.sync_directory(t.OUT)
 d.write_json(t.OUT/'source_labels.private.json',{'y':[int(r['true_label']=='A5C') for r in data.source.diag],'discordant':[r['true_label']!=r['directory_label'] for r in data.source.diag],'patient':[r['patient'] for r in data.source.diag],'video':[r['video'] for r in data.source.diag]})
 d.write_json(t.PUB/'readout_receipt.json',{'run_id':'A56','reader_freeze_git_commit':reader['git_commit'],'models_freeze_git_commit':freeze['git_commit'],'new_source_model_forwards':266,'source_model_forwards_this_invocation':new,'committed_source_forwards_reused_from_clean_prefix':266-new,'rows_per_model':133,'score_columns':5,'raw_logits':True,'private_score_sha256':t.sha(path),'all_cells_complete':True});print(json.dumps({'once_source_readout_complete':True,'model_forwards':new}),flush=True)


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['execute','supervise']);a=p.parse_args()
 if a.stage=='supervise':
  from scripts import supervise_annotation_coverage as supervisor
  supervisor.OUT=t.OUT;supervisor.COMMANDS={'readout':('scripts/readout_source_retention.py','execute',600,True)};sys.argv=[sys.argv[0],'readout'];supervisor.main()
 else:
  lock=d.acquire_lock(t.OUT/'owner.lock');start=time.time();ok=False
  try:readout();ok=True
  finally:
   u=resource.getrusage(resource.RUSAGE_SELF);d.write_json(t.OUT/'readout_worker_process.json',{'pid':os.getpid(),'ok':ok,'CPU_seconds':u.ru_utime+u.ru_stime,'wall_seconds':time.time()-start,'GPU_job_wall_seconds':time.time()-start});lock.close()
