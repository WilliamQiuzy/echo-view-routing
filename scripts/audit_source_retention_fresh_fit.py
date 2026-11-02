#!/usr/bin/env python3
"""A56 fixed FIT-only CPU checkpoint/output parity; never source diagnostic inference."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import hashlib,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from scripts import train_source_retention as t
from echo_routing.temporal.pw_transfer import state_sha


def main():
 # Indices0/568 are fixed before finalmodels/readout, and already in the32FITpanel.
 t.verify();manifest=t.j(t.PUB/'models_manifest.json');data=t.Inputs();checks={};n=0
 for arm in ('Z','R'):
  row=manifest['models'][arm];rd=t.OUT/'models'/arm;assert t.sha(rd/'final.pt')==row['bundle_sha256'] and t.sha(rd/'optimizer_final.pt')==row['optimizer_final_sha256'];bundle=torch.load(rd/'final.pt',map_location='cpu',weights_only=True);st=torch.load(rd/'optimizer_final.pt',map_location='cpu',weights_only=True);assert bundle['identity']==st['identity']==row['identity'] and bundle['update']==st['update']==1200;assert state_sha(bundle['model'])==state_sha(st['model'])==row['final_full_state_sha256'];del st;m=t.build('cpu');m.load_state_dict(bundle['model']);recorded=np.load(rd/'fit_final.private.npz')['source'];errors=[]
  for index,position in [(0,0),(568,31)]:
   window,marker,label=data.source_window(index)
   with torch.no_grad():q=t.logits(m,window,marker,device='cpu').numpy()[0]
   error=float(np.max(np.abs(q-recorded[position])));assert error<=1e-4,(arm,index,error);errors.append({'FIT_parent_index':index,'existing_panel_row':position,'full5logit_max_abs':error});n+=1
  assert state_sha(t.flat(m))==row['final_full_state_sha256'];checks[arm]={'model_optimizer_bundle_fullstate_identical':True,'CPU_FIT_checks':errors,'read_only_state_unchanged':True};del m,bundle
 assert t.sha(t.a51.BASE)==t.j(t.PUB/'input_manifest.json')['Base_A08_sha256'];t.write(t.PUB/'independent_fresh_fit.json',{'passed':True,'models_manifest_sha256':t.sha(t.PUB/'models_manifest.json'),'script_sha256':t.sha(Path(__file__)),'models':checks,'new_CPU_FIT_forwards':n,'new_GPU_forwards':0,'new_backwards':0,'source_diagnostic_forwards':0,'selected_indices':[0,568],'tolerance':1e-4,'epoch':time.time()});print(json.dumps({'fresh_FIT_checkpoint_parity_passed':True,'CPU_forwards':n,'models':checks}),flush=True)


if __name__=='__main__':main()
