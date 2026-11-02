#!/usr/bin/env python3
"""Download only the official named A57 checkpoint; no model forward."""
import hashlib,json,os,resource,time
from pathlib import Path
import gdown
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/a57_dense_sequence_v1';DEST=ROOT/'checkpoints/a57_jepa21/vjepa2_1_vitb_mimic_pt169_c60.pt'
FILE_ID='102kUdwIAv9Sw0QlSyThHQwyrPtVMe9lh'
def main():
 start=time.time();RUN.mkdir(parents=True,exist_ok=True);DEST.parent.mkdir(parents=True,exist_ok=True)
 (RUN/'source_acquisition_pid.json').write_text(json.dumps({'pid':os.getpid(),'start_epoch':start,'GPU':False,'file_id':FILE_ID})+'\n')
 assert not DEST.exists(),'preserve existing source; no blind overwrite'
 result=gdown.download(id=FILE_ID,output=str(DEST),quiet=False,use_cookies=False);assert result and DEST.is_file()
 h=hashlib.sha256()
 with DEST.open('rb') as f:
  for block in iter(lambda:f.read(16*1024*1024),b''):h.update(block)
 usage=resource.getrusage(resource.RUSAGE_SELF)
 receipt={'publisher':'https://github.com/bowang-lab/EchoJEPA','upstream_commit':'77208e1a8521c0e82f79579eebe4b22f46c5ccbd','official_folder':'https://drive.google.com/drive/folders/1RFEXMe8TTcABMBz4H_qtiLB43K9jD_lf','official_relative_name':'vjepa 2.1/vjepa2_1_vitb_mimic_pt169_c60.pt','file_id':FILE_ID,'path':str(DEST.relative_to(ROOT)),'bytes':DEST.stat().st_size,'sha256':h.hexdigest(),'pid':os.getpid(),'start_epoch':start,'end_epoch':time.time(),'CPU_core_seconds':usage.ru_utime+usage.ru_stime,'network_wall_seconds':time.time()-start,'HTTP_status_final_URL':'not exposed by gdown; officialfolder/fileID/fullSHA retained','new_model_forwards':0}
 (RUN/'source_acquisition.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)
if __name__=='__main__':main()
