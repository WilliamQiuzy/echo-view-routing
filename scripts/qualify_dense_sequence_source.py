#!/usr/bin/env python3
"""A57 source admission: strict CPU state only, zero model forwards."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import hashlib,json,resource,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from echo_routing.temporal import jepa21_source as source
RUN=ROOT/'runs/a57_dense_sequence_v1';PUB=ROOT/'docs/results/2026-11-02_dense_target_sequence'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(16*1024*1024),b''):h.update(b)
 return h.hexdigest()
def git(*args):return subprocess.check_output(['git','-C',str(source.UPSTREAM),*args],text=True).strip()
def main():
 start=time.time();torch.set_num_threads(1);acq=json.loads((RUN/'source_acquisition.json').read_text());path=ROOT/acq['path'];assert acq['sha256']==source.SOURCE_SHA==sha(path);assert git('rev-parse','HEAD')==source.COMMIT;assert not git('diff','--name-only','HEAD');m=source.build_encoder();ck=torch.load(path,map_location='cpu',weights_only=True,mmap=True);state=source.strict_target_state(ck,m);inactive=source.inactive_video_names([n for n,p in m.named_parameters()]);active=[n for n,p in m.named_parameters() if n not in inactive];assert len(state)==158 and len(inactive)==9
 loaded={}
 for name,module in list(sys.modules.items()):
  f=getattr(module,'__file__',None)
  if f and Path(f).resolve().is_relative_to(source.UPSTREAM.resolve()):
   p=Path(f).resolve();rel=str(p.relative_to(source.UPSTREAM));blob=subprocess.check_output(['git','-C',str(source.UPSTREAM),'show',source.COMMIT+':'+rel]);assert hashlib.sha256(blob).hexdigest()==sha(p);loaded[rel]=sha(p)
 # Preserve every source tensor, including explicitly unused video-path parameters.
 dst=ROOT/'checkpoints/a57_jepa21/target_encoder.pt';torch.save(state,dst);usage=resource.getrusage(resource.RUSAGE_SELF);rec={'passed':True,'publisher':'https://github.com/bowang-lab/EchoJEPA','source_checkpoint':acq,'official_model_repo':'https://github.com/facebookresearch/vjepa2','official_model_commit':source.COMMIT,'loaded_model_subset_SHA':loaded,'license':'MIT','license_sha256':sha(source.UPSTREAM/'LICENSE'),'strict_target_tensors':len(state),'missing_extra_dropped_filled':0,'source_dtype':'FP32','target_encoder_sha256':sha(dst),'target_encoder_bytes':dst.stat().st_size,'total_parameters':sum(p.numel() for p in m.parameters()),'active_video_parameters':sum(p.numel() for n,p in m.named_parameters() if n in active),'active_video_tensors':len(active),'inactive_video_names':inactive,'inactive_video_reason':'image-only modality/patch branch and deepnorm0..2 unused by official video forward returning last normalized layer; retained exactly and explicitly frozen forJ','encoder_kwargs':source.KW,'feature_contract':'official default forward(training=False): final block11 normalized by norms_block3; [B,8x14x14,768] at16x224 expected, actual shape checked before training','normalization':{'scale':1/255,'mean':[.485,.456,.406],'std':[.229,.224,.225],'source':'official VJEPA2preprocessor ImageNet normalization;224/16-frame research input with officialRoPE scaling, not an upstream task reproduction'},'pretraining_overlap_EV9V':'unknown','model_forwards':0,'CPU_core_seconds':usage.ru_utime+usage.ru_stime,'wall_seconds':time.time()-start};PUB.mkdir(parents=True,exist_ok=True);(PUB/'source.json').write_text(json.dumps(rec,indent=2)+'\n');print(json.dumps({k:rec[k] for k in ['passed','strict_target_tensors','total_parameters','active_video_parameters','active_video_tensors','target_encoder_sha256','CPU_core_seconds','model_forwards']}),flush=True)
if __name__=='__main__':main()
