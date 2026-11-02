#!/usr/bin/env python3
"""A54 source-only frozen DINO patch descriptor screen, no target readout."""
import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES','0');os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,hashlib,json,resource,sys,time,urllib.request,tarfile,io
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from echo_routing.temporal import dense_cooccurrence as dc,direct_view_training as dt
from echo_routing.temporal.dev_train import lr_factor
from scripts import train_direct_view_evidence as old
OUT=ROOT/'runs/a54_dense_covariance_v1';PUB=ROOT/'docs/results/2026-11-01_dense_cooccurrence';ARMS=('C','S');CARD='docs/superpowers/plans/2026-11-01-dense-selfsupervised-cooccurrence.md'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def arrsha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,x):dt.write_json(p,x)
def codepins():
 paths=[CARD,'scripts/train_dense_cooccurrence.py','scripts/audit_dense_cooccurrence.py','scripts/supervise_dense_cooccurrence.py','echo_routing/temporal/dense_cooccurrence.py','scripts/train_direct_view_evidence.py','echo_routing/temporal/direct_view_training.py','echo_routing/temporal/dev_train.py']
 return {p:sha(ROOT/p) for p in paths}
def intake():
 assert not (PUB/'source.json').exists(),'source identity already admitted'
 def get(url):
  with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'A54-research-source-qualification'}),timeout=60) as r:return r.read()
 commit=json.loads(get('https://api.github.com/repos/facebookresearch/dinov2/commits/main'))['sha'];base='https://raw.githubusercontent.com/facebookresearch/dinov2/'+commit+'/'
 meta={p:get(base+p) for p in ['README.md','LICENSE','MODEL_CARD.md']};assert b'Apache License' in meta['LICENSE'] and b'Apache License 2.0' in meta['README.md']
 url='https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth';assert url.encode() in meta['README.md']
 blob=get('https://api.github.com/repos/facebookresearch/dinov2/tarball/'+commit);(OUT/'upstream.tar.gz').write_bytes(blob)
 with tarfile.open(fileobj=io.BytesIO(blob),mode='r:gz') as t:
  members=t.getmembers();prefix=members[0].name.split('/')[0]+'/'
  for m in members:
   if not m.isfile():continue
   name=m.name[len(prefix):];p=Path(name);assert not p.is_absolute() and '..' not in p.parts
   q=OUT/'upstream'/p;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(t.extractfile(m).read())
 for p,b in meta.items():(OUT/p).write_bytes(b)
 with urllib.request.urlopen(url,timeout=60) as r,(OUT/'weights.pth').open('xb') as f:
  while b:=r.read(1024*1024):f.write(b)
 data=old.Inputs();np.save(OUT/'plan.npy',data.plan)
 rec={'run_id':'A54','upstream_commit':commit,'official_weight_url':url,'weight_sha256':sha(OUT/'weights.pth'),'archive_sha256':sha(OUT/'upstream.tar.gz'),'license_sha256':sha(OUT/'LICENSE'),'metadata_sha256':{p:sha(OUT/p) for p in meta},'upstream_code':{str(p.relative_to(OUT/'upstream')):sha(p) for p in (OUT/'upstream').rglob('*.py')},'reference_source_scores_sha256':sha(old.OUT/'source_scores.private.npz'),'reference_source_readout_sha256':sha(old.PUB/'source_readout.json'),'reference_source_freeze_sha256':sha(old.OUT/'source_models_freeze.json'),'bound_sha256':sha(data.path),'plan_sha256':sha(OUT/'plan.npy'),'code':codepins(),'fit':569,'diagnostic':133,'labels':'same verified manual point labels, source-only reused diagnostic patients','pretrained_weight_contract':'official pinned DINOv2 README/LICENSE Apache2; generic pretraining overlap unknown','new_source_model_forwards':0}
 write(PUB/'source.json',rec);print(json.dumps({'source_intake':True,'upstream_commit':commit,'weight_sha256':rec['weight_sha256']}))
def verify():
 m=read(PUB/'source.json');assert sha(old.OUT/'source_scores.private.npz')==m['reference_source_scores_sha256'] and sha(old.PUB/'source_readout.json')==m['reference_source_readout_sha256'] and sha(old.OUT/'source_models_freeze.json')==m['reference_source_freeze_sha256'];assert sha(OUT/'weights.pth')==m['weight_sha256'] and sha(old.Inputs().path)==m['bound_sha256'] and sha(OUT/'plan.npy')==m['plan_sha256']
 patch=read(PUB/'execution_patch.json') if (PUB/'execution_patch.json').exists() else None
 for p,h in m['code'].items():
  if patch and p in patch['corrected_code_sha256']:
   assert patch['original_code_sha256'][p]==h and sha(ROOT/p)==patch['corrected_code_sha256'][p],p
  else:assert sha(ROOT/p)==h,p
 for p,h in m['upstream_code'].items():assert sha(OUT/'upstream'/p)==h,p
 return m
def model():
 torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 m=torch.hub.load(str(OUT/'upstream'),'dinov2_vits14',source='local',pretrained=False);s=torch.load(OUT/'weights.pth',map_location='cpu',weights_only=True);m.load_state_dict(s,strict=True);m.eval().requires_grad_(False)
 return m.cuda(),len(s)
def forward(m,image):
 assert image.dtype==np.uint8 and image.shape==(224,224,3)
 x=torch.from_numpy(image.copy()).permute(2,0,1)[None].cuda().float()/255;x=(x-torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None])/torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None]
 with torch.no_grad():t=m.forward_features(x)['x_norm_patchtokens']
 assert t.shape==(1,256,384);return t[0].cpu().numpy()
def extract():
 rec=verify();fr=read(OUT/'input_freeze.json');assert fr['actual_git_objects_verified'] and fr['source_sha256']==sha(PUB/'source.json');d=old.Inputs();m,keys=model();p=dc.projection();before=sha(OUT/'weights.pth')
 check=OUT/'fit_checks.json'
 if not check.exists():
  a=forward(m,d.window(d.fit[0])[8]);b=forward(m,d.window(d.fit[0])[8]);assert np.array_equal(a,b);assert len(m.state_dict())==keys and not any(v.requires_grad for v in m.parameters())
  # Two more distinct fixed FIT checks; no diagnostic classifier scores.
  forward(m,d.window(d.fit[1])[8]);forward(m,d.window(d.fit[-1])[8]);write(check,{'passed':True,'strict_state_keys':keys,'frozen_backbone':True,'FIT_GPU_forwards':4,'repeat_max_abs':float(abs(a-b).max())})
 progress=OUT/'feature_progress.json';dest=OUT/'tokens.private.npy';n=702
 if progress.exists():
  q=read(progress);assert q['pending'] is None and dest.exists(),'ambiguous neural forward must not repeat';T=np.lib.format.open_memmap(dest,mode='r+')
 else:
  T=np.lib.format.open_memmap(dest,mode='w+',dtype=np.float32,shape=(n,256,384));T[:]=np.nan;T.flush();q={'done':0,'pending':None};write(progress,q)
 rows=d.fit+d.diagnostic
 while q['done']<n:
  i=q['done'];q['pending']=i;write(progress,q);T[i]=forward(m,d.window(rows[i])[8]);T.flush();fd=os.open(dest,os.O_RDWR);os.fsync(fd);os.close(fd);q={'done':i+1,'pending':None};write(progress,q)
  if (i+1)%100==0:print(json.dumps({'frozen_source_rows':i+1,'total':702}),flush=True)
 del T;T=np.load(dest,mmap_mode='r');assert T.shape==(702,256,384) and np.isfinite(T).all();E={arm:np.stack([dc.descriptor(t,p,arm=='S') for t in T]) for arm in ARMS};arrays={};entries={}
 for arm,X in E.items():
  mu=X[:569].mean(0);sd=X[:569].std(0);sd[sd<1e-6]=1;arrays[arm]=X;arrays[arm+'_mean']=mu;arrays[arm+'_scale']=sd;entries[arm]={'descriptor_sha256':arrsha(X),'fit_mean_sha256':arrsha(mu),'fit_scale_sha256':arrsha(sd)}
 np.savez(OUT/'features.private.npz',**arrays);assert sha(OUT/'weights.pth')==before
 write(PUB/'features_manifest.json',{'source_sha256':sha(PUB/'source.json'),'tokens_sha256':sha(dest),'features_sha256':sha(OUT/'features.private.npz'),'cells':entries,'formal_frozen_source_forwards':702,'fit_check_GPU_forwards':4,'classifier_scores':0,'backbone_optimizer_updates':0,'checks':read(check)});print(json.dumps({'feature_complete':True,'classifier_scores':0}))
def step(h,opt,x,y,u):
 opt.param_groups[0]['lr']=1e-3*lr_factor(u,60,600);opt.zero_grad(set_to_none=True);loss=torch.nn.functional.binary_cross_entropy_with_logits(h(torch.from_numpy(x)).flatten(),torch.from_numpy(y));loss.backward();norm=float(torch.nn.utils.clip_grad_norm_(h.parameters(),1));opt.step();return {'u':u,'loss':float(loss.detach()),'grad_norm':norm,'plan_sha256':None,'pooled_logits':8}
def train():
 assert read(PUB/'checks.json')['passed'];verify();fr=read(OUT/'features_freeze.json');assert fr['actual_git_objects_verified'] and fr['features_sha256']==sha(PUB/'features_manifest.json');features=read(PUB/'features_manifest.json');assert features['features_sha256']==sha(OUT/'features.private.npz');e=np.load(OUT/'features.private.npz');d=old.Inputs();y=np.array([r['true_label']=='A5C' for r in d.fit],float);plan=np.load(OUT/'plan.npy');entries={}
 for arm in ARMS:
  rd=OUT/'heads'/arm;rd.mkdir(parents=True,exist_ok=True);claim=rd/'claim.json';assert not claim.exists(),'formal head cannot silently restart';write(claim,{'source_sha256':sha(PUB/'source.json'),'feature_manifest_sha256':sha(PUB/'features_manifest.json')});h=torch.nn.Linear(912,1,dtype=torch.float64)
  with torch.no_grad():h.weight.zero_();h.bias.fill_(-2)
  opt=torch.optim.AdamW(h.parameters(),lr=.001,weight_decay=.0001);X=(e[arm][:569]-e[arm+'_mean'])/e[arm+'_scale'];logs=[]
  # Synthetic computational repeat/forced-identical and disk3+3 resume are independently tested before formal stage.
  for u,ids in enumerate(plan):
   def persist(v):
    v['plan_sha256']=arrsha(ids);logs.append(v);tmp=rd/'resume.tmp';torch.save({'head':h.state_dict(),'optimizer':opt.state_dict(),'updates':u+1},tmp);fd=os.open(tmp,os.O_RDWR);os.fsync(fd);os.close(fd);tmp.replace(rd/'resume.pt');jp=rd/'updates.private.jsonl';jp.write_text(''.join(json.dumps(z)+'\n' for z in logs));fd=os.open(jp,os.O_RDWR);os.fsync(fd);os.close(fd);dt.sync_directory(rd)
   dt.claimed_update(rd,u,lambda:step(h,opt,X[ids],y[ids],u),persist)
  (rd/'resume.pt').replace(rd/'final.pt');ck=torch.load(rd/'final.pt',weights_only=True,map_location='cpu');assert ck['updates']==600;entries[arm]={'head_sha256':sha(rd/'final.pt'),'journal_sha256':sha(rd/'updates.private.jsonl'),'updates':600,'active_parameters':913,'fit_logits':4800,'CNN_updates':0}
 write(PUB/'heads_manifest.json',{'models':entries,'source_sha256':sha(PUB/'source.json'),'features_sha256':sha(PUB/'features_manifest.json'),'code':codepins(),'head_updates':1200,'source_diagnostic_scores_before_freeze':0})
def readout():
 verify();h=read(PUB/'heads_manifest.json');fr=read(OUT/'heads_freeze.json');assert fr['actual_git_objects_verified'] and fr['heads_manifest_sha256']==sha(PUB/'heads_manifest.json');frozen_features=read(PUB/'features_manifest.json');assert frozen_features['features_sha256']==sha(OUT/'features.private.npz') and h['source_sha256']==sha(PUB/'source.json') and h['features_sha256']==sha(PUB/'features_manifest.json');audit=read(PUB/'independent_heads.json');assert audit['passed'] and audit['heads_manifest_sha256']==sha(PUB/'heads_manifest.json') and audit['source_sha256']==h['source_sha256'] and audit['features_sha256']==h['features_sha256'];e=np.load(OUT/'features.private.npz');d=old.Inputs();values={}
 for a in ARMS:
  assert sha(OUT/'heads'/a/'final.pt')==h['models'][a]['head_sha256'];dest=OUT/f'claim_readout_{a}.json';assert not dest.exists(),'one diagnostic readout only';write(dest,{'head_sha256':h['models'][a]['head_sha256'],'rows':133});ck=torch.load(OUT/'heads'/a/'final.pt',map_location='cpu',weights_only=True);X=(e[a][569:]-e[a+'_mean'])/e[a+'_scale'];values[a]=X@ck['head']['weight'].numpy()[0]+float(ck['head']['bias'][0])
 with (OUT/'scores.private.npz').open('xb') as f:np.savez(f,**values)
 y=np.array([r['true_label']=='A5C' for r in d.diagnostic]);discord=np.array([r['true_label']!=r['directory_label'] for r in d.diagnostic]);metrics={}
 for a,s in values.items():
  q=dt.source_metrics(y,s,discord);q['AUC']=float(np.mean((s[y,None]>s[None,~y])+.5*(s[y,None]==s[None,~y])));metrics[a]=q
 # Exact original sourceG and geometry caches are reused, no extra G/source forward.
 oldscores=np.load(old.OUT/'source_scores.private.npz');g=oldscores['G'];geo=oldscores['geometry'];gauc=float(np.mean((g[y,None]>g[None,~y])+.5*(g[y,None]==g[None,~y])));gm=dt.source_metrics(y,geo,discord)
 C,S=metrics['C'],metrics['S'];absolute=C['balanced_accuracy']>=.8 and C['AUC']>=max(.85,gauc+.03) and C['fp']<=5 and C['tp']>=28 and C['balanced_accuracy']>=gm['balanced_accuracy']+.10;contrast=C['balanced_accuracy']-S['balanced_accuracy']>=.03 and C['tp']-S['tp']>=4 and C['fp']<=S['fp'];paired={k:{'shared':int((x&z).sum()),'C_only':int((x&~z).sum()),'S_only':int((~x&z).sum())} for k,x,z in [('correct_A5C',y&(values['C']>0),y&(values['S']>0)),('A4C_errors',~y&(values['C']>0),~y&(values['S']>0))]}
 write(PUB/'readout.json',{'metrics':metrics,'paired':paired,'C_absolute_pass':absolute,'C_minus_S_association_pass':contrast,'qualified_for_future_target_card':absolute and contrast,'scores_sha256':sha(OUT/'scores.private.npz'),'source_classifier_scores':266,'fresh_neural_backbone_forwards_in_classifier_readout':0,'target_scores_or_full_F1_results':0,'scope':'Reused source diagnostic569fit/133diagnostic,36/10patients; no source/target anatomical truth or clinical result.'});print(json.dumps({'metrics':metrics,'absolute':absolute,'association':contrast}))
def main():
 parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['intake','extract','train','readout']);a=parser.parse_args();OUT.mkdir(exist_ok=True,parents=True);PUB.mkdir(exist_ok=True,parents=True);lock=dt.acquire_lock(OUT/'owner.lock');start=time.time();ok=False
 try:globals()[a.stage]();ok=True
 finally:
  u=resource.getrusage(resource.RUSAGE_SELF);write(OUT/f'{a.stage}_worker_process.json',{'pid':os.getpid(),'CPU_seconds':u.ru_utime+u.ru_stime,'wall_seconds':time.time()-start,'GPU_job_wall_seconds':time.time()-start if a.stage=='extract' else 0,'ok':ok});lock.close()
if __name__=='__main__':main()
