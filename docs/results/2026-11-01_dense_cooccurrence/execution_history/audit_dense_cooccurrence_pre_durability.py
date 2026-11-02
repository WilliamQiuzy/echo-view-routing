#!/usr/bin/env python3
"""A54 independent direct token arithmetic, NumPy optimizer and score audit."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse,copy,json,hashlib,math,resource,sys,time,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from scripts import train_dense_cooccurrence as t
O,P=t.OUT,t.PUB

def checks():
 z=np.load(O/'features.private.npz');d=t.old.Inputs();ids=np.load(O/'plan.npy')[0];y=np.array([r['true_label']=='A5C' for r in d.fit],float)[ids];X=(z['C'][ids]-z['C_mean'])/z['C_scale'];h=torch.nn.Linear(912,1,dtype=torch.float64)
 with torch.no_grad():h.weight.zero_();h.bias.fill_(-2)
 loss=torch.nn.functional.binary_cross_entropy_with_logits(h(torch.from_numpy(X)).flatten(),torch.from_numpy(y));loss.backward();s=np.full(8,-2.);p=1/(1+np.exp(-s));coef=(p-y)/8;err=float(abs(h.weight.grad.numpy()[0]-X.T@coef).max());assert err<1e-12 and abs(float(loss)-np.logaddexp(0,s).mean()+np.mean(y*s))<1e-12
 def new():
  q=torch.nn.Linear(912,1,dtype=torch.float64)
  with torch.no_grad():q.weight.zero_();q.bias.fill_(-2)
  return q,torch.optim.AdamW(q.parameters(),lr=.001,weight_decay=.0001)
 a,ao=new();b,bo=new()
 for u in range(6):t.step(a,ao,X,y,u)
 for u in range(3):t.step(b,bo,X,y,u)
 path=O/'check_resume.pt';torch.save({'head':b.state_dict(),'opt':bo.state_dict()},path);c,co=new();ck=torch.load(path,map_location='cpu',weights_only=True);c.load_state_dict(ck['head']);co.load_state_dict(ck['opt'])
 for u in range(3,6):t.step(c,co,X,y,u)
 assert all(torch.equal(v,c.state_dict()[k]) for k,v in a.state_dict().items())
 forced,fo=new()
 for u in range(6):t.step(forced,fo,X.copy(),y.copy(),u)
 assert all(torch.equal(v,forced.state_dict()[k]) for k,v in a.state_dict().items())
 t.write(P/'checks.json',{'passed':True,'independent_fit_BCE_gradient_max_abs':err,'real_fit_repeat_forced_identical_and_disk3plus3_resume_exact':True,'source_diagnostic_scores':0,'CNN_updates':0,'check_head_updates':18,'check_fit_logits':152})

def final():
 m=t.verify();fm=t.read(P/'features_manifest.json');assert t.sha(O/'tokens.private.npy')==fm['tokens_sha256'] and t.sha(O/'features.private.npz')==fm['features_sha256'];tokens=np.load(O/'tokens.private.npy',mmap_mode='r');e=np.load(O/'features.private.npz');p=np.linalg.qr(np.random.default_rng(2026110154).standard_normal((384,32)))[0];shift=np.random.default_rng(2026110155).choice(256,32,replace=False);feature_error={}
 for arm in ['C','S']:
  X=[]
  for tokens_i in tokens:
   x=tokens_i.astype(float);a=x@p;a-=a.mean(0)
   if arm=='S':a=np.column_stack([np.r_[a[-int(s):,j],a[:-int(s),j]] if s else a[:,j] for j,s in enumerate(shift)])
   cov=np.einsum('pi,pj->ij',a,a)/256;X.append(np.r_[x.mean(0),cov[np.triu_indices(32)]])
  X=np.asarray(X);err=float(abs(X-e[arm]).max());assert err<1e-10;feature_error[arm]=err;mu=X[:569].mean(0);sd=X[:569].std(0);sd[sd<1e-6]=1;assert np.max(abs(mu-e[arm+'_mean']))<1e-10 and np.max(abs(sd-e[arm+'_scale']))<1e-10
 heads=t.read(P/'heads_manifest.json');d=t.old.Inputs();plan=np.load(O/'plan.npy');y=np.array([r['true_label']=='A5C' for r in d.fit],float);results={};replayed_heads={};independent_scores={}
 for arm in ['C','S']:
  X=(e[arm][:569]-e[arm+'_mean'])/e[arm+'_scale'];w=np.zeros(912);b=-2.;mw=np.zeros(912);vw=np.zeros(912);mb=vb=0.;rd=O/'heads'/arm;logs=[json.loads(s) for s in (rd/'updates.private.jsonl').read_text().splitlines()];assert [v['u'] for v in logs]==list(range(600))
  for u,ids in enumerate(plan):
   z=X[ids]@w+b;pr=1/(1+np.exp(-z));c=(pr-y[ids])/8;gw=X[ids].T@c;gb=c.sum();norm=math.sqrt(float(gw@gw+gb*gb));clip=min(1.,1/(norm+1e-6));gw*=clip;gb*=clip;lr=.001*((u+1)/60 if u<60 else .5*(1+math.cos(math.pi*(u-60)/540)));mw=.9*mw+.1*gw;vw=.999*vw+.001*gw*gw;mb=.9*mb+.1*gb;vb=.999*vb+.001*gb*gb;w*=1-lr*.0001;b*=1-lr*.0001;w-=lr/(1-.9**(u+1))*mw/(np.sqrt(vw/(1-.999**(u+1)))+1e-8);b-=lr/(1-.9**(u+1))*mb/(math.sqrt(vb/(1-.999**(u+1)))+1e-8);assert logs[u]['plan_sha256']==t.arrsha(ids);assert abs(logs[u]['loss']-(np.logaddexp(0,z)-y[ids]*z).mean())<1e-9
  ck=torch.load(rd/'final.pt',map_location='cpu',weights_only=True);assert t.sha(rd/'final.pt')==heads['models'][arm]['head_sha256'] and t.sha(rd/'updates.private.jsonl')==heads['models'][arm]['journal_sha256'];err=max(float(abs(w-ck['head']['weight'].numpy()[0]).max()),abs(b-float(ck['head']['bias'][0])));assert err<1e-9 and ck['updates']==600;opt=ck['optimizer'];states=[opt['state'][k] for k in opt['param_groups'][0]['params']];assert all(int(x['step'])==600 for x in states);assert max(float(abs(states[0]['exp_avg'].numpy()[0]-mw).max()),float(abs(states[0]['exp_avg_sq'].numpy()[0]-vw).max()),abs(float(states[1]['exp_avg'])-mb),abs(float(states[1]['exp_avg_sq'])-vb))<1e-9
  replayed_heads[arm]=(w.copy(),b);fit=X@w+b;results[arm]={'updates':600,'all_AdamW_steps_replayed':True,'head_optimizer_max_abs':err,'fit_tp':int(((fit>0)&(y==1)).sum()),'fit_fp':int(((fit>0)&(y==0)).sum())}
 if (P/'readout.json').exists():
  rr=t.read(P/'readout.json');independent_scores={arm:((e[arm][569:]-e[arm+'_mean'])/e[arm+'_scale'])@w+b for arm,(w,b) in replayed_heads.items()};scores=np.load(O/'scores.private.npz');yy=np.array([r['true_label']=='A5C' for r in d.diagnostic]);disc=np.array([r['true_label']!=r['directory_label'] for r in d.diagnostic]);actual={}
  for arm,s in independent_scores.items():
   assert np.max(abs(scores[arm]-s))<1e-8;q=(s>0);tp=int((q&yy).sum());fp=int((q&~yy).sum());tn=int((~q&~yy).sum());fn=int((~q&yy).sum());ba=.5*(tp/39+tn/94);auc=sum(1 if a>b else .5 if a==b else 0 for a in s[yy] for b in s[~yy])/(39*94);v=rr['metrics'][arm];assert [tp,tn,fp,fn]==[v[k] for k in ['tp','tn','fp','fn']] and abs(ba-v['balanced_accuracy'])<1e-12 and abs(auc-v['AUC'])<1e-12;actual[arm]={'tp':tp,'fp':fp,'BA':ba,'AUC':auc}
  old=np.load(t.old.OUT/'source_scores.private.npz');g=old['G'];ga=sum(1 if a>b else .5 if a==b else 0 for a in g[yy] for b in g[~yy])/(39*94);geo=old['geometry']>0;gb=.5*((geo&yy).sum()/39+(~geo&~yy).sum()/94);C,S=actual['C'],actual['S'];absolute=C['BA']>=.8 and C['AUC']>=max(.85,ga+.03) and C['fp']<=5 and C['tp']>=28 and C['BA']>=gb+.10;contrast=C['BA']-S['BA']>=.03 and C['tp']-S['tp']>=4 and C['fp']<=S['fp'];assert [absolute,contrast,absolute and contrast]==[rr[k] for k in ['C_absolute_pass','C_minus_S_association_pass','qualified_for_future_target_card']]
  for key,mask in [('correct_A5C',yy),('A4C_errors',~yy)]:
   c=mask&(scores['C']>0);s=mask&(scores['S']>0);assert rr['paired'][key]=={'shared':int((c&s).sum()),'C_only':int((c&~s).sum()),'S_only':int((~c&s).sum())}
  assert t.sha(O/'scores.private.npz')==rr['scores_sha256'];readout_verified=True
 else:readout_verified=False
 t.write(P/('independent_final.json' if readout_verified else 'independent_heads.json'),{'passed':True,'all702_token_descriptors_independently_reconstructed':True,'max_feature_error':feature_error,'all_fit_scalers_recomputed':True,'all1200_AdamW_updates_and_optimizer_states_reconstructed':True,'models':results,'source_readout_verified':readout_verified,'new_neural_forwards':0,'independent_fit_training_logits':9600,'independent_source_score_reconstructions':266 if readout_verified else 0,'new_target_readouts':0})
 print(json.dumps({'audit_passed':True,'readout_verified':readout_verified}))

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('stage',choices=['checks','final']);v=a.parse_args();start=time.time();ok=False
 try:globals()[v.stage]();ok=True
 finally:
  u=resource.getrusage(resource.RUSAGE_SELF);t.write(O/f'audit_{v.stage}_{"post" if (P/"readout.json").exists() else "pre"}_worker_process.json',{'pid':os.getpid(),'ok':ok,'CPU_seconds':u.ru_utime+u.ru_stime,'wall_seconds':time.time()-start,'GPU_job_wall_seconds':0})
