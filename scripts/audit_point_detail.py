#!/usr/bin/env python3
"""Independent CPU binding/state/cache audit, no new diagnostic model forwards."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse,hashlib,json,resource,sys,time
from collections import defaultdict
from fractions import Fraction
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import av,numpy as np,torch
from PIL import Image
OUT=ROOT/'runs/a52_point_detail_v1';PUB=ROOT/'docs/results/2026-11-01_point_detail_a5c'
PRIOR=ROOT/'runs/a51_verified_direct_view_pilot_v1';SOURCE=ROOT/'runs/a51_discriminative_evidence_v1/source'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,z):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(z,indent=2,allow_nan=False)+'\n')
def state_sha(state):
 h=hashlib.sha256()
 for n,t in sorted(state.items()):
  a=t.detach().cpu().contiguous().numpy();h.update(n.encode());h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
 return h.hexdigest()

def binding():
 b=json.loads((OUT/'bound.private.json').read_text());s=json.loads((PRIOR/'selection.private.json').read_text());old={r['id']:r for r in s['requests']};rows=b['rows'];assert len(rows)==702
 sealed=json.loads((ROOT/'docs/results/2026-11-01_direct_view_evidence/pilot/input_manifest.json').read_text());assert sha(PRIOR/'bound.private.json')==sealed['bound_sha256'];oldbound=json.loads((PRIOR/'bound.private.json').read_text());assert sha(PRIOR/'selection.private.json')==oldbound['selection_sha256']==b['selection_sha256'];assert sha(PRIOR/'bound.private.json')==b['prior_binding_sha256'];assert [r['id'] for r in rows]==[r['id'] for r in s['requests']] and len({r['id'] for r in rows})==702
 roles=defaultdict(set);hashes={a:defaultdict(list) for a in ('raw','H','R')};by=defaultdict(list);counts={};z=np.load(OUT/'points.private.npz')['H']
 for j,r in enumerate(rows):
  q=old[r['id']]
  for k in ['centre','frames','patient','role','true_label','directory_label','video']:assert r[k]==q[k]
  assert r['centre']==r['frames'][8]
  roles[r['role']].add(r['patient']);by[(r['directory_label'],r['video'])].append((j,r))
 assert roles['fit'].isdisjoint(roles['diagnostic']) and [len(roles[k]) for k in ['fit','diagnostic']]==[36,10]
 for v in b['videos']:
  key=v['folder'],v['video'];p=SOURCE/'videos'/key[0]/key[1];assert sha(p)==v['sha256'];need={r['centre'] for _,r in by[key]};raw={};pts=[]
  with av.open(str(p)) as c:
   stream=c.streams.video[0];stream.thread_type='NONE'
   for i,f in enumerate(c.decode(stream)):
    t=Fraction(f.pts)*Fraction(f.time_base)*30;assert t.denominator==1;pts.append(t.numerator)
    if i in need:raw[i]=f.to_ndarray(format='rgb24')
  assert pts==v['native_pts_indices'] and len(pts)==v['decoded_frames']
  for j,r in by[key]:
   x=raw[r['centre']];h,w=x.shape[:2];nw,nh=round(448*w/max(w,h)),round(448*h/max(w,h));im=Image.fromarray(x).resize((nw,nh),Image.Resampling.BILINEAR);H=Image.new('RGB',(448,448));H.paste(im,((448-nw)//2,(448-nh)//2));R=H.resize((224,224),Image.Resampling.BILINEAR).resize((448,448),Image.Resampling.BILINEAR)
   assert sha_bytes(x)==r['raw_point_sha256'];assert sha_bytes(np.asarray(H))==r['H_sha256'];assert sha_bytes(np.asarray(R))==r['R_sha256'];assert np.array_equal(z[j],np.asarray(H));hashes['raw'][r['raw_point_sha256']].append(r)
   for a in ['H','R']:hashes[a][r[a+'_sha256']].append(r)
 for same in [v for group in hashes.values() for v in group.values()]:assert len({r['patient'] for r in same})==len({r['role'] for r in same})==len({r['true_label'] for r in same})==1
 fit=[r for r in rows if r['role']=='fit'];diag=[r for r in rows if r['role']=='diagnostic'];assert len(fit)==569 and len(diag)==133
 plan=np.load(OUT/'plan.npy');assert plan.shape==(600,8) and set(plan.ravel())==set(range(569))
 labels=np.array([r['true_label']=='A5C' for r in fit]);assert np.all(labels[plan].sum(1)==4)
 from sklearn.linear_model import LogisticRegression
 x=np.asarray([r['geometry'] for r in fit]);mean,scale=x.mean(0),x.std(0);scale[scale<1e-12]=1;g=LogisticRegression(C=1.,solver='lbfgs',max_iter=1000,random_state=2026110151).fit((x-mean)/scale,labels);ctrl=b['geometry']
 assert np.array_equal(mean,np.asarray(ctrl['mean'])) and np.array_equal(scale,np.asarray(ctrl['scale'])) and np.max(abs(g.coef_[0]-ctrl['coef']))<1e-12 and abs(g.intercept_[0]-ctrl['intercept'])<1e-12 and ctrl['prior']==float(labels.mean())
 result={'passed':True,'bound_sha256':sha(OUT/'bound.private.json'),'all702_raw_markers_H_R_and_cached_H_recomputed':True,'all68_AVI_SHA_and_full_PTS_verified':True,'fit_diagnostic':[569,133],'patient_counts':[36,10],'unique_raw_H_R_points':{a:len(v) for a,v in hashes.items()},'same_label_multiplicities_retained':True,'no_crosspatient_or_crossfold_point_duplicates':True,'plan600x8_fullcoverage_balanced_true_labels':True,'diagnostic_model_forwards':0}
 dump(PUB/'independent_binding.json',result)

def sha_bytes(a):return hashlib.sha256(np.asarray(a).tobytes()).hexdigest()

def independent_model():
 from torch import nn
 from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
 from torchvision.ops.misc import FrozenBatchNorm2d
 class Model(nn.Module):
  def __init__(self):
   super().__init__();self.backbone=resnet_fpn_backbone(backbone_name='resnet50',weights=None,norm_layer=FrozenBatchNorm2d)
   for a in self.backbone.modules():
    if isinstance(a,FrozenBatchNorm2d):a.eps=0
   self.backbone.load_state_dict(torch.load(ROOT/'runs/a48_valve_source_intake_v1/backbone_G.pt',map_location='cpu',weights_only=True))
   for p in self.backbone.parameters():p.requires_grad_(True)
   self.head=nn.Linear(1024,1);nn.init.zeros_(self.head.weight);nn.init.constant_(self.head.bias,-2)
 return Model().eval()

def models():
 man=json.loads((PUB/'models_manifest.json').read_text());inputs=json.loads((PUB/'input_manifest.json').read_text());assert man['input_manifest_sha256']==sha(PUB/'input_manifest.json');assert sha(ROOT/'runs/a48_valve_source_intake_v1/backbone_G.pt')==inputs['base_sha256'];entries={}
 for arm in ['H','R']:
  rd=OUT/'source_train'/arm;e=man['models'][arm];assert e['identity']=={'run_id':'A52','arm':arm,'input_manifest_sha256':sha(PUB/'input_manifest.json'),'plan_sha256':inputs['plan_sha256']};assert sha(rd/'final.pt')==e['model_sha256'] and sha(rd/'optimizer_final.pt')==e['optimizer_final_sha256'] and sha(rd/'updates.private.jsonl')==e['journal_sha256']
  ck=torch.load(rd/'final.pt',map_location='cpu',weights_only=True);st=torch.load(rd/'optimizer_final.pt',map_location='cpu',weights_only=True);assert ck['updates']==st['update']==600 and ck['identity']==st['identity']==e['identity']
  m=independent_model();initial={n:v.clone() for n,v in m.state_dict().items()};assert state_sha(initial)==e['initial_state_sha256'];m.load_state_dict(ck['model'],strict=True);final=m.state_dict();assert state_sha(final)==e['full_state_sha256']==state_sha(st['model'])
  live={n for n,p in m.named_parameters()};assert set(e['live_drift'])==live and all(torch.count_nonzero(final[n]-initial[n]) for n in live)
  assert all(torch.equal(v,initial[n]) for n,v in final.items() if n not in live)
  opt=st['optimizer'];groups=opt['param_groups'];ordered=[(n,p) for n,p in m.named_parameters() if n.startswith('backbone.')]+[(n,p) for n,p in m.named_parameters() if n.startswith('head.')];ids=[i for g in groups for i in g['params']]
  assert len(ids)==len(set(ids))==len(live)==len(opt['state']);assert set(ids)==set(opt['state'])
  assert len(groups)==2 and len(groups[0]['params'])==len(ordered)-2 and len(groups[1]['params'])==2
  for pid,(name,p) in zip(ids,ordered):
   q=opt['state'][pid];assert int(q['step'])==600 and q['exp_avg'].shape==p.shape==q['exp_avg_sq'].shape;assert q['exp_avg'].dtype==p.dtype==q['exp_avg_sq'].dtype and torch.isfinite(q['exp_avg']).all() and torch.isfinite(q['exp_avg_sq']).all() and (q['exp_avg_sq']>=0).all()
  from echo_routing.temporal.dev_train import lr_factor
  for j,g in enumerate(groups):assert abs(g['lr']-[1e-5,1e-3][j]*lr_factor(599,60,600))<1e-15 and g['base_lr']==[1e-5,1e-3][j] and g['weight_decay']==1e-4 and g['betas']==(.9,.999) and g['eps']==1e-8
  logs=[json.loads(x) for x in (rd/'updates.private.jsonl').read_text().splitlines()];assert [r['u'] for r in logs]==list(range(600));assert all(r['physical_forwards']==r['physical_backwards']==8 for r in logs)
  plan=np.load(OUT/'plan.npy');assert all(r['plan_row_sha256']==hashlib.sha256(plan[u].tobytes()).hexdigest() for u,r in enumerate(logs));assert all(any(r['groups'][g]>0 for r in logs[1:]) for g in logs[0]['groups'])
  guard=json.loads((rd/'update_guard.json').read_text());assert guard=={'pending':None,'last_committed':599}
  assert sha(rd/'fit_init.npy')==e['fit_init_sha256'] and sha(rd/'fit_final.npy')==e['fit_final_sha256'];assert (np.load(rd/'fit_init.npy')==-2).all()
  rng=st['CPU_rng'];assert rng.dtype==torch.uint8 and rng.numel()==5056;torch.set_rng_state(rng);assert st['CUDA_rng'].dtype==torch.uint8 and st['CUDA_rng'].numel()>0
  entries[arm]={'updates':600,'all_live_parameters_moved':len(live),'optimizer_complete600':True,'frozen_BN_equal':True,'forwards':4800,'backwards':4800,'full_state_sha256':state_sha(final)}
 assert man['models']['H']['initial_state_sha256']==man['models']['R']['initial_state_sha256']
 dump(PUB/'independent_models.json',{'passed':True,'models_manifest_sha256':sha(PUB/'models_manifest.json'),'models':entries,'additional_model_forwards':0,'RNG_scope':'CPU set_state accepted, CUDA state structure audited; production3plus3 tests real restore'})

def metrics(y,s,disc):
 p=s>0;tp=int((p&y).sum());tn=int((~p&~y).sum());fp=int((p&~y).sum());fn=int((~p&y).sum());auc=float(((s[y,None]>s[None,~y])+.5*(s[y,None]==s[None,~y])).mean())
 return {'tp':tp,'tn':tn,'fp':fp,'fn':fn,'balanced_accuracy':.5*(tp/(tp+fn)+tn/(tn+fp)),'A4C_recall':tn/(tn+fp),'A5C_recall':tp/(tp+fn),'AUROC':auc,'discordant_accuracy':float((p[disc]==y[disc]).mean())}

def final():
 r=json.loads((PUB/'readout.json').read_text());current=json.loads((PUB/'input_manifest.json').read_text());assert sha(OUT/'bound.private.json')==current['bound_sha256'];assert sha(ROOT/'runs/a48_valve_source_intake_v1/backbone_G.pt')==current['base_sha256'];b=json.loads((OUT/'bound.private.json').read_text());rows=[x for x in b['rows'] if x['role']=='diagnostic'];y=np.array([x['true_label']=='A5C' for x in rows]);disc=np.array([x['true_label']!=x['directory_label'] for x in rows]);z=np.load(OUT/'scores.private.npz');assert sha(OUT/'scores.private.npz')==r['score_cache_sha256'];ctrl=b['geometry'];x=np.asarray([a['geometry'] for a in rows]);assert np.max(abs(z['geometry']-(((x-ctrl['mean'])/ctrl['scale'])@np.asarray(ctrl['coef'])+ctrl['intercept'])))<1e-12
 assert np.array_equal(z['constant'],np.full(133,np.log(ctrl['prior']/(1-ctrl['prior']))));m={a:metrics(y,z[a],disc) for a in z.files}
 for a in m:
  for k,v in m[a].items():assert abs(v-r['metrics'][a][k])<1e-12,(a,k)
 checks={}
 for a in ['H','R']:
  q=m[a];checks[a]={'BA':q['balanced_accuracy']>=.8,'A4C_recall':q['A4C_recall']>=.7,'A5C_recall':q['A5C_recall']>=.7,'A4C_protection':q['fp']/94<=4/94+.02+1e-12,'AUC_floor':q['AUROC']>=.85,'AUC_increment':q['AUROC']>=.8074195308237861+.03,'image_over_geometry':q['balanced_accuracy']>=m['geometry']['balanced_accuracy']+.1,'image_over_constant':q['balanced_accuracy']>=m['constant']['balanced_accuracy']+.1}
 assert checks==r['source_gates']['absolute_checks'];passed=any(all(v.values()) for v in checks.values());assert passed==r['source_gates']['absolute_pass']
 h,low=m['H'],m['R'];detail={'BA_increment':h['balanced_accuracy']>=low['balanced_accuracy']+.05,'no_extra_A4C_harm':h['A4C_recall']>=low['A4C_recall'],'AUC_increment':h['AUROC']>=low['AUROC']+.03};assert detail==r['source_gates']['detail_checks'] and all(detail.values())==r['source_gates']['detail_increment_pass']
 man=json.loads((PUB/'models_manifest.json').read_text());fr=json.loads((OUT/'models_freeze.json').read_text());assert fr==r['models_freeze'] and fr['manifest_sha256']==sha(PUB/'models_manifest.json')
 for a in ['H','R']:
  cell=json.loads((OUT/'cells'/f'{a}.progress.json').read_text());complete=json.loads((OUT/'cells'/f'{a}.complete.json').read_text());assert complete=={'identity':cell['identity'],'rows':133,'sha256':sha(OUT/'cells'/f'{a}.npy')};assert cell['identity']['bound_sha256']==current['bound_sha256'];assert man['input_manifest_sha256']==sha(PUB/'input_manifest.json') and man['models'][a]['identity']['input_manifest_sha256']==sha(PUB/'input_manifest.json');assert cell['done']==133 and cell['pending'] is None and cell['identity']['model_sha256']==man['models'][a]['model_sha256'];assert np.array_equal(np.load(OUT/'cells'/f'{a}.npy'),z[a]);assert (OUT/'cells'/f'{a}.progress.json').stat().st_mtime>=fr['freeze_epoch']
 old_report=json.loads((ROOT/'docs/results/2026-11-01_direct_view_evidence/pilot/source_readout.json').read_text());assert sha(PRIOR/'source_scores.private.npz')==old_report['raw_score_cache_sha256'];oldbound=json.loads((PRIOR/'bound.private.json').read_text());assert [x['id'] for x in rows]==[x['id'] for x in oldbound['requests'] if x['role']=='diagnostic'];paired={};G=np.load(PRIOR/'source_scores.private.npz')['G'];scores={'G':G,**{a:z[a] for a in ['H','R']}}
 for a,c in [('H','G'),('R','G'),('H','R')]:
  p=scores[a]>0;q=scores[c]>0;paired[a+'_vs_'+c]={'A5C_both':int((p&q&y).sum()),'A5C_A_only':int((p&~q&y).sum()),'A5C_B_only':int((~p&q&y).sum()),'A4C_both_harm':int((p&q&~y).sum()),'A4C_A_only_harm':int((p&~q&~y).sum()),'A4C_B_only_harm':int((~p&q&~y).sum())}
 fit=[x for x in b['rows'] if x['role']=='fit'];fy=np.array([x['true_label']=='A5C' for x in fit]);learning={}
 for a in ['H','R']:
  s=np.load(OUT/'source_train'/a/'fit_final.npy');learning[a]={'fit_correct':int(((s>0)==fy).sum()),'fit_total':569,'fit_BCE':float((np.logaddexp(0,s)-fy*s).mean())}
 dump(PUB/'independent_final.json',{'passed':True,'cache_only_no_new_model_forwards':True,'metrics':m,'absolute_checks':checks,'absolute_pass':passed,'detail_checks':detail,'detail_increment_pass':all(detail.values()),'paired_changes':paired,'fit_learning':learning,'score_cache_sha256':sha(OUT/'scores.private.npz'),'models_freeze':fr,'target_models_or_readouts':0})

def main():
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['binding','models','final']);a=p.parse_args();torch.set_num_threads(1);start=time.time();ok=False
 try:globals()[a.stage]();ok=True
 finally:
  u=resource.getrusage(resource.RUSAGE_SELF);dump(OUT/f'audit_{a.stage}_worker_process.json',{'pid':os.getpid(),'stage':a.stage,'ok':ok,'CPU_seconds':u.ru_utime+u.ru_stime,'wall_seconds':time.time()-start})
if __name__=='__main__':main()
