#!/usr/bin/env python3
"""Independent NumPy A53 reconstruction, preserved reference evaluator."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse,hashlib,json,math,resource,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from scripts import train_risk_constrained_a5c as t
from scripts import train_direct_view_evidence as src
OUT=ROOT/'runs/a53_frozen_view_transfer_v1';PUB=ROOT/'docs/results/2026-11-01_frozen_direct_view_transfer';ARMS=('G','L','W')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def dump(p,z):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(z,indent=2,allow_nan=False)+'\n')

def inputs(f,arm,scales=None):
 c=src.ctx();F=t.Fold(c,0,f);O=t.t31.outputs(c.a31,0,f);M=read(PUB/'input_manifest.json');assert sha(c.a31.out/f'outputs_s0_f{f}.npz')==M['folds'][str(f)]['outputs_sha256'];g=O['emb'][F.need].astype(float);s=np.zeros_like(g) if arm=='G' else np.load(OUT/f'features_{arm}_f{f}.npy').astype(float);fm=read(PUB/'feature_manifest.json')
 if arm!='G':assert sha(OUT/f'features_{arm}_f{f}.npy')==fm['features'][f'{arm}_f{f}']['feature_sha256']
 index=np.full(len(F.W['seq']),-1,np.int64);index[F.need]=np.arange(len(F.need));referenced=set()
 for row in F.pl:
  for k in row:
   lo,hi=F.pk['offsets'][k:k+2];referenced.update((F.pk['rows'][lo:hi]-F.off).tolist())
 ids=index[np.array(sorted(referenced))];assert (ids>=0).all()
 if scales is None:
  gm,gs=g[ids].mean(0),g[ids].std(0);gs[gs<1e-6]=1
  sm,ss=(np.zeros(512),np.ones(512)) if arm=='G' else (s[ids].mean(0),s[ids].std(0));ss[ss<1e-6]=1;scales=(gm,gs,sm,ss)
 gm,gs,sm,ss=scales;window=np.concatenate([(g-gm)/gs,np.zeros_like(g) if arm=='G' else (s-sm)/ss],1);X=np.zeros((len(F.pk['base']),1024));near=t.t27.nearest_rows(F.W,F.seqs);membership_error=0
 for k in np.flatnonzero(F.el):
  rr=near[int(F.pk['seq'][k])][int(F.pk['a'][k]):int(F.pk['b'][k])];X[k]=window[index[rr]].mean(0);unique,count=np.unique(rr,return_counts=True);lo,hi=F.pk['offsets'][k:k+2];assert np.array_equal(unique,F.pk['rows'][lo:hi]-F.off);membership_error=max(membership_error,float(abs(count/len(rr)-F.pk['weights'][lo:hi]).max()))
 assert membership_error<1e-12;return F,X,window,tuple(scales),membership_error

def features():
 fm=read(PUB/'feature_manifest.json');M=read(PUB/'input_manifest.json');assert fm['input_manifest_sha256']==sha(PUB/'input_manifest.json');records={}
 for key,e in fm['features'].items():
  p=OUT/f'features_{key}.npy';assert sha(p)==e['feature_sha256'];E=np.load(p);r=read(OUT/f'features_{key}.json');assert r['identity']==e['identity'] and r['pending'] is None and r['done']==e['rows'];assert E.shape==(e['rows'],512) and np.isfinite(E).all();records[key]={'rows':e['rows'],'feature_sha256':sha(p),'complete':True}
 dm=read(PUB/'fit_design_manifest.json');z=np.load(OUT/'fit_scalers.private.npz');assert sha(OUT/'fit_scalers.private.npz')==dm['fit_scalers_sha256'];errs={}
 for f in (0,1):
  for arm in ARMS:
   F,X,win,sc,err=inputs(f,arm);key=f'{arm}_f{f}';assert np.max(abs(np.stack(sc)-z[key]))<1e-12;assert digest(np.stack(sc))==dm['items'][key]['scales_sha256'];errs[key]=err
  assert len({dm['items'][f'{a}_f{f}']['G_scales_sha256'] for a in ARMS})==1
 # Independent fresh FIT backbone evaluation of two fixed windows per source; no head predictions.
 fresh={}
 for arm in ['L','W']:
  m,h=src.restore(torch.load(ROOT/'runs/a51_verified_direct_view_pilot_v1/source_train'/arm/'final.pt',map_location='cpu',weights_only=True),'cpu');m.eval();F=t.Fold(src.ctx(),0,0);rows=F.need[[0,len(F.need)-1]];fr,idx,mk=F.frames(rows);values=[]
  import echo_routing.vit.target_mvit as tm
  with torch.no_grad():
   for j in range(2):values.append(m.embed(tm.to_rgb(fr[idx[j:j+1]]),tm.marker_planes(mk[j:j+1])).numpy()[0])
  cached=np.load(OUT/f'features_{arm}_f0.npy')[[0,len(F.need)-1]];fresh[arm]=float(abs(np.asarray(values)-cached).max());assert fresh[arm]<3e-5
 dump(PUB/'independent_features.json',{'passed':True,'all4frozen_feature_cells':records,'independent_full_membership_error':errs,'fit_scaler_independently_recomputed':True,'feature_manifest_sha256':sha(PUB/'feature_manifest.json'),'fit_design_manifest_sha256':sha(PUB/'fit_design_manifest.json'),'target_classifier_scores':0,'GPU_forwards_in_audit':0,'fresh_CPU_fit_backbone_forwards':4,'fresh_CPU_fit_embedding_max_abs':fresh})

def replay(X,plan):
 w=np.zeros(1024);bias=-2.;mw=np.zeros(1024);vw=np.zeros(1024);mb=vb=0.;losses=[]
 for u,ids in enumerate(plan):
  x=X[ids];z=x@w+bias;p=1/(1+np.exp(-z));coeff=np.r_[p[0]-1,p[1:]/3];gw=x.T@coeff;gb=coeff.sum();norm=math.sqrt(float(gw@gw+gb*gb));clip=min(1.,1/(norm+1e-6));gw*=clip;gb*=clip;lr=.001*((u+1)/60 if u<60 else .5*(1+math.cos(math.pi*(u-60)/540)));mw=.9*mw+.1*gw;vw=.999*vw+.001*gw*gw;mb=.9*mb+.1*gb;vb=.999*vb+.001*gb*gb;w*=1-lr*.0001;bias*=1-lr*.0001;w-=lr/(1-.9**(u+1))*mw/(np.sqrt(vw/(1-.999**(u+1)))+1e-8);bias-=lr/(1-.9**(u+1))*mb/(math.sqrt(vb/(1-.999**(u+1)))+1e-8);losses.append(float(np.logaddexp(0,-z[0])+np.logaddexp(0,z[1:]).mean()))
 return w,bias,mw,vw,mb,vb,losses

def heads():
 M=read(PUB/'heads_manifest.json');assert M['input_manifest_sha256']==sha(PUB/'input_manifest.json') and M['feature_manifest_sha256']==sha(PUB/'feature_manifest.json');results={}
 for f in (0,1):
  for arm in ARMS:
   key=f'{arm}_f{f}';e=M['models'][key];rd=OUT/'heads'/key;assert sha(rd/'final.pt')==e['head_sha256'] and sha(rd/'optimizer_final.pt')==e['optimizer_final_sha256'] and sha(rd/'updates.private.jsonl')==e['journal_sha256'];ck=torch.load(rd/'final.pt',map_location='cpu',weights_only=True);st=torch.load(rd/'optimizer_final.pt',map_location='cpu',weights_only=True);assert ck['updates']==st['update']==600 and ck['identity']==st['identity']==e['identity'];assert e['identity']['input_manifest_sha256']==sha(PUB/'input_manifest.json')
   F,X,_,sc,_=inputs(f,arm,tuple(v.numpy() for v in ck['scales']));assert F.psha==e['identity']['plan_sha256'];w,b,mw,vw,mb,vb,losses=replay(X,F.pl);err=max(float(abs(w-ck['head']['weight'].numpy()[0]).max()),abs(b-float(ck['head']['bias'][0])));assert err<1e-9,(key,err)
   logs=[json.loads(a) for a in (rd/'updates.private.jsonl').read_text().splitlines()];assert [a['u'] for a in logs]==list(range(600)) and all(a['plan_row_sha256']==digest(F.pl[u]) for u,a in enumerate(logs));assert max(abs(logs[u]['loss']-losses[u]) for u in range(600))<1e-9;assert all(a['pooled_run_logits']==4 and a['CNN_backwards']==0 and a['loss_backwards']==1 for a in logs)
   opt=st['optimizer'];ids=opt['param_groups'][0]['params'];assert len(ids)==2==len(opt['state']);sw,sb=[opt['state'][i] for i in ids];assert int(sw['step'])==int(sb['step'])==600
   assert max(float(abs(sw['exp_avg'].numpy()[0]-mw).max()),float(abs(sw['exp_avg_sq'].numpy()[0]-vw).max()),abs(float(sb['exp_avg'])-mb),abs(float(sb['exp_avg_sq'])-vb))<1e-9
   assert all(torch.equal(ck['head'][n],st['head'][n]) for n in ck['head']);assert read(rd/'update_guard.json')=={'pending':None,'last_committed':599}
   if arm=='G':assert (w[512:]==0).all() and not torch.count_nonzero(ck['head']['weight'][:,512:])
   results[key]={'updates':600,'NumPy_all600_AdamW_head_max_abs':err,'head_sha256':sha(rd/'final.pt'),'structural_zero_G_block':arm=='G','pooled_run_logits':2400,'loss_backwards':600,'CNN_updates':0}
 dump(PUB/'independent_heads.json',{'passed':True,'heads_manifest_sha256':sha(PUB/'heads_manifest.json'),'models':results,'all_3600_updates_reconstructed':True,'new_classifier_scores_in_audit':0})

def final():
 M=read(PUB/'heads_manifest.json');R=read(PUB/'readout.json');raw=read(PUB/'raw_complete.json');assert raw['cache_sha256']==sha(OUT/'readout.private.npz')==R['raw_cache_sha256'];cache=np.load(OUT/'readout.private.npz');c=src.ctx();a8=t.rsh.a08_ctx();data=t.t31.vpr.training_data(c.a31.a15.a10.a07p);counts={};gates={};sets={a:[] for a in ARMS};source_sets={a:set() for a in ARMS}
 for held in (0,1):
  F=t.Fold(c,0,held);D=t.d32.fold_data(c.d,0,held);D['M']=t.d32.set_masks(D);P={'seqs':D['seqs'],'n_of':dict(data.n),'label_of':dict(data.label)};ev=t.bc.evalset(P,list(range(len(D['seqs']))),f'heldout_f{held}');base=t.tables_pred(F.pk['base'],D,a8.cfg,P,ev);assert base['sequence_row']==R['cells'][str(held)]['Base']['sequence_row'] and base['counts']==R['cells'][str(held)]['Base']['counts'];assert np.array_equal(F.need,cache[f'rows_f{held}']);near=t.t27.nearest_rows(F.W,F.seqs);lookup=np.full(len(F.W['seq']),-1,np.int64);lookup[F.need]=np.arange(len(F.need))
  for arm in ARMS:
   key=f'{arm}_f{1-held}';path=OUT/'heads'/key/'final.pt';ck=torch.load(path,map_location='cpu',weights_only=True);_,_,window,_,_=inputs(held,arm,tuple(v.numpy() for v in ck['scales']));scores=window@ck['head']['weight'].numpy()[0]+float(ck['head']['bias'][0]);assert np.max(abs(scores-cache[f'scores_{arm}_f{held}']))<1e-10
   cell=read(OUT/f'scores_{arm}_f{held}.json');assert cell['state']=='complete' and cell['identity']['head_sha256']==sha(path) and cell['sha256']==sha(OUT/f'scores_{arm}_f{held}.npy');assert np.array_equal(np.load(OUT/f'scores_{arm}_f{held}.npy'),scores)
   pred=F.pk['base'].copy()
   for k in np.flatnonzero(F.el):
    refs=near[int(F.pk['seq'][k])][int(F.pk['a'][k]):int(F.pk['b'][k])]
    if scores[lookup[refs]].mean()>0:pred[k]=3
   table=t.tables_pred(pred,D,a8.cfg,P,ev);expected=R['cells'][str(held)][arm];assert table['sequence_row']==expected['sequence_row'] and table['counts']==expected['counts'];fix=(F.pk['base']==2)&(F.pk['labels']==3)&(pred==3);harm=(F.pk['base']==2)&(F.pk['labels']==2)&(pred==3);vsource={ref[0] for k in np.flatnonzero(fix) for ref in F.seqs[int(F.pk['seq'][k])]['refs'][int(F.pk['a'][k]):int(F.pk['b'][k])] if data.label[ref[0]]==3};source_sets[arm]|=vsource;assert len(vsource)==expected['S1_source_count'] and int(fix.sum()-harm.sum())==expected['net_fixed_count'];assert int(fix.sum())==expected['S1_fixed_count'] and int(harm.sum())==expected['A4C_harmed_count'];assert np.array_equal(pred[~F.el],F.pk['base'][~F.el]);sets[arm].append((fix,harm));counts[f'{arm}_f{held}']={'fix':int(fix.sum()),'harm':int(harm.sum()),'full_F1':table['sequence_row']['segment_f1_50_multi'],'short_misses':table['counts']['fn_short']}
 # Recompute every strict gate from validated evaluator rows, no production gate helper.
 for arm in ARMS:
  actual_fixed=sum(int(x.sum()) for x,y in sets[arm]);actual_harm=sum(int(y.sum()) for x,y in sets[arm]);actual_mean=sum(counts[f'{arm}_f{h}']['full_F1'] for h in (0,1))/2;assert R['oof'][arm]=={'A5C_fixed':actual_fixed,'A4C_harmed':actual_harm,'S1_source_count':len(source_sets[arm]),'F1_mean':actual_mean}
  v=[R['cells'][str(h)][arm] for h in (0,1)];b=[R['cells'][str(h)]['Base'] for h in (0,1)];delta=lambda k:[v[j]['sequence_row'][k]-b[j]['sequence_row'][k] for j in (0,1)];o=R['oof'][arm];g={'fix':o['A5C_fixed']>=11 and all(x['S1_fixed_count']>=1 and x['S1_source_count']>=1 for x in v) and o['S1_source_count']>=3,'harm':o['A4C_harmed']<=14,'net':all(x['net_fixed_count']>0 for x in v) and o['A5C_fixed']-o['A4C_harmed']>=5,'F1':min(delta('segment_f1_50_multi'))>0 and sum(delta('segment_f1_50_multi'))/2>=.003,'native':min(delta('native_macro_f1'))>=-.002,'boundary':min(delta('boundary_recall_multi'))>=-.005,'short':sum(x['counts']['fn_short'] for x in v)<=79,'structure':all(x['non_A4C_labels_unchanged'] for x in v)};assert {**g,'all':all(g.values())}==R['gates']['absolute'][arm];gates[arm]=all(g.values())
 for A,B in [('L','W'),('L','G'),('W','G')]:
  expected=R['paired'][A+'_vs_'+B]
  for j,kind in enumerate(['fix','harm']):assert expected[kind]=={'shared':sum(int((sets[A][h][j]&sets[B][h][j]).sum()) for h in (0,1)),'A_only':sum(int((sets[A][h][j]&~sets[B][h][j]).sum()) for h in (0,1)),'B_only':sum(int((~sets[A][h][j]&sets[B][h][j]).sum()) for h in (0,1))}
 for control in ['G','W']:
  value=R['oof']['L']['A5C_fixed']>=R['oof'][control]['A5C_fixed']+5 and R['oof']['L']['A4C_harmed']<=R['oof'][control]['A4C_harmed'] and all(R['cells'][str(h)]['L']['sequence_row']['segment_f1_50_multi']>R['cells'][str(h)][control]['sequence_row']['segment_f1_50_multi'] for h in (0,1));assert value==R['gates']['source_contribution']['L_vs_'+control]
 dump(PUB/'independent_final.json',{'passed':True,'complete_raw_score_and_membership_and_label_reconstruction':True,'unchanged_reference_evaluator_reproduced_from_independent_labels':True,'counts':counts,'strict_absolute_pass':gates,'paired_fix_harm_and_contribution_recomputed':True,'fresh_neural_backbone_forwards':0,'independent_algebraic_window_score_reconstructions':31401,'new_production_score_readouts':0,'raw_cache_sha256':sha(OUT/'readout.private.npz')})

def main():
 a=argparse.ArgumentParser();a.add_argument('stage',choices=['features','heads','final']);args=a.parse_args();torch.set_num_threads(1);start=time.time();ok=False
 try:globals()[args.stage]();ok=True
 finally:
  u=resource.getrusage(resource.RUSAGE_SELF);dump(OUT/f'audit_{args.stage}_worker_process.json',{'pid':os.getpid(),'ok':ok,'CPU_seconds':u.ru_utime+u.ru_stime,'wall_seconds':time.time()-start,'GPU_job_wall_seconds':0})
if __name__=='__main__':main()
