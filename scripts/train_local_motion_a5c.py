"""A46 sole-owner stage runner. Private field/case caches never leave the server."""
import os

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
import argparse
import copy
import hashlib
import inspect
import itertools
import json
import math
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch.nn import functional as TF

import scripts.train_a4c_support as a45
import scripts.train_risk_constrained_a5c as t
from echo_routing.temporal import local_motion as lm

CFG='configs/experiments/video_local_motion_a5c_v1.yaml'
CARD='docs/superpowers/plans/2026-10-28-local-motion-a5c.md'
OUT=ROOT/'runs/video_local_motion_a5c_v1';RES=ROOT/'docs/results/2026-10-28_local_motion_a5c'
ARMS=['F','S'];U=600
sha=a45.sha;digest=a45.digest;write=a45.write

def publish(name,x):
    write(OUT/name,x);write(RES/name,x)

def coord(phase,**kw):
    old=json.loads((OUT/'coordination.json').read_text()) if (OUT/'coordination.json').exists() else {}
    write(OUT/'coordination.json',old|{'round':'A46','owner':'Codex','phase':phase,'pid':os.getpid(),'at':t.now(),**kw})

def pins(ctx):
    assert sha(ROOT/'models_frozen/video_within_view_v1/all_s0/final.pt')=='9b144925f32fac7d725d13231a132abeb6635fa44b6fe0dbcec7f5b59d035bbc'
    assert sha(ctx.a31.out/'split.json')==t.PINS['split']
    for f in [0,1]:
        assert sha(ctx.a31.out/f'outputs_s0_f{f}.npz')==t.PINS['outputs'][0][str(f)]
        assert sha(ctx.a31.out/'data'/f'runs_s0_f{f}.npz')==t.PINS['runs'][0][str(f)]

def pair_map(F):
    refs=t.t37.wt.window_refs(F.seqs,F.W['seq'][F.need],F.W['s'][F.need]);pairs=[];index={};pi=np.full((len(refs),15),-1,np.int32);counts={'valid':0,'repeat':0,'splice_or_gap':0}
    for j,row in enumerate(refs):
        for k,(x,y) in enumerate(itertools.pairwise(row)):
            if lm.valid_pair(x,y):
                key=(x,y)
                if key not in index:index[key]=len(pairs);pairs.append(key)
                pi[j,k]=index[key];counts['valid']+=1
            else:counts['repeat' if x==y else 'splice_or_gap']+=1
    return pairs,pi,counts

def loadF(ctx,f,device=None):
    F=t.Fold(ctx,0,f);O=t.t31.outputs(ctx.a31,0,f);F.emb=np.asarray(O['emb'],np.float32)
    z=np.load(OUT/f'input_f{f}.npz');assert np.array_equal(z['rows'],F.need);F.pi=z['pairs'];F.perms=z['perms'];F.at=np.full(len(F.W['seq']),-1,np.int64);F.at[F.need]=np.arange(len(F.need))
    if device is not None:F.bank=torch.from_numpy(np.load(OUT/f'flow_f{f}.npy')).to(device);F.embd=torch.from_numpy(F.emb).to(device)
    return F

def prepare(ctx):
    pins(ctx);ss=[];out={}
    for f in [0,1]:
        F=t.Fold(ctx,0,f);pairs,pi,c=pair_map(F);ps=np.stack([lm.tile_permutation(f,r) for r in F.need]);sess={t.session_of(r[0]) for q in F.seqs for r in q['refs']};ss.append(sess)
        assert len(F.need)==[5030,5437][f] and len(F.L['positives'])==[34,69][f] and len(F.L['negatives'])==[714,699][f] and len(F.L['other'])==[1,3][f]
        assert F.psha==t.EXPECTED_PLAN[0][f]['sha256'];ex=t.rc.exposure(F.pk,F.pl);assert ex['distinct_negatives']==len(F.L['negatives'])
        # Independent nearest-centre multiplicities for every original eligible run.
        near=t.t27.nearest_rows(F.W,F.seqs)
        for i in F.L['eligible']:
            lo,hi=F.pk['offsets'][i:i+2];qi=int(F.pk['seq'][i]);a,b=int(F.pk['a'][i]),int(F.pk['b'][i]);r,w=np.unique(near[qi][a:b],return_counts=True)
            assert np.array_equal(r,F.pk['rows'][lo:hi]-F.off) and np.max(abs(w/(b-a)-F.pk['weights'][lo:hi]))<1e-12
        p=OUT/f'pairrefs_f{f}.json';write(p,pairs)
        np.savez(OUT/f'input_f{f}.npz',rows=F.need,pairs=pi,perms=ps,plan=F.pl,markers=F.W['marker'][F.need])
        out[str(f)]={'eligible_runs':len(F.L['eligible']),'rows':len(F.need),'grid':len(F.W['seq']),'fit_positive':len(F.L['positives']),'fit_negative':len(F.L['negatives']),'other':len(F.L['other']),'unique_pairs':len(pairs),'pair_counts':c,'flow_bytes':len(pairs)*2*56*56*2,'pairrefs_sha256':sha(p),'inputs_sha256':sha(OUT/f'input_f{f}.npz'),'permutation_sha256':digest(ps),'plan_sha256':F.psha,'exposure':ex,'memberships_verified':True,'sessions':len(sess),'original_output_sha256':t.PINS['outputs'][0][str(f)],'runs_sha256':t.PINS['runs'][0][str(f)]}
    assert not ss[0]&ss[1]
    assert sum(x['flow_bytes'] for x in out.values())+100_000_000<3e9
    out['isolation']=True;out['written']=t.now();out['config_sha256']=sha(ROOT/CFG);out['contract_sha256']=sha(ROOT/CARD)
    publish('preparation.json',out);return out

def raft_model(device):
    from torchvision.models.optical_flow import Raft_Small_Weights, raft_small
    w=Raft_Small_Weights.C_T_V2;p=OUT/'raft_small_C_T_V2-01064c6d.pth'
    if not p.exists():torch.hub.download_url_to_file(w.url,str(p),hash_prefix='01064c6d',progress=False)
    m=raft_small(weights=None).eval().to(device);m.load_state_dict(torch.load(p,map_location=device,weights_only=True));m.requires_grad_(False)
    return m,w,p

def flow(ctx):
    assert (OUT/'preparation.json').exists();device='cuda';m,w,wp=raft_model(device);prep=json.loads((OUT/'preparation.json').read_text());out={}
    for f in [0,1]:
        done=OUT/f'flow_done_f{f}.json';path=OUT/f'flow_f{f}.npy'
        if done.exists():
            out[str(f)]=json.loads(done.read_text());assert sha(path)==out[str(f)]['flow_sha256'];continue
        F=t.Fold(ctx,0,f);pairs=json.loads((OUT/f'pairrefs_f{f}.json').read_text());assert sha(OUT/f'pairrefs_f{f}.json')==prep[str(f)]['pairrefs_sha256']
        pp=OUT/f'flow_progress_f{f}.json';first=json.loads(pp.read_text())['next'] if pp.exists() else 0
        bank=np.lib.format.open_memmap(path,mode='r+' if first else 'w+',dtype=np.float16,shape=(len(pairs),2,56,56));st=time.time();coord('flow_preparation',fold=f,next_pair=first,total_pairs=len(pairs))
        for k in range(first,len(pairs),32):
            chunk=pairs[k:k+32];refs=sorted({tuple(r) for p in chunk for r in p});at={r:i for i,r in enumerate(refs)};fr=t.t37.rr.decode_frames(refs,F.fp,0);x=torch.from_numpy(fr).permute(0,3,1,2).to(device)
            im1=x[[at[tuple(p[0])] for p in chunk]];im2=x[[at[tuple(p[1])] for p in chunk]];im1,im2=w.transforms()(im1,im2)
            with torch.no_grad():raw=m(im1,im2,num_flow_updates=12)[-1];z=TF.avg_pool2d(raw,4).div(8).clamp(-4,4)
            assert torch.isfinite(z).all();bank[k:k+len(chunk)]=z.cpu().numpy().astype(np.float16)
            if (k//32)%20==0 or k+32>=len(pairs):
                bank.flush();write(pp,{'next':min(k+32,len(pairs)),'total':len(pairs),'pid':os.getpid(),'device':torch.cuda.get_device_name(0),'written':t.now()});print(json.dumps({'flow_fold':f,'pairs':min(k+32,len(pairs)),'total':len(pairs),'elapsed':round(time.time()-st,1)}),flush=True)
        bank.flush();stats={'pairs':len(pairs),'flow_sha256':sha(path),'finite':bool(np.isfinite(bank).all()),'zero_element_share':float((bank==0).mean()),'clipped_element_share':float((abs(bank)==4).mean()),'abs_mean':float(np.abs(bank.astype(np.float32)).mean()),'max_abs':float(abs(bank).max()),'dtype':'float16','shape':list(bank.shape),'seconds':time.time()-st};assert stats['finite'] and stats['abs_mean']>0
        write(done,stats);out[str(f)]=stats
    import torchvision
    rec={'folds':out,'raft_weight_sha256':sha(wp),'raft_url':w.url,'raft_pretraining':str(w.meta['_docs']),'raft_source_sha256':sha(inspect.getfile(type(m))),'torch':torch.__version__,'torchvision':torchvision.__version__,'iterations':12,'official_transform':'[-1,1], no resize, original224','written':t.now()}
    publish('flow.json',rec);return {'flow_complete':True,'pairs':sum(v['pairs'] for v in out.values())}

def new_model(f,device):
    torch.manual_seed(2026102846+f);m=lm.MotionRepair().eval().to(device);opt=torch.optim.AdamW(m.parameters(),lr=.001,betas=(.9,.999),eps=1e-8,weight_decay=.0001);return m,opt

def batch(F,rows,arm,zero=False):
    j=F.at[rows];assert (j>=0).all();x=lm.motion_inputs(F.bank,F.pi[j],F.W['marker'][rows],F.perms[j] if arm=='S' else None)
    if zero:x[:,:2]=0
    return x,F.embd[torch.as_tensor(rows,device=F.embd.device)]

def step(m,opt,F,u,arm):
    st=time.time();rows,v,seg=F.occ(u);x,e=batch(F,rows,arm);vv=torch.as_tensor(v,dtype=torch.float32,device=x.device);sg=torch.as_tensor(seg,device=x.device)
    lr=.001*t.t31.lr_factor(u,60,U)
    for g in opt.param_groups:g['lr']=lr
    opt.zero_grad(set_to_none=True);s=m(x,e);loss,r=lm.loss_from_scores(s,vv,sg);loss.backward()
    groups={'motion':m.motion,'appearance':m.appearance,'fusion':m.fusion};gs={k:float(sum(p.grad.detach().double().square().sum() for p in module.parameters())) for k,module in groups.items()}
    gn=float(torch.nn.utils.clip_grad_norm_(m.parameters(),1));opt.step();torch.cuda.synchronize()
    return {'u':u,'loss':float(loss.detach()),'run_scores':r.detach().cpu().tolist(),'grad_norm':gn,'grad_sq':gs,'lr':lr,'n_occ':len(rows),'rows_sha256':digest(rows),'seconds':time.time()-st,'pid':os.getpid(),'device':torch.cuda.get_device_name(0)}

def state(m,opt,log):return {'model':copy.deepcopy(m.state_dict()),'opt':copy.deepcopy(opt.state_dict()),'log':copy.deepcopy(log)}
def delta(a,b):return max(float((a[k]-b[k]).abs().max()) for k in a)

def checks(ctx):
    pins(ctx);assert json.loads((OUT/'flow.json').read_text())['iterations']==12;F=loadF(ctx,0,'cuda');m,o=new_model(0,'cuda');init=copy.deepcopy(m.state_dict());rows,_,_=F.occ(0);x,e=batch(F,rows,'F')
    assert torch.equal(m(x,e),torch.full((len(rows),),-2.,device='cuda'))
    logs=[step(m,o,F,u,'F') for u in range(6)];goal=copy.deepcopy(m.state_dict());assert logs[0]['grad_sq']['motion']==0 and all(logs[-1]['grad_sq'][k]>0 for k in ['motion','appearance','fusion'])
    assert all(not torch.equal(v,init[k]) for k,v in goal.items())
    mm,oo=new_model(0,'cuda');[step(mm,oo,F,u,'F') for u in range(6)];rep=delta(goal,mm.state_dict());assert rep==0
    mr,orr=new_model(0,'cuda');ll=[step(mr,orr,F,u,'F') for u in range(3)];st=state(mr,orr,ll);mr2,or2=new_model(0,'cuda');mr2.load_state_dict(st['model']);or2.load_state_dict(st['opt']);[step(mr2,or2,F,u,'F') for u in range(3,6)];res=delta(goal,mr2.state_dict());assert res==0
    # Full production chain rule vs individual-window re-forward, after motion gradients become live.
    rows,v,seg=F.occ(0);x,e=batch(F,rows,'F');vv=torch.tensor(v,dtype=torch.float32,device='cuda');sg=torch.tensor(seg,device='cuda');m.zero_grad(set_to_none=True);scores=m(x,e);loss,_=lm.loss_from_scores(scores,vv,sg);loss.backward();ga=torch.cat([p.grad.flatten().double() for p in m.parameters()]);sl=scores.detach().requires_grad_(True);ll,_=lm.loss_from_scores(sl,vv,sg);g=torch.autograd.grad(ll,sl)[0];m.zero_grad(set_to_none=True)
    s2=[]
    for k in range(len(rows)):
        sk=m(x[k:k+1],e[k:k+1]);sk.backward(g[k:k+1]);s2.append(sk.detach())
    gb=torch.cat([p.grad.flatten().double() for p in m.parameters()]);rel=float(torch.linalg.vector_norm(ga-gb)/torch.linalg.vector_norm(ga));scoredev=float((torch.cat(s2)-scores.detach()).abs().max());assert rel<1e-4 and scoredev<1e-4
    # Maximum planned actual CNN batch, for memory/runtime projection, with true backward.
    ue=int(np.argmax([len(F.occ(u)[0]) for u in range(U)]));torch.cuda.reset_peak_memory_stats();maxrec=step(m,o,F,ue,'F');mem=torch.cuda.max_memory_allocated()/1e9;assert mem<40
    total_est=sum(np.mean([len(t.Fold(ctx,0,f).occ(u)[0]) for u in range(U)]) for f in [0,1])*2*U*maxrec['seconds']/maxrec['n_occ'];assert total_est<5*3600
    # Frozen RAFT official transform, batching parity and elementary displacement sanity.
    raft,w,wp=raft_model('cuda');pair=json.loads((OUT/'pairrefs_f0.json').read_text())[0];fr=t.t37.rr.decode_frames([tuple(pair[0]),tuple(pair[1])],F.fp,0);im=torch.from_numpy(fr).permute(0,3,1,2).to('cuda');p,q=w.transforms()(im[:1],im[1:]);assert torch.equal(p,im[:1].float()/255*2-1)
    with torch.no_grad():
        single=raft(p,q,num_flow_updates=12)[-1];double=raft(p.repeat(2,1,1,1),q.repeat(2,1,1,1),num_flow_updates=12)[-1][:1];same=raft(p,p,num_flow_updates=12)[-1];trans=raft(p,torch.roll(p,4,-1),num_flow_updates=12)[-1]
    parity=float((single-double).abs().max());sameerr=float(same.abs().median());dx=float(trans[:,0,32:-32,32:-32].median());assert parity<.005 and sameerr<.5 and 1<dx<7
    cached=np.load(OUT/'flow_f0.npy',mmap_mode='r')[0].astype(np.float32);recomputed=TF.avg_pool2d(single,4).div(8).clamp(-4,4)[0].cpu().numpy().astype(np.float16).astype(np.float32);cachedev=float(abs(cached-recomputed).max());assert cachedev<.002
    r={'ok':True,'parameters':sum(p.numel() for p in m.parameters()),'state_tensors':len(init),'repeat6_max_abs':rep,'forced_F_path_max_abs':rep,'resume3plus3_max_abs':res,'production_gradient_rel':rel,'perwindow_score_max_abs':scoredev,'first_step_motion_zero':True,'all_components_live_by_update6':True,'all_state_tensors_moved':True,'max_batch_occ':maxrec['n_occ'],'peak_gpu_gb':mem,'projected_training_gpu_hours':total_est/3600,'raft_batch_parity_max_abs':parity,'same_image_flow_abs_median':sameerr,'synthetic_dx4_estimated_median':dx,'cached_vs_fresh_flow_max_abs':cachedev,'raft_unchanged_sha256':sha(wp),'written':t.now()};publish('checks.json',r);return r

def train(ctx):
    pins(ctx);assert json.loads((OUT/'checks.json').read_text())['ok'];prep=json.loads((OUT/'preparation.json').read_text());models={}
    for f in [0,1]:
        F=loadF(ctx,f,'cuda')
        for arm in ARMS:
            key=f'{arm}_f{f}';rd=OUT/'train'/key;rd.mkdir(parents=True,exist_ok=True);ident=hashlib.sha256((sha(ROOT/CFG)+sha(ROOT/CARD)+sha(ROOT/'echo_routing/temporal/local_motion.py')+sha(Path(__file__))+prep[str(f)]['inputs_sha256']+json.loads((OUT/'flow.json').read_text())['folds'][str(f)]['flow_sha256']+key).encode()).hexdigest()
            if (rd/'manifest.json').exists():
                mm=json.loads((rd/'manifest.json').read_text());assert mm['identity']==ident and sha(rd/'final.pt')==mm['final_sha256'];models[key]=mm;continue
            assert not (rd/'final.pt').exists(),'unmanifested final requires audit, do not retrain'
            m,o=new_model(f,'cuda');init=copy.deepcopy(m.state_dict());log=[]
            if (rd/'resume.pt').exists():
                st=torch.load(rd/'resume.pt',map_location='cuda',weights_only=False);assert st['identity']==ident;m.load_state_dict(st['model']);o.load_state_dict(st['opt']);log=st['log']
            coord('formal_training',model=key,update=len(log),device=torch.cuda.get_device_name(0),identity=ident)
            for u in range(len(log),U):
                rec=step(m,o,F,u,arm);assert math.isfinite(rec['loss']) and math.isfinite(rec['grad_norm']);log.append(rec)
                if (u+1)%50==0:
                    pr={'model':key,'update':u+1,'total':U,'pid':os.getpid(),'device':rec['device'],'loss':rec['loss'],'window_occurrences':sum(x['n_occ'] for x in log),'at':t.now()};write(OUT/'progress.json',pr);print(json.dumps(pr),flush=True)
                if (u+1)%150==0 and u+1<U:
                    tmp=rd/'resume.tmp';torch.save(state(m,o,log)|{'identity':ident},tmp);tmp.replace(rd/'resume.pt')
            steps=[int(v['step'].item()) for v in o.state.values()];assert len(log)==U and set(steps)=={U} and len(steps)==len(list(m.parameters()))
            moved={k:not torch.equal(v,init[k]) for k,v in m.state_dict().items()};assert all(moved.values()) and all(torch.isfinite(v).all() for v in m.state_dict().values())
            torch.save({k:v.detach().cpu() for k,v in m.state_dict().items()},rd/'final.pt');(rd/'updates.jsonl').write_text(''.join(json.dumps(v)+'\n' for v in log));torch.save({'steps':steps,'final_update':U,'parameter_names':list(dict(m.named_parameters()))},rd/'optimizer_steps.pt')
            man={'arm':arm,'fit_fold':f,'identity':ident,'updates':U,'optimizer_steps':steps,'parameters':sum(p.numel() for p in m.parameters()),'state_tensors':len(moved),'all_state_tensors_moved':moved,'final_sha256':sha(rd/'final.pt'),'updates_sha256':sha(rd/'updates.jsonl'),'optimizer_receipt_sha256':sha(rd/'optimizer_steps.pt'),'plan_sha256':F.psha,'inputs_sha256':prep[str(f)]['inputs_sha256'],'flow_sha256':sha(OUT/f'flow_f{f}.npy'),'initial_state_sha256':digest(torch.cat([v.flatten().cpu() for v in init.values()]).numpy()),'occurrences':sum(v['n_occ'] for v in log),'pid':os.getpid(),'device':torch.cuda.get_device_name(0),'finished':t.now()};write(rd/'manifest.json',man);models[key]=man
            if (rd/'resume.pt').exists():(rd/'resume.pt').unlink()
        del F;torch.cuda.empty_cache()
    man={'models':models,'config_sha256':sha(ROOT/CFG),'contract_sha256':sha(ROOT/CARD),'base_sha256':'9b144925f32fac7d725d13231a132abeb6635fa44b6fe0dbcec7f5b59d035bbc','code_sha256':{p:sha(ROOT/p) for p in ['scripts/train_local_motion_a5c.py','echo_routing/temporal/local_motion.py']},'preparation_sha256':sha(OUT/'preparation.json'),'flow_manifest_sha256':sha(OUT/'flow.json'),'checks_sha256':sha(OUT/'checks.json'),'written':t.now()};publish('weights.json',man);coord('trained_pending_git_freeze');return {'models':4,'real_steps':2400}

def freeze(commit,msha):
    assert commit and sha(OUT/'weights.json')==sha(RES/'weights.json')==msha
    man=json.loads((OUT/'weights.json').read_text())
    for k,v in man['models'].items():assert sha(OUT/'train'/k/'final.pt')==v['final_sha256']
    for p,h in man['code_sha256'].items():assert sha(ROOT/p)==h
    write(OUT/'frozen.json',{'commit':commit,'manifest_sha256':msha,'at':t.now()});coord('frozen_pending_once_readout',manifest_commit=commit);return {'frozen':4,'commit':commit}

def score_rows(m,F,arm,zero=False):
    out=[]
    with torch.no_grad():
        for j in range(0,len(F.need),32):
            rows=F.need[j:j+32];x,e=batch(F,rows,arm,zero);out.append(m(x,e).cpu().numpy())
    return np.concatenate(out)

def readout(ctx):
    assert not (OUT/'readout.json').exists() and not (OUT/'readout_started.json').exists(),'once readout already started; audit/resume cache, never repeat'
    frozen=json.loads((OUT/'frozen.json').read_text());assert sha(OUT/'weights.json')==frozen['manifest_sha256'];man=json.loads((OUT/'weights.json').read_text());write(OUT/'readout_started.json',{'at':t.now(),'pid':os.getpid(),'frozen':frozen});coord('once_readout')
    data=t.t31.vpr.training_data(ctx.a31.a15.a10.a07p);a8=t.rsh.a08_ctx();cells={};cache={};sources={k:[] for k in ['F','S','Z_F','Z_S']};alls={k:[] for k in ['Base',*sources]};ys=[]
    for h in [0,1]:
        F=loadF(ctx,h,'cuda');D=t.d32.fold_data(ctx.d,0,h);D['M']=t.d32.set_masks(D);pk=D['pk'];assert np.array_equal(F.pk['base'],pk['base']);preds={'Base':pk['base'].copy()}
        for arm in ARMS:
            key=f'{arm}_f{1-h}';rd=OUT/'train'/key;assert sha(rd/'final.pt')==man['models'][key]['final_sha256'];m,_=new_model(1-h,'cuda');m.load_state_dict(torch.load(rd/'final.pt',map_location='cuda',weights_only=True))
            for zero in [False,True]:
                name=f'Z_{arm}' if zero else arm;s=score_rows(m,F,arm,zero);r=np.full(len(pk['base']),-np.inf);lookup=np.full(len(F.W['seq']),np.nan);lookup[F.need]=s
                for i in np.flatnonzero(pk['base']==2):
                    lo,hi=pk['offsets'][i:i+2];rows=pk['rows'][lo:hi]-F.off;r[i]=np.dot(lookup[rows].astype(np.float64),pk['weights'][lo:hi])
                preds[name]=lm.deploy(pk['base'],r);cache[f'score_{name}_h{h}']=s;cache[f'r_{name}_h{h}']=r
        P={'seqs':D['seqs'],'n_of':dict(data.n),'label_of':dict(data.label)};ev=t.bc.evalset(P,list(range(len(D['seqs']))),f'heldout_f{h}');cell={}
        for name,p in preds.items():
            row=t.tables_pred(p,D,a8.cfg,P,ev);ps=row.pop('preds');row.pop('pred');row['boundaries']=t.bounds(D,ps);ex,recs=t.a5c_extras(D,p);row.update(ex);row['non_A4C_unchanged']=bool(np.array_equal(p[pk['base']!=2],pk['base'][pk['base']!=2]));row['changed']=int((p!=pk['base']).sum());cell[name]=row;alls[name].append(p);cache[f'pred_{name}_h{h}']=p
            if name in sources:sources[name].append(recs)
        cache[f'rows_h{h}']=F.need;cache[f'label_h{h}']=pk['labels'];ys.append(pk['labels']);cells[f'heldout_f{h}']=cell
        write(OUT/f'readout_fold{h}.json',cell);np.savez(OUT/f'private_readout_fold{h}.npz',**{k:v for k,v in cache.items() if k.endswith(f'h{h}')});del F;torch.cuda.empty_cache()
    yy=np.concatenate(ys);bb=np.concatenate(alls['Base']);oof={}
    for name,ps in alls.items():
        p=np.concatenate(ps);fx=int(((bb!=yy)&(p==yy)).sum());ha=int(((bb==yy)&(p!=yy)).sum());oof[name]={'fixed':fx,'broken':ha,'net':fx-ha,'A5C_fixed':int(((yy==3)&(bb==2)&(p==3)).sum()),'correct_A4C_harmed':int(((yy==2)&(bb==2)&(p!=2)).sum()),'F1_mean':float(np.mean([cells[f'heldout_f{h}'][name]['sequence_row']['segment_f1_50_multi'] for h in [0,1]])),'missed_short':sum(cells[f'heldout_f{h}'][name]['counts']['fn_short'] for h in [0,1]),'boundaries_deleted':sum(cells[f'heldout_f{h}'][name]['boundaries']['deleted_by_merge'] for h in [0,1])}
    gates={name:a45.package(cells,name,sources[name]) for name in ['F','S']};fc=[cells[f'heldout_f{h}']['F'] for h in [0,1]];sc=[cells[f'heldout_f{h}']['S'] for h in [0,1]];con={'fix_each_fold':all(f['S1_fixed']['fixed']>=s['S1_fixed']['fixed'] for f,s in zip(fc,sc)),'fix_plus5_oof':oof['F']['A5C_fixed']-oof['S']['A5C_fixed']>=5,'harms_no_more_each_fold':all(f['correct_A4C_kept']['kept']>=s['correct_A4C_kept']['kept'] for f,s in zip(fc,sc)),'net_plus5_oof':oof['F']['net']-oof['S']['net']>=5,'F1_each_fold':all(f['sequence_row']['segment_f1_50_multi']>=s['sequence_row']['segment_f1_50_multi'] for f,s in zip(fc,sc)),'absolute_F':gates['F']['passed']};con['passed']=all(con.values())
    np.savez(OUT/'private_readout.npz',**cache);r={'cells':cells,'oof':oof,'gates':gates,'representation_contribution':con,'frozen':frozen,'private_cache_sha256':sha(OUT/'private_readout.npz'),'written':t.now()};publish('readout.json',r);coord('readout_complete_pending_independent_audit');return {'oof':oof,'gates':gates,'contribution':con}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',required=True,choices=['prepare','flow','check','train','freeze','readout']);ap.add_argument('--commit',default='');ap.add_argument('--manifest-sha',default='');args=ap.parse_args();OUT.mkdir(parents=True,exist_ok=True);RES.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    ctx=a45.context();started=t.now();st=time.time();ok=False;kind='cpu' if args.stage in ['prepare','freeze'] else 'gpu'
    try:
        r={'prepare':lambda:prepare(ctx),'flow':lambda:flow(ctx),'check':lambda:checks(ctx),'train':lambda:train(ctx),'freeze':lambda:freeze(args.commit,args.manifest_sha),'readout':lambda:readout(ctx)}[args.stage]();ok=True;print(json.dumps(r,indent=1,default=float),flush=True)
    finally:t.ledger_append(OUT/'ledger.jsonl',f'A46 {args.stage} actual stage wall',kind,started,time.time()-st,ok,'bootstrap outside stage and root active intervals accounted separately; GPU stage wall includes pixel I/O')
if __name__=='__main__':main()
