"""A45, one-class A4C denoising support versus nonspecific reconstruction; CPU only."""
import os,sys,time,json,math,hashlib,argparse,copy
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
import scripts.train_risk_constrained_a5c as t
from echo_routing.temporal import a4c_support as s

CFG='configs/experiments/video_a4c_support_v1.yaml'
OUT=ROOT/'runs/video_a4c_support_v1';RESULTS=ROOT/'docs/results/2026-10-28_a4c_support'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,default=float)+'\n')
def context():return t.Ctx(t.load_cfg(SimpleNamespace(config=t.ROOT/'configs/experiments/video_risk_constrained_a5c_v1.yaml',set=[])), 'configs/experiments/video_risk_constrained_a5c_v1.yaml')
def pins():
    assert sha(ROOT/'runs/video_risk_constrained_a5c_v1/weights_s0.json')=='2f7cfae0c41b442f0374d0d2b5566f89bfaf2d762cd7d2f4eac99b654f7ecb02'
    assert sha(ROOT/'runs/video_risk_constrained_a5c_v1/readout_cache_s0.npz')=='526ab49bc0b69af56b23f8a1757921695127b834e642761080b9f3f8bcf7a5ef'
    assert sha(ROOT/'models_frozen/video_within_view_v1/all_s0/final.pt')=='9b144925f32fac7d725d13231a132abeb6635fa44b6fe0dbcec7f5b59d035bbc'
def load_fold(ctx,f,inner=True):
    O=t.t31.outputs(ctx.a31,0,f);assert sha(ctx.a31.out/f'outputs_s0_f{f}.npz')==t.PINS['outputs'][0][str(f)]
    seqs=t.t31.load_pool(ctx.a31,f);W=t.t31.pool_W(seqs);R=t.t31.runs(ctx.a31,0,f);pk=R['P'];off=O['offset']
    assert sha(ctx.a31.out/'data'/f'runs_s0_f{f}.npz')==t.PINS['runs'][0][str(f)]
    x=s.run_embeddings(pk,O['emb'],off);out={'pk':pk,'seqs':seqs,'x':x,'off':off,'W':W,'emb':O['emb']}
    if not inner:return out
    allsess=sorted({t.session_of(r[0]) for q in seqs for r in q['refs']});perm=np.random.default_rng([20261028,45,f,1]).permutation(allsess);proper=set(perm[:len(perm)//2]);cal=set(perm[len(perm)//2:]);fp=[]
    for i in range(len(pk['base'])):
        lo,hi=pk['offsets'][i:i+2];q=seqs[int(pk['seq'][i])];samples=np.unique(W['s'][pk['rows'][lo:hi]-off].ravel());fp.append({t.session_of(q['refs'][int(j)][0]) for j in samples})
    fit=s.partition_mask(fp,proper);ca=s.partition_mask(fp,cal);negative=(pk['base']==2)&(pk['labels']==2);positive=(pk['base']==2)&(pk['labels']==3)
    ni=np.flatnonzero(fit&negative);pi=np.flatnonzero(fit&positive);ci=np.flatnonzero(ca&negative)
    assert not (proper&cal) and not (set().union(*(fp[i] for i in np.flatnonzero(fit))) & set().union(*(fp[i] for i in np.flatnonzero(ca))))
    assert len(ni)>=2 and len(pi)>=1 and len(ci)>=99, ('insufficient fixed inner split',f,len(ni),len(pi),len(ci))
    mu,sd=s.normalizer(x[ni]);a,u=s.plans(ni,pi,f)
    out.update(neg=ni,pos=pi,cal=ci,mu=mu,sd=sd,plans={'A':a,'U':u},footprints=fp,proper=proper,cal_sessions=cal)
    return out
def summary(F):
    return {'proper_A4C':len(F['neg']),'proper_A5C':len(F['pos']),'calibration_A4C':len(F['cal']),'proper_sessions':len(F['proper']),'calibration_sessions':len(F['cal_sessions']),'footprint_disjoint':True,'run_input_sha256':digest(F['x']),'proper_neg_ids_sha256':digest(F['neg']),'proper_pos_ids_sha256':digest(F['pos']),'calibration_ids_sha256':digest(F['cal']),'scaler_sha256':digest(np.stack([F['mu'],F['sd']])),'plan_sha256':{a:digest(p) for a,p in F['plans'].items()},'frozen_output_sha256':t.PINS['outputs'][0][str(F['f'])]}
def prepare(ctx):
    pins();r={}
    for f in [0,1]:
        F=load_fold(ctx,f);F['f']=f;r[str(f)]=summary(F)
        np.savez(OUT/f'inner_f{f}.npz',neg=F['neg'],pos=F['pos'],cal=F['cal'],mu=F['mu'],sd=F['sd'],plan_A=F['plans']['A'],plan_U=F['plans']['U'])
    write(OUT/'preparation.json',r);write(RESULTS/'preparation.json',r);return r
def new_model(f):
    torch.manual_seed(2026102845+f);m=s.SupportAE().eval();opt=torch.optim.AdamW(m.parameters(),lr=.001,betas=(.9,.999),eps=1e-8,weight_decay=.0001);g=torch.Generator().manual_seed(2026102845+f);return m,opt,g
def step(m,opt,g,x,ids,u):
    z=torch.from_numpy(np.asarray(x[ids],np.float32));noise=torch.randn(z.shape,generator=g)*.1
    for group in opt.param_groups:group['lr']=.001*.5*(1+math.cos(math.pi*u/600))
    opt.zero_grad(set_to_none=True);pred=m(z+noise);loss=((pred-z)**2).mean();loss.backward();gn=float(torch.nn.utils.clip_grad_norm_(m.parameters(),1.));opt.step()
    return {'u':u,'loss':float(loss.detach()),'grad_norm':gn,'lr':opt.param_groups[0]['lr'],'batch_sha256':digest(ids),'noise_sha256':digest(noise.numpy()),'pid':os.getpid(),'device':'cpu'}
def restore(m,opt,g,state):m.load_state_dict(state['model']);opt.load_state_dict(state['opt']);g.set_state(state['noise'])
def state(m,opt,g):return {'model':copy.deepcopy(m.state_dict()),'opt':copy.deepcopy(opt.state_dict()),'noise':g.get_state().clone()}
def checks(ctx):
    prep=prepare(ctx);F=load_fold(ctx,0);x=(F['x']-F['mu'])/F['sd'];plan=F['plans']['A'];m,opt,g=new_model(0);before=copy.deepcopy(m.state_dict());logs=[step(m,opt,g,x,plan[u],u) for u in range(6)];target=copy.deepcopy(m.state_dict())
    m2,o2,g2=new_model(0);[step(m2,o2,g2,x,plan[u],u) for u in range(6)];repeat=max(float((target[k]-m2.state_dict()[k]).abs().max()) for k in target)
    m3,o3,g3=new_model(0);[step(m3,o3,g3,x,plan[u],u) for u in range(3)];st=state(m3,o3,g3);mr,orr,gr=new_model(0);restore(mr,orr,gr,st);[step(mr,orr,gr,x,plan[u],u) for u in range(3,6)];resume=max(float((target[k]-mr.state_dict()[k]).abs().max()) for k in target)
    assert torch.equal(gr.get_state(),g.get_state());assert repeat==resume==0
    moved=sum(not torch.equal(v,before[k]) for k,v in target.items());assert moved==len(target)==8 and all(np.isfinite(u['loss']) and u['grad_norm']>0 for u in logs)
    # Independent full-model FP64 MSE gradient expression, not the production mean operator.
    md=copy.deepcopy(m).double();z=torch.from_numpy(np.asarray(x[plan[0][:3]],np.float64));yp=md(z);loss=((yp-z)**2).mean();ga=torch.autograd.grad(loss,tuple(md.parameters()),retain_graph=True);ref=sum((yp.flatten()[j]-z.flatten()[j])**2 for j in range(z.numel()))/z.numel();gb=torch.autograd.grad(ref,tuple(md.parameters()));err=max(float((a-b).abs().max()) for a,b in zip(ga,gb));assert err<1e-10
    for f in [0,1]:
        ff=load_fold(ctx,f);pk=ff['pk'];near=t.t27.nearest_rows(ff['W'],ff['seqs']);dev=0.
        for i in range(len(pk['base'])):
            qi=int(pk['seq'][i]);a,b=int(pk['a'][i]),int(pk['b'][i]);explicit=ff['emb'][near[qi][a:b]].astype(np.float64).mean(axis=0);dev=max(dev,float(np.max(abs(explicit-ff['x'][i]))))
        assert dev<1e-6;prep[str(f)]['explicit_repeated_sample_mean_max_abs']=dev
    r={'ok':True,'preparation':prep,'repeat6_max_abs':repeat,'forced_identical_data_path_max_abs':repeat,'resume3plus3_max_abs':resume,'noise_rng_resume_exact':True,'moved_state_tensors':moved,'parameters':sum(p.numel() for p in m.parameters()),'mse_gradient_reference_max_abs':err,'finished':t.now()};write(OUT/'checks.json',r);write(RESULTS/'checks.json',r);return r
def train(ctx):
    assert json.loads((OUT/'checks.json').read_text())['ok'];models={}
    for f in [0,1]:
        F=load_fold(ctx,f);F['f']=f; x=(F['x']-F['mu'])/F['sd']
        for arm in ['A','U']:
            rd=OUT/'train'/f'{arm}_f{f}';rd.mkdir(parents=True,exist_ok=True);assert not (rd/'final.pt').exists(),'training already exists; no duplicate'
            m,opt,g=new_model(f);init=copy.deepcopy(m.state_dict());fitids=F['neg'] if arm=='A' else np.r_[F['neg'],F['pos']];initial=float(s.errors(m,F['x'][fitids],F['mu'],F['sd']).mean());plan=F['plans'][arm];logs=[];first=0
            if (rd/'resume.pt').exists():
                st=torch.load(rd/'resume.pt',map_location='cpu',weights_only=False);assert st['plan']==digest(plan);restore(m,opt,g,st);logs=st['log'];first=len(logs)
            for u in range(first,600):
                rec=step(m,opt,g,x,plan[u],u);logs.append(rec)
                if (u+1)%100==0:
                    prog={'arm':arm,'fit_fold':f,'updates':u+1,'loss':rec['loss'],'pid':os.getpid(),'device':'cpu','written':t.now()};write(OUT/'progress.json',prog);print(json.dumps(prog),flush=True)
                if (u+1)%150==0:torch.save(state(m,opt,g)|{'plan':digest(plan),'log':logs},rd/'resume.pt')
            assert len(logs)==600 and len(opt.state)==8 and all(v['step'].item()==600 for v in opt.state.values())
            assert all(torch.isfinite(v).all() for v in m.state_dict().values());assert all(not torch.equal(v,init[k]) for k,v in m.state_dict().items())
            final=float(s.errors(m,F['x'][fitids],F['mu'],F['sd']).mean());assert final<initial,'no reconstruction learning'
            torch.save({'model':m.state_dict(),'mu':torch.from_numpy(F['mu']),'sd':torch.from_numpy(F['sd'])},rd/'final.pt');(rd/'updates.jsonl').write_text(''.join(json.dumps(u)+'\n' for u in logs))
            calerr=s.errors(m,F['x'][F['cal']],F['mu'],F['sd']);threshold=s.calibrate(calerr);np.save(rd/'calibration_errors.npy',calerr)
            assert np.array_equal(s.pvalues(calerr,calerr)<=.01,s.outlier(calerr,threshold))
            man={'arm':arm,'fit_fold':f,'updates':600,'optimizer_steps':600,'parameters':140064,'final_sha256':sha(rd/'final.pt'),'updates_sha256':sha(rd/'updates.jsonl'),'calibration_errors_sha256':sha(rd/'calibration_errors.npy'),'threshold':threshold,'calibration_n':len(calerr),'calibration_false_rejections':int((calerr>threshold).sum()),'initial_fit_mse':initial,'final_fit_mse':final,'initial_model_sha256':digest(torch.cat([v.flatten() for v in init.values()]).numpy()),'final_all8_tensors_moved':True,'input':summary(F),'pid':os.getpid(),'device':'cpu'}
            write(rd/'manifest.json',man);models[f'{arm}_f{f}']=man;assert sha(rd/'final.pt')==man['final_sha256'];(rd/'resume.pt').unlink()
    man={'models':models,'config_sha256':sha(ROOT/CFG),'contract_sha256':sha(ROOT/'docs/superpowers/plans/2026-10-28-a4c-support-training.md'),'code_sha256':{p:sha(ROOT/p) for p in ['scripts/train_a4c_support.py','echo_routing/temporal/a4c_support.py']},'A44_manifest_sha256':sha(ROOT/'runs/video_risk_constrained_a5c_v1/weights_s0.json'),'A44_private_cache_sha256':sha(ROOT/'runs/video_risk_constrained_a5c_v1/readout_cache_s0.npz'),'written':t.now()}
    write(OUT/'weights.json',man);write(RESULTS/'weights.json',man);return {'models':4,'updates':2400}
def freeze(commit,manifest_sha):
    assert sha(OUT/'weights.json')==sha(RESULTS/'weights.json')==manifest_sha
    man=json.loads((OUT/'weights.json').read_text())
    for k,v in man['models'].items():
        rd=OUT/'train'/k;assert sha(rd/'final.pt')==v['final_sha256'] and sha(rd/'calibration_errors.npy')==v['calibration_errors_sha256']
    write(OUT/'frozen.json',{'commit':commit,'manifest_sha256':manifest_sha,'written':t.now()});return {'frozen':4,'commit':commit}
def package(cells,policy,sources):
    fs=[cells[f'heldout_f{h}'][policy] for h in [0,1]];bs=[cells[f'heldout_f{h}']['Base'] for h in [0,1]];fix=sum(v['S1_fixed']['fixed'] for v in fs);harm=sum(v['correct_A4C_kept']['runs']-v['correct_A4C_kept']['kept'] for v in fs);net=[v['runs']['all']['net_fixed'] for v in fs];df=[v['sequence_row']['segment_f1_50_multi']-b['sequence_row']['segment_f1_50_multi'] for v,b in zip(fs,bs)]
    g={'S1_ok':fix>=11,'source_ok':len(set.union(*sources))>=3 and all(v['S1_fixed']['fixed']>=1 and len(src)>=1 for v,src in zip(fs,sources)),'A4C_ok':harm<=14,'nonA4C_invariant':all(v['non_A4C_unchanged'] for v in fs),'net_per_fold_ok':all(n>=0 for n in net),'net_oof_ok':sum(net)>=5,'F1_per_fold_ok':all(d>=0 for d in df),'F1_mean_ok':sum(df)/2>=.001,'native_ok':all(v['sequence_row']['native_macro_f1']-b['sequence_row']['native_macro_f1']>=-.003 for v,b in zip(fs,bs)),'boundary_recall_ok':all(v['sequence_row']['boundary_recall_multi']-b['sequence_row']['boundary_recall_multi']>=-.005 for v,b in zip(fs,bs)),'short_misses_ok':sum(v['counts']['fn_short'] for v in fs)<=sum(v['counts']['fn_short'] for v in bs)+3,'not_noop':sum(v['changed'] for v in fs)>0};g['passed']=all(g.values());return g
def readout(ctx):
    assert not (OUT/'readout.json').exists(),'once-only readout already exists';frozen=json.loads((OUT/'frozen.json').read_text());assert sha(OUT/'weights.json')==frozen['manifest_sha256'];man=json.loads((OUT/'weights.json').read_text());old=json.loads((ROOT/'runs/video_risk_constrained_a5c_v1/readout_s0.json').read_text());prior=np.load(ROOT/'runs/video_risk_constrained_a5c_v1/readout_cache_s0.npz',allow_pickle=False);data=t.t31.vpr.training_data(ctx.a31.a15.a10.a07p);a8=t.rsh.a08_ctx();cells={};cache={};sources={k:[] for k in ['N_A','N_U','C_A','C_U']};alls={k:[] for k in ['Base','N','C',*sources]};ys=[]
    for h in [0,1]:
        F=load_fold(ctx,h,False);D=t.d32.fold_data(ctx.d,0,h);D['M']=t.d32.set_masks(D);pk=D['pk'];assert np.array_equal(pk['base'],F['pk']['base']);el=np.flatnonzero(pk['base']==2);x=F['x'];errs={};rp={}
        for p in ['C','N']:
            rp[p]=np.full(len(pk['base']),-np.inf);rp[p][el]=prior[f'r_{p}_heldout_f{h}']
        for a in ['A','U']:
            key=f'{a}_f{1-h}';rd=OUT/'train'/key;assert sha(rd/'final.pt')==man['models'][key]['final_sha256'];st=torch.load(rd/'final.pt',map_location='cpu',weights_only=True);m=s.SupportAE().eval();m.load_state_dict(st['model']);errs[a]=s.errors(m,x,st['mu'].numpy(),st['sd'].numpy());cache[f'error_{a}_h{h}']=errs[a]
        preds={'Base':pk['base'].copy(),**{p:prior[f'run_pred_{p}_heldout_f{h}'] for p in ['C','N']}}
        for p in ['N','C']:
            for a in ['A','U']:preds[f'{p}_{a}']=s.deploy(pk['base'],rp[p],errs[a],man['models'][f'{a}_f{1-h}']['threshold'])
        P={'seqs':D['seqs'],'n_of':dict(data.n),'label_of':dict(data.label)};ev=t.bc.evalset(P,list(range(len(D['seqs']))),f'heldout_f{h}');cell={}
        for k,p in preds.items():
            row=t.tables_pred(p,D,a8.cfg,P,ev);ps=row.pop('preds');row.pop('pred');row['boundaries']=t.bounds(D,ps);ex,recs=t.a5c_extras(D,p);row.update(ex);row['non_A4C_unchanged']=bool(np.array_equal(p[pk['base']!=2],pk['base'][pk['base']!=2]));row['changed']=int((p!=pk['base']).sum());cell[k]=row;alls[k].append(p);cache[f'pred_{k}_h{h}']=p
            if k in sources:sources[k].append(recs)
            if k in ['Base','C','N']:assert row['sequence_row']==old['cells'][f'heldout_f{h}'][k]['sequence_row'] and row['counts']==old['cells'][f'heldout_f{h}'][k]['counts']
        cells[f'heldout_f{h}']=cell;ys.append(pk['labels']);cache[f'x_h{h}']=x;cache[f'label_h{h}']=pk['labels'];cache[f'r_C_h{h}']=rp['C'];cache[f'r_N_h{h}']=rp['N']
    yy=np.concatenate(ys);bb=np.concatenate(alls['Base']);oof={}
    for k,ps in alls.items():
        p=np.concatenate(ps);fx=int(((bb!=yy)&(p==yy)).sum());ha=int(((bb==yy)&(p!=yy)).sum());oof[k]={'fixed':fx,'broken':ha,'net':fx-ha,'A5C_fixed':int(((yy==3)&(bb==2)&(p==3)).sum()),'correct_A4C_harmed':int(((yy==2)&(bb==2)&(p!=2)).sum()),'F1_mean':float(np.mean([cells[f'heldout_f{h}'][k]['sequence_row']['segment_f1_50_multi'] for h in [0,1]])),'missed_short':sum(cells[f'heldout_f{h}'][k]['counts']['fn_short'] for h in [0,1]),'boundaries_deleted':sum(cells[f'heldout_f{h}'][k]['boundaries']['deleted_by_merge'] for h in [0,1])}
    gates={p:package(cells,p,sources[p]) for p in sources};aa=[cells[f'heldout_f{h}']['N_A'] for h in [0,1]];uu=[cells[f'heldout_f{h}']['N_U'] for h in [0,1]];con={'retain80_each_fold':all(a['S1_fixed']['fixed']>=math.ceil(.8*u['S1_fixed']['fixed']) for a,u in zip(aa,uu)),'harms_no_more_each_fold':all(a['correct_A4C_kept']['kept']>=u['correct_A4C_kept']['kept'] for a,u in zip(aa,uu)),'harms_5_fewer_oof':oof['N_U']['broken']-oof['N_A']['broken']>=5,'net5_more_oof':oof['N_A']['net']-oof['N_U']['net']>=5,'F1_no_less_each_fold':all(a['sequence_row']['segment_f1_50_multi']>=u['sequence_row']['segment_f1_50_multi'] for a,u in zip(aa,uu)),'absolute_A_passed':gates['N_A']['passed']};con['passed']=all(con.values())
    np.savez(OUT/'private_readout.npz',**cache);r={'cells':cells,'oof':oof,'gates':gates,'primary_contribution':con,'frozen':frozen,'private_cache_sha256':sha(OUT/'private_readout.npz'),'written':t.now()};write(OUT/'readout.json',r);write(RESULTS/'readout.json',r);return {'oof':oof,'gates':gates,'contribution':con}
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['prepare','check','train','freeze','readout'],required=True);ap.add_argument('--commit',default='');ap.add_argument('--manifest-sha',default='');args=ap.parse_args();OUT.mkdir(parents=True,exist_ok=True);RESULTS.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1);torch.use_deterministic_algorithms(True);ctx=context();pins();started=t.now();st=time.time();ok=False
    try:
        fn={'prepare':lambda:prepare(ctx),'check':lambda:checks(ctx),'train':lambda:train(ctx),'freeze':lambda:freeze(args.commit,args.manifest_sha),'readout':lambda:readout(ctx)}[args.stage];res=fn();ok=True;print(json.dumps(res,indent=1,default=float),flush=True)
    finally:t.ledger_append(OUT/'ledger.jsonl',f'A45 {args.stage} actual CPU stage','cpu',started,time.time()-st,ok,'No GPU; frozen visual embeddings are the intended model input')
if __name__=='__main__':main()
