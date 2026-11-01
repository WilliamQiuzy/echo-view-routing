#!/usr/bin/env python3
"""A52 fixed point-frame H/R source experiment, conditional target remains gated."""
from __future__ import annotations
import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES','0');os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse,copy,hashlib,json,resource,sys,time
from collections import defaultdict
from fractions import Fraction
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import av,numpy as np,torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from echo_routing.temporal import point_detail as pd,direct_view_training as dt
from echo_routing.temporal.pw_transfer import state_sha,BASE_SHA
from echo_routing.temporal.dev_train import lr_factor
from scripts.prepare_direct_view_evidence import sha
OUT=ROOT/'runs/a52_point_detail_v1';PUB=ROOT/'docs/results/2026-11-01_point_detail_a5c'
PRIOR=ROOT/'runs/a51_verified_direct_view_pilot_v1';SOURCE=ROOT/'runs/a51_discriminative_evidence_v1/source'
BASE=ROOT/'runs/a48_valve_source_intake_v1/backbone_G.pt'
CARD='docs/superpowers/plans/2026-11-01-point-detail-supervision.md'
ARMS=('H','R')
write=dt.write_json


def metric(y,s,discord):
    a=dt.source_metrics(y,s,discord);p=s[y];n=s[~y]
    a.update(A4C_recall=a['a4c_recall'],A5C_recall=a['a5c_recall'],AUROC=float(np.mean((p[:,None]>n[None,:])+.5*(p[:,None]==n[None,:]))))
    return a


def project_files():
    paths=[Path(CARD),Path('scripts/train_point_detail.py'),Path('scripts/audit_point_detail.py'),Path('scripts/supervise_point_detail.py'),Path('echo_routing/temporal/point_detail.py')]
    for module in list(sys.modules.values()):
        p=getattr(module,'__file__',None)
        if p:
            p=Path(p).resolve()
            if p.is_relative_to(ROOT) and '.venv' not in p.parts and p.suffix=='.py':paths.append(p.relative_to(ROOT))
    return {str(p):sha(ROOT/p) for p in sorted(set(paths))}


def prepare():
    if (OUT/'bound.private.json').exists():raise RuntimeError('point inputs already bound; no replacement')
    selection=json.loads((PRIOR/'selection.private.json').read_text());bound=json.loads((PRIOR/'bound.private.json').read_text())
    assert len(selection['requests'])==len(bound['requests'])==702
    assert sha(BASE)==BASE_SHA
    by=defaultdict(list)
    for r in selection['requests']:by[(r['directory_label'],r['video'])].append(r)
    cache=[];records={};seen={a:defaultdict(list) for a in ('raw','H','R')}
    for v in bound['videos']:
        key=v['folder'],v['video'];path=SOURCE/'videos'/key[0]/key[1];assert sha(path)==v['sha256']
        need={int(r['centre']) for r in by[key]};raw={};pts=[]
        with av.open(str(path)) as c:
            s=c.streams.video[0];s.thread_type='NONE';w,h=s.width,s.height;fps=float(s.average_rate)
            for i,fr in enumerate(c.decode(s)):
                q=Fraction(fr.pts)*Fraction(fr.time_base)*30;assert q.denominator==1;pts.append(q.numerator)
                if i in need:raw[i]=fr.to_ndarray(format='rgb24')
        assert pts==v['native_pts_indices'] and len(pts)==v['decoded_frames'] and set(raw)==need
        for r in by[key]:
            assert r['centre']==r['frames'][8], 'point must equal sealed marker slot8'
            x=raw[r['centre']];H=pd.point_canvas(x,'H');R=pd.point_canvas(x,'R');rh=hashlib.sha256(x.tobytes()).hexdigest();seen['raw'][rh].append(r)
            rec={**r,'raw_point_sha256':rh,'H_sha256':hashlib.sha256(H.tobytes()).hexdigest(),'R_sha256':hashlib.sha256(R.tobytes()).hexdigest(),
              'geometry':[w,h,fps,len(pts),float(x.mean()),float(x.std()),float((x.max(-1).astype(float)-x.min(-1)).mean()),w/h]}
            records[r['id']]=rec;cache.append((r['id'],H))
            for a in ARMS:seen[a][rec[a+'_sha256']].append(r)
    for kind,hashed in seen.items():
        for same in hashed.values():
            assert len({r['patient'] for r in same})==1,kind+' point overlap across patients'
            assert len({r['role'] for r in same})==1,kind+' point overlap across folds'
            assert len({r['true_label'] for r in same})==1,kind+' point has conflicting labels'
    rows=[records[r['id']] for r in selection['requests']];lookup=dict(cache);images=np.stack([lookup[r['id']] for r in rows])
    OUT.mkdir(parents=True,exist_ok=True);np.savez_compressed(OUT/'points.private.npz',H=images)
    fit=[r for r in rows if r['role']=='fit'];y=np.asarray([r['true_label']=='A5C' for r in fit],int);assert len(fit)==569
    plan=dt.source_plan(y);np.save(OUT/'plan.npy',plan)
    x=np.asarray([r['geometry'] for r in fit]);mean,scale=x.mean(0),x.std(0);scale[scale<1e-12]=1
    g=LogisticRegression(C=1.,solver='lbfgs',max_iter=1000,random_state=2026110151).fit((x-mean)/scale,y)
    control={'mean':mean.tolist(),'scale':scale.tolist(),'coef':g.coef_[0].tolist(),'intercept':float(g.intercept_[0]),'prior':float(y.mean())}
    private={'selection_sha256':sha(PRIOR/'selection.private.json'),'prior_binding_sha256':sha(PRIOR/'bound.private.json'),'rows':rows,'geometry':control,'videos':bound['videos']}
    write(OUT/'bound.private.json',private)
    rec={'run_id':'A52','scope':'exploratory reused source diagnostic; direct point-view labels, no anatomy truth','fit':569,'diagnostic':133,
      'source_patients':[36,10],'source_true_counts':{'fit':[330,239],'diagnostic':[94,39]},'excluded_fit_markers':5,
      'bound_sha256':sha(OUT/'bound.private.json'),'points_sha256':sha(OUT/'points.private.npz'),'plan_sha256':sha(OUT/'plan.npy'),
      'plan_covers569':set(plan.ravel())==set(range(569)),'unique_point_hashes':{a:len(v) for a,v in seen.items()},'within_partition_same_label_multiplicity_retained':True,'cross_patient_point_overlap':0,
      'base_sha256':sha(BASE),'Base_A08_sha256':sha(ROOT/'models_frozen/video_within_view_v1/all_s0/final.pt'),
      'code':project_files(),'versions':{'torch':torch.__version__,'numpy':np.__version__,'Pillow':Image.__version__},'diagnostic_model_forwards':0,'formal_models':0,
      'source_arms':ARMS,'updates':600,'occurrences_per_update':8,'all_body_FPN_live':True,'source_target_budget_shared_with_A51':True}
    write(PUB/'input_manifest.json',rec);print(json.dumps({k:v for k,v in rec.items() if k!='code'}))


def verify():
    m=json.loads((PUB/'input_manifest.json').read_text())
    for key,p in [('bound_sha256',OUT/'bound.private.json'),('points_sha256',OUT/'points.private.npz'),('plan_sha256',OUT/'plan.npy'),('base_sha256',BASE)]:assert sha(p)==m[key],key
    assert sha(ROOT/'models_frozen/video_within_view_v1/all_s0/final.pt')==m['Base_A08_sha256']
    for p,h in m['code'].items():assert sha(ROOT/p)==h,p
    return m


class Inputs:
    def __init__(self):
        self.bound=json.loads((OUT/'bound.private.json').read_text());self.rows=self.bound['rows'];self.fit=[i for i,r in enumerate(self.rows) if r['role']=='fit'];self.diag=[i for i,r in enumerate(self.rows) if r['role']=='diagnostic']
        self.H=np.load(OUT/'points.private.npz')['H'];assert self.H.shape==(702,448,448,3)
        self.R=np.stack([np.asarray(Image.fromarray(x).resize((224,224),Image.Resampling.BILINEAR).resize((448,448),Image.Resampling.BILINEAR)) for x in self.H]);self.plan=np.load(OUT/'plan.npy')
        for i,r in enumerate(self.rows):
            for a in ARMS:assert hashlib.sha256(getattr(self,a)[i].tobytes()).hexdigest()==r[a+'_sha256']
    def tensor(self,index,arm):return pd.normalise(torch.from_numpy(getattr(self,arm)[index:index+1]).to('cuda'))


def build():
    torch.manual_seed(2026110152);return pd.build(torch.load(BASE,weights_only=True,map_location='cpu')).cuda()


def opt_for(m):
    return torch.optim.AdamW([{'params':list(m.backbone.parameters()),'lr':1e-5,'base_lr':1e-5},{'params':list(m.head.parameters()),'lr':1e-3,'base_lr':1e-3}],betas=(.9,.999),eps=1e-8,weight_decay=1e-4)


def groups(m):
    return {'stem':list(m.backbone.body.conv1.parameters()),**{f'layer{k}':list(getattr(m.backbone.body,f'layer{k}').parameters()) for k in range(1,5)},'FPN':list(m.backbone.fpn.parameters()),'head':list(m.head.parameters())}


def step(m,opt,data,u,arm,force=None):
    for group in opt.param_groups:group['lr']=group['base_lr']*lr_factor(u,60,600)
    opt.zero_grad(set_to_none=True);loss_sum=0.;a=force or arm
    for j in data.plan[u]:
        i=data.fit[int(j)];s=m(data.tensor(i,a));y=s.new_tensor([float(data.rows[i]['true_label']=='A5C')]);loss=torch.nn.functional.binary_cross_entropy_with_logits(s,y)/8;loss.backward();loss_sum+=float(loss.detach())
    norms={k:float(sum((p.grad.double().square().sum() for p in ps if p.grad is not None),torch.tensor(0.,device='cuda')).sqrt()) for k,ps in groups(m).items()}
    clip=float(torch.nn.utils.clip_grad_norm_(m.parameters(),1.));assert np.isfinite(clip)
    opt.step()
    return {'u':u,'loss':loss_sum,'groups':norms,'clip':clip,'physical_forwards':8,'physical_backwards':8,'pid':os.getpid(),'device':torch.cuda.get_device_name(0),'plan_row_sha256':hashlib.sha256(data.plan[u].tobytes()).hexdigest()}


def state(m):return {n:x.detach().cpu().clone() for n,x in m.state_dict().items()}


def checkpoint(m,opt,u,identity):return {'model':state(m),'optimizer':opt.state_dict(),'update':u,'identity':identity,'CPU_rng':torch.get_rng_state(),'CUDA_rng':torch.cuda.get_rng_state()}


def save_torch(path,obj):
    tmp=path.with_suffix('.tmp')
    with tmp.open('wb') as f:torch.save(obj,f);f.flush();os.fsync(f.fileno())
    tmp.replace(path);dt.sync_directory(path.parent)


def check():
    manifest=verify();data=Inputs();m=build();opt=opt_for(m);initial=state(m);live={n for n,p in m.named_parameters() if p.requires_grad}
    assert len(live)==len(list(m.parameters())) and len({id(p) for g in opt.param_groups for p in g['params']})==len(live)
    ids=[data.fit[int(i)] for i in data.plan[0][:4]];y=np.array([data.rows[i]['true_label']=='A5C' for i in ids],float)
    scores=torch.cat([m(data.tensor(i,'H')) for i in ids]);assert scores.detach().cpu().tolist()==[-2.]*4
    loss=torch.nn.functional.binary_cross_entropy_with_logits(scores,torch.tensor(y,dtype=torch.float32,device='cuda'));loss.backward()
    formula=abs(float(loss.detach())-float(np.mean(np.logaddexp(0.,-2.)+2*y)));biaserr=abs(float(m.head.bias.grad)-float((1/(1+np.exp(2))-y).mean()))
    grads={n:p.grad.detach().clone() for n,p in m.named_parameters()};opt.zero_grad(set_to_none=True)
    for i in ids:(torch.nn.functional.binary_cross_entropy_with_logits(m(data.tensor(i,'H')),torch.tensor([float(data.rows[i]['true_label']=='A5C')],device='cuda'))/4).backward()
    graderr=max(float((p.grad-grads[n]).abs().max()) for n,p in m.named_parameters());assert max(formula,biaserr,graderr)<1e-5
    def execute(arm,force=None,st=None,stop=6):
        a=build();o=opt_for(a);start=0
        if st is not None:a.load_state_dict(st['model']);o.load_state_dict(st['optimizer']);start=st['update'];torch.set_rng_state(st['CPU_rng']);torch.cuda.set_rng_state(st['CUDA_rng'])
        logs=[step(a,o,data,u,arm,force) for u in range(start,stop)];return a,o,logs
    del m,opt,grads;torch.cuda.empty_cache()
    a,o,logs=execute('H');expected=state(a);assert all(logs[0]['groups'][k]==0 for k in logs[0]['groups'] if k!='head');assert all(any(r['groups'][k]>0 for r in logs[1:]) for k in groups(a))
    assert all(torch.equal(v,initial[n]) for n,v in expected.items() if n not in live);del a,o
    for arm,force in [('H',None),('R','H')]:
        a,o,_=execute(arm,force);assert state_sha(state(a))==state_sha(expected);del a,o
    a,o,_=execute('H',stop=3);st=copy.deepcopy(checkpoint(a,o,3,{}));del a,o;a,o,_=execute('H',st=st);assert state_sha(state(a))==state_sha(expected);del a,o,st
    a,o,_=execute('R');torch.cuda.synchronize();start=time.time()
    for u in range(6,12):step(a,o,data,u,'R')
    torch.cuda.synchronize();sec=(time.time()-start)/6
    result={'passed':True,'input_manifest_sha256':sha(PUB/'input_manifest.json'),'independent_BCE_error':formula,'independent_bias_gradient_error':biaserr,'full_vs_micro_gradient_max_abs':graderr,
      'all_trainable_parameters':len(live),'repeat_exact':True,'forced_H_R_exact':True,'resume3plus3_exact':True,'frozen_BN_unchanged':True,'first_backbone_grad_zero_expected':True,'later_all_seven_groups_nonzero':True,
      'seconds_per_update':sec,'source_forecast_seconds':sec*1200*1.3+180,'conditional_target_forecast_seconds':sec/8*60612*2*1.4+360,'peak_GPU_bytes':torch.cuda.max_memory_allocated(),'diagnostic_model_forwards':0}
    write(PUB/'checks.json',result);print(json.dumps(result))


@torch.no_grad()
def score_fit(m,data,arm):return np.array([float(m(data.tensor(i,arm))) for i in data.fit],np.float32)


def train():
    man=verify();check=json.loads((PUB/'checks.json').read_text());assert check['passed'] and check['input_manifest_sha256']==sha(PUB/'input_manifest.json')
    fr=json.loads((OUT/'prelaunch_freeze.json').read_text());assert fr['input_manifest_sha256']==sha(PUB/'input_manifest.json') and fr['actual_git_objects_verified']
    audit=json.loads((PUB/'independent_binding.json').read_text());assert audit['passed'] and audit['bound_sha256']==man['bound_sha256']
    data=Inputs();entries={}
    for arm in ARMS:
        rd=OUT/'source_train'/arm;rd.mkdir(parents=True,exist_ok=True);dt.sync_directory(rd.parent)
        ident={'run_id':'A52','arm':arm,'input_manifest_sha256':sha(PUB/'input_manifest.json'),'plan_sha256':man['plan_sha256']}
        if (rd/'manifest.json').exists():
            e=json.loads((rd/'manifest.json').read_text());assert e['identity']==ident and e['updates']==600 and sha(rd/'final.pt')==e['model_sha256'];entries[arm]=e;continue
        m=build();opt=opt_for(m);initial=state(m);rp=rd/'resume.pt';claim=rd/'claim.json';jp=rd/'updates.private.jsonl';begin=0;logs=[]
        if claim.exists():
            assert json.loads(claim.read_text())==ident
            if not rp.exists():rp=rd/'optimizer_final.pt'
            if not rp.exists():raise RuntimeError('claimed model without safe full state')
            st=torch.load(rp,map_location='cpu',weights_only=False);assert st['identity']==ident;m.load_state_dict(st['model']);opt.load_state_dict(st['optimizer']);begin=st['update'];torch.set_rng_state(st['CPU_rng']);torch.cuda.set_rng_state(st['CUDA_rng']);logs=[json.loads(x) for x in jp.read_text().splitlines()];dt.validate_recovery(rd,begin,len(logs))
        else:
            with claim.open('x') as f:json.dump(ident,f);f.flush();os.fsync(f.fileno())
            dt.sync_directory(rd);np.save(rd/'fit_init.npy',score_fit(m,data,arm))
        with jp.open('a') as jf:
            def persist(rec):jf.write(json.dumps(rec)+'\n');jf.flush();os.fsync(jf.fileno());logs.append(rec)
            for u in range(begin,600):
                rec=dt.claimed_update(rd,u,lambda:step(m,opt,data,u,arm),persist)
                if (u+1)%25==0:
                    write(OUT/'progress.json',{'run_id':'A52','arm':arm,'update':u+1,'pid':os.getpid(),'device':rec['device'],'updated_epoch':time.time()});print(json.dumps({'arm':arm,'update':u+1,'loss':rec['loss']}),flush=True)
                if (u+1)%150==0:save_torch(rd/'resume.pt',checkpoint(m,opt,u+1,ident))
        assert len(logs)==600 and [r['u'] for r in logs]==list(range(600));assert {int(v['step']) for v in opt.state.values()}=={600}
        final=state(m);live={n for n,p in m.named_parameters() if p.requires_grad};assert all(torch.equal(initial[n],v) for n,v in final.items() if n not in live)
        assert all(any(r['groups'][k]>0 for r in logs[1:]) for k in groups(m));np.save(rd/'fit_final.npy',score_fit(m,data,arm))
        save_torch(rd/'final.pt',{'model':final,'identity':ident,'base_sha256':man['base_sha256'],'full_state_sha256':state_sha(final),'updates':600})
        (rd/'resume.pt').replace(rd/'optimizer_final.pt');dt.sync_directory(rd)
        e={'identity':ident,'updates':600,'model_sha256':sha(rd/'final.pt'),'optimizer_final_sha256':sha(rd/'optimizer_final.pt'),'full_state_sha256':state_sha(final),
          'initial_state_sha256':state_sha(initial),'live_tensors':len(live),'live_drift':{n:float((v.double()-initial[n].double()).norm()) for n,v in final.items() if n in live},
          'frozen_BN_unchanged':True,'optimizer_step_values':[600],'optimizer_parameters':len(opt.state),'journal_sha256':sha(jp),'fit_init_sha256':sha(rd/'fit_init.npy'),'fit_final_sha256':sha(rd/'fit_final.npy'),
          'physical_training_forwards':4800,'physical_training_backwards':4800}
        assert all(v>0 for v in e['live_drift'].values());write(rd/'manifest.json',e);entries[arm]=e;del m,opt,initial,final;torch.cuda.empty_cache()
    write(PUB/'models_manifest.json',{'run_id':'A52','models':entries,'input_manifest_sha256':sha(PUB/'input_manifest.json'),'updates_total':1200,'diagnostic_model_forwards':0})


def readout():
    man=verify();models=json.loads((PUB/'models_manifest.json').read_text());fr=json.loads((OUT/'models_freeze.json').read_text());assert fr['manifest_sha256']==sha(PUB/'models_manifest.json') and fr['actual_git_objects_verified']
    audit=json.loads((PUB/'independent_models.json').read_text());assert audit['passed'] and audit['models_manifest_sha256']==sha(PUB/'models_manifest.json')
    data=Inputs();scores={}
    for arm in ARMS:
        p=OUT/'source_train'/arm/'final.pt';assert sha(p)==models['models'][arm]['model_sha256'];ck=torch.load(p,map_location='cpu',weights_only=True);m=build();m.load_state_dict(ck['model']);assert state_sha(state(m))==models['models'][arm]['full_state_sha256']
        identity={'arm':arm,'model_sha256':sha(p),'bound_sha256':man['bound_sha256'],'rows':133};cell=dt.open_cell(OUT/'cells',arm,identity,133)
        with torch.no_grad():
            while (j:=dt.claim_row(cell)) is not None:dt.finish_row(cell,j,float(m(data.tensor(data.diag[j],arm))))
        dt.seal_cell(cell);scores[arm]=np.asarray(cell['scores']).copy();dt.close_cell(cell);del m;torch.cuda.empty_cache()
    ctrl=data.bound['geometry'];rows=[data.rows[i] for i in data.diag];x=np.array([r['geometry'] for r in rows]);scores['geometry']=((x-ctrl['mean'])/ctrl['scale'])@np.asarray(ctrl['coef'])+ctrl['intercept'];scores['constant']=np.full(133,np.log(ctrl['prior']/(1-ctrl['prior'])))
    dest=OUT/'scores.private.npz'
    with dest.open('xb') as f:np.savez_compressed(f,**scores);f.flush();os.fsync(f.fileno())
    y=np.array([r['true_label']=='A5C' for r in rows]);disc=np.array([r['true_label']!=r['directory_label'] for r in rows]);metrics={a:metric(y,s,disc) for a,s in scores.items()}
    result={'run_id':'A52','metrics':metrics,'source_gates':pd.source_gates(metrics,metrics['geometry']['balanced_accuracy'],metrics['constant']['balanced_accuracy']),
      'score_cache_sha256':sha(dest),'source_model_forwards':266,'models_freeze':fr,'target_models_or_readouts':0,'source_diagnostic_reused_from_A51':True}
    write(PUB/'readout.json',result);print(json.dumps(result))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','check','train','readout']);args=ap.parse_args();torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    OUT.mkdir(parents=True,exist_ok=True);lock=dt.acquire_lock(OUT/'exclusive_worker.lock');start=time.time();ok=False
    try:globals()[args.stage]();ok=True
    finally:
        lock.close();r=resource.getrusage(resource.RUSAGE_SELF);write(OUT/f'{args.stage}_worker_process.json',{'pid':os.getpid(),'stage':args.stage,'ok':ok,'start_epoch':start,'end_epoch':time.time(),'CPU_seconds':r.ru_utime+r.ru_stime,'GPU_job_wall_seconds':time.time()-start if args.stage!='prepare' else 0})
if __name__=='__main__':main()
