"""A47 sole-owner stages. Private frames, references, features and cases stay server-side.

PanEcho: CarDS-Yale/PanEcho836c46e. Released weights CC BY-NC-SA4.0;
upstream code AGPLv3. Noncommercial research, no task-output pseudo-labels.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
import argparse
import ast
import copy
import hashlib
import inspect
import json
import math
import sys
import time
import urllib.request
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
import torchvision
from torch import nn
from echo_routing.temporal import clinical_pretraining as cp
import scripts.train_a4c_support as a45
import scripts.train_risk_constrained_a5c as t

CFG='configs/experiments/video_clinical_pretrained_a5c_v1.yaml'
CARD='docs/superpowers/plans/2026-10-28-clinical-pretrained-a5c.md'
OUT=ROOT/'runs/video_clinical_pretrained_a5c_v1'
RES=ROOT/'docs/results/2026-10-28_clinical_pretrained_a5c'
REV='836c46ef6bc4eace50b7dc36704ecef2d33affde'
PURL='https://github.com/CarDS-Yale/PanEcho/releases/download/v1.0/panecho.pt'
CODE=['scripts/train_clinical_pretrained_a5c.py','echo_routing/temporal/clinical_pretraining.py']
ARMS=['P','G'];U=600
sha=a45.sha;digest=a45.digest;write=a45.write


def publish(name, value):
    write(OUT/name,value);write(RES/name,value)


def coord(phase, **kw):
    p=OUT/'coordination.json';old=json.loads(p.read_text()) if p.exists() else {}
    write(p,old|{'round':'A47','owner':'Codex','phase':phase,'at':t.now(),'pid':os.getpid(),**kw})


def configure():
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True


def pins(ctx, source):
    base=ROOT/f'models_frozen/video_within_view_v1/all_s{source}/final.pt'
    expected=json.loads((ROOT/'runs/video_local_motion_a5c_v1/weights.json').read_text())['base_sha256'] if source==0 else None
    if expected: assert sha(base)==expected
    assert sha(ctx.a31.out/'split.json')==t.PINS['split']
    for f in [0,1]:
        assert sha(ctx.a31.out/f'outputs_s{source}_f{f}.npz')==t.PINS['outputs'][source][str(f)]
        assert sha(ctx.a31.out/'data'/f'runs_s{source}_f{f}.npz')==t.PINS['runs'][source][str(f)]
    return sha(base)


def folder_bytes(path):
    return sum(p.stat().st_size for p in path.rglob('*') if p.is_file()) if path.exists() else 0


def new_bytes():
    return folder_bytes(OUT)+folder_bytes(ROOT/'runs/a47_public_evidence_intake_v1')+50_000_000


def conditional(source):
    assert source in [0,2]
    if source==2:
        r=json.loads((OUT/'readout_s0.json').read_text())
        assert any(g['passed'] for g in r['gates'].values()), 'source2 absolute package gate did not open'
        assert json.loads((OUT/'independent_audit_s0.json').read_text())['ok']
        assert json.loads((OUT/'source2_budget_authorized.json').read_text())['within_shared_cap']


def prepare(ctx, source):
    conditional(source);base=pins(ctx,source);result={};sessions=[]
    for f in [0,1]:
        F=t.Fold(ctx,source,f);refs=t.t37.wt.window_refs(F.seqs,F.W['seq'][F.need],F.W['s'][F.need])
        target=cp.marker_refs(refs,F.W['marker'][F.need]);unique=[];at={};idx=[]
        for ref in target:
            if ref not in at: at[ref]=len(unique);unique.append(ref)
            idx.append(at[ref])
        sess={t.session_of(r[0]) for q in F.seqs for r in q['refs']};sessions.append(sess)
        if source==0:
            assert len(F.need)==[5030,5437][f] and {k:len(v) for k,v in F.L.items()}==t.EXPECTED_LISTS[0][f]
            assert F.psha==t.EXPECTED_PLAN[0][f]['sha256']
        ex=t.rc.exposure(F.pk,F.pl);assert ex['distinct_negatives']==len(F.L['negatives'])
        near=t.t27.nearest_rows(F.W,F.seqs)
        for i in F.L['eligible']:
            lo,hi=F.pk['offsets'][i:i+2];q=int(F.pk['seq'][i]);aa,bb=int(F.pk['a'][i]),int(F.pk['b'][i]);rr,ct=np.unique(near[q][aa:bb],return_counts=True)
            assert np.array_equal(rr,F.pk['rows'][lo:hi]-F.off) and np.max(abs(ct/(bb-aa)-F.pk['weights'][lo:hi]))<1e-12
        key=f's{source}_f{f}';rp=OUT/f'refs_{key}.private.json';write(rp,unique)
        ip=OUT/f'input_{key}.npz';np.savez(ip,rows=F.need,index=np.asarray(idx,np.int32),plan=F.pl,markers=F.W['marker'][F.need])
        result[str(f)]={'eligible_runs':len(F.L['eligible']),'rows':len(F.need),'unique_frames':len(unique),'grid':len(F.W['seq']),'fit_positive':len(F.L['positives']),'fit_negative':len(F.L['negatives']),'other':len(F.L['other']),'map_bytes_per_arm':len(unique)*768*7*7*2,'references_sha256':sha(rp),'inputs_sha256':sha(ip),'plan_sha256':F.psha,'exposure':ex,'membership_verified':True,'sessions':len(sess),'original_output_sha256':t.PINS['outputs'][source][str(f)],'runs_sha256':t.PINS['runs'][source][str(f)]}
    assert not sessions[0]&sessions[1]
    projected=new_bytes()+sum(v['map_bytes_per_arm'] for v in result.values())*2+250_000_000
    assert projected<3e9, f'feature-cache forecast exceeds shared disk cap: {projected}'
    result|={'source':source,'base_sha256':base,'session_isolation':True,'projected_final_shared_bytes':projected,'config_sha256':sha(ROOT/CFG),'contract_sha256':sha(ROOT/CARD),'written':t.now()}
    publish(f'preparation_s{source}.json',result);coord('prepared',source=source);return result


def image_model(arm, device='cpu'):
    m=torchvision.models.convnext_tiny(weights=None);m.classifier[-1]=nn.Identity()
    m.load_state_dict(torch.load(OUT/f'sources/{arm}.pt',map_location='cpu',weights_only=True),strict=True)
    return m.eval().requires_grad_(False).to(device)


def acquire():
    src=OUT/'sources';src.mkdir(exist_ok=True);files={}
    for name in ['src/models.py','hubconf.py','README.md','LICENSES.md','content/tasks.md']:
        p=src/('panecho_'+name.replace('/','_'))
        if not p.exists():p.write_bytes(urllib.request.urlopen(f'https://raw.githubusercontent.com/CarDS-Yale/PanEcho/{REV}/{name}',timeout=60).read())
        files[name]=sha(p)
    w=torchvision.models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
    downloads={};image_hashes={};head_names=[]
    expected=torchvision.models.convnext_tiny(weights=None);expected.classifier[-1]=nn.Identity()
    for arm,url in [('P',PURL),('G',w.url)]:
        p=src/('panecho.pt' if arm=='P' else Path(w.url).name)
        if not p.exists():torch.hub.download_url_to_file(url,str(p),progress=False)
        if arm=='P':assert p.stat().st_size==505054604
        obj=torch.load(p,map_location='cpu',weights_only=True)
        if arm=='P':
            head_names=cp.clinical_head_names(obj['weights'])
        state=cp.extract_image_state(obj,expected.state_dict(),arm)
        expected.load_state_dict(state,strict=True);torch.save(state,src/f'{arm}.pt')
        downloads[arm]={'url':url,'bytes':p.stat().st_size,'sha256':sha(p),'safe_weights_only_load':True}
        image_hashes[arm]={'sha256':sha(src/f'{arm}.pt'),'bytes':(src/f'{arm}.pt').stat().st_size,'keys':len(state),'parameters':sum(v.numel() for v in state.values())}
        del obj,state
    ps=torch.load(src/'P.pt',weights_only=True);gs=torch.load(src/'G.pt',weights_only=True)
    changed=[k for k in ps if not torch.equal(ps[k],gs[k])];assert len(changed)>100
    man={'panecho_code_revision':REV,'panecho_sources_sha256':files,'downloads':downloads,'image_states':image_hashes,'clinical_heads_unused':head_names,'temporal_transformer_unused':True,'changed_image_tensors_P_vs_G':len(changed),'weights_license':'CC-BY-NC-SA-4.0','upstream_other_materials_license':'AGPLv3','torch':torch.__version__,'torchvision':torchvision.__version__,'torchvision_convnext_source_sha256':sha(inspect.getfile(type(expected))),'code_sha256':{p:sha(ROOT/p) for p in CODE},'config_sha256':sha(ROOT/CFG),'contract_sha256':sha(ROOT/CARD),'written':t.now()}
    publish('sources.json',man);coord('acquired_pending_actual_git_source_freeze');return man


def source_freeze(commit,msha):
    assert commit and sha(OUT/'sources.json')==sha(RES/'sources.json')==msha
    m=json.loads((OUT/'sources.json').read_text())
    for arm in ARMS:assert sha(OUT/f'sources/{arm}.pt')==m['image_states'][arm]['sha256']
    for p,h in m['code_sha256'].items():assert sha(ROOT/p)==h
    publish('source_frozen.json',{'commit':commit,'manifest_sha256':msha,'at':t.now(),'root_verified_actual_git_objects':True})
    # Only re-downloadable acquisition duplicates; immutable extracted states remain.
    for name in ['panecho.pt',Path(m['downloads']['G']['url']).name]:(OUT/'sources'/name).unlink(missing_ok=True)
    return {'source_frozen':commit}


def normalize(fr):
    x=torch.from_numpy(fr).permute(0,3,1,2).to('cuda').float()/255
    return (x-x.new_tensor([.485,.456,.406])[None,:,None,None])/x.new_tensor([.229,.224,.225])[None,:,None,None]


def features(ctx,source):
    conditional(source);pins(ctx,source);sf=json.loads((OUT/'source_frozen.json').read_text());assert sha(OUT/'sources.json')==sf['manifest_sha256']
    prep=json.loads((OUT/f'preparation_s{source}.json').read_text());result={}
    for arm in ARMS:
        m=image_model(arm,'cuda')
        for f in [0,1]:
            key=f'{arm}_s{source}_f{f}';path=OUT/f'maps_{key}.npy';done=OUT/f'maps_{key}.json'
            if done.exists():
                rec=json.loads(done.read_text());assert sha(path)==rec['sha256'];result[key]=rec;continue
            assert not path.exists(),'incomplete feature cache: explicit engineering recovery required'
            F=t.Fold(ctx,source,f);refs=json.loads((OUT/f'refs_s{source}_f{f}.private.json').read_text());assert sha(OUT/f'refs_s{source}_f{f}.private.json')==prep[str(f)]['references_sha256']
            bank=np.lib.format.open_memmap(path,mode='w+',dtype=np.float16,shape=(len(refs),768,7,7));start=time.time()
            for j in range(0,len(refs),32):
                fr=t.t37.rr.decode_frames([tuple(v) for v in refs[j:j+32]],F.fp,0);x=normalize(fr)
                with torch.no_grad():z=m.features(x)
                assert z.shape[1:]==(768,7,7) and torch.isfinite(z).all();bank[j:j+len(x)]=z.cpu().numpy().astype(np.float16)
                if j%640==0 or j+32>=len(refs):
                    bank.flush();coord('frozen_feature_extraction',source=source,arm=arm,fold=f,frames=min(j+32,len(refs)),total=len(refs),device=torch.cuda.get_device_name(0));print(json.dumps({'feature':key,'frames':min(j+32,len(refs)),'total':len(refs),'seconds':time.time()-start}),flush=True)
            bank.flush();rec={'sha256':sha(path),'shape':list(bank.shape),'dtype':'float16','finite':bool(np.isfinite(bank).all()),'seconds':time.time()-start,'source_state_sha256':sha(OUT/f'sources/{arm}.pt')};assert rec['finite'];write(done,rec);result[key]=rec
        del m;torch.cuda.empty_cache()
    assert new_bytes()<3e9
    publish(f'features_s{source}.json',{'banks':result,'source_frozen':sf,'written':t.now()});return {'feature_banks':len(result),'shared_new_bytes':new_bytes()}


def loadF(ctx,source,f,arm,device='cuda'):
    F=t.Fold(ctx,source,f);inp=np.load(OUT/f'input_s{source}_f{f}.npz');assert np.array_equal(inp['rows'],F.need) and np.array_equal(inp['plan'],F.pl)
    F.at=np.full(len(F.W['seq']),-1,np.int64);F.at[F.need]=inp['index']
    F.bank=torch.from_numpy(np.load(OUT/f'maps_{arm}_s{source}_f{f}.npy')).to(device)
    F.embd=torch.from_numpy(np.asarray(t.t31.outputs(ctx.a31,source,f)['emb'],np.float32)).to(device)
    return F


def new_model(source,f,device='cuda'):
    torch.manual_seed(2026102847+10*source+f);m=cp.ClinicalRepair().eval().to(device)
    return m,torch.optim.AdamW(m.parameters(),lr=.001,betas=(.9,.999),eps=1e-8,weight_decay=.0001)


def batch(F,rows):
    idx=F.at[rows];assert (idx>=0).all()
    return F.bank[torch.as_tensor(idx,device=F.bank.device)].float(),F.embd[torch.as_tensor(rows,device=F.embd.device)]


def step(m,opt,F,u):
    start=time.time();rows,v,slots=F.occ(u);x,e=batch(F,rows);weights=torch.as_tensor(v,dtype=torch.float32,device=x.device);sg=torch.as_tensor(slots,device=x.device)
    lr=.001*t.t31.lr_factor(u,60,600)
    for group in opt.param_groups:group['lr']=lr
    opt.zero_grad(set_to_none=True);scores=m(x,e);loss,r=cp.loss_from_scores(scores,weights,sg);loss.backward()
    grads={name:float(sum(p.grad.detach().double().square().sum() for p in mod.parameters())) for name,mod in [('tokens',m.tokens),('spatial',m.spatial),('appearance',m.appearance),('fusion',m.fusion)]}
    gn=float(torch.nn.utils.clip_grad_norm_(m.parameters(),1));opt.step();torch.cuda.synchronize()
    return {'u':u,'loss':float(loss.detach()),'run_scores':r.detach().cpu().tolist(),'lr':lr,'grad_norm':gn,'grad_sq':grads,'rows_sha256':digest(rows),'n_occ':len(rows),'pid':os.getpid(),'device':torch.cuda.get_device_name(0),'seconds':time.time()-start}


def state(m,o,logs):return {'model':copy.deepcopy(m.state_dict()),'opt':copy.deepcopy(o.state_dict()),'log':copy.deepcopy(logs)}
def delta(a,b):return max(float((a[k]-b[k]).abs().max()) for k in a)


def checks(ctx,source):
    conditional(source);pins(ctx,source);F=loadF(ctx,source,0,'P');m,o=new_model(source,0);initial=copy.deepcopy(m.state_dict());rows,_,_=F.occ(0);x,e=batch(F,rows)
    assert torch.equal(m(x,e),torch.full((len(rows),),-2.,device='cuda'))
    logs=[step(m,o,F,u) for u in range(6)];goal=copy.deepcopy(m.state_dict());assert all(logs[0]['grad_sq'][k]==0 for k in ['tokens','spatial','appearance']) and all(logs[-1]['grad_sq'][k]>0 for k in logs[-1]['grad_sq']);assert all(not torch.equal(v,initial[k]) for k,v in goal.items())
    mm,oo=new_model(source,0);[step(mm,oo,F,u) for u in range(6)];repeat=delta(goal,mm.state_dict());assert repeat==0
    # Force the G path to the same P bank: named arm cannot change any other condition.
    forced=loadF(ctx,source,0,'G');forced.bank=F.bank;mm,oo=new_model(source,0);[step(mm,oo,forced,u) for u in range(6)];equal=delta(goal,mm.state_dict());assert equal==0;del forced
    mm,oo=new_model(source,0);ll=[step(mm,oo,F,u) for u in range(3)];ss=state(mm,oo,ll);mm,oo=new_model(source,0);mm.load_state_dict(ss['model']);oo.load_state_dict(ss['opt']);[step(mm,oo,F,u) for u in range(3,6)];resume=delta(goal,mm.state_dict());assert resume==0
    rows,v,slots=F.occ(0);x,e=batch(F,rows);vv=torch.as_tensor(v,dtype=torch.float32,device='cuda');sg=torch.as_tensor(slots,device='cuda');m.zero_grad(set_to_none=True);scores=m(x,e);loss,_=cp.loss_from_scores(scores,vv,sg);loss.backward();ga=torch.cat([p.grad.flatten().double() for p in m.parameters()]);z=scores.detach().requires_grad_();ll,_=cp.loss_from_scores(z,vv,sg);grad=torch.autograd.grad(ll,z)[0];m.zero_grad(set_to_none=True)
    indiv=[]
    for j in range(len(rows)):
        sj=m(x[j:j+1],e[j:j+1]);sj.backward(grad[j:j+1]);indiv.append(sj.detach())
    gb=torch.cat([p.grad.flatten().double() for p in m.parameters()]);relative=float(torch.linalg.vector_norm(ga-gb)/torch.linalg.vector_norm(ga));scoreerr=float((scores.detach()-torch.cat(indiv)).abs().max());assert relative<1e-4 and scoreerr<1e-4
    # Independent formula and structural-zero synthetic occurrence, still re-forwarded.
    vd=vv.double();sd=scores.detach().double();direct=torch.stack([sum(sd[j]*vd[j] for j in range(len(sd)) if int(sg[j])==k) for k in range(4)]);reference=torch.logaddexp(direct.new_zeros(()),-direct[0])+torch.logaddexp(direct.new_zeros(3),direct[1:]).mean();assert abs(float(reference)-float(loss.detach()))<1e-5
    ue=int(np.argmax([len(F.occ(u)[0]) for u in range(U)]));torch.cuda.reset_peak_memory_stats();timing=step(m,o,F,ue);memory=torch.cuda.max_memory_allocated()/1e9;assert memory<40
    prep=json.loads((OUT/f'preparation_s{source}.json').read_text());estimate=sum(prep[str(f)]['exposure']['occurrences_mean_update'] for f in [0,1])*1200*timing['seconds']/timing['n_occ'];assert estimate<5*3600
    # Execute the exact pinned upstream ImageEncoder class, only disabling its constructor download.
    tree=ast.parse((OUT/'sources/panecho_src_models.py').read_text());node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='ImageEncoder')
    namespace={'torch':torch,'torchvision':torchvision};exec(compile(ast.Module(body=[node],type_ignores=[]),'pinned_panecho_ImageEncoder','exec'),namespace)
    original_factory=torchvision.models.convnext_tiny
    def no_download(*args,**kwargs):kwargs['weights']=None;return original_factory(*args,**kwargs)
    torchvision.models.convnext_tiny=no_download
    try:upstream=namespace['ImageEncoder']('convnext_tiny').eval().cuda()
    finally:torchvision.models.convnext_tiny=original_factory
    refs=json.loads((OUT/f'refs_s{source}_f0.private.json').read_text());select=[0,len(refs)//2,len(refs)-1];fr=t.t37.rr.decode_frames([tuple(refs[j]) for j in select],F.fp,0);xx=normalize(fr);parity={}
    assert np.array_equal(fr,t.t37.rr.decode_frames([tuple(refs[j]) for j in select],F.fp,0))
    for arm in ARMS:
        im=image_model(arm,'cuda');upstream.model.load_state_dict(im.state_dict(),strict=True);captured={};hook=upstream.model.features.register_forward_hook(lambda mod,ins,out:captured.update(map=out.detach()))
        with torch.no_grad():glob=upstream(xx);maps=im.features(xx);composed=im.classifier(im.avgpool(maps))
        hook.remove();err=float((glob-composed).abs().max());maperr=float((maps-captured['map']).abs().max());cached=np.load(OUT/f'maps_{arm}_s{source}_f0.npy',mmap_mode='r')[select].astype(np.float32);cacheerr=float(abs(cached-maps.cpu().numpy().astype(np.float16).astype(np.float32)).max());assert err<1e-5 and maperr<1e-5 and cacheerr<.02
        parity[arm]={'official_image_embedding_max_abs':err,'official_features_max_abs':maperr,'cache_fresh_fp16_max_abs':cacheerr};del im
    r={'ok':True,'source':source,'parameters':sum(p.numel() for p in m.parameters()),'repeat6_max_abs':repeat,'forced_equal_P_G_max_abs':equal,'resume3plus3_max_abs':resume,'production_gradient_relative':relative,'individual_window_score_max_abs':scoreerr,'formula_error':abs(float(reference)-float(loss.detach())),'all_components_live_by6':True,'first_step_upstream_zero':True,'all_state_tensors_moved':True,'maximum_batch_occurrences':timing['n_occ'],'peak_gpu_gb':memory,'projected_training_gpu_hours':estimate/3600,'source_parity':parity,'written':t.now()};publish(f'checks_s{source}.json',r);return r


def train(ctx,source):
    conditional(source);base=pins(ctx,source);assert json.loads((OUT/f'checks_s{source}.json').read_text())['ok'];prep=json.loads((OUT/f'preparation_s{source}.json').read_text());features_manifest=json.loads((OUT/f'features_s{source}.json').read_text());models={}
    for f in [0,1]:
        for arm in ARMS:
            F=loadF(ctx,source,f,arm);key=f'{arm}_s{source}_f{f}';rd=OUT/'train'/key;rd.mkdir(parents=True,exist_ok=True)
            identity=hashlib.sha256(''.join([sha(ROOT/CFG),sha(ROOT/CARD),*[sha(ROOT/p) for p in CODE],prep[str(f)]['inputs_sha256'],features_manifest['banks'][key]['sha256'],key]).encode()).hexdigest()
            if (rd/'manifest.json').exists():
                man=json.loads((rd/'manifest.json').read_text());assert man['identity']==identity and sha(rd/'final.pt')==man['final_sha256'];models[key]=man;continue
            assert not (rd/'final.pt').exists(),'unmanifested final: audit, never retrain'
            m,o=new_model(source,f);initial=copy.deepcopy(m.state_dict());logs=[]
            if (rd/'resume.pt').exists():
                ss=torch.load(rd/'resume.pt',map_location='cuda',weights_only=False);assert ss['identity']==identity;m.load_state_dict(ss['model']);o.load_state_dict(ss['opt']);logs=ss['log']
            coord('formal_training',source=source,model=key,update=len(logs),device=torch.cuda.get_device_name(0))
            for u in range(len(logs),U):
                rec=step(m,o,F,u);assert math.isfinite(rec['loss']) and math.isfinite(rec['grad_norm']);logs.append(rec)
                if (u+1)%50==0:
                    pr={'model':key,'update':u+1,'total':600,'pid':os.getpid(),'device':rec['device'],'loss':rec['loss'],'window_occurrences':sum(v['n_occ'] for v in logs),'at':t.now()};write(OUT/'progress.json',pr);print(json.dumps(pr),flush=True)
                if (u+1)%150==0 and u+1<600:
                    p=rd/'resume.tmp';torch.save(state(m,o,logs)|{'identity':identity},p);p.replace(rd/'resume.pt')
            steps=[int(v['step']) for v in o.state.values()];assert len(logs)==600 and set(steps)=={600} and len(steps)==len(list(m.parameters()))
            moved={k:not torch.equal(v,initial[k]) for k,v in m.state_dict().items()};assert all(moved.values()) and all(torch.isfinite(v).all() for v in m.state_dict().values())
            torch.save({k:v.detach().cpu() for k,v in m.state_dict().items()},rd/'final.pt');(rd/'updates.jsonl').write_text(''.join(json.dumps(v)+'\n' for v in logs));torch.save({'steps':steps,'parameter_names':list(dict(m.named_parameters())),'final_update':600},rd/'optimizer_steps.pt')
            man={'arm':arm,'source':source,'fit_fold':f,'identity':identity,'updates':600,'optimizer_steps':steps,'parameters':197521,'all_state_tensors_moved':moved,'final_sha256':sha(rd/'final.pt'),'updates_sha256':sha(rd/'updates.jsonl'),'optimizer_receipt_sha256':sha(rd/'optimizer_steps.pt'),'plan_sha256':F.psha,'inputs_sha256':prep[str(f)]['inputs_sha256'],'features_sha256':features_manifest['banks'][key]['sha256'],'initial_state_sha256':digest(torch.cat([v.flatten().cpu() for v in initial.values()]).numpy()),'occurrences':sum(v['n_occ'] for v in logs),'pid':os.getpid(),'device':torch.cuda.get_device_name(0),'finished':t.now()};write(rd/'manifest.json',man);models[key]=man;(rd/'resume.pt').unlink(missing_ok=True);del F;torch.cuda.empty_cache()
    pins(ctx,source)
    man={'models':models,'source':source,'base_sha256':base,'source_manifest_sha256':sha(OUT/'sources.json'),'source_freeze':json.loads((OUT/'source_frozen.json').read_text()),'config_sha256':sha(ROOT/CFG),'contract_sha256':sha(ROOT/CARD),'code_sha256':{p:sha(ROOT/p) for p in CODE},'loaded_repository_code_sha256':t.t38.loaded_code(),'parent_config_sha256':{p:sha(ROOT/p) for p in [ctx.config,t.A31_CONFIG,t.A32_CONFIG]},'preparation_sha256':sha(OUT/f'preparation_s{source}.json'),'features_manifest_sha256':sha(OUT/f'features_s{source}.json'),'checks_sha256':sha(OUT/f'checks_s{source}.json'),'written':t.now()};publish(f'weights_s{source}.json',man);coord('trained_pending_actual_git_freeze',source=source);return {'models':4,'actual_updates':2400}


def freeze(source,commit,msha):
    assert commit and sha(OUT/f'weights_s{source}.json')==sha(RES/f'weights_s{source}.json')==msha
    man=json.loads((OUT/f'weights_s{source}.json').read_text())
    for key,v in man['models'].items():assert sha(OUT/'train'/key/'final.pt')==v['final_sha256']
    for p,h in man['code_sha256'].items():assert sha(ROOT/p)==h
    publish(f'frozen_s{source}.json',{'commit':commit,'manifest_sha256':msha,'at':t.now(),'root_verified_actual_git_objects':True});coord('frozen_pending_once_readout',source=source);return {'frozen':4}


def package(cells,name,sources):
    fs=[cells[f'heldout_f{h}'][name] for h in [0,1]];bs=[cells[f'heldout_f{h}']['Base'] for h in [0,1]]
    denom=sum(v['S1_fixed']['runs'] for v in fs);protected=sum(v['correct_A4C_kept']['runs'] for v in fs)
    fix=sum(v['S1_fixed']['fixed'] for v in fs);harm=sum(v['correct_A4C_kept']['runs']-v['correct_A4C_kept']['kept'] for v in fs);net=[v['runs']['all']['net_fixed'] for v in fs];df=[v['sequence_row']['segment_f1_50_multi']-b['sequence_row']['segment_f1_50_multi'] for v,b in zip(fs,bs)]
    g={'S1_ok':fix>=math.ceil(.10*denom),'source_ok':len(set.union(*sources))>=3 and all(v['S1_fixed']['fixed']>=1 and len(src)>=1 for v,src in zip(fs,sources)),'A4C_ok':harm<=math.floor(.01*protected),'nonA4C_invariant':all(v['non_A4C_unchanged'] for v in fs),'net_per_fold_ok':all(n>=0 for n in net),'net_oof_ok':sum(net)>=5,'F1_per_fold_ok':all(d>=0 for d in df),'F1_mean_ok':sum(df)/2>=.001,'native_ok':all(v['sequence_row']['native_macro_f1']-b['sequence_row']['native_macro_f1']>=-.003 for v,b in zip(fs,bs)),'boundary_recall_ok':all(v['sequence_row']['boundary_recall_multi']-b['sequence_row']['boundary_recall_multi']>=-.005 for v,b in zip(fs,bs)),'short_misses_ok':sum(v['counts']['fn_short'] for v in fs)<=sum(v['counts']['fn_short'] for v in bs)+3,'not_noop':sum(v['changed'] for v in fs)>0};g['passed']=all(g.values());return g


def readout(ctx,source):
    conditional(source);pins(ctx,source);assert not (OUT/f'readout_started_s{source}.json').exists(),'once readout already started; audit saved cache rather than repeat'
    frozen=json.loads((OUT/f'frozen_s{source}.json').read_text());assert sha(OUT/f'weights_s{source}.json')==frozen['manifest_sha256'];man=json.loads((OUT/f'weights_s{source}.json').read_text());write(OUT/f'readout_started_s{source}.json',{'at':t.now(),'pid':os.getpid(),'frozen':frozen});coord('once_opposite_fold_readout',source=source)
    data=t.t31.vpr.training_data(ctx.a31.a15.a10.a07p);a8=t.rsh.a08_ctx();cells={};cache={};sources={k:[] for k in ARMS};alls={k:[] for k in ['Base',*ARMS]};ys=[]
    for h in [0,1]:
        D=t.d32.fold_data(ctx.d,source,h);D['M']=t.d32.set_masks(D);pk=D['pk'];preds={'Base':pk['base'].copy()}
        for arm in ARMS:
            F=loadF(ctx,source,h,arm);key=f'{arm}_s{source}_f{1-h}';rd=OUT/'train'/key;assert sha(rd/'final.pt')==man['models'][key]['final_sha256'];m,_=new_model(source,1-h);m.load_state_dict(torch.load(rd/'final.pt',map_location='cuda',weights_only=True));ss=[]
            with torch.no_grad():
                for j in range(0,len(F.need),32):ss.append(m(*batch(F,F.need[j:j+32])).cpu().numpy())
            scores=np.concatenate(ss);lookup=np.full(len(F.W['seq']),np.nan);lookup[F.need]=scores;r=np.full(len(pk['base']),-np.inf)
            for i in np.flatnonzero(pk['base']==2):
                lo,hi=pk['offsets'][i:i+2];rows=pk['rows'][lo:hi]-F.off;r[i]=np.dot(lookup[rows].astype(np.float64),pk['weights'][lo:hi])
            preds[arm]=cp.deploy(pk['base'],r);cache[f'score_{arm}_h{h}']=scores;cache[f'r_{arm}_h{h}']=r;del F;torch.cuda.empty_cache()
        P={'seqs':D['seqs'],'n_of':dict(data.n),'label_of':dict(data.label)};ev=t.bc.evalset(P,list(range(len(D['seqs']))),f'heldout_f{h}');cell={}
        for name,pred in preds.items():
            row=t.tables_pred(pred,D,a8.cfg,P,ev);ps=row.pop('preds');row.pop('pred');row['boundaries']=t.bounds(D,ps);ex,recs=t.a5c_extras(D,pred);row.update(ex);row['non_A4C_unchanged']=bool(np.array_equal(pred[pk['base']!=2],pk['base'][pk['base']!=2]));row['changed']=int((pred!=pk['base']).sum());cell[name]=row;alls[name].append(pred);cache[f'pred_{name}_h{h}']=pred
            if name in sources:sources[name].append(recs)
        cache[f'rows_h{h}']=t.Fold(ctx,source,h).need;cache[f'label_h{h}']=pk['labels'];ys.append(pk['labels']);cells[f'heldout_f{h}']=cell
        write(OUT/f'readout_s{source}_fold{h}.json',cell);np.savez(OUT/f'private_readout_s{source}_fold{h}.npz',**{k:v for k,v in cache.items() if k.endswith(f'h{h}')})
    yy=np.concatenate(ys);bb=np.concatenate(alls['Base']);oof={}
    for name,ps in alls.items():
        pred=np.concatenate(ps);fx=int(((bb!=yy)&(pred==yy)).sum());ha=int(((bb==yy)&(pred!=yy)).sum());oof[name]={'fixed':fx,'broken':ha,'net':fx-ha,'A5C_fixed':int(((yy==3)&(bb==2)&(pred==3)).sum()),'correct_A4C_harmed':int(((yy==2)&(bb==2)&(pred==3)).sum()),'F1_mean':float(np.mean([cells[f'heldout_f{h}'][name]['sequence_row']['segment_f1_50_multi'] for h in [0,1]])),'missed_short':sum(cells[f'heldout_f{h}'][name]['counts']['fn_short'] for h in [0,1]),'boundaries_deleted':sum(cells[f'heldout_f{h}'][name]['boundaries']['deleted_by_merge'] for h in [0,1])}
    gates={arm:package(cells,arm,sources[arm]) for arm in ARMS};pc=[cells[f'heldout_f{h}']['P'] for h in [0,1]];gc=[cells[f'heldout_f{h}']['G'] for h in [0,1]]
    inc={'fix_each_fold':all(p['S1_fixed']['fixed']>=g['S1_fixed']['fixed'] for p,g in zip(pc,gc)),'fix_plus5_oof':oof['P']['A5C_fixed']>=oof['G']['A5C_fixed']+5,'harms_no_more_each_fold':all(p['correct_A4C_kept']['kept']>=g['correct_A4C_kept']['kept'] for p,g in zip(pc,gc)),'net_plus5_oof':oof['P']['net']>=oof['G']['net']+5,'F1_each_fold':all(p['sequence_row']['segment_f1_50_multi']>=g['sequence_row']['segment_f1_50_multi'] for p,g in zip(pc,gc)),'absolute_P':gates['P']['passed']};inc['passed']=all(inc.values())
    np.savez(OUT/f'private_readout_s{source}.npz',**cache);result={'source':source,'cells':cells,'oof':oof,'gates':gates,'pretrained_state_increment':inc,'frozen':frozen,'private_cache_sha256':sha(OUT/f'private_readout_s{source}.npz'),'written':t.now()};publish(f'readout_s{source}.json',result);coord('readout_complete_pending_independent_audit',source=source);return {'oof':oof,'gates':gates,'increment':inc}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',required=True,choices=['prepare','acquire','source-freeze','features','check','train','freeze','readout']);ap.add_argument('--source',type=int,default=0);ap.add_argument('--commit',default='');ap.add_argument('--manifest-sha',default='');args=ap.parse_args();OUT.mkdir(parents=True,exist_ok=True);RES.mkdir(parents=True,exist_ok=True);configure();start=time.time();started=t.now();ok=False;kind='gpu' if args.stage in ['features','check','train','readout'] else 'cpu'
    try:
        ctx=a45.context();fn={'prepare':lambda:prepare(ctx,args.source),'acquire':acquire,'source-freeze':lambda:source_freeze(args.commit,args.manifest_sha),'features':lambda:features(ctx,args.source),'check':lambda:checks(ctx,args.source),'train':lambda:train(ctx,args.source),'freeze':lambda:freeze(args.source,args.commit,args.manifest_sha),'readout':lambda:readout(ctx,args.source)}[args.stage];result=fn();ok=True;print(json.dumps(result,indent=1,default=float),flush=True)
    finally:
        if kind=='gpu':t.ledger_append(OUT/'ledger.jsonl',f'A47 {args.stage} source{args.source} stage wall','gpu',started,time.time()-start,ok,'includes I/O while stage active; no future/idle precharge')
        t.ledger_append(OUT/'ledger.jsonl',f'A47 {args.stage} source{args.source} process CPU','cpu',started,time.process_time(),ok,'actual process CPU including imports; root active wall separately')
if __name__=='__main__':main()
