"""A48 licensed valve-localization spatial transfer; sole Codex queue.
All references, pixels, detailed logs and score caches remain server-side.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
import argparse
import copy
import hashlib
import inspect
import json
import math
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
import torchvision
from torch import nn
from echo_routing.temporal import clinical_pretraining as cp
from echo_routing.temporal import valve_localized as vl
import scripts.train_a4c_support as a45
import scripts.train_risk_constrained_a5c as t

CFG='configs/experiments/video_valve_localized_a5c_v1.yaml'
CARD='docs/superpowers/plans/2026-10-29-valve-localized-a5c.md'
OUT=ROOT/'runs/video_valve_localized_a5c_v1'
RES=ROOT/'docs/results/2026-10-29_valve_localized_a5c'
SRC=ROOT/'runs/a48_valve_source_intake_v1'
CODE=['scripts/train_valve_localized_a5c.py','scripts/audit_valve_localized_a5c.py','echo_routing/temporal/valve_localized.py']
ARMS=['R','G'];U=600
sha=a45.sha;digest=a45.digest;write=a45.write
def publish(name, value):
    write(OUT/name,value);write(RES/name,value)


def coord(phase, **kw):
    p=OUT/'coordination.json';old=json.loads(p.read_text()) if p.exists() else {}
    write(p,old|{'round':'A48','owner':'Codex','phase':phase,'at':t.now(),'pid':os.getpid(),**kw})


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
    return folder_bytes(OUT)+folder_bytes(SRC)+80_000_000


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
        result[str(f)]={'eligible_runs':len(F.L['eligible']),'rows':len(F.need),'unique_frames':len(unique),'grid':len(F.W['seq']),'fit_positive':len(F.L['positives']),'fit_negative':len(F.L['negatives']),'other':len(F.L['other']),'ram_pixel_bytes':len(unique)*224*224*3,'references_sha256':sha(rp),'inputs_sha256':sha(ip),'plan_sha256':F.psha,'exposure':ex,'membership_verified':True,'sessions':len(sess),'original_output_sha256':t.PINS['outputs'][source][str(f)],'runs_sha256':t.PINS['runs'][source][str(f)]}
    assert not sessions[0]&sessions[1]
    projected=new_bytes()+700_000_000
    assert projected<3e9, f'feature-cache forecast exceeds shared disk cap: {projected}'
    result|={'source':source,'base_sha256':base,'session_isolation':True,'projected_final_shared_bytes':projected,'config_sha256':sha(ROOT/CFG),'contract_sha256':sha(ROOT/CARD),'written':t.now()}
    publish(f'preparation_s{source}.json',result);coord('prepared',source=source);return result


def source_manifest():
    src=json.loads((SRC/'sources.json').read_text())
    for arm in ARMS:
        assert sha(SRC/f'backbone_{arm}.pt')==src['sources'][arm]['backbone_sha256']
    src|={'code_sha256':{p:sha(ROOT/p) for p in CODE},'loaded_repository_code_sha256':t.t38.loaded_code(),'config_sha256':sha(ROOT/CFG),'contract_sha256':sha(ROOT/CARD),'written':t.now()}
    publish('sources.json',src);return src


def source_freeze(commit,msha):
    assert commit and sha(OUT/'sources.json')==sha(RES/'sources.json')==msha
    man=json.loads((OUT/'sources.json').read_text())
    for p,h in man['code_sha256'].items():assert sha(ROOT/p)==h
    publish('source_frozen.json',{'commit':commit,'manifest_sha256':msha,'at':t.now(),'root_verified_actual_git_objects':True})
    return {'source_frozen':commit}


def loadF(ctx,source,f):
    F=t.Fold(ctx,source,f);inp=np.load(OUT/f'input_s{source}_f{f}.npz');assert np.array_equal(inp['rows'],F.need) and np.array_equal(inp['plan'],F.pl)
    prep=json.loads((OUT/f'preparation_s{source}.json').read_text())
    rp=OUT/f'refs_s{source}_f{f}.private.json';assert sha(rp)==prep[str(f)]['references_sha256']
    refs=[tuple(v) for v in json.loads(rp.read_text())]
    chunks=[]
    for j in range(0,len(refs),128):chunks.append(t.t37.rr.decode_frames(refs[j:j+128],F.fp,0))
    F.bank=torch.from_numpy(np.concatenate(chunks));del chunks
    F.at=np.full(len(F.W['seq']),-1,np.int64);F.at[F.need]=inp['index']
    F.embd=torch.from_numpy(np.asarray(t.t31.outputs(ctx.a31,source,f)['emb'],np.float32)).cuda()
    F.refs=refs
    assert F.bank.dtype==torch.uint8 and F.bank.shape[1:]==(224,224,3)
    return F


def new_model(source,f,arm='R',device='cuda',force=None):
    state=torch.load(SRC/f'backbone_{force or arm}.pt',map_location='cpu',weights_only=True)
    backbone=vl.make_backbone(state)
    torch.manual_seed(2026102948+10*source+f);m=vl.ValveRepair(backbone).eval().to(device)
    groups=[{'params':[p for n,p in m.backbone.named_parameters() if n.startswith('body.layer4.')],'lr':1e-5,'base_lr':1e-5},
            {'params':list(m.backbone.fpn.parameters()),'lr':1e-5,'base_lr':1e-5},
            {'params':list(m.head.parameters()),'lr':1e-3,'base_lr':1e-3}]
    assert all(p.requires_grad for g in groups for p in g['params'])
    opt=torch.optim.AdamW(groups,betas=(.9,.999),eps=1e-8,weight_decay=1e-4)
    return m,opt


def batch(F,rows):
    idx=F.at[rows];assert (idx>=0).all()
    return F.bank[torch.as_tensor(idx)],F.embd[torch.as_tensor(rows,device='cuda')]


def step(m,opt,F,u):
    start=time.time();rows,v,slots=F.occ(u);x,e=batch(F,rows)
    weights=torch.as_tensor(v,dtype=torch.float32,device='cuda');sg=torch.as_tensor(slots,device='cuda')
    for g in opt.param_groups:g['lr']=g['base_lr']*t.t31.lr_factor(u,60,600)
    opt.zero_grad(set_to_none=True);rec=vl.two_pass_backward(m,x,e,weights,sg,4)
    pp=[(n,p) for n,p in m.named_parameters() if p.requires_grad]
    assert all(p.grad is not None for n,p in pp)
    norms=torch.stack([p.grad.detach().double().square().sum() for n,p in pp]);assert torch.isfinite(norms).all()
    values=norms.cpu().tolist();zero=[n for (n,p),z in zip(pp,values) if z==0]
    gs={key:sum(z for (n,p),z in zip(pp,values) if n.startswith(prefix)) for key,prefix in [('layer4','backbone.body.layer4.'),('fpn','backbone.fpn.'),('head','head.')]}
    gn=float(torch.nn.utils.clip_grad_norm_([p for n,p in pp],1));assert math.isfinite(gn);opt.step();torch.cuda.synchronize()
    return rec|{'u':u,'lr':[g['lr'] for g in opt.param_groups],'grad_norm':gn,'grad_sq':gs,'zero_grad_tensors':zero,'rows_sha256':digest(rows),'weights_sha256':digest(v),'slots_sha256':digest(slots),'n_occ':len(rows),'pid':os.getpid(),'device':torch.cuda.get_device_name(0),'seconds':time.time()-start}


def cpu_state(m):return {k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
def delta(a,b):return max(float((a[k].cpu()-b[k].cpu()).abs().max()) for k in a)
def live(m,logs):return all(any(n not in rec['zero_grad_tensors'] for rec in logs[1:]) for n,p in m.named_parameters() if p.requires_grad)
def frozen_equal(m,initial):
    trained={n for n,p in m.named_parameters() if p.requires_grad}
    return all(torch.equal(v.cpu(),initial[k]) for k,v in m.state_dict().items() if k not in trained)


def official(arm):
    from torchvision.models.detection import fasterrcnn_resnet50_fpn,FasterRCNN_ResNet50_FPN_Weights
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    import torchvision.models.detection.faster_rcnn as module
    # Use recorded local COCO source through the licensed constructor, never a new download.
    generic=torch.load(SRC/'fasterrcnn_resnet50_fpn_coco-258fb6c6.pth',map_location='cpu',weights_only=True)
    enum=FasterRCNN_ResNet50_FPN_Weights.COCO_V1
    original=type(enum).get_state_dict
    type(enum).get_state_dict=lambda self,*a,**kw:generic
    try:full=fasterrcnn_resnet50_fpn(weights=enum,weights_backbone=None).eval()
    finally:type(enum).get_state_dict=original
    if arm=='R':
        full.roi_heads.box_predictor=FastRCNNPredictor(full.roi_heads.box_predictor.cls_score.in_features,4)
        full.load_state_dict(torch.load(SRC/'model_avdetector.pth',map_location='cpu',weights_only=True),strict=True)
    return full


def checks(ctx,source):
    conditional(source);pins(ctx,source)
    sf=json.loads((OUT/'source_frozen.json').read_text());assert sha(OUT/'sources.json')==sf['manifest_sha256']
    F=loadF(ctx,source,0);parity={};select=[0,len(F.refs)//2,len(F.refs)-1]
    fr=t.t37.rr.decode_frames([F.refs[j] for j in select],F.fp,0);assert np.array_equal(fr,F.bank[select].numpy())
    for arm in ARMS:
        full=official(arm).cuda();m,o=new_model(source,0,arm)
        assert all(torch.equal(v,full.backbone.state_dict()[k]) for k,v in m.backbone.state_dict().items())
        frames=torch.from_numpy(fr);xx=frames.permute(0,3,1,2).cuda().float()/255
        with torch.no_grad():
            image1,_=full.transform(list(xx),None);image2,_=m.transform(list(xx),None)
            assert torch.equal(image1.tensors,image2.tensors)
            maps1=full.backbone(image1.tensors);maps2=m.backbone(image2.tensors)
        err=max(float((maps1[str(i)]-maps2[str(i)]).abs().max()) for i in range(4));assert err==0
        assert sum(p.numel() for p in m.head.parameters())==294281
        parity[arm]={'official_backbone_tensor_identity':True,'official_transform_exact':True,'all_four_maps_max_abs':err,'FrozenBN_eps':0.}
        del full,m,o,maps1,maps2;torch.cuda.empty_cache()
    m,o=new_model(source,0);initial=cpu_state(m);rows,v,slots=F.occ(0);x,e=batch(F,rows)
    with torch.no_grad():assert torch.equal(torch.cat([m(x[j:j+4],e[j:j+4]) for j in range(0,len(rows),4)]),torch.full((len(rows),),-2.,device='cuda'))
    logs=[step(m,o,F,u) for u in range(6)];goal=cpu_state(m);assert live(m,logs) and frozen_equal(m,initial)
    assert logs[0]['grad_sq']['layer4']==logs[0]['grad_sq']['fpn']==0
    del m,o;torch.cuda.empty_cache()
    repeat={}
    for variant in ['repeat','forcedG','resume']:
        mm,oo=new_model(source,0,'G' if variant=='forcedG' else 'R',force='R' if variant=='forcedG' else None)
        for u in range(6):
            if variant=='resume' and u==3:
                ss=cpu_state(mm);osave=copy.deepcopy(oo.state_dict());del mm,oo;mm,oo=new_model(source,0);mm.load_state_dict(ss);oo.load_state_dict(osave);del ss,osave
            step(mm,oo,F,u)
        repeat[variant]=delta(goal,mm.state_dict());assert repeat[variant]==0;del mm,oo;torch.cuda.empty_cache()
    m,o=new_model(source,0);m.load_state_dict(goal);trained=[p for p in m.parameters() if p.requires_grad]
    umin=int(np.argmin([len(F.occ(u)[0]) for u in range(U)]));rows,v,slots=F.occ(umin);x,e=batch(F,rows);vv=torch.tensor(v,dtype=torch.float32,device='cuda');sg=torch.tensor(slots,device='cuda')
    m.zero_grad(set_to_none=True);scores=m(x,e);loss,r=cp.loss_from_scores(scores,vv,sg);loss.backward();ga=[p.grad.detach().clone() for p in trained]
    z=scores.detach();r64=torch.stack([(z.double()[sg==k]*vv.double()[sg==k]).sum() for k in range(4)]);reference=torch.logaddexp(r64.new_zeros(()),-r64[0])+torch.logaddexp(r64.new_zeros(3),r64[1:]).mean()
    co64=torch.where(sg==0,-torch.sigmoid(-r64[0])*vv.double(),torch.sigmoid(r64[sg])*vv.double()/3)
    zz=z.requires_grad_();ll,_=cp.loss_from_scores(zz,vv,sg);co=torch.autograd.grad(ll,zz)[0];ce=float((co.double()-co64).abs().max());assert ce<1e-6
    m.zero_grad(set_to_none=True);rec=vl.two_pass_backward(m,x,e,vv,sg,4)
    num=sum((a.double()-p.grad.double()).square().sum() for a,p in zip(ga,trained));den=sum(a.double().square().sum() for a in ga);rel=float(torch.sqrt(num/den));assert rel<1e-4
    with torch.no_grad():individual=torch.cat([m(x[j:j+1],e[j:j+1]) for j in range(len(x))])
    scoreerr=float((individual-scores.detach()).abs().max());assert scoreerr<1e-4
    # Add a duplicated zero coefficient occurrence while preserving every original run weight.
    xz=torch.cat([x,x[:1]]);ez=torch.cat([e,e[:1]]);wz=torch.cat([vv,vv.new_zeros(1)]);sz=torch.cat([sg,sg[:1]]);m.zero_grad(set_to_none=True);zr=vl.two_pass_backward(m,xz,ez,wz,sz,4);assert zr['zero_coefficients']>=1 and zr['second_occurrences']==len(xz)
    del ga,scores,xz,ez;torch.cuda.empty_cache();umax=int(np.argmax([len(F.occ(u)[0]) for u in range(U)]));torch.cuda.reset_peak_memory_stats();timing=step(m,o,F,umax);memory=torch.cuda.max_memory_allocated()/1e9;assert memory<40
    prep=json.loads((OUT/f'preparation_s{source}.json').read_text());occ=0
    for f in [0,1]:
        ff=t.Fold(ctx,source,f);occ+=sum(len(ff.occ(u)[0]) for u in range(600))*2
    forecast=occ*timing['seconds']/timing['n_occ']*1.25+1200;assert forecast<5*3600
    result={'ok':True,'source':source,'parameters':sum(p.numel() for p in m.parameters()),'trainable':sum(p.numel() for p in m.parameters() if p.requires_grad),'head_parameters':294281,'repeats_max_abs':repeat,'all_selected_tensor_gradients_live':True,'all_frozen_unchanged':True,'first_step_structural_zero':True,'production_gradient_relative':rel,'formula_error':abs(float(reference)-float(loss.detach())),'coefficient_max_abs':ce,'batch_invariance_max_abs':scoreerr,'zero_occurrence_real_backward':True,'source_parity':parity,'max_occurrences':timing['n_occ'],'peak_gpu_gb':memory,'forecast_training_plus_readout_gpu_hours':forecast/3600,'timing':timing,'written':t.now()}
    assert result['formula_error']<1e-5;publish(f'checks_s{source}.json',result);return result


def train(ctx,source):
    conditional(source);base=pins(ctx,source);checks=json.loads((OUT/f'checks_s{source}.json').read_text());assert checks['ok']
    assert json.loads((OUT/'resource_authorization.json').read_text())['within_shared_cap']
    prep=json.loads((OUT/f'preparation_s{source}.json').read_text());models={}
    for f in [0,1]:
        F=loadF(ctx,source,f)
        for arm in ARMS:
            key=f'{arm}_s{source}_f{f}';rd=OUT/'train'/key;rd.mkdir(parents=True,exist_ok=True)
            identity=hashlib.sha256(''.join([sha(ROOT/CFG),sha(ROOT/CARD),*[sha(ROOT/p) for p in CODE],prep[str(f)]['inputs_sha256'],sha(SRC/f'backbone_{arm}.pt'),key]).encode()).hexdigest()
            if (rd/'manifest.json').exists():
                man=json.loads((rd/'manifest.json').read_text());assert man['identity']==identity and sha(rd/'final.pt')==man['final_sha256'];models[key]=man;continue
            assert not (rd/'final.pt').exists(),'unmanifested final: audit, never retrain'
            m,o=new_model(source,f,arm);initial=cpu_state(m);logs=[]
            if (rd/'resume.pt').exists():
                ss=torch.load(rd/'resume.pt',map_location='cuda',weights_only=False);assert ss['identity']==identity;m.load_state_dict(ss['model']);o.load_state_dict(ss['opt']);logs=ss['logs'];del ss
            coord('formal_training',source=source,model=key,update=len(logs),device=torch.cuda.get_device_name(0))
            for u in range(len(logs),U):
                rec=step(m,o,F,u);logs.append(rec)
                if (u+1)%25==0:
                    pr={'model':key,'update':u+1,'total':600,'pid':os.getpid(),'device':rec['device'],'loss':rec['loss'],'window_occurrences':sum(v['n_occ'] for v in logs),'at':t.now()};write(OUT/'progress.json',pr);print(json.dumps(pr),flush=True)
                if (u+1)%150==0 and u+1<600:
                    tmp=rd/'resume.tmp';torch.save({'model':m.state_dict(),'opt':o.state_dict(),'logs':logs,'identity':identity},tmp);tmp.replace(rd/'resume.pt')
            pp=[(n,p) for n,p in m.named_parameters() if p.requires_grad];steps=[int(o.state[p]['step']) for n,p in pp];assert set(steps)=={600} and len(logs)==600 and live(m,logs) and frozen_equal(m,initial)
            moved={n:not torch.equal(p.detach().cpu(),initial[n]) for n,p in pp};assert all(moved.values())
            torch.save(cpu_state(m),rd/'final.pt');(rd/'updates.jsonl').write_text(''.join(json.dumps(v)+'\n' for v in logs));torch.save({'steps':steps,'parameter_names':[n for n,p in pp],'final_update':600},rd/'optimizer_steps.pt')
            man={'arm':arm,'source':source,'fit_fold':f,'identity':identity,'updates':600,'optimizer_steps':steps,'trainable_parameters':sum(p.numel() for n,p in pp),'all_trainable_moved':moved,'all_trainable_gradients_observed_nonzero':True,'all_frozen_unchanged':True,'final_sha256':sha(rd/'final.pt'),'updates_sha256':sha(rd/'updates.jsonl'),'optimizer_receipt_sha256':sha(rd/'optimizer_steps.pt'),'plan_sha256':F.psha,'inputs_sha256':prep[str(f)]['inputs_sha256'],'source_state_sha256':sha(SRC/f'backbone_{arm}.pt'),'initial_tensor_sha256':{k:digest(v.numpy()) for k,v in initial.items()},'occurrences':sum(v['n_occ'] for v in logs),'pid':os.getpid(),'device':torch.cuda.get_device_name(0),'finished':t.now()};write(rd/'manifest.json',man);models[key]=man;(rd/'resume.pt').unlink(missing_ok=True);del m,o,initial;torch.cuda.empty_cache()
        del F
    pins(ctx,source)
    man={'models':models,'source':source,'base_sha256':base,'source_manifest_sha256':sha(OUT/'sources.json'),'source_freeze':json.loads((OUT/'source_frozen.json').read_text()),'config_sha256':sha(ROOT/CFG),'contract_sha256':sha(ROOT/CARD),'code_sha256':{p:sha(ROOT/p) for p in CODE},'loaded_repository_code_sha256':t.t38.loaded_code(),'parent_config_sha256':{p:sha(ROOT/p) for p in [ctx.config,t.A31_CONFIG,t.A32_CONFIG]},'preparation_sha256':sha(OUT/f'preparation_s{source}.json'),'checks_sha256':sha(OUT/f'checks_s{source}.json'),'written':t.now()};publish(f'weights_s{source}.json',man);coord('trained_pending_actual_git_freeze',source=source);return {'models':4,'actual_updates':2400}


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


def readout(ctx,source,cache_only=False):
    conditional(source);pins(ctx,source)
    if cache_only:assert (OUT/f'readout_started_s{source}.json').exists()
    else:assert not (OUT/f'readout_started_s{source}.json').exists(),'once readout already started; recover saved cache only'
    frozen=json.loads((OUT/f'frozen_s{source}.json').read_text());assert sha(OUT/f'weights_s{source}.json')==frozen['manifest_sha256'];man=json.loads((OUT/f'weights_s{source}.json').read_text())
    if not cache_only:write(OUT/f'readout_started_s{source}.json',{'at':t.now(),'pid':os.getpid(),'frozen':frozen})
    coord('cache_only_evaluation' if cache_only else 'once_opposite_fold_readout',source=source)
    data=t.t31.vpr.training_data(ctx.a31.a15.a10.a07p);a8=t.rsh.a08_ctx();cells={};cache={};sources={k:[] for k in ARMS};alls={k:[] for k in ['Base',*ARMS]};ys=[]
    for h in [0,1]:
        D=t.d32.fold_data(ctx.d,source,h);D['M']=t.d32.set_masks(D);pk=D['pk'];preds={'Base':pk['base'].copy()}
        for arm in ARMS:
            if cache_only:
                saved=np.load(OUT/f'scores_{arm}_s{source}_h{h}.private.npz');scores=saved['scores'];r=saved['r'];assert np.array_equal(saved['rows'],t.Fold(ctx,source,h).need)
                preds[arm]=cp.deploy(pk['base'],r);cache[f'score_{arm}_h{h}']=scores;cache[f'r_{arm}_h{h}']=r;continue
            F=loadF(ctx,source,h);key=f'{arm}_s{source}_f{1-h}';rd=OUT/'train'/key;assert sha(rd/'final.pt')==man['models'][key]['final_sha256'];m,_=new_model(source,1-h,arm);m.load_state_dict(torch.load(rd/'final.pt',map_location='cuda',weights_only=True));ss=[]
            with torch.no_grad():
                for j in range(0,len(F.need),4):ss.append(m(*batch(F,F.need[j:j+4])).cpu().numpy())
            scores=np.concatenate(ss);lookup=np.full(len(F.W['seq']),np.nan);lookup[F.need]=scores;r=np.full(len(pk['base']),-np.inf)
            for i in np.flatnonzero(pk['base']==2):
                lo,hi=pk['offsets'][i:i+2];rows=pk['rows'][lo:hi]-F.off;r[i]=np.dot(lookup[rows].astype(np.float64),pk['weights'][lo:hi])
            np.savez(OUT/f'scores_{arm}_s{source}_h{h}.private.npz',scores=scores,r=r,rows=F.need);preds[arm]=cp.deploy(pk['base'],r);cache[f'score_{arm}_h{h}']=scores;cache[f'r_{arm}_h{h}']=r;del F;torch.cuda.empty_cache()
        P={'seqs':D['seqs'],'n_of':dict(data.n),'label_of':dict(data.label)};ev=t.bc.evalset(P,list(range(len(D['seqs']))),f'heldout_f{h}');cell={}
        for name,pred in preds.items():
            row=t.tables_pred(pred,D,a8.cfg,P,ev);ps=row.pop('preds');row.pop('pred');row['boundaries']=t.bounds(D,ps);ex,recs=t.a5c_extras(D,pred);row.update(ex);row['non_A4C_unchanged']=bool(np.array_equal(pred[pk['base']!=2],pk['base'][pk['base']!=2]));row['changed']=int((pred!=pk['base']).sum());cell[name]=row;alls[name].append(pred);cache[f'pred_{name}_h{h}']=pred
            if name in sources:sources[name].append(recs)
        cache[f'rows_h{h}']=t.Fold(ctx,source,h).need;cache[f'label_h{h}']=pk['labels'];ys.append(pk['labels']);cells[f'heldout_f{h}']=cell
        write(OUT/f'readout_s{source}_fold{h}.json',cell);np.savez(OUT/f'private_readout_s{source}_fold{h}.npz',**{k:v for k,v in cache.items() if k.endswith(f'h{h}')})
    yy=np.concatenate(ys);bb=np.concatenate(alls['Base']);oof={}
    for name,ps in alls.items():
        pred=np.concatenate(ps);fx=int(((bb!=yy)&(pred==yy)).sum());ha=int(((bb==yy)&(pred!=yy)).sum());oof[name]={'fixed':fx,'broken':ha,'net':fx-ha,'A5C_fixed':int(((yy==3)&(bb==2)&(pred==3)).sum()),'correct_A4C_harmed':int(((yy==2)&(bb==2)&(pred==3)).sum()),'F1_mean':float(np.mean([cells[f'heldout_f{h}'][name]['sequence_row']['segment_f1_50_multi'] for h in [0,1]])),'missed_short':sum(cells[f'heldout_f{h}'][name]['counts']['fn_short'] for h in [0,1]),'boundaries_deleted':sum(cells[f'heldout_f{h}'][name]['boundaries']['deleted_by_merge'] for h in [0,1])}
    gates={arm:package(cells,arm,sources[arm]) for arm in ARMS};pc=[cells[f'heldout_f{h}']['R'] for h in [0,1]];gc=[cells[f'heldout_f{h}']['G'] for h in [0,1]]
    inc={'fix_each_fold':all(p['S1_fixed']['fixed']>=g['S1_fixed']['fixed'] for p,g in zip(pc,gc)),'fix_plus5_oof':oof['R']['A5C_fixed']>=oof['G']['A5C_fixed']+5,'harms_no_more_each_fold':all(p['correct_A4C_kept']['kept']>=g['correct_A4C_kept']['kept'] for p,g in zip(pc,gc)),'net_plus5_oof':oof['R']['net']>=oof['G']['net']+5,'F1_each_fold':all(p['sequence_row']['segment_f1_50_multi']>=g['sequence_row']['segment_f1_50_multi'] for p,g in zip(pc,gc)),'absolute_R':gates['R']['passed']};inc['passed']=all(inc.values())
    np.savez(OUT/f'private_readout_s{source}.npz',**cache);result={'source':source,'cells':cells,'oof':oof,'gates':gates,'pretrained_state_increment':inc,'frozen':frozen,'private_cache_sha256':sha(OUT/f'private_readout_s{source}.npz'),'written':t.now()};publish(f'readout_s{source}.json',result);coord('readout_complete_pending_independent_audit',source=source);return {'oof':oof,'gates':gates,'increment':inc}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',required=True,choices=['prepare','sources','source-freeze','check','train','freeze','readout','readout-cache']);ap.add_argument('--source',type=int,default=0);ap.add_argument('--commit',default='');ap.add_argument('--manifest-sha',default='');args=ap.parse_args();OUT.mkdir(parents=True,exist_ok=True);RES.mkdir(parents=True,exist_ok=True);configure();start=time.time();started=t.now();ok=False;kind='gpu' if args.stage in ['check','train','readout'] else 'cpu'
    try:
        ctx=a45.context();fn={'prepare':lambda:prepare(ctx,args.source),'sources':source_manifest,'source-freeze':lambda:source_freeze(args.commit,args.manifest_sha),'check':lambda:checks(ctx,args.source),'train':lambda:train(ctx,args.source),'freeze':lambda:freeze(args.source,args.commit,args.manifest_sha),'readout':lambda:readout(ctx,args.source),'readout-cache':lambda:readout(ctx,args.source,True)}[args.stage];result=fn();ok=True;print(json.dumps(result,indent=1,default=float),flush=True)
    finally:
        if kind=='gpu':t.ledger_append(OUT/'ledger.jsonl',f'A48 {args.stage} source{args.source} stage wall','gpu',started,time.time()-start,ok,'includes I/O and bootstrap; no future/idle precharge')
        t.ledger_append(OUT/'ledger.jsonl',f'A48 {args.stage} source{args.source} process CPU','cpu',started,time.process_time(),ok,'actual process CPU including imports; root active wall separately')
if __name__=='__main__':main()
