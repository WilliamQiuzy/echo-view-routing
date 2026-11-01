#!/usr/bin/env python3
"""PW-T1-C: user-authorized bounded execution of the unchanged PW-T1 comparison."""
from __future__ import annotations
import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES','0');os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse
import copy
import hashlib
import json
from pathlib import Path
import signal
import resource
import sys
import time
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from echo_routing.temporal import pw_transfer as pwt
from echo_routing.temporal import pw_transfer_live as durable
from echo_routing.temporal.joint_visual import load_frame_u8
from echo_routing.temporal.dev_train import lr_factor
import scripts.diagnose_a5c_image_dependence as d
t=d.t44
OUT=ROOT/'runs/pw_purpose_transfer_live_v1';PUB=ROOT/'docs/results/2026-11-01_pw_purpose_transfer'
BASE=ROOT/'runs/a48_valve_source_intake_v1/backbone_G.pt'
ARMS=('G','S','L');CARD='docs/superpowers/plans/2026-11-01-pw-purpose-transfer-continuation.md'


def sources():
    assert d.sha(BASE)==pwt.BASE_SHA
    g=torch.load(BASE,map_location='cpu',weights_only=True);ss={'G':None}
    source_manifest=json.loads((ROOT/'docs/results/2026-10-30_pw_small_pilot/models_manifest.json').read_text())
    for a in ('S','L'):
        p=ROOT/'runs/pw_small_pilot_v1'/f'{a}.pt';assert d.sha(p)==pwt.SOURCE_SHA[a]
        ck=torch.load(p,map_location='cpu',weights_only=True);assert ck['steps']==600
        assert pwt.state_sha(ck['model'])==source_manifest['models'][a]['model_state_sha256']
        ss[a]=ck['model']
    return g,ss


def images(F):
    start,cpu=time.time(),time.process_time();memo={};frames={};h=hashlib.sha256()
    for row in F.need:
        seq=F.seqs[int(F.W['seq'][row])];pos=int(F.W['s'][row,int(F.W['marker'][row])])
        v,f,g=seq['refs'][pos];key=(v,int(f),round(float(g),6))
        if key not in memo:
            paths=F.fp(v);memo[key]=load_frame_u8(paths[min(int(f),len(paths)-1)],key[2]).copy()
        frame=memo[key];frames[int(row)]=frame
        h.update(np.asarray([row],np.int64).tobytes());h.update(frame.tobytes())
    return frames,{'rows':len(frames),'distinct_marker_inputs':len(memo),'marker_pixels_sha256':h.hexdigest(),
                   'decode_wall_seconds':time.time()-start,'decode_CPU_seconds':time.process_time()-cpu}


def prepare():
    if (OUT/'input_manifest.json').exists():raise RuntimeError('PW-T1 inputs already prepared')
    c=d.ctx();g,ss=sources();folds={}
    for f in (0,1):
        F=t.Fold(c,0,f);_,rec=images(F)
        assert (len(F.L['positives']),len(F.L['negatives']),len(F.L['other']))==((34,714,1),(69,699,3))[f]
        assert F.psha==('3b3be337e7d22a1f1a96196cfc4460b4425ae367f407f6a16a01e5b2cc5c3efc','2a1dd83d9e8b5f2025ccb6e4bd90587bae15ee9b42bb27fe7f6c71ab6ed463a8')[f]
        folds[str(f)]={**rec,'eligible_runs':int(F.el.sum()),'positives':len(F.L['positives']),
                       'protected_negatives':len(F.L['negatives']),'other':len(F.L['other']),
                       'plan_sha256':F.psha,'training_occurrences':int(sum(len(F.occ(u)[0]) for u in range(600))),
                       'pool_sha256':d.sha(c.a31.out/f'pool_{f}.npy'),'runs_sha256':d.sha(c.a31.out/'data'/f'runs_s0_f{f}.npz')}
    old=json.loads((ROOT/'docs/results/2026-10-31_pw_purpose_transfer/input_manifest.json').read_text())
    for f in (0,1):
        for k in ('marker_pixels_sha256','plan_sha256','pool_sha256','runs_sha256','training_occurrences','rows'):
            assert folds[str(f)][k]==old['folds'][str(f)][k]
    code=d.module_files();code[CARD]=d.sha(ROOT/CARD)
    original_card='docs/superpowers/plans/2026-10-31-pw-purpose-transfer.md';code[original_card]=d.sha(ROOT/original_card)
    cfg='configs/experiments/video_risk_constrained_a5c_v1.yaml';code[cfg]=d.sha(ROOT/cfg)
    rec={'run_id':'PW-T1-C','arms':ARMS,'source_files_sha256':pwt.SOURCE_SHA,'folds':folds,'code':code,
         'base_A08_sha256':d.sha(ROOT/'models_frozen/video_within_view_v1/all_s0/final.pt'),
         'formal_source_quantity_gate_passed':False,'original_source_quantity_gate_and_pool_preserved':True,'source_pairs_fit_heldout':[19,6],
         'source_manifest_sha256':d.sha(ROOT/'docs/results/2026-10-30_pw_small_pilot/models_manifest.json'),
         'training_updates_per_model':600,'new_loss_or_gate':False,'new_original_test_read':False}
    d.dump(OUT/'input_manifest.json',rec);d.dump(PUB/'input_manifest.json',rec)
    print(json.dumps({'stage':'prepared','folds':folds}),flush=True)


def verify_input():
    m=json.loads((OUT/'input_manifest.json').read_text())
    fr=json.loads((OUT/'input_freeze.json').read_text());assert fr['manifest_sha256']==d.sha(OUT/'input_manifest.json')
    for p,h in m['code'].items():assert d.sha(ROOT/p)==h,'PW-T1 source code changed: '+p
    return m


def model(g,ss,arm,device):
    torch.manual_seed(2026103151)
    return pwt.build(g,ss[arm]).to(device).eval()


def opt_for(m):
    return torch.optim.AdamW([{'params':[p for n,p in m.named_parameters() if p.requires_grad and n.startswith('backbone.')],'lr':1e-5,'base_lr':1e-5},
                             {'params':list(m.head.parameters()),'lr':1e-3,'base_lr':1e-3}],betas=(.9,.999),eps=1e-8,weight_decay=1e-4)


def step(m,opt,F,frames,u,device):
    for group in opt.param_groups:group['lr']=group['base_lr']*lr_factor(u,60,600)
    opt.zero_grad(set_to_none=True);rows,v,seg=F.occ(u)
    wall,cpu=time.time(),time.process_time()
    r=pwt.two_pass(m,frames,rows,torch.from_numpy(v).float().to(device),torch.from_numpy(seg).to(device),device)
    grad={n:bool(torch.count_nonzero(p.grad).item()) if p.grad is not None else False for n,p in m.named_parameters() if p.requires_grad}
    norm=torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad],1.)
    assert torch.isfinite(norm) and np.isfinite(r['loss']) and r['replay_max_abs']<=1e-5
    opt.step();torch.cuda.synchronize()
    return {**r,'step':u+1,'plan_row_sha256':hashlib.sha256(F.pl[u].tobytes()).hexdigest(),
            'occurrence_rows_sha256':hashlib.sha256(np.asarray(rows,np.int64).tobytes()).hexdigest(),
            'gradient_nonzero':grad,'gradient_norm':float(norm),'wall_seconds':time.time()-wall,'CPU_seconds':time.process_time()-cpu}


@torch.no_grad()
def score_rows(m,frames,rows,device):
    start,cpu=time.time(),time.process_time()
    s=np.array([float(m(pwt.normalise(torch.from_numpy(frames[int(r)][None]).to(device)))[0]) for r in rows],np.float32)
    torch.cuda.synchronize()
    return s,{'wall_seconds':time.time()-start,'CPU_seconds':time.process_time()-cpu}


def full_fit(m,F,frames,device):
    s,tim=score_rows(m,frames,F.need,device);full=np.full(len(F.W['seq']),np.nan,np.float32);full[F.need]=s
    r=t.rc.run_weighted_mean(F.pk,full,F.off,F.el)
    return {'positive_recovered':int((r[F.L['positives']]>0).sum()),'protected_harmed':int((r[F.L['negatives']]>0).sum()),
            'eligible_runs':int(F.el.sum()),**tim},s


def check(device):
    im=verify_input();c=d.ctx();F=t.Fold(c,0,0);frames,px=images(F);assert px['marker_pixels_sha256']==im['folds']['0']['marker_pixels_sha256']
    g,ss=sources();start=time.time();timings=[]
    for arm in ('G','S','L'):
        probe=model(g,ss,arm,'cpu');assert len([p for p in probe.parameters() if p.requires_grad])==28
        assert torch.count_nonzero(probe.head.weight)==0 and probe.head.bias.item()==-2;del probe
    def execute(updates,bundle=None,optimizer=None,forced_state=None):
        m=model(g,ss,'G',device) if forced_state is None else pwt.build(g,forced_state).to(device).eval()
        if bundle is not None:pwt.restore_bundle(m,bundle,pwt.BASE_SHA)
        opt=opt_for(m)
        if optimizer is not None:opt.load_state_dict(optimizer)
        log=[step(m,opt,F,frames,u,device) for u in updates]
        return m,opt,log
    m,opt,log=execute(range(6));expected=pwt.state_sha(m.state_dict())
    assert all(log[0]['gradient_nonzero'][n] is False for n in log[0]['gradient_nonzero'] if n.startswith('backbone.'))
    assert all(any(q['gradient_nonzero'][n] for q in log[1:]) for n in log[0]['gradient_nonzero'])
    timings=[q for q in log[2:]];trainable_bytes=sum(p.numel()*p.element_size() for p in m.parameters() if p.requires_grad)
    del m,opt
    forced=model(g,ss,'G','cpu').state_dict()
    m,opt,repeat=execute(range(6),forced_state=forced);assert pwt.state_sha(m.state_dict())==expected;del m,opt,forced
    m,opt,_=execute(range(3));bundle=pwt.make_bundle(m,pwt.BASE_SHA,{'step':3});ost=copy.deepcopy(opt.state_dict())
    p=OUT/'resume_check.pt';s=time.time();torch.save({'bundle':bundle,'optimizer':ost},p);save_seconds=time.time()-s
    loaded=torch.load(p,map_location='cpu',weights_only=True);p.unlink();del m,opt
    m,opt,resumed=execute(range(3,6),loaded['bundle'],loaded['optimizer']);assert pwt.state_sha(m.state_dict())==expected;del m,opt
    # Real production chain rule versus a single retained graph on fit-only four-row fixture.
    a=model(g,ss,'G',device);b=model(g,ss,'G',device)
    with torch.no_grad():a.head.weight.fill_(.001);b.load_state_dict(a.state_dict())
    rr=F.need[:4];v=torch.ones(4,device=device);seg=torch.arange(4,device=device)
    pwt.two_pass(a,frames,rr,v,seg,device)
    score=torch.cat([b(pwt.normalise(torch.from_numpy(frames[int(r)][None]).to(device))) for r in rr])
    (torch.nn.functional.softplus(-score[0])+torch.nn.functional.softplus(score[1:]).mean()).backward()
    dif=max(float((pa.grad-pb.grad).abs().max()) for pa,pb in zip(a.parameters(),b.parameters()) if pa.requires_grad)
    den=max(float(pb.grad.abs().max()) for pb in b.parameters() if pb.requires_grad);assert dif/max(den,1e-20)<=1e-5
    del a,b
    m=model(g,ss,'G',device);s,tim=score_rows(m,frames,F.need[:12],device);assert np.array_equal(s,np.full(12,-2,np.float32))
    # Independent per-reference decode / ImageNet normalization equality on fixed fit rows.
    for row in F.need[[0,len(F.need)//2,-1]]:
        seq=F.seqs[int(F.W['seq'][row])];v0,f0,g0=seq['refs'][int(F.W['s'][row,int(F.W['marker'][row])])];paths=F.fp(v0)
        assert np.array_equal(frames[int(row)],load_frame_u8(paths[min(int(f0),len(paths)-1)],round(float(g0),6)))
        independently_normalised=(frames[int(row)].astype(np.float32).transpose(2,0,1)/255-np.asarray([.485,.456,.406],np.float32)[:,None,None])/np.asarray([.229,.224,.225],np.float32)[:,None,None]
        assert np.array_equal(pwt.normalise(torch.from_numpy(frames[int(row)][None])).numpy()[0],independently_normalised)
    del m;torch.cuda.empty_cache()
    rec={'passed':True,'repeat6_bitwise':True,'forced_same_source_bitwise':True,'resume3_plus3_bitwise':True,'two_pass_gradient_relative_error':dif/max(den,1e-20),
         'all_selected_gradients_seen':True,'Init_minus2':True,'timed_updates':[{k:q[k] for k in ('occurrences','wall_seconds','CPU_seconds')} for q in timings],
         'score_seconds_per_row':tim['wall_seconds']/12,'score_CPU_per_row':tim['CPU_seconds']/12,'compact_checkpoint_save_seconds':save_seconds,
         'trainable_bytes':trainable_bytes,'decode':px,'wall_seconds':time.time()-start}
    d.dump(OUT/'checks.json',rec);d.dump(PUB/'checks.json',rec);print(json.dumps(rec),flush=True)


def save_resume(path,m,opt,u,identity,log):
    marker=path.with_suffix('.valid.json');durable.write_json(marker,{'valid':False,'step':u})
    path.unlink(missing_ok=True)
    torch.save({'bundle':pwt.make_bundle(m,pwt.BASE_SHA,{'step':u,'identity':identity}),'optimizer':opt.state_dict(),
                'step':u,'identity':identity,'log':log,'CPU_rng':torch.get_rng_state(),'CUDA_rng':torch.cuda.get_rng_state()},path)
    with path.open('rb') as f:os.fsync(f.fileno())
    durable.write_json(marker,{'valid':True,'step':u,'sha256':d.sha(path)})


def train(device):
    im=verify_input();projection=json.loads((OUT/'launch_projection.json').read_text());assert projection['eligible']
    resource.setrlimit(resource.RLIMIT_CPU,(int(projection['train_CPU_limit_seconds']),int(projection['train_CPU_limit_seconds'])+5))
    signal.alarm(int(projection['train_wall_limit_seconds']));c=d.ctx();g,ss=sources()
    manifest=json.loads((OUT/'models_manifest.json').read_text()) if (OUT/'models_manifest.json').exists() else {'models':{},'input_manifest_sha256':d.sha(OUT/'input_manifest.json')}
    assert manifest['input_manifest_sha256']==d.sha(OUT/'input_manifest.json')
    for f in (0,1):
        F=t.Fold(c,0,f);frames,px=images(F);assert px['marker_pixels_sha256']==im['folds'][str(f)]['marker_pixels_sha256']
        for arm in ARMS:
            rd=OUT/'train'/f'{arm}_f{f}';rd.mkdir(parents=True,exist_ok=True)
            if sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())>projection['namespace_peak_limit_bytes']:raise RuntimeError('namespace peak budget reached')
            if (rd/'manifest.json').exists():
                entry=json.loads((rd/'manifest.json').read_text())
                for fn,k in [('final.pt','final_bundle_sha256'),('head.pt','head_sha256'),('updates.private.jsonl','logs_sha256'),('fit_scores.npz','fit_score_cache_sha256')]:assert d.sha(rd/fn)==entry[k]
                assert entry['steps']==600 and entry['plan_sha256']==F.psha and entry['source_sha256']==pwt.SOURCE_SHA[arm]
                durable.validate_journal([json.loads(x) for x in (rd/'updates.private.jsonl').read_text().splitlines()],600,inflight=(rd/'update.inflight.json').exists())
                manifest['models'][f'{arm}_f{f}']=entry;continue
            if (rd/'final.pt').exists():raise RuntimeError('incomplete finalization; verify durable600resume before repair, no repeated training')
            m=model(g,ss,arm,device);opt=opt_for(m);live={n for n,p in m.named_parameters() if p.requires_grad}
            initial={n:v.detach().cpu().clone() for n,v in m.state_dict().items()};identity=f'{arm}_f{f}_{F.psha}_{pwt.SOURCE_SHA[arm]}'
            log=[];begin=0;resume=rd/'resume.pt';valid=rd/'resume.valid.json';journal=rd/'updates.private.jsonl';inflight=rd/'update.inflight.json'
            disklog=[json.loads(x) for x in journal.read_text().splitlines()] if journal.exists() else []
            if resume.exists():
                marker=json.loads(valid.read_text());assert marker['valid'] and marker['sha256']==d.sha(resume)
                st=torch.load(resume,map_location='cpu',weights_only=True);assert st['identity']==identity
                pwt.restore_bundle(m,st['bundle'],pwt.BASE_SHA);opt.load_state_dict(st['optimizer']);begin=st['step'];log=st['log']
                durable.validate_journal(disklog,begin,inflight=inflight.exists());assert disklog==log
                torch.set_rng_state(st['CPU_rng']);torch.cuda.set_rng_state(st['CUDA_rng'])
            else:durable.validate_journal(disklog,0,inflight=inflight.exists())
            if begin==0:
                ff0,s0=full_fit(m,F,frames,device);assert np.array_equal(s0,np.full(len(s0),-2,np.float32))
                d.dump(rd/'fit_init.json',ff0)
            start=time.time()
            for u in range(begin,600):
                durable.write_json(inflight,{'step':u+1,'identity':identity,'before_optimizer_execution':True})
                rec=step(m,opt,F,frames,u,device);log.append(rec)
                with journal.open('a') as fh:fh.write(json.dumps(rec)+'\n');fh.flush();os.fsync(fh.fileno())
                if (u+1)%150==0:save_resume(resume,m,opt,u+1,identity,log)
                inflight.unlink()
                if u==0 or (u+1)%25==0:
                    progress={'arm':arm,'fit_fold':f,'step':u+1,'total_steps':600,'loss':rec['loss'],'occurrences':sum(q['occurrences'] for q in log),
                              'pid':os.getpid(),'device':torch.cuda.get_device_name(0),'updated_epoch':time.time(),'elapsed_model_seconds':time.time()-start}
                    d.dump(OUT/'progress.json',progress);print(json.dumps(progress),flush=True)
            ff1,s1=full_fit(m,F,frames,device)
            final=m.state_dict();frozen=[n for n in final if n not in live]
            assert all(torch.equal(final[n].cpu(),initial[n]) for n in frozen)
            drift={n:float((final[n].cpu()-initial[n]).abs().max()) for n in live};assert all(v>0 for v in drift.values())
            assert all(any(q['gradient_nonzero'][n] for q in log[1:]) for n in live)
            steps=[int(v['step']) for v in opt.state.values()];assert min(steps)==max(steps)==600
            bundle=pwt.make_bundle(m,pwt.BASE_SHA,{'step':600,'arm':arm,'fit_fold':f,'source_sha256':pwt.SOURCE_SHA[arm],'plan_sha256':F.psha})
            torch.save(bundle,rd/'final.pt');torch.save({k:v for k,v in bundle['trainable'].items() if k.startswith('head.')},rd/'head.pt')
            check_m=pwt.restore_bundle(pwt.build(g),torch.load(rd/'final.pt',map_location='cpu',weights_only=True),pwt.BASE_SHA)
            assert pwt.state_sha(check_m.state_dict())==pwt.state_sha(final);del check_m
            np.savez_compressed(rd/'fit_scores.npz',rows=F.need,scores=s1)
            durable.validate_journal([json.loads(x) for x in journal.read_text().splitlines()],600)
            optimizer_steps={n:int(opt.state[p]['step']) for n,p in m.named_parameters() if p.requires_grad}
            durable.write_json(rd/'optimizer_steps.json',optimizer_steps)
            record={'arm':arm,'fit_fold':f,'steps':600,'optimizer_step_min_max':[min(steps),max(steps)],'occurrences':sum(q['occurrences'] for q in log),
                    'training_forwards':sum(q['physical_forwards'] for q in log),'training_backwards':sum(q['physical_backwards'] for q in log),
                    'final_bundle_sha256':d.sha(rd/'final.pt'),'head_sha256':d.sha(rd/'head.pt'),'complete_full_state_sha256':bundle['full_state_sha256'],
                    'frozen_state_sha256':bundle['frozen_state_sha256'],'source_sha256':pwt.SOURCE_SHA[arm],'shared_G_sha256':pwt.BASE_SHA,'plan_sha256':F.psha,
                    'initial_full_state_sha256':pwt.state_sha(initial),'all_selected_gradients_nonzero':True,'drift_by_selected_tensor':drift,'all_frozen_tensors_unchanged':True,
                    'Init':json.loads((rd/'fit_init.json').read_text()),'final_fit':ff1,'logs_sha256':d.sha(rd/'updates.private.jsonl'),'fit_score_cache_sha256':d.sha(rd/'fit_scores.npz'),
                    'optimizer_steps_sha256':d.sha(rd/'optimizer_steps.json'),'input_manifest_sha256':d.sha(OUT/'input_manifest.json')}
            d.dump(rd/'manifest.json',record);manifest['models'][f'{arm}_f{f}']=record
            resume.unlink(missing_ok=True);valid.unlink(missing_ok=True)
            d.dump(OUT/'models_manifest.json',manifest);d.dump(PUB/'models_manifest.json',manifest)
            del m,opt,initial,final,bundle;torch.cuda.empty_cache()


def package_gates(rec):
    gates={};base=rec['cells']
    for arm in ARMS:
        cells=[base[str(h)][arm] for h in (0,1)];bb=[base[str(h)]['Base'] for h in (0,1)]
        dr=lambda key:[cells[h]['sequence_row'][key]-bb[h]['sequence_row'][key] for h in (0,1)]
        g={'S1':sum(x['S1_fixed_count'] for x in cells)>=11 and all(x['S1_fixed_count']>=1 and x['S1_source_count']>=1 for x in cells) and rec['oof'][arm]['S1_source_count']>=3,
           'A4C':sum(x['A4C_harmed_count'] for x in cells)<=14,'net':all(x['net_fixed_count']>=0 for x in cells) and sum(x['net_fixed_count'] for x in cells)>=5,
           'F1':min(dr('segment_f1_50_multi'))>=0 and float(np.mean(dr('segment_f1_50_multi')))>=.001,
           'native':min(dr('native_macro_f1'))>=-.003,'boundary':min(dr('boundary_recall_multi'))>=-.005,
           'short':sum(x['counts']['fn_short'] for x in cells)<=sum(x['counts']['fn_short'] for x in bb)+3,
           'structure':all(x['non_A4C_labels_unchanged'] for x in cells)}
        gates[arm]={**g,'all':all(g.values())}
    contribution={}
    for control in ('G','S'):
        l=[base[str(h)]['L'] for h in (0,1)];b=[base[str(h)][control] for h in (0,1)]
        contribution[control]=all(l[h]['S1_fixed_count']>=b[h]['S1_fixed_count'] and l[h]['A4C_harmed_count']<=b[h]['A4C_harmed_count'] and l[h]['sequence_row']['segment_f1_50_multi']>=b[h]['sequence_row']['segment_f1_50_multi'] for h in (0,1)) and sum(x['S1_fixed_count'] for x in l)>=sum(x['S1_fixed_count'] for x in b)+5 and sum(x['net_fixed_count'] for x in l)>=sum(x['net_fixed_count'] for x in b)+5
    rec['gates']={'absolute':gates,'L_contribution_vs_controls':contribution,'clinical_purpose_increment':gates['L']['all'] and all(contribution.values())}


def cell_identity(man,arm,h):
    entry=man['models'][f'{arm}_f{1-h}']
    return {'model_bundle_sha256':entry['final_bundle_sha256'],'input_manifest_sha256':man['input_manifest_sha256'],'fit_fold':1-h,'heldout_fold':h}


def structural():
    """Historical cache/evaluator parity only, before new formal models or readouts."""
    verify_input();c=d.ctx();a8=t.rsh.a08_ctx();data=t.t31.vpr.training_data(c.a31.a15.a10.a07p)
    old=json.loads((ROOT/'docs/results/2026-10-28_risk_constrained_a5c/readout_s0.json').read_text());cache=np.load(c.out/'readout_cache_s0.npz');report={}
    for h in (0,1):
        F=t.Fold(c,0,h);D=t.d32.fold_data(c.d,0,h);D['M']=t.d32.set_masks(D)
        P={'seqs':D['seqs'],'n_of':dict(data.n),'label_of':dict(data.label)};ev=t.bc.evalset(P,list(range(len(D['seqs']))),f'heldout_f{h}')
        assert np.array_equal(D['M']['S1'],(F.pk['base']==2)&(F.pk['labels']==3))
        for arm in ('Base','C','N'):
            pred=cache[f'run_pred_{arm}_heldout_f{h}'];tab=t.tables_pred(pred,D,a8.cfg,P,ev);ob=old['cells'][f'heldout_f{h}'][arm]
            assert tab['sequence_row']==ob['sequence_row'] and tab['counts']==ob['counts']
            ex,recs=t.a5c_extras(D,pred);assert tab['S1_fixed']==ob['S1_fixed'] and ex['correct_A4C_kept']==ob['correct_A4C_kept']
            if arm!='Base':
                scores=cache[f'scores_{arm}_heldout_f{h}'];rr,pp=durable.reference_labels(F.pk,F.need,scores,F.off)
                assert np.array_equal(pred,pp) and np.max(np.abs(rr[F.el]-cache[f'r_{arm}_heldout_f{h}']))<1e-12
            report[f'{arm}_f{h}']={'full_evaluator_and_counts_equal':True,'source_recordings':len(recs)}
    d.dump(OUT/'structural_checks.json',{'passed':True,'historical_only':True,'new_model_forwards':0,'cells':report});d.dump(PUB/'structural_checks.json',{'passed':True,'historical_only':True,'new_model_forwards':0,'cells':report})


def readout(device):
    complete=OUT/'readout.raw_complete.json'
    if complete.exists():
        assert json.loads(complete.read_text())['cache_sha256']==d.sha(OUT/'readout_cache.npz');return
    if (OUT/'readout_cache.npz').exists():raise RuntimeError('raw cache exists without completion seal; CPU-only recovery required')
    fr=json.loads((OUT/'models_freeze.json').read_text());assert fr['manifest_sha256']==d.sha(OUT/'models_manifest.json')
    man=json.loads((OUT/'models_manifest.json').read_text());assert set(man['models'])=={f'{a}_f{f}' for a in ARMS for f in (0,1)}
    im=verify_input();g,ss=sources();c=d.ctx();cache={};count=0
    for h in (0,1):
        F=t.Fold(c,0,h);frames,px=images(F);assert px['marker_pixels_sha256']==im['folds'][str(h)]['marker_pixels_sha256'];cache[f'rows_f{h}']=F.need
        for arm in ARMS:
            key=f'{arm}_heldout{h}';identity=cell_identity(man,arm,h)
            if durable.claim_cell(OUT/'raw_cells',key,identity):
                rd=OUT/'train'/f'{arm}_f{1-h}';assert d.sha(rd/'final.pt')==identity['model_bundle_sha256']
                bundle=torch.load(rd/'final.pt',map_location='cpu',weights_only=True);m=pwt.restore_bundle(pwt.build(g),bundle,pwt.BASE_SHA).to(device)
                scores,_=score_rows(m,frames,F.need,device)
                durable.finish_cell(OUT/'raw_cells',key,identity,F.need,scores);del m,bundle;torch.cuda.empty_cache()
            rows,scores=durable.load_cell(OUT/'raw_cells',key,identity);assert np.array_equal(rows,F.need)
            cache[f'scores_{arm}_f{h}']=scores;count+=len(rows)
            d.dump(OUT/'readout_progress.json',{'completed_rows':count,'total_rows':31401,'last_cell':key,'pid':os.getpid(),'updated_epoch':time.time()})
    assert count==31401
    with (OUT/'readout_cache.npz').open('xb') as f:np.savez_compressed(f,**cache);f.flush();os.fsync(f.fileno())
    durable.write_json(complete,{'cache_sha256':d.sha(OUT/'readout_cache.npz'),'rows':count,'all_six_cells_durable':True,'before_any_aggregation_or_evaluator':True})


def evaluate():
    if (OUT/'readout.json').exists():raise RuntimeError('aggregate readout already exists')
    complete=json.loads((OUT/'readout.raw_complete.json').read_text());assert complete['cache_sha256']==d.sha(OUT/'readout_cache.npz')
    cache=np.load(OUT/'readout_cache.npz');c=d.ctx();a8=t.rsh.a08_ctx();data=t.t31.vpr.training_data(c.a31.a15.a10.a07p)
    old=json.loads((ROOT/'docs/results/2026-10-28_risk_constrained_a5c/readout_s0.json').read_text())
    rec={'cells':{},'oof':{},'raw_cache_sha256':complete['cache_sha256']};sets={a:[] for a in ARMS};source_sets={a:set() for a in ARMS}
    for h in (0,1):
        F=t.Fold(c,0,h);D=t.d32.fold_data(c.d,0,h);D['M']=t.d32.set_masks(D)
        P={'seqs':D['seqs'],'n_of':dict(data.n),'label_of':dict(data.label)};ev=t.bc.evalset(P,list(range(len(D['seqs']))),f'heldout_f{h}')
        y,b=F.pk['labels'],F.pk['base'];assert np.array_equal(D['pk']['base'],b) and np.array_equal(D['pk']['labels'],y)
        base=t.tables_pred(b,D,a8.cfg,P,ev);bp=base.pop('preds');base.pop('pred');base['boundaries']=t.bounds(D,bp);base.update(t.a5c_extras(D,b)[0])
        ob=old['cells'][f'heldout_f{h}']['Base'];assert base['sequence_row']==ob['sequence_row'] and base['counts']==ob['counts']
        cell={'Base':base,'Base_reproduced':True};assert np.array_equal(cache[f'rows_f{h}'],F.need)
        for arm in ARMS:
            s=cache[f'scores_{arm}_f{h}'];S=np.full(len(F.W['seq']),np.nan,np.float32);S[F.need]=s
            r=t.rc.run_weighted_mean(F.pk,S,F.off,F.el);pred=t.rc.one_way(b,r)
            ref_r,ref_p=durable.reference_labels(F.pk,F.need,s,F.off);assert np.max(np.abs(r[F.el]-ref_r[F.el]))<1e-12 and np.array_equal(pred,ref_p)
            table=t.tables_pred(pred,D,a8.cfg,P,ev);preds=table.pop('preds');table.pop('pred');ex,vids=t.a5c_extras(D,pred);table.update(ex);source_sets[arm]|=vids
            fx=(b==2)&(y==3)&(pred==3);harm=(b==2)&(y==2)&(pred==3)
            table.update({'S1_fixed_count':int(fx.sum()),'A4C_harmed_count':int(harm.sum()),'net_fixed_count':int(fx.sum()-harm.sum()),'S1_source_count':len(vids),
                'non_A4C_labels_unchanged':bool(np.array_equal(pred[b!=2],b[b!=2])),'boundaries':t.bounds(D,preds)})
            cell[arm]=table;sets[arm].append((fx,harm))
        rec['cells'][str(h)]=cell
    for arm in ARMS:
        rec['oof'][arm]={'A5C_fixed':sum(int(x[0].sum()) for x in sets[arm]),'A4C_harmed':sum(int(x[1].sum()) for x in sets[arm]),'S1_source_count':len(source_sets[arm]),
          'F1_mean':float(np.mean([rec['cells'][str(h)][arm]['sequence_row']['segment_f1_50_multi'] for h in (0,1)]))}
    rec['paired']={}
    for a,b in [('L','G'),('L','S'),('S','G')]:
        rec['paired'][a+'_vs_'+b]={kind:{'shared':sum(int((sets[a][h][j]&sets[b][h][j]).sum()) for h in (0,1)),
          a+'_only':sum(int((sets[a][h][j]&~sets[b][h][j]).sum()) for h in (0,1)),b+'_only':sum(int((~sets[a][h][j]&sets[b][h][j]).sum()) for h in (0,1))} for j,kind in enumerate(('fixes','harms'))}
    package_gates(rec);rec['source2_launched']=False
    d.dump(OUT/'readout.json',rec);d.dump(PUB/'readout.json',rec);print(json.dumps({'oof':rec['oof'],'gates':rec['gates']}),flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=('prepare','check','structural','train','readout','evaluate'),required=True);args=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);start,cpu=time.time(),time.process_time()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True,warn_only=False);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    try:
        if args.stage=='prepare':return prepare()
        if args.stage=='structural':return structural()
        if args.stage=='evaluate':return evaluate()
        device=torch.device('cuda');torch.cuda.set_device(0)
        {'check':check,'train':train,'readout':readout}[args.stage](device)
    finally:
        with (OUT/'stages.jsonl').open('a') as f:f.write(json.dumps({'stage':args.stage,'wall_seconds':time.time()-start,'process_CPU_seconds':time.process_time()-cpu,'pid':os.getpid()})+'\n')


if __name__=='__main__':main()
