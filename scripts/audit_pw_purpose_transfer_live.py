#!/usr/bin/env python3
"""Independent model/journal and cache-only full-readout reconstruction for PW-T1-C."""
import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES','0')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from echo_routing.temporal.pw_pilot import make_model
from echo_routing.temporal import pw_transfer_live as ref
from echo_routing.temporal.joint_visual import load_frame_u8
import scripts.train_pw_purpose_transfer_live as run
t=run.t
OUT=run.OUT;PUB=run.PUB;ARMS=('G','S','L')


def canonical(state):
    h=hashlib.sha256()
    for name in sorted(state):
        a=state[name].detach().cpu().contiguous().numpy()
        for b in (name.encode(),str(a.dtype).encode(),str(a.shape).encode(),a.tobytes()):h.update(b)
    return h.hexdigest()


def exact_plan(pk,f):
    positives=np.flatnonzero((pk['base']==2)&(pk['labels']==3));negative=np.flatnonzero((pk['base']==2)&(pk['labels']==2))
    perm=np.random.default_rng([20261028,44,0,f,1000]).permutation(negative)
    return np.asarray([[int(np.random.default_rng([20261028,44,0,f,u]).choice(positives)),*[int(perm[(3*u+k)%len(perm)]) for k in range(3)]] for u in range(600)],np.int64)


def membership(F):
    """Fresh nearest-centre argmin and multiplicities; first tie stays earlier."""
    maps={};count=0
    for i in np.flatnonzero(F.el):
        seq=int(F.pk['seq'][i]);q=F.seqs[seq]
        if seq not in maps:
            rr=np.flatnonzero(F.W['seq']==seq);centres=F.W['centre'][rr]
            maps[seq]=rr[np.abs(np.arange(len(q['labels']))[:,None]-centres[None,:]).argmin(1)]
        a,b=int(F.pk['a'][i]),int(F.pk['b'][i]);counter=Counter(maps[seq][a:b].tolist())
        rr=np.asarray(sorted(counter),np.int64);ww=np.asarray([counter[int(j)]/(b-a) for j in rr])
        lo,hi=F.pk['offsets'][i:i+2]
        assert np.array_equal(rr,F.pk['rows'][lo:hi]-F.off)
        assert np.allclose(ww,F.pk['weights'][lo:hi],atol=1e-12,rtol=0);count+=1
    return count


def models():
    if (OUT/'independent_models.json').exists():raise RuntimeError('model audit already complete; no repeated fit forward')
    start,cpu=time.time(),time.process_time();im=run.verify_input();man=json.loads((OUT/'models_manifest.json').read_text())
    assert set(man['models'])=={f'{a}_f{f}' for a in ARMS for f in (0,1)}
    g=torch.load(run.BASE,map_location='cpu',weights_only=True);assert ref.sha(run.BASE)==im['source_files_sha256']['G']
    base=make_model(g)
    with torch.no_grad():base.head.weight.zero_();base.head.bias.fill_(-2)
    fullG={n:v.clone() for n,v in base.state_dict().items()};live={n for n,p in base.named_parameters() if p.requires_grad};assert len(live)==28
    sources={'G':fullG}
    for arm in ('S','L'):
        p=ROOT/'runs/pw_small_pilot_v1'/f'{arm}.pt';assert ref.sha(p)==im['source_files_sha256'][arm]
        ck=torch.load(p,map_location='cpu',weights_only=True);assert ck['steps']==600
        state=ck['model'];state['head.weight']=torch.zeros_like(state['head.weight']);state['head.bias']=torch.full_like(state['head.bias'],-2)
        assert all(torch.equal(state[n],v) for n,v in fullG.items() if n not in live);sources[arm]=state
    c=run.d.ctx();report={};total_occ=0;fresh=0;max_error=0.
    for f in (0,1):
        F=t.Fold(c,0,f);pl=exact_plan(F.pk,f);assert np.array_equal(pl,F.pl)
        assert hashlib.sha256(pl.tobytes()).hexdigest()==im['folds'][str(f)]['plan_sha256'];checked=membership(F)
        fitrows=F.need[[0,len(F.need)//2,-1]];frames={}
        for row in fitrows:
            sq=F.seqs[int(F.W['seq'][row])];v,frame,gamma=sq['refs'][int(F.W['s'][row,int(F.W['marker'][row])])];paths=F.fp(v)
            frames[int(row)]=load_frame_u8(paths[min(int(frame),len(paths)-1)],round(float(gamma),6)).copy()
        for arm in ARMS:
            key=f'{arm}_f{f}';rd=OUT/'train'/key;e=man['models'][key]
            for fn,k in [('final.pt','final_bundle_sha256'),('head.pt','head_sha256'),('fit_scores.npz','fit_score_cache_sha256'),('updates.private.jsonl','logs_sha256'),('optimizer_steps.json','optimizer_steps_sha256')]:assert ref.sha(rd/fn)==e[k]
            bundle=torch.load(rd/'final.pt',map_location='cpu',weights_only=True)
            assert bundle['metadata']=={'step':600,'arm':arm,'fit_fold':f,'source_sha256':im['source_files_sha256'][arm],'plan_sha256':im['folds'][str(f)]['plan_sha256']}
            assert set(bundle['trainable'])==live and bundle['base_sha256']==im['source_files_sha256']['G']
            complete={n:v for n,v in fullG.items()};complete.update(bundle['trainable'])
            assert canonical(complete)==bundle['full_state_sha256']==e['complete_full_state_sha256']
            assert canonical({n:v for n,v in complete.items() if n not in live})==bundle['frozen_state_sha256']==e['frozen_state_sha256']
            assert canonical(sources[arm])==e['initial_full_state_sha256']
            drift={n:float((complete[n]-sources[arm][n]).abs().max()) for n in live};assert all(v>0 and v==e['drift_by_selected_tensor'][n] for n,v in drift.items())
            head=torch.load(rd/'head.pt',map_location='cpu',weights_only=True);assert set(head)=={'head.weight','head.bias'} and all(torch.equal(head[n],complete[n]) for n in head)
            optimizer=json.loads((rd/'optimizer_steps.json').read_text());assert set(optimizer)==live and set(optimizer.values())=={600}
            logs=[json.loads(q) for q in (rd/'updates.private.jsonl').read_text().splitlines()];ref.validate_journal(logs,600,inflight=(rd/'update.inflight.json').exists())
            occurrence_total=0
            for u,q in enumerate(logs):
                rows=[]
                for i in pl[u]:
                    a,b=F.pk['offsets'][int(i):int(i)+2];rows.extend((F.pk['rows'][a:b]-F.off).tolist())
                n=len(rows);occurrence_total+=n
                assert q['occurrences']==n and q['physical_forwards']==2*n and q['physical_backwards']==n
                assert q['plan_row_sha256']==hashlib.sha256(pl[u].tobytes()).hexdigest()
                assert q['occurrence_rows_sha256']==hashlib.sha256(np.asarray(rows,np.int64).tobytes()).hexdigest()
                assert np.isfinite(q['loss']) and np.isfinite(q['gradient_norm']) and q['replay_max_abs']<=1e-5
            assert all(any(q['gradient_nonzero'][n] for q in logs[1:]) for n in live)
            assert e['steps']==600 and e['occurrences']==occurrence_total==im['folds'][str(f)]['training_occurrences']
            assert e['training_forwards']==2*occurrence_total and e['training_backwards']==occurrence_total
            assert e['plan_sha256']==im['folds'][str(f)]['plan_sha256'] and e['source_sha256']==im['source_files_sha256'][arm]
            fit=np.load(rd/'fit_scores.npz');assert np.array_equal(fit['rows'],F.need)
            means,pred=ref.reference_labels(F.pk,F.need,fit['scores'],F.off);pos=(F.pk['base']==2)&(F.pk['labels']==3);neg=(F.pk['base']==2)&(F.pk['labels']==2)
            assert int((pred[pos]==3).sum())==e['final_fit']['positive_recovered'] and int((pred[neg]==3).sum())==e['final_fit']['protected_harmed']
            m=make_model(g);m.load_state_dict(complete,strict=True);m.to('cuda').eval()
            with torch.no_grad():
                for row in fitrows:
                    x=torch.from_numpy(frames[int(row)]).permute(2,0,1)[None].to('cuda').float()/255
                    x=(x-torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None])/torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None]
                    val=float(m(x)[0]);expected=float(fit['scores'][np.flatnonzero(F.need==row)[0]]);err=abs(val-expected)
                    assert err<=1e-4;max_error=max(max_error,err);fresh+=1
            del m;torch.cuda.empty_cache();total_occ+=occurrence_total
            report[key]={'updates':600,'all28selected_tensors_changed':True,'all_frozen_tensors_exact_G':True,'complete_model_reconstruction_and_source_SHA_pass':True,'optimizer_steps_all600':True,
                'occurrences':occurrence_total,'true_forwards':2*occurrence_total,'true_backwards':occurrence_total,'all_gradients_seen':True,'all_eligible_memberships_independently_checked':checked,'final_fit_recomputed':True}
    assert total_occ==90918 and not list((OUT/'train').rglob('resume.pt'))
    assert ref.sha(ROOT/'models_frozen/video_within_view_v1/all_s0/final.pt')==im['base_A08_sha256']
    result={'independent_model_audit_pass':True,'models':report,'total_updates':3600,'total_occurrences':total_occ,'total_training_forwards':2*total_occ,'total_training_backwards':total_occ,
        'fit_only_reconstruction_forwards':fresh,'fit_reconstruction_max_abs_error':max_error,'new_heldout_forwards':0,'A08_unchanged':True,'wall_seconds':time.time()-start,'process_CPU_seconds':time.process_time()-cpu}
    run.d.dump(OUT/'independent_models.json',result);run.d.dump(PUB/'independent_models.json',result);print(json.dumps({k:v for k,v in result.items() if k!='models'}),flush=True)


def final():
    start,cpu=time.time(),time.process_time();run.verify_input();rec=json.loads((OUT/'readout.json').read_text());modelaudit=json.loads((OUT/'independent_models.json').read_text());assert modelaudit['independent_model_audit_pass']
    seal=json.loads((OUT/'readout.raw_complete.json').read_text());assert seal['cache_sha256']==ref.sha(OUT/'readout_cache.npz')==rec['raw_cache_sha256']
    cache=np.load(OUT/'readout_cache.npz');man=json.loads((OUT/'models_manifest.json').read_text());c=run.d.ctx();a8=t.rsh.a08_ctx();data=t.t31.vpr.training_data(c.a31.a15.a10.a07p)
    fixed={a:[] for a in ARMS};harmed={a:[] for a in ARMS};src={a:[] for a in ARMS};metrics={a:[] for a in (*ARMS,'Base')};passed={};n1=n4=members=0
    for h in (0,1):
        F=t.Fold(c,0,h);members+=membership(F);D=t.d32.fold_data(c.d,0,h);P={'seqs':D['seqs'],'n_of':dict(data.n),'label_of':dict(data.label)};ev=t.bc.evalset(P,list(range(len(D['seqs']))),f'heldout_f{h}')
        y,b=F.pk['labels'],F.pk['base'];n1+=int(((b==2)&(y==3)).sum());n4+=int(((b==2)&(y==2)).sum());assert np.array_equal(cache[f'rows_f{h}'],F.need)
        for arm in (*ARMS,'Base'):
            if arm=='Base':pred=b.copy()
            else:
                rows,scores=ref.load_cell(OUT/'raw_cells',f'{arm}_heldout{h}',run.cell_identity(man,arm,h))
                assert np.array_equal(rows,F.need) and np.array_equal(scores,cache[f'scores_{arm}_f{h}'])
                means,pred=ref.reference_labels(F.pk,rows,scores,F.off)
                fixed[arm].append((b==2)&(y==3)&(pred==3));harmed[arm].append((b==2)&(y==2)&(pred==3))
                sources=set()
                for i in np.flatnonzero(fixed[arm][-1]):
                    q=F.seqs[int(F.pk['seq'][i])];aa,bb=int(F.pk['a'][i]),int(F.pk['b'][i])
                    vs={v for v,_,_ in q['refs'][aa:bb] if data.label[v]==int(y[i])}
                    assert vs==set(D['true_src'][i]);sources|=vs
                src[arm].append(sources);cell=rec['cells'][str(h)][arm]
                assert int(fixed[arm][-1].sum())==cell['S1_fixed_count'] and int(harmed[arm][-1].sum())==cell['A4C_harmed_count'] and len(sources)==cell['S1_source_count']
                assert np.array_equal(pred[b!=2],b[b!=2]) and cell['non_A4C_labels_unchanged']
            # Independent direct per-sample replacement, followed by pristine original evaluator.
            pp={q['id']:D['base_preds'][q['id']].copy() for q in D['seqs']}
            for i,label in enumerate(pred):
                q=D['seqs'][int(F.pk['seq'][i])];aa,bb=int(F.pk['a'][i]),int(F.pk['b'][i]);pp[q['id']][aa:bb]=label
            sr=t.t31.seq_rows(a8.cfg,P,pp,ev);counts=t.rsh.counts_of(D['seqs'],sr);cell=rec['cells'][str(h)][arm]
            assert sr['row']==cell['sequence_row'] and counts==cell['counts']
            metrics[arm].append((sr['row'],counts))
            total=sum(int((np.diff(pp[q['id']])!=0).sum()) for q in D['seqs']);basebounds=sum(int((np.diff(D['base_preds'][q['id']])!=0).sum()) for q in D['seqs'])
            assert cell['boundaries']=={'base_boundaries':basebounds,'final_boundaries':total,'deleted_by_merge':basebounds-total}
        for arm in ARMS:
            assert rec['cells'][str(h)][arm]['net_fixed_count']==int(fixed[arm][-1].sum()-harmed[arm][-1].sum())
    assert (n1,n4,members)==(103,1413,1520)
    for arm in ARMS:
        fx=[int(x.sum()) for x in fixed[arm]];hm=[int(x.sum()) for x in harmed[arm]];df=[metrics[arm][h][0]['segment_f1_50_multi']-metrics['Base'][h][0]['segment_f1_50_multi'] for h in (0,1)]
        g={'S1':sum(fx)>=11 and min(fx)>=1 and min(len(x) for x in src[arm])>=1 and len(set.union(*src[arm]))>=3,'A4C':sum(hm)<=14,
           'net':min(a-b for a,b in zip(fx,hm))>=0 and sum(fx)-sum(hm)>=5,'F1':min(df)>=0 and np.mean(df)>=.001,
           'native':min(metrics[arm][h][0]['native_macro_f1']-metrics['Base'][h][0]['native_macro_f1'] for h in (0,1))>=-.003,
           'boundary':min(metrics[arm][h][0]['boundary_recall_multi']-metrics['Base'][h][0]['boundary_recall_multi'] for h in (0,1))>=-.005,
           'short':sum(q[1]['fn_short'] for q in metrics[arm])<=sum(q[1]['fn_short'] for q in metrics['Base'])+3,'structure':True}
        g={k:bool(v) for k,v in g.items()};g['all']=all(g.values());passed[arm]=g;assert g==rec['gates']['absolute'][arm]
        oof=rec['oof'][arm];assert oof['A5C_fixed']==sum(fx) and oof['A4C_harmed']==sum(hm) and oof['S1_source_count']==len(set.union(*src[arm]))
        assert oof['F1_mean']==float(np.mean([v[0]['segment_f1_50_multi'] for v in metrics[arm]]))
    for a,b in [('L','G'),('L','S'),('S','G')]:
        for kind,ss in [('fixes',fixed),('harms',harmed)]:
            both=onlya=onlyb=0
            for h in (0,1):
                both+=int(np.count_nonzero(ss[a][h]&ss[b][h]));onlya+=int(np.count_nonzero(ss[a][h]&~ss[b][h]));onlyb+=int(np.count_nonzero(~ss[a][h]&ss[b][h]))
            assert rec['paired'][a+'_vs_'+b][kind]=={'shared':both,a+'_only':onlya,b+'_only':onlyb}
    increments={}
    for control in ('G','S'):
        lf=[int(q.sum()) for q in fixed['L']];cf=[int(q.sum()) for q in fixed[control]];lh=[int(q.sum()) for q in harmed['L']];ch=[int(q.sum()) for q in harmed[control]]
        increments[control]=all(lf[h]>=cf[h] and lh[h]<=ch[h] and metrics['L'][h][0]['segment_f1_50_multi']>=metrics[control][h][0]['segment_f1_50_multi'] for h in (0,1)) and sum(lf)>=sum(cf)+5 and sum(lf)-sum(lh)>=sum(cf)-sum(ch)+5
    assert increments==rec['gates']['L_contribution_vs_controls'] and bool(passed['L']['all'] and all(increments.values()))==rec['gates']['clinical_purpose_increment']
    result={'independent_final_cache_audit_pass':True,'eligible_runs_checked':members,'A5C_denominator':n1,'correct_A4C_denominator':n4,'all_membership_and_weighted_means_reconstructed':True,'all_raw_cell_model_fit_fold_identities_verified':True,'all_one_way_per_sample_labels_and_full_evaluator_metrics_recomputed':True,'source_attribution_and_paired_fix_harm_overlaps_recomputed':True,'all_absolute_and_increment_gates_recomputed':True,'new_model_forwards':0,'raw_cache_sha256':seal['cache_sha256'],'wall_seconds':time.time()-start,'process_CPU_seconds':time.process_time()-cpu}
    run.d.dump(OUT/'independent_final.json',result);run.d.dump(PUB/'independent_final.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=('models','final'),required=True);args=ap.parse_args()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    {'models':models,'final':final}[args.mode]()
