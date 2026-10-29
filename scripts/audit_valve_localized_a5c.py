"""Independent A48 marker, frozen-map, functional-head and once-cache audit.

Never train or repeat full held-out inference. Only the fixed first/middle/last
window rows are freshly forwarded for parity; all outcomes use saved scores.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import json
import math
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch.nn import functional as fn
import scripts.train_valve_localized_a5c as a


def functional_model(state,x,e):
    x=fn.layer_norm(x.permute(0,2,3,1),(1024,),state['tokens.weight'],state['tokens.bias'],eps=1e-5).permute(0,3,1,2)
    x=fn.gelu(fn.conv2d(x,state['spatial.0.weight'],state['spatial.0.bias'])).flatten(1)
    x=fn.gelu(fn.linear(x,state['spatial.3.weight'],state['spatial.3.bias']))
    e=fn.layer_norm(e,(512,),state['appearance.weight'],state['appearance.bias'],eps=1e-5)
    x=fn.gelu(fn.linear(torch.cat([x,e],1),state['fusion.0.weight'],state['fusion.0.bias']))
    return fn.linear(x,state['fusion.2.weight'],state['fusion.2.bias']).flatten()


def auc(pos,neg):
    return float(((pos[:,None]>neg[None,:]).sum()+.5*(pos[:,None]==neg[None,:]).sum())/(len(pos)*len(neg)))


def softplus_reference(x):
    """Independent stable formula, including torch's beta1/threshold20 branch.

    logaddexp stays finite for extreme finite inputs. At20 the nonlinear
    branch remains active, including its derivative.
    """
    return torch.where(x>20,x,torch.logaddexp(torch.zeros_like(x),x))


def audit(ctx,source):
    a.pins(ctx,source)
    result=json.loads((a.OUT/f'readout_s{source}.json').read_text());weights=json.loads((a.OUT/f'weights_s{source}.json').read_text());prep=json.loads((a.OUT/f'preparation_s{source}.json').read_text());source_man=json.loads((a.OUT/'sources.json').read_text());cache=np.load(a.OUT/f'private_readout_s{source}.npz',allow_pickle=False)
    assert a.sha(a.OUT/f'private_readout_s{source}.npz')==result['private_cache_sha256']
    assert a.sha(a.OUT/'sources.json')==weights['source_manifest_sha256']
    for arm in ['R','G']:assert a.sha(a.SRC/f'backbone_{arm}.pt')==source_man['sources'][arm]['backbone_sha256']
    data=a.t.t31.vpr.training_data(ctx.a31.a15.a10.a07p);cfg=a.t.rsh.a08_ctx().cfg
    allpred={k:[] for k in ['Base','R','G']};labels=[];out={};sources={k:[] for k in ['R','G']};cells={};maximum=0.;all_sessions=[]
    for h in [0,1]:
        F=a.t.Fold(ctx,source,h);D=a.t.d32.fold_data(ctx.d,source,h);pk=F.pk;yy=pk['labels'];base=pk['base'];inp=np.load(a.OUT/f'input_s{source}_f{h}.npz');refs=json.loads((a.OUT/f'refs_s{source}_f{h}.private.json').read_text());assert a.sha(a.OUT/f'input_s{source}_f{h}.npz')==prep[str(h)]['inputs_sha256'] and a.sha(a.OUT/f'refs_s{source}_f{h}.private.json')==prep[str(h)]['references_sha256']
        # Direct original reference indexing, without the production window/marker helper.
        distinct=[];index={};sess=set()
        for j,row in enumerate(F.need):
            q=F.seqs[int(F.W['seq'][row])];slot=int(F.W['marker'][row]);ref=tuple(q['refs'][int(F.W['s'][row,slot])])
            if ref not in index:index[ref]=len(distinct);distinct.append(ref)
            assert inp['index'][j]==index[ref] and tuple(refs[index[ref]])==ref and inp['markers'][j]==slot
        assert [tuple(v) for v in refs]==distinct and np.array_equal(inp['rows'],F.need) and np.array_equal(inp['plan'],F.pl)
        sess={a.t.session_of(r[0]) for q in F.seqs for r in q['refs']};all_sessions.append(sess)
        # Expand every sample to its nearest original centre, earlier centre on ties.
        near={}
        for qi,q in enumerate(F.seqs):
            rows=np.flatnonzero(F.W['seq']==qi);cent=F.W['centre'][rows]
            near[qi]=rows[np.abs(np.arange(len(q['labels']))[:,None]-cent[None,:]).argmin(1)]
        fit=a.t.Fold(ctx,source,1-h);expected_updates=[]
        for planrow in fit.pl:
            rr=[]
            for i in planrow:
                lo,hi=fit.pk['offsets'][int(i):int(i)+2];rr.extend((fit.pk['rows'][lo:hi]-fit.off).tolist())
            expected_updates.append(np.asarray(rr,np.int64))
        emb=a.t.t31.outputs(ctx.a31,source,h)['emb'];preds={'Base':base.copy()};means_by_arm={};parity={};membership_count=0
        for arm in ['R','G']:
            key=f'{arm}_s{source}_f{1-h}';rd=a.OUT/'train'/key;man=weights['models'][key]
            assert a.sha(rd/'final.pt')==man['final_sha256'] and a.sha(rd/'updates.jsonl')==man['updates_sha256'] and a.sha(rd/'optimizer_steps.pt')==man['optimizer_receipt_sha256']
            logs=[json.loads(v) for v in (rd/'updates.jsonl').read_text().splitlines()];assert [v['u'] for v in logs]==list(range(600))
            opt=torch.load(rd/'optimizer_steps.pt',map_location='cpu',weights_only=True);assert set(opt['steps'])=={600} and opt['final_update']==600
            assert sum(v['n_occ'] for v in logs)==sum(len(v) for v in expected_updates)==man['occurrences']
            for log,rows in zip(logs,expected_updates):
                u=log['u'];rr,v,slots=fit.occ(u)
                assert log['rows_sha256']==a.digest(rows)==a.digest(rr) and log['n_occ']==len(rows) and log['pid']==man['pid'] and log['device']=='NVIDIA L40S'
                assert log['weights_sha256']==a.digest(v) and log['slots_sha256']==a.digest(slots)
                assert log['first_occurrences']==log['second_occurrences']==len(rows) and log['replay_max_abs']<=1e-4
            state=torch.load(rd/'final.pt',map_location='cpu',weights_only=True);initial,_=a.new_model(source,1-h,arm,device='cpu')
            names=[n for n,z in initial.named_parameters() if z.requires_grad];assert opt['parameter_names']==names and len(opt['steps'])==len(names)
            assert all(any(n not in v['zero_grad_tensors'] for v in logs[1:]) for n in names)
            for k,v in state.items():
                assert torch.isfinite(v).all() and a.digest(initial.state_dict()[k].numpy())==man['initial_tensor_sha256'][k]
                assert (not torch.equal(v,initial.state_dict()[k])) if k in names else torch.equal(v,initial.state_dict()[k])
            del initial
            # Fixed label-free first/middle/last original windows. Fresh licensed backbone,
            # explicit normalization/resize/pooling and functional head; no full inference repeat.
            select=np.asarray([0,len(F.need)//2,len(F.need)-1]);images=[]
            for j in select:
                row=F.need[j];ref=refs[int(inp['index'][j])];vid,frame,gamma=ref;paths=F.fp(vid)
                images.append(a.t.jv.load_frame_u8(paths[min(int(frame),len(paths)-1)],round(float(gamma),6)))
            xx=torch.tensor(np.stack(images),device='cuda').permute(0,3,1,2).float()/255
            xx=(xx-xx.new_tensor([.485,.456,.406])[None,:,None,None])/xx.new_tensor([.229,.224,.225])[None,:,None,None]
            xx=fn.interpolate(xx,size=(800,800),mode='bilinear',align_corners=False)
            full=a.official(arm).cuda();full.backbone.load_state_dict({k[len('backbone.'):]:v for k,v in state.items() if k.startswith('backbone.')},strict=True)
            with torch.no_grad():
                pyramid=full.backbone(xx);x=torch.cat([fn.adaptive_avg_pool2d(pyramid[str(i)],(14,14)) for i in range(4)],dim=1)
                hs={k[len('head.'):]:v.cuda() for k,v in state.items() if k.startswith('head.')};e=torch.tensor(np.asarray(emb[F.need[select]],np.float32),device='cuda');ss=functional_model(hs,x,e).cpu().numpy()
            err=float(abs(ss-cache[f'score_{arm}_h{h}'][select]).max());maximum=max(maximum,err);assert err<1e-4
            parity[arm]={'independent_normalization_resize_four_scale_head_max_abs':err,'full_source_original_loader':True}
            # Independently differentiable scalar formula and functional head in float64.
            prod=a.vl.ValveHead().cuda().double();prod.load_state_dict(hs);xd=x.detach().double();ed=e.double();ww=torch.tensor([1.,0.,1.,1.,1.],device='cuda',dtype=torch.float64);slots=torch.tensor([0,1,1,2,3],device='cuda');take=torch.tensor([0,1,1,2,2],device='cuda');prod.zero_grad(set_to_none=True);loss,_=a.cp.loss_from_scores(prod(xd[take],ed[take]),ww,slots);loss.backward();ga=torch.cat([z.grad.flatten() for z in prod.parameters()]);ind={k:v.detach().double().requires_grad_() for k,v in hs.items()};s=functional_model(ind,xd[take],ed[take]);rr=torch.stack([sum(s[j]*ww[j] for j in range(5) if int(slots[j])==k) for k in range(4)]);ll=softplus_reference(-rr[0])+softplus_reference(rr[1:]).mean();exact=torch.logaddexp(rr.new_zeros(()),-rr[0])+torch.logaddexp(rr.new_zeros(3),rr[1:]).mean();matherr=float(abs(ll.detach()-exact.detach()));assert matherr<=4.2e-9;grads=torch.autograd.grad(ll,list(ind.values()));gb=torch.cat([z.flatten() for z in grads]);gerr=float(torch.linalg.vector_norm(ga-gb)/torch.linalg.vector_norm(ga));diag={'gradient_relative':gerr,'gradient_max_abs':float((ga-gb).abs().max()),'production_gradient_norm':float(torch.linalg.vector_norm(ga)),'functional_gradient_norm':float(torch.linalg.vector_norm(gb)),'head_forward_max_abs':float((prod(xd[take],ed[take])-s).abs().max()),'loss_abs_difference':float(abs(loss.detach()-ll.detach())),'exact_math_loss_difference':matherr,'max_abs_fixed_subset_score':float(s.detach().abs().max())};a.write(a.OUT/f'gradient_diagnostic_s{source}_h{h}_{arm}.private.json',diag);assert gerr<1e-10,json.dumps(diag);parity[arm]['functional_head_gradient_relative']=gerr;parity[arm]['exact_math_loss_difference']=matherr
            del full,pyramid,prod,ind,state,hs,x,xd,ed,ga,gb;torch.cuda.empty_cache()
            runmeans=np.full(len(base),-np.inf);pred=base.copy();lookup={int(row):float(s) for row,s in zip(F.need,cache[f'score_{arm}_h{h}'])}
            for i in np.flatnonzero(base==2):
                qi=int(pk['seq'][i]);aa,bb=int(pk['a'][i]),int(pk['b'][i]);samples=near[qi][aa:bb];rows,ct=np.unique(samples,return_counts=True);lo,hi=pk['offsets'][i:i+2]
                assert np.array_equal(rows,pk['rows'][lo:hi]-F.off) and np.max(abs(ct/(bb-aa)-pk['weights'][lo:hi]))<1e-12;membership_count+=1
                runmeans[i]=sum(lookup[int(row)] for row in samples)/(bb-aa)
                if runmeans[i]>0:pred[i]=3
            assert np.max(abs(runmeans[base==2]-cache[f'r_{arm}_h{h}'][base==2]))<1e-10 and np.array_equal(pred,cache[f'pred_{arm}_h{h}'])
            assert np.array_equal(pred[base!=2],base[base!=2]);preds[arm]=pred;means_by_arm[arm]=runmeans;torch.cuda.empty_cache()
        P={'seqs':D['seqs'],'n_of':dict(data.n),'label_of':dict(data.label)};ev=a.t.bc.evalset(P,list(range(len(D['seqs']))),f'heldout_f{h}');cell={}
        for name,pred in preds.items():
            seqpred={str(q['id']):np.full(len(q['labels']),-99,np.int64) for q in F.seqs}
            for i in range(len(pred)):
                q=F.seqs[int(pk['seq'][i])];seqpred[str(q['id'])][int(pk['a'][i]):int(pk['b'][i])]=pred[i]
            assert all((v>=0).all() for v in seqpred.values());sr=a.t.t31.seq_rows(cfg,P,seqpred,ev);counts=a.t.rsh.counts_of(F.seqs,sr);published=result['cells'][f'heldout_f{h}'][name];assert sr['row']==published['sequence_row'] and counts==published['counts']
            fix=(base==2)&(yy==3)&(pred==3);harm=(base==2)&(yy==2)&(pred==3);recordings=set()
            for i in np.flatnonzero(fix):
                q=F.seqs[int(pk['seq'][i])];aa,bb=int(pk['a'][i]),int(pk['b'][i]);recordings.update(q['refs'][j][0] for j in range(aa,bb) if int(q['labels'][j])==3)
            assert int(fix.sum())==published['S1_fixed']['fixed'] and int(harm.sum())==published['correct_A4C_kept']['runs']-published['correct_A4C_kept']['kept']
            if name!='Base':sources[name].append(recordings);assert len(recordings)==published['S1_fixed_sources']['true_class_recordings']
            oldbound=sum(int(np.count_nonzero(np.diff(D['base_preds'][str(q['id'])]))) for q in F.seqs);newbound=sum(int(np.count_nonzero(np.diff(seqpred[str(q['id'])]))) for q in F.seqs);assert oldbound-newbound==published['boundaries']['deleted_by_merge']
            cell[name]={'fix':int(fix.sum()),'harm':int(harm.sum()),'net':int(((base!=yy)&(pred==yy)).sum()-((base==yy)&(pred!=yy)).sum()),'F1':sr['row']['segment_f1_50_multi'],'native':sr['row']['native_macro_f1'],'boundary_recall':sr['row']['boundary_recall_multi'],'short':counts['fn_short'],'source_count':len(recordings),'deleted':oldbound-newbound};allpred[name].append(pred)
        paired={}
        for typ,mask in [('fix',(base==2)&(yy==3)),('harm',(base==2)&(yy==2))]:
            p=mask&(preds['R']==3);g=mask&(preds['G']==3);paired[typ]={'shared':int((p&g).sum()),'R_only':int((p&~g).sum()),'G_only':int((g&~p).sum())}
        out[str(h)]={'every_marker_reference_reconstructed':True,'eligible_memberships_explicitly_expanded':membership_count,'all600_optimizer_and_actual_rows_verified':True,'paired_R_G':paired,'functional_and_feature_parity':parity,'heldout_A5C_vs_correctA4C_AUROC':{arm:auc(means[(base==2)&(yy==3)],means[(base==2)&(yy==2)]) for arm,means in means_by_arm.items()},'run_label_disagreements_R_G':int((preds['R']!=preds['G']).sum()),'sessions':len(sess),'S1_denominator':int(((base==2)&(yy==3)).sum()),'correctA4C_denominator':int(((base==2)&(yy==2)).sum())};cells[h]=cell;labels.append(yy)
    assert not all_sessions[0]&all_sessions[1]
    yy=np.concatenate(labels);bb=np.concatenate(allpred['Base']);oof={}
    for name,pp in allpred.items():
        pred=np.concatenate(pp);fix=int(((bb!=yy)&(pred==yy)).sum());harm=int(((bb==yy)&(pred!=yy)).sum());v={'fixed':fix,'broken':harm,'net':fix-harm,'A5C_fixed':int(((bb==2)&(yy==3)&(pred==3)).sum()),'correct_A4C_harmed':int(((bb==2)&(yy==2)&(pred==3)).sum()),'F1_mean':float(np.mean([cells[h][name]['F1'] for h in [0,1]])),'missed_short':sum(cells[h][name]['short'] for h in [0,1]),'boundaries_deleted':sum(cells[h][name]['deleted'] for h in [0,1])};assert v==result['oof'][name];oof[name]=v
    gates={}
    for name in ['R','G']:
        cc=[cells[h][name] for h in [0,1]];bc=[cells[h]['Base'] for h in [0,1]];df=[c['F1']-b['F1'] for c,b in zip(cc,bc)]
        g={'S1_ok':sum(c['fix'] for c in cc)>=math.ceil(.10*sum(out[str(h)]['S1_denominator'] for h in [0,1])),'source_ok':len(set.union(*sources[name]))>=3 and all(c['fix']>=1 and c['source_count']>=1 for c in cc),'A4C_ok':sum(c['harm'] for c in cc)<=math.floor(.01*sum(out[str(h)]['correctA4C_denominator'] for h in [0,1])),'nonA4C_invariant':all(np.array_equal(p[b!=2],b[b!=2]) for p,b in zip(allpred[name],allpred['Base'])),'net_per_fold_ok':all(c['net']>=0 for c in cc),'net_oof_ok':oof[name]['net']>=5,'F1_per_fold_ok':all(v>=0 for v in df),'F1_mean_ok':sum(df)/2>=.001,'native_ok':all(c['native']-b['native']>=-.003 for c,b in zip(cc,bc)),'boundary_recall_ok':all(c['boundary_recall']-b['boundary_recall']>=-.005 for c,b in zip(cc,bc)),'short_misses_ok':sum(c['short'] for c in cc)<=sum(b['short'] for b in bc)+3,'not_noop':any(not np.array_equal(p,b) for p,b in zip(allpred[name],allpred['Base']))};g['passed']=all(g.values());assert g==result['gates'][name];gates[name]=g
    pp=[cells[h]['R'] for h in [0,1]];gg=[cells[h]['G'] for h in [0,1]];inc={'fix_each_fold':all(p['fix']>=g['fix'] for p,g in zip(pp,gg)),'fix_plus5_oof':oof['R']['A5C_fixed']>=oof['G']['A5C_fixed']+5,'harms_no_more_each_fold':all(p['harm']<=g['harm'] for p,g in zip(pp,gg)),'net_plus5_oof':oof['R']['net']>=oof['G']['net']+5,'F1_each_fold':all(p['F1']>=g['F1'] for p,g in zip(pp,gg)),'absolute_R':gates['R']['passed']};inc['passed']=all(inc.values());assert inc==result['pretrained_state_increment'];a.pins(ctx,source)
    return {'ok':True,'source':source,'folds':out,'oof':oof,'gates':gates,'pretrained_state_increment':inc,'functional_network_subset_max_abs':maximum,'four_actual600_updates_verified':True,'all_trainable_gradients_and_drift_verified':True,'all_frozen_backbone_and_A08_hashes_unchanged':True,'input_session_isolation':True,'audit_code_sha256':a.sha(Path(__file__)),'private_score_cache_sha256':a.sha(a.OUT/f'private_readout_s{source}.npz'),'written':a.t.now()}


def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--source',type=int,default=0);args=p.parse_args();a.configure();start=time.time();started=a.t.now();ok=False
    try:
        r=audit(a.a45.context(),args.source);a.publish(f'independent_audit_s{args.source}.json',r);ok=True;print(json.dumps(r,indent=1))
    finally:
        a.t.ledger_append(a.OUT/'ledger.jsonl',f'A48 independent audit source{args.source} GPU stage wall','gpu',started,time.time()-start,ok,'fixed label-free parity subsets; no full inference repeat')
        a.t.ledger_append(a.OUT/'ledger.jsonl',f'A48 independent audit source{args.source} process CPU','cpu',started,time.process_time(),ok,'includes imports and full cache/evaluator recount')
if __name__=='__main__':main()
