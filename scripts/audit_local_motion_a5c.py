"""Independent A46 cache/mean/label/evaluator audit with explicit functional CNN parity.

No training, threshold search or second complete held-out inference. The fixed
first/middle/last-row parity subset is label free; all result recounts use the
one formal readout's private scores.
"""
import os

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch.nn import functional as fn

import scripts.train_local_motion_a5c as a


def functional_model(state,x,e):
    # Rebuild math without MotionRepair, its forward, or its Sequential modules.
    for idx,gn,stride,pad in [(0,1,(1,2,2),(1,2,2)),(3,4,(1,2,2),(1,1,1)),(6,7,(1,2,2),(1,1,1))]:
        x=fn.conv3d(x,state[f'motion.{idx}.weight'],state[f'motion.{idx}.bias'],stride=stride,padding=pad)
        x=fn.group_norm(x,4 if idx==0 else 8,state[f'motion.{gn}.weight'],state[f'motion.{gn}.bias'],eps=1e-5)
        x=fn.silu(x)
    x=fn.adaptive_avg_pool3d(x,(1,4,4)).flatten(1)
    x=fn.gelu(fn.linear(x,state['motion.11.weight'],state['motion.11.bias']))
    e=fn.layer_norm(e,(512,),state['appearance.weight'],state['appearance.bias'],eps=1e-5)
    x=fn.gelu(fn.linear(torch.cat([e,x],1),state['fusion.0.weight'],state['fusion.0.bias']))
    return fn.linear(x,state['fusion.2.weight'],state['fusion.2.bias']).flatten()

def direct_input(bank,indices,markers,perms=None):
    # Explicit loops copy original vector/time tiles; no production permutation helper.
    z=np.zeros((len(indices),3,15,56,56),np.float32)
    for n,row in enumerate(indices):
        for tt,j in enumerate(row):
            if j>=0:z[n,:2,tt]=bank[int(j)].astype(np.float32)
        if perms is not None:
            original=z[n,:2].copy()
            for dst,src in enumerate(perms[n]):
                dr,dc=divmod(dst,7);sr,sc=divmod(int(src),7)
                z[n,:2,:,dr*8:(dr+1)*8,dc*8:(dc+1)*8]=original[:,:,sr*8:(sr+1)*8,sc*8:(sc+1)*8]
        z[n,2,min(int(markers[n]),14)]=1
    return z

def auc(pos,neg):return float(((pos[:,None]>neg[None,:]).sum()+.5*(pos[:,None]==neg[None,:]).sum())/(len(pos)*len(neg)))

def audit(ctx):
    r=json.loads((a.OUT/'readout.json').read_text());w=json.loads((a.OUT/'weights.json').read_text());flow=json.loads((a.OUT/'flow.json').read_text());prep=json.loads((a.OUT/'preparation.json').read_text());cache=np.load(a.OUT/'private_readout.npz',allow_pickle=False)
    assert a.sha(a.OUT/'private_readout.npz')==r['private_cache_sha256'];a.pins(ctx)
    data=a.t.t31.vpr.training_data(ctx.a31.a15.a10.a07p);basecfg=a.t.rsh.a08_ctx().cfg;allpred={k:[] for k in ['Base','F','S','Z_F','Z_S']};labels=[];out={};sources={k:[] for k in allpred if k!='Base'};cells={};maximum=0.
    for h in [0,1]:
        F=a.t.Fold(ctx,0,h);D=a.t.d32.fold_data(ctx.d,0,h);pk=F.pk;y=pk['labels'];base=pk['base'];inp=np.load(a.OUT/f'input_f{h}.npz');bank=np.load(a.OUT/f'flow_f{h}.npy',mmap_mode='r');pairrefs=json.loads((a.OUT/f'pairrefs_f{h}.json').read_text());assert a.sha(a.OUT/f'input_f{h}.npz')==prep[str(h)]['inputs_sha256'] and a.sha(a.OUT/f'flow_f{h}.npy')==flow['folds'][str(h)]['flow_sha256']
        at={tuple(tuple(x) for x in p):i for i,p in enumerate(pairrefs)};refs=[];all_sessions=set();perms=[]
        for j,row in enumerate(F.need):
            q=F.seqs[int(F.W['seq'][row])];rr=[tuple(q['refs'][int(v)]) for v in F.W['s'][row]];refs.append(rr)
            hh=hashlib.sha256(f'A46|{h}|{int(row)}'.encode()).digest();pp=np.random.default_rng(int.from_bytes(hh[:8],'big')).permutation(49);perms.append(pp);assert np.array_equal(pp,inp['perms'][j])
            for tt in range(15):
                x,z=rr[tt:tt+2];valid=x[0]==z[0] and float(x[2])==float(z[2]) and int(z[1])-int(x[1])==3
                expected=at[(x,z)] if valid else -1;assert int(inp['pairs'][j,tt])==expected
            all_sessions.update(a.t.session_of(v[0]) for v in rr)
        assert np.array_equal(inp['rows'],F.need) and np.array_equal(inp['plan'],F.pl)
        # Independent nearest-centre sample expansion, retaining earlier centre on ties.
        maps={}
        for qi,q in enumerate(F.seqs):
            rows=np.flatnonzero(F.W['seq']==qi);cent=F.W['centre'][rows];maps[qi]=rows[np.abs(np.arange(len(q['labels']))[:,None]-cent[None,:]).argmin(axis=1)]
        preds={'Base':base.copy()};scoremeans={};memberships=0;emb=a.t.t31.outputs(ctx.a31,0,h)['emb'];parity={}
        for arm in ['F','S']:
            key=f'{arm}_f{1-h}';rd=a.OUT/'train'/key;man=w['models'][key];assert a.sha(rd/'final.pt')==man['final_sha256'] and a.sha(rd/'updates.jsonl')==man['updates_sha256'];logs=[json.loads(v) for v in (rd/'updates.jsonl').read_text().splitlines()];assert [v['u'] for v in logs]==list(range(600));opt=torch.load(rd/'optimizer_steps.pt',map_location='cpu',weights_only=True);assert set(opt['steps'])=={600};assert sum(v['n_occ'] for v in logs)==[14778,15528][1-h]
            state=torch.load(rd/'final.pt',map_location='cuda',weights_only=True);init,_=a.new_model(1-h,'cuda');initial=init.state_dict();assert len(state)==man['state_tensors'] and all(torch.isfinite(v).all() and not torch.equal(v,initial[k]) for k,v in state.items());del init
            select=np.array([0,len(F.need)//2,len(F.need)-1]);x=direct_input(bank,inp['pairs'][select],F.W['marker'][F.need[select]],np.asarray(perms)[select] if arm=='S' else None);e=torch.as_tensor(np.asarray(emb[F.need[select]],np.float32),device='cuda');xt=torch.tensor(x,device='cuda')
            for zero in [False,True]:
                name=f'Z_{arm}' if zero else arm
                if zero:xt[:,:2]=0
                with torch.no_grad():ss=functional_model(state,xt,e).cpu().numpy()
                err=float(np.max(np.abs(ss-cache[f'score_{name}_h{h}'][select])));maximum=max(maximum,err);assert err<1e-4;parity[name]=err
                means=np.full(len(base),-np.inf);pred=base.copy();lookup={int(row):float(s) for row,s in zip(F.need,cache[f'score_{name}_h{h}'])}
                for i in np.flatnonzero(base==2):
                    qi=int(pk['seq'][i]);aa,bb=int(pk['a'][i]),int(pk['b'][i]);sample_rows=maps[qi][aa:bb];rows,counts=np.unique(sample_rows,return_counts=True);lo,hi=pk['offsets'][i:i+2]
                    assert np.array_equal(rows,pk['rows'][lo:hi]-F.off) and np.max(abs(counts/(bb-aa)-pk['weights'][lo:hi]))<1e-12;memberships+=1
                    means[i]=sum(lookup[int(rr)] for rr in sample_rows)/(bb-aa)
                    if means[i]>0:pred[i]=3
                assert np.max(abs(means[base==2]-cache[f'r_{name}_h{h}'][base==2]))<1e-10 and np.array_equal(pred,cache[f'pred_{name}_h{h}']);assert np.array_equal(pred[base!=2],base[base!=2]);preds[name]=pred;scoremeans[name]=means
        P={'seqs':D['seqs'],'n_of':dict(data.n),'label_of':dict(data.label)};ev=a.t.bc.evalset(P,list(range(len(D['seqs']))),f'heldout_f{h}');cell={}
        for name,pred in preds.items():
            ps={str(q['id']):np.full(len(q['labels']),-99,np.int64) for q in F.seqs}
            for i in range(len(pred)):
                q=F.seqs[int(pk['seq'][i])];ps[str(q['id'])][int(pk['a'][i]):int(pk['b'][i])]=pred[i]
            assert all((v>=0).all() for v in ps.values());sr=a.t.t31.seq_rows(basecfg,P,ps,ev);counts=a.t.rsh.counts_of(F.seqs,sr);published=r['cells'][f'heldout_f{h}'][name];assert sr['row']==published['sequence_row'] and counts==published['counts']
            fix=(base==2)&(y==3)&(pred==3);harm=(base==2)&(y==2)&(pred==3);rec=set()
            for i in np.flatnonzero(fix):
                q=F.seqs[int(pk['seq'][i])];aa,bb=int(pk['a'][i]),int(pk['b'][i]);rec.update(q['refs'][j][0] for j in range(aa,bb) if int(q['labels'][j])==3)
            assert int(fix.sum())==published['S1_fixed']['fixed'] and int(harm.sum())==published['correct_A4C_kept']['runs']-published['correct_A4C_kept']['kept']
            if name!='Base':sources[name].append(rec);assert len(rec)==published['S1_fixed_sources']['true_class_recordings']
            original_bound=sum(int(np.count_nonzero(np.diff(D['base_preds'][str(q['id'])]))) for q in F.seqs);final_bound=sum(int(np.count_nonzero(np.diff(ps[str(q['id'])]))) for q in F.seqs);assert original_bound-final_bound==published['boundaries']['deleted_by_merge']
            cell[name]={'fix':int(fix.sum()),'harm':int(harm.sum()),'net':int(((base!=y)&(pred==y)).sum()-((base==y)&(pred!=y)).sum()),'F1':sr['row']['segment_f1_50_multi'],'native':sr['row']['native_macro_f1'],'boundary_recall':sr['row']['boundary_recall_multi'],'short':counts['fn_short'],'source_count':len(rec),'deleted':original_bound-final_bound};allpred[name].append(pred)
        pair={}
        for typ,mask in [('fix',(base==2)&(y==3)),('harm',(base==2)&(y==2))]:
            ff=mask&(preds['F']==3);ss=mask&(preds['S']==3);pair[typ]={'shared':int((ff&ss).sum()),'F_only':int((ff&~ss).sum()),'S_only':int((ss&~ff).sum())}
        out[str(h)]={'memberships_explicitly_recounted':memberships,'all_pairs_and_permutations_reconstructed':True,'functional_network_parity':parity,'paired_F_S':pair,'heldout_auroc':{name:auc(means[(base==2)&(y==3)],means[(base==2)&(y==2)]) for name,means in scoremeans.items()},'run_labels_disagree_F_S':int((preds['F']!=preds['S']).sum()),'input_source_sessions':len(all_sessions)};cells[h]=cell;labels.append(y)
    yy=np.concatenate(labels);bb=np.concatenate(allpred['Base']);oof={}
    for name,pp in allpred.items():
        p=np.concatenate(pp);fx=int(((bb!=yy)&(p==yy)).sum());ha=int(((bb==yy)&(p!=yy)).sum());v={'fixed':fx,'broken':ha,'net':fx-ha,'A5C_fixed':int(((bb==2)&(yy==3)&(p==3)).sum()),'correct_A4C_harmed':int(((bb==2)&(yy==2)&(p==3)).sum()),'F1_mean':float(np.mean([cells[h][name]['F1'] for h in [0,1]])),'missed_short':sum(cells[h][name]['short'] for h in [0,1]),'boundaries_deleted':sum(cells[h][name]['deleted'] for h in [0,1])};assert v==r['oof'][name];oof[name]=v
    gates={}
    for name in ['F','S']:
        cc=[cells[h][name] for h in [0,1]];bc=[cells[h]['Base'] for h in [0,1]];d=[c['F1']-b['F1'] for c,b in zip(cc,bc)]
        g={'S1_ok':sum(c['fix'] for c in cc)>=11,'source_ok':len(set.union(*sources[name]))>=3 and all(c['fix']>=1 and c['source_count']>=1 for c in cc),'A4C_ok':sum(c['harm'] for c in cc)<=14,'nonA4C_invariant':True,'net_per_fold_ok':all(c['net']>=0 for c in cc),'net_oof_ok':oof[name]['net']>=5,'F1_per_fold_ok':all(v>=0 for v in d),'F1_mean_ok':sum(d)/2>=.001,'native_ok':all(c['native']-b['native']>=-.003 for c,b in zip(cc,bc)),'boundary_recall_ok':all(c['boundary_recall']-b['boundary_recall']>=-.005 for c,b in zip(cc,bc)),'short_misses_ok':sum(c['short'] for c in cc)<=sum(b['short'] for b in bc)+3,'not_noop':any(not np.array_equal(p,b) for p,b in zip(allpred[name],allpred['Base']))};g['passed']=all(g.values());assert g==r['gates'][name];gates[name]=g
    ff=[cells[h]['F'] for h in [0,1]];ss=[cells[h]['S'] for h in [0,1]];con={'fix_each_fold':all(f['fix']>=s['fix'] for f,s in zip(ff,ss)),'fix_plus5_oof':oof['F']['A5C_fixed']-oof['S']['A5C_fixed']>=5,'harms_no_more_each_fold':all(f['harm']<=s['harm'] for f,s in zip(ff,ss)),'net_plus5_oof':oof['F']['net']-oof['S']['net']>=5,'F1_each_fold':all(f['F1']>=s['F1'] for f,s in zip(ff,ss)),'absolute_F':gates['F']['passed']};con['passed']=all(con.values());assert con==r['representation_contribution']
    a.pins(ctx)
    return {'ok':True,'folds':out,'oof':oof,'gates':gates,'representation_contribution':con,'full_network_subset_max_abs':maximum,'input_model_cache_shas_verified':True,'all_four_600_optimizer_counters_verified':True,'every_state_tensor_independently_moved':True,'audit_code_sha256':a.sha(Path(__file__)),'private_score_cache_sha256':a.sha(a.OUT/'private_readout.npz'),'written':a.t.now()}

def main():
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    st=time.time();started=a.t.now();ok=False
    try:r=audit(a.a45.context());a.publish('independent_audit.json',r);ok=True;print(json.dumps({'ok':True,'oof':r['oof'],'folds':r['folds']},indent=1))
    finally:
        a.t.ledger_append(a.OUT/'ledger.jsonl','A46 independent cache/evaluator/subset audit actual GPU stage wall','gpu',started,time.time()-st,ok,'Small fixed label-free parity subset only, no repeated full inference; full result recount from once cache')
        a.t.ledger_append(a.OUT/'ledger.jsonl','A46 independent audit process CPU incl imports','cpu',started,time.process_time(),ok,'Measured process CPU seconds')
if __name__=='__main__':main()
